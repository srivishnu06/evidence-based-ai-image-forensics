"""Serve the compiled React dashboard and evidence-based CV inference via FastAPI.

Start in project root: python -m uvicorn api_server:app --host 0.0.0.0 --port 8502
All detector and artifact predictions come from the existing saved checkpoints.
"""

import base64
import io
import os
import threading
import time
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "true")
os.environ.setdefault("HF_PARALLEL_LOADING_WORKERS", "2")

import torch
import numpy as np
import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool
from starlette.staticfiles import StaticFiles

from src.constants import ARTIFACT_DISPLAY_NAMES, ARTIFACT_LABELS
from src.inference.pipeline import ForensicsPipeline
from src.vlm.grounded_pipeline import QwenVisualObserver, analyze_image
from src.vlm.qwen_explainer import QwenEvidenceExplainer
from src.vlm.report_builder import build_forensic_report
from src.inference.evidence_quality import summarize_evidence
from src.vlm.structured_observer import StructuredQwenObserver

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "frontend" / "dist"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MAX_DIMENSION = 2048
SUPPORTED_TYPES = {"image/jpeg", "image/png", "image/webp"}
_MODEL_LOCK = threading.Lock()

app = FastAPI(title="Evidence-Based AI Image Forensics", version="1.0.0")


@lru_cache(maxsize=1)
def get_cv_pipeline():
    return ForensicsPipeline(
        checkpoint_path=str(ROOT / "checkpoints" / "deit_region_calibrated.pt"),
        detector_checkpoint_path=str(ROOT / "checkpoints" / "deit_tiny_detector.pt"),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


@lru_cache(maxsize=1)
def get_visual_observer():
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this 4-bit Qwen setup")
    snapshot_dir = (
        Path.home()
        / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots"
    )
    candidates = sorted(
        candidate.parent
        for candidate in snapshot_dir.glob("*/config.json")
        if len(list(candidate.parent.glob("*.safetensors"))) == 2
    )
    if not candidates:
        raise FileNotFoundError("The locally cached Qwen model was not found")
    return StructuredQwenObserver(
        QwenEvidenceExplainer(model_name=str(candidates[-1]), load_in_4bit=True)
    )


def image_data_uri(image):
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")



# The probability map uses a FIXED 0..1 colour scale for every image.
# This never rescales individual images to exaggerate weak patch scores.
def build_visible_evidence_views(image, cv):
    names = [ARTIFACT_DISPLAY_NAMES[label] for label in ARTIFACT_LABELS]
    selected_idx = names.index(cv["strongest_artifact"])
    heat = np.asarray(cv["patch_probabilities"][selected_idx], dtype=np.float32)
    if heat.shape != (14, 14) or not np.isfinite(heat).all():
        raise ValueError("Unexpected or non-finite CV spatial probabilities")
    heat = np.clip(heat, 0, 1)

    # Fixed colour stops from low (navy) to high (red), no per-image scaling.
    levels = np.array([0, .25, .50, .75, 1], dtype=np.float32)
    colours = np.array([
        [19, 43, 82], [46, 136, 183], [99, 197, 191],
        [231, 181, 82], [215, 74, 82],
    ], dtype=np.float32)
    rendered = np.stack([
        np.interp(heat, levels, colours[:, channel])
        for channel in range(3)
    ], axis=-1).astype(np.uint8)
    heat_only = Image.fromarray(rendered, "RGB").resize(
        image.size, Image.Resampling.NEAREST
    )

    # Clear boundaries display patches ABOVE the calibrated spatial threshold.
    # These are probability-map regions, not verified defects.
    cutoff = float(cv["localization_threshold"])
    mask_small = Image.fromarray(
        (heat >= cutoff).astype(np.uint8) * 255, mode="L"
    )
    mask_large = np.asarray(
        mask_small.resize(image.size, Image.Resampling.NEAREST)
    ).copy()
    contours, _ = cv2.findContours(
        mask_large, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    boundaries_bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    thickness = max(2, min(image.size) // 150)
    if contours:
        cv2.drawContours(boundaries_bgr, contours, -1, (15, 23, 40), thickness + 3)
        cv2.drawContours(boundaries_bgr, contours, -1, (65, 230, 255), thickness)
    boundaries = Image.fromarray(
        cv2.cvtColor(boundaries_bgr, cv2.COLOR_BGR2RGB), "RGB"
    )
    return heat_only, boundaries, float(heat.max()), cutoff


def prepare_image(data):
    try:
        with Image.open(io.BytesIO(data)) as opened:
            width, height = opened.size
            if width < 1 or height < 1 or width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail="Image has too many pixels (20 million maximum)",
                )
            opened.load()
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="Cannot decode the uploaded image")

    if max(image.size) > MAX_DIMENSION:
        image.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.Resampling.LANCZOS)
    return image


def analyze_sync(image, include_vlm):
    # Serialize GPU access and prevent simultaneous lazy model initialisations.
    with _MODEL_LOCK:
        start = time.perf_counter()
        analysis = analyze_image(image, get_cv_pipeline(), observer=None)
        cv = analysis["cv_result"]
        review = summarize_evidence(cv)
        report = analysis["forensic_report"]
        observation = None
        vlm_invoked = False
        vlm_error = None

        if include_vlm and report["evidence_available"] and review["has_candidate"]:
            try:
                observation = get_visual_observer().observe(
                    original_image=image,
                    evidence_crop=analysis["evidence_crop"],
                    artifact_category=report["artifact_category"],
                )
                vlm_invoked = bool(observation and observation.strip())
                if not vlm_invoked:
                    observation = None
                report = build_forensic_report(cv, vlm_observation=observation)
            except Exception as exc:
                # The independent CV inference remains available if Qwen fails.
                vlm_error = (
                    "Experimental VLM generation failed ("
                    + type(exc).__name__
                    + "). The independent CV results are unaffected."
                )

        report_text = report["report"]
        if not include_vlm and report["evidence_available"]:
            report_text = report_text.replace(
                "No visual observation is available.",
                "Experimental VLM observation was not requested.",
            )
        if vlm_error:
            report_text += "\n\nVLM STATUS\n" + vlm_error
        if include_vlm and not review["has_candidate"]:
            report_text += "\n\nVLM STATUS\nNo flagged category with extracted spatial candidates; visual observation was skipped."
        report_text += "\n\nEVIDENCE RELIABILITY (AUTOMATED, UNVERIFIED)\n" + review["summary"]

        names = [ARTIFACT_DISPLAY_NAMES[label] for label in ARTIFACT_LABELS]
        scores = cv["artifact_scores"]
        flags = cv["artifact_detected"]
        thresholds = cv["artifact_thresholds"]
        artifacts = [
            {
                "name": name,
                "score": float(scores[name]),
                "threshold": float(thresholds[i]),
                "flagged": bool(flags[name]),
            }
            for i, name in enumerate(names)
        ]

        probability_map, region_boundaries, peak_probability, spatial_cutoff = (
            build_visible_evidence_views(image, cv)
        )

        return {
            "verdict": report["verdict"],
            "ai_probability": float(cv["ai_probability"]),
            "auth_threshold": float(cv["auth_threshold"]),
            "artifact_category": report["artifact_category"],
            "artifact_score": float(cv["strongest_artifact_probability"]),
            "artifact_flagged": bool(report["artifact_flagged"]),
            "evidence_regions": int(report["evidence_regions"]),
            "evidence_review": review,
            "artifact_scores": artifacts,
            "evidence_image": image_data_uri(cv["evidence_overlay"]),
            "evidence_probability_map": image_data_uri(probability_map),
            "evidence_boundaries": image_data_uri(region_boundaries),
            "evidence_peak_probability": peak_probability,
            "localization_threshold": spatial_cutoff,
            "evidence_crop": (
                image_data_uri(analysis["evidence_crop"])
                if analysis["evidence_crop"] is not None
                else None
            ),
            "report": report_text,
            "vlm_observation": observation,
            "vlm_invoked": vlm_invoked,
            "vlm_error": vlm_error,
            "inference_seconds": round(time.perf_counter() - start, 3),
            "processed_image_width": image.width,
            "processed_image_height": image.height,
        }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "frontend_built": (DIST / "index.html").is_file(),
        "cuda_available": torch.cuda.is_available(),
        "cv_loaded": get_cv_pipeline.cache_info().currsize > 0,
        "qwen_loaded": get_visual_observer.cache_info().currsize > 0,
    }


@app.post("/api/analyze")
async def analyze_endpoint(
    file: UploadFile = File(...),
    include_vlm: bool = Form(False),
):
    if file.content_type not in SUPPORTED_TYPES:
        raise HTTPException(status_code=415, detail="Upload a PNG, JPEG or WebP image")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 20 MB limit")
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded image is empty")
    image = prepare_image(data)
    try:
        return await run_in_threadpool(analyze_sync, image, include_vlm)
    except Exception as exc:
        # Do not expose full stack traces to the web client.
        raise HTTPException(
            status_code=500,
            detail=f"CV inference failed: {type(exc).__name__}. Check the server log.",
        ) from exc


@app.get("/forensicslab-icon.svg", include_in_schema=False)
def custom_favicon():
    path = DIST / "forensicslab-icon.svg"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Favicon not built")
    return FileResponse(path, media_type="image/svg+xml")


if (DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=str(DIST / "assets")), name="assets")


@app.get("/")
def frontend_index():
    path = DIST / "index.html"
    if not path.is_file():
        raise HTTPException(status_code=503, detail="React build missing: run npm --prefix frontend run build")
    return FileResponse(path)
