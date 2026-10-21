# Evidence-Based AI Image Forensics using Vision-Language Models

UTS 32040 Industry Project — reproducible handover guide (20 October 2026).

This repository implements a **research prototype**, not an image-authenticity certification service.

## System design

1. **Stage 1 (authenticity):** independent DeiT-Tiny detector, trained on Tiny-GenImage. Predicts an AI-generated model score and a binary verdict using the saved classification threshold. Four models were evaluated (ResNet18, EfficientNet-B0, MobileNetV3-Large, DeiT-Tiny).
2. **Stage 2 (candidate evidence):** region-aware DeiT-Tiny with artifact scores for seven categories and 7 × 14 × 14 spatial maps; separately calibrated artifact and localisation thresholds. This is **not a causal explanation of the Stage 1 classifier**.
3. **Stage 3 (optional language):** Qwen2.5-VL-3B-Instruct loaded in 4-bit on a supported NVIDIA GPU. It inspects the original image plus a *colour-unaltered* crop suggested by Stage 2. Observations are experimental and may be wrong.
4. **Presentation:** React + TypeScript + Vite frontend served with a FastAPI Python backend. The application uses React and FastAPI.

**Critical interpretation:** authenticity prediction ≠ model-selected region ≠ independently verified visible defect. A Stage 2 score or Qwen sentence is not proof of AI origin.

## Running in VS Code

**Yes, VS Code can open, edit, run and debug this project.** It is an IDE, not a substitute for the GPU or model files.

### Best option when time is short: use existing AWS SageMaker GPU

Keep the working SageMaker Space and checkpoints. Edit the project using a VS Code remote connection **if your SageMaker setup provides a supported remote/SSH/SSM pathway**; otherwise continue editing in SageMaker JupyterLab and use VS Code for locally copied source. Do not assume standard Remote-SSH works automatically with every SageMaker Studio Space.

In your existing SageMaker terminal (do not reinstall working dependencies unnecessarily):

```bash
cd ~/ai_image_forensics_improved
python -c "import torch, fastapi, cv2; print('CUDA:', torch.cuda.is_available())"
npm --prefix frontend run build
python -m uvicorn api_server:app --host 127.0.0.1 --port 8502 --workers 1
```

Open this relative SageMaker route on the **same Studio hostname**:

```text
/jupyterlab/default/proxy/8502/
```

For the current Space, the URL used during development was:

```text
https://80fomvjle68xuwc.studio.ap-southeast-2.sagemaker.aws/jupyterlab/default/proxy/8502/
```

This URL requires access to the original AWS Studio Space and will not work after the Space is removed/recreated unless its hostname and resources remain accessible.

If the server is already running, do not launch a second instance on port 8502. Check `curl -f http://127.0.0.1:8502/api/health` first. Restart the server after editing `api_server.py`; rebuild React after editing the frontend.

### Fresh local Linux machine with NVIDIA GPU

**Prerequisites:** Python 3.12, Node.js 20+, npm, working NVIDIA driver/CUDA-capable PyTorch environment, disk space for project + checkpoints + Qwen (~7.1 GiB weights cache) + dependencies. The project was run on a Tesla T4-class GPU with ~14.5 GiB VRAM. Qwen's 4-bit `bitsandbytes` configuration is not a drop-in solution for macOS/Apple Silicon.

Open the **complete** project folder in VS Code (not just this documentation ZIP). In the VS Code terminal:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# First install compatible CUDA-enabled PyTorch and torchvision
# for your NVIDIA driver following pytorch.org/get-started/locally/.
python -m pip install -r requirements.txt
npm --prefix frontend ci
npm --prefix frontend run build
python -m uvicorn api_server:app --host 127.0.0.1 --port 8502 --workers 1
```

Open `http://127.0.0.1:8502/`. Check `http://127.0.0.1:8502/api/health` first if something fails. Configure VS Code's interpreter as `.venv/bin/python`. On Windows, WSL2 + NVIDIA GPU is the more suitable route for this Linux-centred setup; on macOS, CV-only inference may be possible with platform-specific changes, but **4-bit Qwen is not guaranteed to work unchanged**. No full local installation has been tested in this handover.

> Note: the `requirements.txt` describes a **fresh environment**. Do not blindly run it against the currently working SageMaker environment; its preinstalled package stack has had documented `datasets`/`pyarrow`, `fsspec` and `cv2` compatibility problems after restarts. Check existing versions first.

### Non-Python / frontend development

```bash
npm --prefix frontend ci
npm --prefix frontend run build
```

React's production build is `frontend/dist/`. The server serves the compiled build, so Vite's development server is not needed for the normal one-port setup. The frontend sends `multipart/form-data` to the relative endpoint `api/analyze` (POST fields: `file`, `include_vlm`). The API returns verdict, score, thresholds, seven artifact scores, images encoded as data URLs, optional Qwen text, report and inference time.

## Required model assets

The documentation ZIP **does not include** the trained weights, ClearML datasets or the complete updated project source. Keep or transfer them separately. The essential files for inference are:

```text
checkpoints/deit_tiny_detector.pt
checkpoints/deit_region_calibrated.pt
src/inference/pipeline.py
src/inference/evidence.py
src/vlm/grounded_pipeline.py
src/vlm/report_builder.py
src/vlm/qwen_explainer.py
src/models/deit_forensics_spatial.py
src/constants.py
api_server.py
frontend/src/...
frontend/index.html
frontend/package.json
frontend/package-lock.json
```

Additional checkpoints and training scripts are necessary for reproducing **all experiments**; see `docs/REPORT_RESULTS.md` and `docs/REPORT_EVIDENCE_MANIFEST.md`.

For Qwen, the API currently expects the cached `Qwen/Qwen2.5-VL-3B-Instruct` model under `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/` with two `.safetensors` shards. The API sets `HF_HUB_OFFLINE=1` and **will not automatically download** the model. On a fresh machine, acquire the model lawfully from Hugging Face first (for example with `hf download Qwen/Qwen2.5-VL-3B-Instruct` after logging in if necessary), or copy the complete verified cache; copying arbitrary individual shards without configs/tokenizer is insufficient. Alternatively, keep Qwen disabled.

Qwen is optional: default to CV-only to reduce time and unsupported visual claims. If candidate evidence is absent, Qwen is skipped even if enabled. The API must never let the VLM change Stage 1's decision.

## Running tests and basic health checks

```bash
python -m py_compile api_server.py src/vlm/grounded_pipeline.py src/vlm/report_builder.py
python -m pytest -q
curl -f http://127.0.0.1:8502/api/health
```

`pytest`/CLI tests have **not** been rerun on a fresh copy of this handover; additional dependency and dataset configuration may be necessary. For a real image via API:

```bash
curl -sS -X POST \
  -F 'file=@outputs/vlm/step78_live_crops/image_17_original.png;type=image/png' \
  -F 'include_vlm=false' \
  http://127.0.0.1:8502/api/analyze > /tmp/forensics_response.json
```

The response is large because it includes base64-encoded evidence images. If this image is not present, use any locally available PNG/JPEG/WebP. Upload limit is 20 MB and 20 million pixels; the backend downsizes images larger than 2048 pixels on the longest dimension.

## Datasets (for research and training, not required for a standard upload demo)

Versioned ClearML datasets used during the project:

- Tiny-GenImage dataset ID `fa50f8f521e84d869134e01276c31843`: 28,000 training + 7,000 official validation records. Midjourney and Wukong fake generators withheld from Stage 1 training.
- X-AIGD-Labeled dataset ID `e89e43f6109e4020aac0952af699531b`: 918 labelled training records (UID-grouped 780/138 train/internal validation) and 2,419 labelled test records. The test data was **already inspected during development**, so it is not an untouched final benchmark.

For a fresh training environment, run `clearml-init` when appropriate, provision authorised ClearML access, and verify access to the exact dataset snapshots. **Do not commit `.clearml.conf` or API credentials.** The existing versioned datasets are much more trustworthy than trying to reconstruct new splits.

## Model validation and limitations

The main report-ready summaries are in `docs/REPORT_RESULTS.md`, with machine-readable companion `docs/experiment_results.json` and `docs/metric_table.csv`. These numbers were taken from **saved experimental logs provided by the user**, not independently rerun in this handover environment. Use the original SageMaker JSON/CSV outputs as the authoritative records.

- Stage 1 best internal validation DeiT-Tiny AUROC 0.972527, accuracy 0.91875. Separate official validation overall AUROC 0.945786 and unseen-generator AUROC 0.874013.
- Stage 2 selected epoch 21 region-distribution checkpoint internal validation global spatial AP 0.225716, pointing accuracy 0.404580, artifact macro F1 0.444938. Low spatial AP for commonsense/physics and small positive counts remain significant limitations.
- Threshold tuning, epoch selection and some experimental changes used the same small validation split; avoid calling these numbers a blind evaluation. Distinguish Stage 1 and Stage 2 dataset/test protocols.
- VLM evidence descriptions are inconsistent and sometimes unsupported (e.g., the parrot's eye being 'not round' or a statue allegedly missing its head); mark such statements as **unverified**, never as verified evidence.

## Back up research evidence BEFORE new experiments

From a running SageMaker project root, **after copying `scripts/archive_report_evidence.py` into `scripts/`**:

```bash
python scripts/archive_report_evidence.py
```

This creates a ZIP in `report_archive/` containing experiment summaries, important source/config files, selected charts and a SHA-256 manifest of model checkpoints; it **does not copy checkpoint binaries** (to avoid 332 MB duplication) and does not copy the 7.1 GB Qwen cache. Download the ZIP from JupyterLab and retain your checkpoint binaries separately. Check the archive tool's printed exclusions before treating it as a full reproducibility backup.

## Key project files

| Path | Role |
|---|---|
| `frontend/src/App.tsx`, `frontend/src/index.css` | React evidence dashboard |
| `api_server.py` | FastAPI API and React static file server |
| `src/inference/pipeline.py` | Separate CV verdict and artifact inference |
| `src/inference/evidence.py` | Visualisations and region extraction |
| `src/vlm/grounded_pipeline.py` | Optional Qwen observation from unaltered crop |
| `src/vlm/report_builder.py` | Cautious, deterministic report |
| `src/training/` | Full detector/evidence training variants |
| `src/evaluation/` | Quantitative evaluation logic |
| `checkpoints/` | Model weights (must be preserved) |
| `outputs/stage1*/`, `outputs/stage2*/`, `outputs/vlm/` | Actual experimental evidence |

## Data and secret safety

Do not publish ClearML credentials, AWS keys, `.env` files, private datasets, arbitrary user uploads or the entire Hugging Face cache in a public repository. This research prototype should not be deployed as a public authenticity-verification service without hardening, privacy review, provenance tests and external evaluation.
