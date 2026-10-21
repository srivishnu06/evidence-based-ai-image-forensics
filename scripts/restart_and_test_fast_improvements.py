"""Restart only the known single-worker API and check real CV, optionally Qwen."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

import requests

root = Path.cwd()
folder = root / "outputs/api"
folder.mkdir(parents=True, exist_ok=True)
pid_file = folder / "server.pid"
log_file = folder / "server.log"
base = "http://127.0.0.1:8502"
print("=== RESTART BACKEND (PORT 8502) ===", flush=True)
if pid_file.exists():
    pid = int(pid_file.read_text().strip())
    cmd_path = Path(f"/proc/{pid}/cmdline")
    if cmd_path.exists():
        cmd = cmd_path.read_bytes().replace(b"\0", b" ").decode(errors="replace")
        if not all(word in cmd for word in ("uvicorn", "api_server:app", "8502")):
            raise RuntimeError(f"Refusing to stop unrelated PID {pid}: {cmd}")
        os.kill(pid, signal.SIGTERM)
        print("Stopped old API PID:", pid, flush=True)
        for _ in range(40):
            if not cmd_path.exists():
                break
            try:
                with urlopen(base + "/api/health", timeout=.2):
                    time.sleep(.25)
            except Exception:
                break
try:
    with urlopen(base + "/api/health", timeout=2):
        raise RuntimeError("Port 8502 still occupied. Safe-stop refused; examine server PID")
except OSError:
    pass
with log_file.open("a") as log:
    p = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api_server:app", "--host", "127.0.0.1", "--port", "8502", "--workers", "1"],
        cwd=str(root), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
pid_file.write_text(str(p.pid))
for _ in range(60):
    try:
        status = requests.get(base + "/api/health", timeout=2)
        status.raise_for_status()
        break
    except requests.RequestException:
        if p.poll() is not None:
            raise RuntimeError("API exited; see outputs/api/server.log")
        time.sleep(.5)
else:
    raise RuntimeError("API did not become ready; see outputs/api/server.log")
print("New PID:", p.pid, "Health:", status.json(), flush=True)

image_path = root / "outputs/vlm/step78_live_crops/image_17_original.png"
if not image_path.is_file():
    raise RuntimeError(f"Fixed smoke image missing: {image_path}")


def run_analysis(vlm: bool) -> dict:
    with image_path.open("rb") as img:
        response = requests.post(
            base + "/api/analyze",
            files={"file": (image_path.name, img, "image/png")},
            data={"include_vlm": "true" if vlm else "false"},
            timeout=180 if vlm else 80,
        )
    if response.status_code != 200:
        raise RuntimeError(f"Analysis failed HTTP {response.status_code}: {response.text[:800]}")
    return response.json()

print("=== REAL CV REGRESSION TEST ===", flush=True)
r = run_analysis(False)
assert abs(r["ai_probability"] - .9669132828712463) < .003, r["ai_probability"]
assert abs(r["artifact_score"] - .9620969295501709) < .003, r["artifact_score"]
assert r["verdict"] == "AI-generated"
assert len(r["artifact_scores"]) == 7
assert r["evidence_regions"] == 1
assert r["evidence_review"]["human_verified"] is False
assert "EVIDENCE RELIABILITY" in r["report"]
assert r["vlm_invoked"] is False
print("CV UNCHANGED: AI score", round(r["ai_probability"], 5), "artifact", round(r["artifact_score"], 5))
print("Evidence status:", r["evidence_review"]["status"], "peak", r["evidence_review"]["spatial_peak"])
print("CV REGRESSION: PASS", flush=True)

if "--with-vlm" in sys.argv:
    print("=== QWEN STRUCTURED OBSERVATION TEST (OPTIONAL) ===", flush=True)
    v = run_analysis(True)
    print("VLM invoked:", v["vlm_invoked"], "VLM error:", v.get("vlm_error"))
    print("OBSERVATION:\n" + str(v.get("vlm_observation"))[:1900])
    print("AI score unchanged:", abs(v["ai_probability"] - r["ai_probability"]) < 1e-6)
    assert abs(v["ai_probability"] - r["ai_probability"]) < 1e-6
    assert abs(v["artifact_score"] - r["artifact_score"]) < 1e-6
    output = root / "outputs/improvement/step93_vlm_smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    keep = {k: v.get(k) for k in ["verdict", "ai_probability", "artifact_category", "artifact_score", "evidence_review", "vlm_invoked", "vlm_observation", "vlm_error", "inference_seconds"]}
    output.write_text(json.dumps(keep, indent=2))
    print("Saved VLM evaluation evidence:", output)
    if v.get("vlm_error") or not v["vlm_invoked"]:
        print("ATTENTION: CV remains working but new VLM path needs debugging.")
    else:
        print("QWEN INTEGRATION: PASS (visual claims NOT independently verified)")
print("RESULT: STAGE 92+93 SMOKE COMPLETED", flush=True)
