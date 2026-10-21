#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo '=== 92+93: PATCH EVIDENCE REVIEW AND STRUCTURED QWEN ==='
python scripts/apply_fast_improvements.py

echo '=== VERIFY PYTHON ==='
python -m py_compile api_server.py src/inference/evidence_quality.py src/vlm/structured_observer.py
PYTHONPATH="$PWD" python -m pytest -q tests/test_improvement_protocol.py

echo '=== REBUILD FRONTEND (MULTILINE OBSERVATIONS) ==='
npm --prefix frontend run build

echo '=== SAFELY RESTART PORT 8502 AND TEST REAL MODELS ==='
PYTHONPATH="$PWD" python scripts/restart_and_test_fast_improvements.py --with-vlm

echo '=== BASELINE HASHES ARE UNCHANGED ==='
sha256sum checkpoints/deit_tiny_detector.pt checkpoints/deit_region_calibrated.pt

echo '=== STAGES 92 + 93 COMPLETE; CHECK VLM OUTPUT ABOVE ==='
