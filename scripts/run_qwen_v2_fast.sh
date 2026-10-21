#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
echo '=== STAGE 93B: REPAIR VLM RESPONSE FORMAT ==='
python scripts/install_qwen_observer_v2.py
python -m pytest -q tests/test_observer_v2.py
python -m py_compile src/vlm/structured_observer.py
echo '=== RESTART API AND TEST REAL CV + QWEN ==='
python scripts/restart_and_test_fast_improvements.py --with-vlm
echo '=== ASSERT STRUCTURED DISPLAY NEVER RETURNS RAW JSON JUNK ==='
python - <<'PY'
import json
from pathlib import Path
file = Path('outputs/improvement/step93_vlm_smoke.json')
response = json.loads(file.read_text())
text = str(response.get('vlm_observation') or '')
assert response.get('vlm_invoked'), 'Qwen observer was not invoked'
assert not response.get('vlm_error'), response['vlm_error']
assert text.startswith('QWEN VISUAL OBSERVATION'), text[:350]
assert 'EVIDENCE STATUS' in text
assert 'Reliability: Unverified' not in text  # old observer response
assert 'I/O{' not in text and '```json' not in text
print('Qwen output format: PASS')
print('Visual text accepted:', 'No dependable visual description' not in text)
print('AUTOMATED FACTUAL VALIDATION: NOT POSSIBLE')
print('=== COMPLETE STAGE 93B ===')
PY
