"""Upgrade only the Qwen observer, preserving a timestamped backup."""
from pathlib import Path
from datetime import datetime
import shutil
import ast

root = Path.cwd()
target = root / "src/vlm/structured_observer.py"
candidate = root / "src/vlm/structured_observer_v2.py"
api = root / "api_server.py"
assert target.is_file() and candidate.is_file() and api.is_file(), "Missing project files: run in repository root"
assert "StructuredQwenObserver" in api.read_text(), "Stage 92+93 API observer not installed"
source = candidate.read_text()
ast.parse(source)
if 'VERSION = "2026-10-20-plain-observation-v2"' in target.read_text():
    print("Qwen observer v2 already installed; skipping overwrite")
else:
    folder = root / "report_archive/api_backups"
    folder.mkdir(parents=True, exist_ok=True)
    backup = folder / ("structured_observer_before_v2_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".py")
    shutil.copy2(target, backup)
    shutil.copy2(candidate, target)
    print("Backed up previous observer:", backup)
    print("Installed simpler, fail-closed Qwen observer:", target)
print("No model weights, CV thresholds, or React code modified")
