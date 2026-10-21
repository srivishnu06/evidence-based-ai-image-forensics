"""Fail-closed, idempotent API patch for Step 92+93. No checkpoint changes."""
from pathlib import Path
import ast
import datetime
import shutil

ROOT = Path.cwd()
api_path = ROOT / "api_server.py"
if not api_path.is_file():
    raise SystemExit("ERROR: Run from project root with api_server.py")
for path in [
    ROOT / "src/inference/evidence_quality.py",
    ROOT / "src/vlm/structured_observer.py",
]:
    if not path.is_file():
        raise SystemExit(f"ERROR: expected module missing: {path}")
original = api_path.read_text()
if "from src.inference.evidence_quality import summarize_evidence" in original:
    print("Already installed: existing API patch found. No changes made.")
    raise SystemExit(0)

new = original

def replace(old, fresh, name):
    global new
    count = new.count(old)
    if count != 1:
        raise RuntimeError(f"{name}: expected unique insertion point, got {count}")
    new = new.replace(old, fresh, 1)

replace(
    "from src.vlm.report_builder import build_forensic_report\n",
    "from src.vlm.report_builder import build_forensic_report\n"
    "from src.inference.evidence_quality import summarize_evidence\n"
    "from src.vlm.structured_observer import StructuredQwenObserver\n",
    "new imports",
)
replace(
    "return QwenVisualObserver(\n        QwenEvidenceExplainer(model_name=str(candidates[-1]), load_in_4bit=True)\n    )",
    "return StructuredQwenObserver(\n        QwenEvidenceExplainer(model_name=str(candidates[-1]), load_in_4bit=True)\n    )",
    "neutral structured observer",
)
replace(
    '        cv = analysis["cv_result"]\n',
    '        cv = analysis["cv_result"]\n'
    '        review = summarize_evidence(cv)\n',
    "read-only evidence diagnostic",
)
replace(
    '        if include_vlm and report["evidence_available"]:\n',
    '        if include_vlm and report["evidence_available"] and review["has_candidate"]:\n',
    "VLM invocation evidence gate",
)
replace(
    '        if vlm_error:\n            report_text += "\\n\\nVLM STATUS\\n" + vlm_error\n',
    '        if vlm_error:\n            report_text += "\\n\\nVLM STATUS\\n" + vlm_error\n'
    '        if include_vlm and not review["has_candidate"]:\n'
    '            report_text += "\\n\\nVLM STATUS\\nNo flagged category with extracted spatial candidates; visual observation was skipped."\n'
    '        report_text += "\\n\\nEVIDENCE RELIABILITY (AUTOMATED, UNVERIFIED)\\n" + review["summary"]\n',
    "append reliability section to forensic report",
)
replace(
    '            "evidence_regions": int(report["evidence_regions"]),\n',
    '            "evidence_regions": int(report["evidence_regions"]),\n'
    '            "evidence_review": review,\n',
    "export evidence diagnostic in JSON",
)
ast.parse(new, filename=str(api_path))
backup = ROOT / "report_archive" / "api_backups"
backup.mkdir(parents=True, exist_ok=True)
name = "api_server.before_step92_93_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".py"
shutil.copy2(api_path, backup / name)
api_path.write_text(new)
# Make multi-line structured VLM observations legible in the existing React card.
css_path = ROOT / "frontend/src/index.css"
if css_path.is_file():
    css = css_path.read_text()
    anchor = ".vlm-answer{"
    if anchor in css and "white-space:pre-wrap" not in css[css.index(anchor):css.index(anchor)+150]:
        css_backup = backup / ("index_css.before_step92_93_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".css")
        shutil.copy2(css_path, css_backup)
        css_path.write_text(css.replace(anchor, anchor + "white-space:pre-wrap;", 1))
        print("CSS UPDATED for multi-line VLM details; backup:", css_backup)
print("API PATCHED:", api_path)
print("ORIGINAL BACKUP:", backup / name)
print("Checkpoints, model inference, numerical thresholds and frontend not modified.")
