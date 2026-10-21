"""Safely align the frontend's display names with the trained model's seven classes."""
from pathlib import Path

path = Path("frontend/src/App.tsx")
if not path.is_file():
    raise SystemExit("frontend/src/App.tsx not found; run from the project root")
source = path.read_text()
replacements = {
    "'Color inconsistency'": "'Colour inconsistency'",
    "'Commonsense inconsistency'": "'Commonsense violation'",
    "'Physics inconsistency'": "'Physics violation'",
}
updated = source
for old, new in replacements.items():
    if old in updated:
        assert updated.count(old) == 1, f"Unexpected duplicate: {old}"
        updated = updated.replace(old, new)
        print(f"Corrected display label: {old} -> {new}")
    elif new in updated:
        print(f"Already correct: {new}")
    else:
        raise SystemExit(f"Frontend label missing, no changes saved: {old}")
if updated != source:
    backup = path.with_name("App.tsx.before_model_labels.bak")
    if backup.exists():
        raise SystemExit("Backup exists; refusing to overwrite")
    backup.write_text(source)
    path.write_text(updated)
    print("Saved label changes; original preserved at", backup)
print("Frontend category labels now match Python model outputs")
