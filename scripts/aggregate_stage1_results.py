from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path("outputs/stage1_evaluation")
MODELS = ["resnet18", "efficientnet_b0", "mobilenet_v3_large", "deit_tiny"]
rows = []

for model in MODELS:
    path = ROOT / model / "final_evaluation.json"
    if not path.exists():
        print(f"Skipping {model}: {path} not found")
        continue

    data = json.loads(path.read_text())
    overall = data["splits"]["overall"]
    seen = data["splits"]["seen"]
    unseen = data["splits"]["unseen"]
    rows.append({
        "model": model,
        "threshold": data["threshold"],
        "accuracy": overall["accuracy"],
        "balanced_accuracy": overall["balanced_accuracy"],
        "precision": overall["precision"],
        "recall": overall["recall"],
        "specificity": overall["specificity"],
        "f1": overall["f1"],
        "auroc": overall["auroc"],
        "average_precision": overall["average_precision"],
        "mcc": overall["mcc"],
        "seen_f1": seen["f1"],
        "seen_auroc": seen["auroc"],
        "unseen_f1": unseen["f1"],
        "unseen_auroc": unseen["auroc"],
    })

if not rows:
    raise SystemExit("No final evaluation JSON files found.")

df = pd.DataFrame(rows)
out = ROOT / "stage1_model_comparison.csv"
df.to_csv(out, index=False)
print(df.to_string(index=False))
print("\nSaved:", out)
