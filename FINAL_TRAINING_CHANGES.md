# Final training corrections

This revision is based on the finalized project plan and the inspected dataset-preparation results.

## Data

- Stage 1 defaults to ClearML Dataset `fa50f8f521e84d869134e01276c31843`.
- Stage 2 defaults to ClearML Dataset `e89e43f6109e4020aac0952af699531b`.
- Training no longer downloads project datasets directly from Hugging Face.
- Dataset parquet files are discovered recursively so the code is robust to ClearML's local-copy folder layout.
- Image values can be decoded from either Hugging Face `Image` objects or parquet `{bytes, path}` structures.

## Stage 1

- Four models supported under one protocol.
- 15 max epochs, patience 4.
- AdamW + 10% warm-up + cosine decay.
- AMP + gradient clipping.
- Mild augmentation only.
- Internal split stratified by label and generator.
- Midjourney + Wukong held out from training.
- AUROC checkpoint selection.
- Internal-validation F1 threshold tuning.
- Best threshold saved in checkpoint.
- Expanded binary metrics and calibration metrics.
- Overall/seen/unseen/per-generator final evaluation.
- CSV/JSON output artifacts.
- Parameters, latency and peak GPU memory logged.

## Stage 2

- ClearML X-AIGD labeled dataset only.
- Multi-label stratified internal validation.
- 30 max epochs, patience 5.
- Automatic artifact-class positive weighting using only the training fold.
- Authenticity head remains frozen.
- Seven artifact thresholds tuned on internal validation.
- Localization threshold tuned on internal validation.
- Thresholds saved in the forensic checkpoint.
- Expanded artifact/localization metrics and per-class result CSV.

## Prototype

- Inference now uses saved validation-tuned thresholds rather than fixed hard-coded thresholds.
