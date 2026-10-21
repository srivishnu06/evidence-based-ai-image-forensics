# ClearML usage

The project uses ClearML for both **dataset versioning** and **experiment tracking**.

## Final dataset IDs

```text
Tiny-GenImage v1.0
fa50f8f521e84d869134e01276c31843

X-AIGD-Labeled v1.0
e89e43f6109e4020aac0952af699531b
```

Training scripts resolve these IDs with `clearml.Dataset.get()` and `get_local_copy()` before loading the parquet files.

## Configure AWS once

```bash
clearml-init
```

Never commit `~/clearml.conf` or API credentials.

## Stage-1 task logging

Each run records:

- model and hyperparameters
- exact ClearML Dataset ID
- train/internal-validation counts
- generator mapping and holdout generators
- losses and LR curve
- Accuracy
- Balanced Accuracy
- Precision
- Recall / Sensitivity
- Specificity
- F1
- AUROC
- Average Precision (AUPRC)
- MCC
- tuned internal-validation threshold
- best epoch/checkpoint
- parameter count
- single-image latency
- peak GPU memory
- `history.csv` and `summary.json`

Final evaluator tasks additionally record overall, seen-generator, unseen-generator and per-generator results.

## Stage-2 task logging

Stage 2 records:

- exact X-AIGD Dataset ID
- multi-label stratification details
- artifact positive counts/weights
- total/artifact/localization losses
- artifact macro/micro F1 during validation
- localization IoU/pixel F1 during validation
- best checkpoint
- seven validation-tuned artifact thresholds
- validation-tuned localization threshold
- final held-out `labeled_test` metrics in a separate testing task
