# Architecture and experiment logic

## Stage 1

Four ImageNet-pretrained detectors are fine-tuned on the same Tiny-GenImage split:

- ResNet18 — conventional residual CNN baseline
- EfficientNet-B0 — efficiency/accuracy CNN
- MobileNetV3-Large — mobile/low-compute CNN
- DeiT-Tiny — lightweight Vision Transformer and Stage-2 forensic backbone

Input is `224 x 224 RGB`.

Midjourney and Wukong fake samples are excluded from the Stage-1 training pool. The remaining training pool is split internally with stratification by `(label, generator)`. The internal split is used for early stopping, checkpoint selection and threshold tuning. The official Tiny-GenImage validation split is held back for final overall/seen/unseen/per-generator evaluation.

Best Stage-1 checkpoint criterion: **internal validation AUROC**.

Binary threshold criterion: **internal validation F1**.

## Why DeiT uses a 14 x 14 evidence grid

DeiT-Tiny uses 16 x 16 patches at 224 x 224 input resolution:

`224 / 16 = 14`

so it produces 196 spatial patch tokens arranged as `14 x 14`. The evidence head predicts seven artifact scores for every token, producing:

`[B, 7, 14, 14]`

## Stage 2

The Stage-1 DeiT checkpoint initializes the forensic model.

Outputs:

1. authenticity: `[B, 2]`
2. artifact type: `[B, 7]`
3. artifact localization: `[B, 7, 14, 14]`

The authenticity head is frozen because X-AIGD labeled examples are AI-generated evidence samples rather than a balanced real/fake training set.

X-AIGD `labeled_train` is split with multi-label stratification to preserve the rare artifact categories as well as possible. The artifact loss uses training-fold positive weights because the inspected dataset is strongly imbalanced. Patch BCE remains unweighted initially; additional patch weighting or Dice loss should only be introduced if validation shows sparse-mask collapse.

Default objective:

`L = 1.0 * BCE_artifact + 0.5 * BCE_localization`

Seven artifact thresholds and one localization threshold are tuned on the internal validation fold and frozen before final `labeled_test` evaluation.

## XAI and VLM

Learned patch evidence maps are the primary spatial evidence. Grad-CAM is secondary attribution for the global authenticity decision. Qwen2.5-VL-3B-Instruct is inference-only and verbalizes already-produced CV evidence; it is not the forensic decision maker.
