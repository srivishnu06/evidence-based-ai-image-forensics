# Recorded Experimental Results — Report Reference

**Status:** Collected from user-provided SageMaker logs through 20 October 2026. These are records of the experiments, **not newly reproduced measurements**. Original outputs in the SageMaker repository are authoritative.

## Research goal and separation of responsibilities

- Stage 1: classify image as AI-generated or authentic, using a **separate** DeiT-Tiny detector.
- Stage 2: predict multiple artifact categories and localise **candidate** evidence using 7 × 14 × 14 patch maps.
- Stage 3: Qwen2.5-VL-3B-Instruct gives an **experimental, fallible** visual observation from the original image and an unaltered crop.
- Neither Stage 2 nor the VLM is a causal explanation of Stage 1, and none proves origin.

## Stage 1 — Tiny-GenImage

Internal training 21,600 images; internal validation 2,400. Generator IDs 4 (Midjourney) and 8 (Wukong) were withheld from Stage 1 training. Detector checkpoint selected on internal validation.

| Detector | Best epoch | Internal val AUROC | Internal val accuracy | Internal FPR |
|---|---:|---:|---:|---:|
| ResNet18 | 9 | 0.9654 | 0.9042 | 0.1021 |
| EfficientNet-B0 | 11 | 0.9550 | 0.8871 | 0.1164 |
| MobileNetV3-Large | 10 | 0.9614 | 0.8938 | 0.1057 |
| DeiT-Tiny | 2 | 0.9725 | 0.9187 | 0.0743 |

**Separate Stage 1 evaluation (recorded earlier; metrics are not from internal validation):**

| Detector | Overall AUROC | Accuracy | Unseen-generator AUROC |
|---|---:|---:|---:|
| ResNet18 | 0.9261 | 0.8526 | 0.8375 |
| EfficientNet-B0 | 0.9082 | 0.8261 | 0.8010 |
| MobileNetV3-Large | 0.9160 | 0.8434 | 0.8071 |
| DeiT-Tiny | 0.9458 | 0.8766 | 0.8740 |

**Deployed:** `checkpoints/deit_tiny_detector.pt`, internal validation threshold **0.514108**. Evaluated probabilities are classifier scores and should not be presented as calibrated probabilities of true provenance.

## Stage 2 — X-AIGD evidence detection

UID-grouped split: 780 training, 138 internal validation from 918 labelled training images. Separate labelled-test pool: 2,419; some labelled-test results were inspected during development.

| Model / selection | Global spatial AP | Present-artifact AP | Pointing accuracy | Artifact macro F1 | All-map IoU @ 0.10 |
|---|---:|---:|---:|---:|---:|
| Original BCE (best val loss, epoch 13) | 0.0914 | 0.2508 | 0.3053 | 0.4322 | 0.0834 |
| Conv decoder (best val loss, epoch 11) | 0.1953 | 0.2818 | 0.2214 | 0.4168 | 0.1501 |
| Region-distribution (best val loss, epoch 11) | 0.1994 | 0.3367 | 0.3244 | 0.4181 | 0.1435 |
| Region-distribution (best spatial AP, epoch 21) | 0.2257 | 0.3706 | 0.4046 | 0.4449 | 0.1593 |

**Recorded relative gain:** selected region model spatial AP vs original BCE: **147.1%** (relative change on the recorded validation comparison, not a blind generalisation claim).

**Do not mix two metric families:** the training summaries report tuned macro F1 of 0.5073 (original), 0.5096 (conv), and 0.5124 (region best-loss). The checkpoint-comparison files instead give 0.4322, 0.4168 and 0.4181 (and 0.4449 for the best-spatial-AP region model). These results require their respective selection/threshold protocols to be reported separately; they are not contradictory raw measurements of the same declared protocol.

### Deployed region-aware checkpoint: per-category validation results

| Category | Positive images | Spatial AP | Present-artifact AP | Pointing accuracy | Artifact F1 |
|---|---:|---:|---:|---:|---:|
| Edge / shape | 91 | 0.3597 | 0.4181 | 0.4505 | 0.8454 |
| Texture | 43 | 0.0861 | 0.2786 | 0.3023 | 0.4200 |
| Colour | 7 | 0.0241 | 0.3983 | 0.2857 | 0.2222 |
| Symbol / text | 32 | 0.2330 | 0.4243 | 0.5312 | 0.7222 |
| Semantics | 67 | 0.1909 | 0.4018 | 0.4328 | 0.5873 |
| Commonsense | 13 | 0.0097 | 0.1778 | 0.1538 | 0.2222 |
| Physics | 9 | 0.0070 | 0.1643 | 0.2222 | 0.0952 |

**Deployed weights:** `checkpoints/deit_region_calibrated.pt` derive from the best-spatial-AP checkpoint at epoch **21**; category thresholds `[0.53, 0.16, 0.53, 0.82, 0.13, 0.57, 0.2]`, localisation threshold **0.14**. The 0.14 threshold was selected using the same internal validation split, with recorded all-map F1 ≈ **0.292** and IoU ≈ **0.171**; these are not independent test-set measures.

### Interpretation and limitations

- Stage 2 is a **multi-label** task; a selected top category can differ from a category annotated for the image. Reporting top-category match across a 7-image convenience sample is not a formal top-1 model accuracy.
- Severe rarity: positive support was only 7 for colour, 9 for physics and 13 for commonsense. Per-category metrics for these are unstable.
- Physics artifact classification F1 deteriorated to 0.0952 on the selected spatial checkpoint, despite a small spatial AP/pointing increase. Report this trade-off.
- Global spatial AP aggregates many background patches; present-artifact AP has a different denominator. Do not equate AP with accuracy.
- The displayed heatmap is at absolute probability scale, but visible region borders do not establish an actual defect or causally explain the Stage 1 decision.

## Qwen VLM review — qualitative, not a validated score

| Case | CV observation | Qwen response | Human review |
|---|---|---|---|
| Sports poster (val 5) | Flagged symbol/text; selected top/bottom lettering patches | Inconsistent across prompt versions, occasionally denying text irregularity | Unreliable; lettering difficult to read, no proof of origin |
| Building (val 17) | Edge/shape candidate | Once confused red overlay with building colour; later claimed a Buddha statue was headless | Unsupported speculation; Qwen now receives an unaltered crop |
| Parrot user upload | Stage 1 AI score 0.6659; Stage 2 semantic 0.6548, 4 categories flagged, 2 regions | “The parrot's eye is not round.” | Visual support not established; true provenance unknown |
| 7-image category probe | Qwen invoked 5, skipped 2 because no threshold-supported region | Several generic descriptions; one possible flower-edge irregularity | Diagnostic convenience sample, not a faithful population-level measure |

**Recommendation for final report:** treat the Qwen component as an exploratory, unverified explanation generator; preserve the exact classifier verdict; include manual explanation-quality evaluation and representative failures. Longer text is not the same as faithfulness.

## Reproducibility / caveats that must remain in the final report

1. Stage 1 internal validation, official evaluation and withheld-generator subsets are distinct; label the split for each metric.
2. Stage 2 uses a UID-grouped 780/138 split, with validation reused for checkpoint selection/calibration.
3. The X-AIGD labelled-test data were inspected during development; do **not** describe them as untouched or blind test results.
4. Some stage2 model training-summary macro F1 values were tuned with thresholds and differ from checkpoint-comparison metrics. Report thresholding and model selection explicitly.
5. VLM qualitative results do not establish actual image provenance, and no aggregate independent VLM faithfulness score has been measured.
6. A genuine independent external validation is needed before claiming robust open-world AI detection.

## Authoritative source locations on AWS

- `outputs/stage1/*_full/summary.json`; `outputs/stage1_evaluation/*/per_generator_metrics.csv`
- `outputs/stage2/deit_forensics_full/summary.json`; `outputs/stage2/deit_spatial_decoder_full/summary.json`; `outputs/stage2/deit_region_full/summary.json`
- `outputs/stage2/region_checkpoint_comparison.json`, `region_per_category_validation.json`, `spatial_decoder_validation_comparison.json`, `region_threshold_calibration.json`
- `outputs/stage2/region_visual_qa_7_categories.png`; `outputs/vlm/step75_category_review/results.json`
- `checkpoints/deit_tiny_detector.pt`; `checkpoints/deit_region_calibrated.pt`

The paired `docs/experiment_results.json` and `evidence_uploaded_from_chat/` package allow reproduction of the *record* but do not replace raw SageMaker files or model binaries.

