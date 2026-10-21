# Time-Constrained Improvement Plan — Keep Working Baseline

**Recommendation:** freeze the functional React/FastAPI demo and the two deployed checkpoints before changing model behaviour. Do not spend remaining days on unvalidated large-model upgrades. Prioritise measurement and a few reversible small experiments.

## Phase 0: Freeze / preserve baseline (15–30 minutes)

Run `python scripts/archive_report_evidence.py` from the live SageMaker repository to make an experiment-results archive. Download and store the archive off-instance; also preserve the actual 2 deployed checkpoint binaries. Collect `python -m pip freeze` and `npm --prefix frontend ls --depth=0` if necessary; DO NOT publish secrets. Set a visible baseline folder/tag; avoid overwriting deployed checkpoints or model selection values.

## Experiment A: Low-risk evidence gating and selection (first; 1–2 hours)

- Use **existing frozen predictions**, not new training. Identify when the strongest artifact category score is high but localisation is diffuse or the peak score is modest.
- Compare (i) current highest artifact-score category, (ii) only categories whose score exceeds category-specific thresholds, and (iii) variants requiring a minimum localisation signal as a display/abstention criterion.
- Evaluate on a **predeclared sample**; report abstention rate, evidence precision, number of false-looking regions, and impact on annotated-category recall. Do not tune repeatedly on the 138-image internal validation and then call it independent.
- Do not alter Stage 1 verdict based on Stage 2 or Qwen agreement. Do not label any automatically selected region "verified".
- Expected benefit: fewer unsupported interpretations; this is a **product reliability improvement**, not necessarily a higher detector AUROC or Stage 2 spatial AP.

## Experiment B: Better VLM observation protocol (1–2 hours)

- Keep Qwen 3B 4-bit initially. Feed original image plus one or two *unaltered* CV-selected crops, with one succinct, **neutral** question per image, avoiding "there is an artifact" priming.
- Request structured fields: `observed_detail`, `specific_location`, `evidence_for_category` (`supports`/`unclear`/`contradicts`), `uncertainty`. Allow abstention.
- Do not require Qwen to assert an anomaly; never let it change the authenticity verdict. Separate the deterministic CV report from a visibly unverified VLM observation.
- Longer reports (e.g. 80–130 words) should be produced **deterministically** from exact CV fields and a short VLM observation, without turning unsupported text into confident paragraphs.
- On 15–25 *predeclared* examples, manually assess category relevance, crop-grounding, hallucinations, uncertainty and abstention. Include images with no extracted regions; report counts with denominators, not invented quality scores.
- Preserve failing examples (parrot-eye shape, statue head, generic scenes) as evidence of limitations.

## Experiment C: Stage 2 controlled improvement (optional if time, 2–4 hours plus review)

- Primary objective: improve spatial AP and/or evidence precision **without unacceptable rare-category classification regressions**. Stage 2 is limited by small training/validation sample and extremely low positive counts for colour, commonsense and physics.
- Before running, inspect existing `train_evidence_region.py`, `spatial_distribution_loss.py`, and class weights. Try one controlled change only (e.g., mild adjustment to distribution-loss weight or class-weight clipping), same UID split and seed, with new checkpoint filename and separate ClearML task.
- Evaluate on the same declared development metrics (global spatial AP, present-artifact AP, pointing, macro-F1, per-category F1, IoU at the *same* threshold); keep a strict fixed decision rule for deployment.
- Baseline to beat: **selected epoch21** global spatial AP 0.225716, pointing 0.404580, macro F1 0.444938 (checkpoint-comparison protocol). If improvement isn't clear, revert to `deit_region_calibrated.pt`.
- Never replace/recalibrate the known-good deployed checkpoint until the candidate passes evaluation. Avoid repeating many trials on the same small val set.

## Stage 1 improvement: later / only if external labelled data available

- Stage 1 baseline evaluation AUROC 0.945786 overall, 0.874013 on unseen fake generators; cannot infer accuracy on a user-uploaded parrot of unknown provenance.
- Highest-value test: evaluate on a separate real/generated source and under JPEG compression/resizing, reporting AUROC, FPR, accuracy, calibration and subgroup metrics. Keep training and evaluation generator identity disjoint.
- Consider calibration or a fixed-seed lightweight ensemble only if measured on a separate calibration split and evaluated separately. Do not blindly raise the decision threshold based on the parrot.

## Stop / fallback rule

If one low-cost gating experiment and one VLM evaluation do not yield credible, independently reviewable gains, **use the current functional demo as the final version**, with documented limitations and finished report/video. Put study and writeup ahead of nonessential refactoring. Existing results already show a meaningful method comparison; improvements are optional, not guaranteed.
