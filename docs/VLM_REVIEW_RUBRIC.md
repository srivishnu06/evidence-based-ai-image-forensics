# VLM Explanation Review Rubric (Proposed; not a measured score)

Review a **fixed prespecified sample** of examples, including failures, authentic images, fake images and missing-evidence cases. Use an assessor who sees original + unaltered crop separately from CV predictions when feasible.

For each response record:

1. `image_id`, ground-truth source and artifact annotation availability.
2. `cv_verdict`, `cv_artifact`, `candidate_region_count`, `vlm_invoked`.
3. `visible_alignment` = 0 (unsupported), 1 (ambiguous/partial), 2 (clearly visually supported).
4. `hallucinated_specifics` = yes/no, with exact unsupported phrase.
5. `specificity` = 0 (generic), 1 (category-level), 2 (specific visible region/feature).
6. `uncertainty_appropriate` = yes/no; especially when no observable defect.
7. `image_quality_or_resolution_issue` = yes/no.
8. `reviewer_note` with concrete visual reason.

Report mean and counts, with number actually reviewed. Inter-rater agreement if possible; do not count Qwen affirming Stage 2 as independent ground truth. Preserve raw responses and all prompts.
