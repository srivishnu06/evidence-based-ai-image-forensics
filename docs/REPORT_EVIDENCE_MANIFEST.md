# Evidence Archive Inventory and Provenance

## What was captured

- User-provided SageMaker training/evaluation log snippets: `Pasted text(1).txt` (Stage 2 smoke), `Pasted text(2).txt` (region full training), `Pasted text(3).txt` (Step 90B summary inspection), and `Pasted text.txt` (earlier environment/dependency troubleshooting). These represent user-supplied console output, not fresh measurements.
- Validation visuals from prior project chats: `xaigd_ground_truth_overlay.png`, `stage2_predicted_evidence_comparison.png`, `stage2_evidence_absolute_scale_verified.png`, `region_visual_qa_7_categories.png`, `category_collage.png`, `crop_row5.png`, `step73_evidence_region_inspection.png`, `unaltered_crop.png`.
- React and Streamlit interface screenshots from the uploaded conversation and saved report examples `forensic_report.txt`, `forensic_analysis_report.txt`.
- Structured numeric record in `docs/experiment_results.json` and human-readable `docs/REPORT_RESULTS.md`.

## Not contained in the documentation package

- Live SageMaker training checkpoints (about 332 MB total), Qwen cache (about 7.1 GB), datasets and original source repository are **not** included. Transfer/back them up separately.
- Saved full per-generator metrics and actual training/evaluation JSON/CSV in AWS are **not automatically copied** from this chat. Use `scripts/archive_report_evidence.py` **on SageMaker** to download those authoritative results in a separate ZIP.
- Raw parrot provenance information is unknown; the parrot screenshots alone cannot be used to score authenticity correctness.

## Verification and report use

- Exact comparisons can be checked against the AWS files listed in `REPORT_RESULTS.md`; the docs include rounded values for display and a structured JSON with source precision.
- Some training-summary macro F1 metrics are tuned using category thresholds; do not mix with separately computed checkpoint-comparison macro F1.
- Visual screenshots are illustrative examples rather than systematic explanation-quality evaluation.
- Bundle generated on 2026-10-20 using only available user-provided records. A checksum manifest is included in the evidence ZIP.
