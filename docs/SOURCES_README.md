# Archived source data and references

- `Pasted text(3).txt` is the user-supplied full Step 90B numeric output (Stage 1 and Stage 2 training summary inspection).
- `region_checkpoint_comparison.json`, `region_per_category_validation.json`, and `spatial_decoder_validation_comparison.json` are structured transcriptions of the Step 90C console output supplied on 2026-10-20 (corresponding authoritative JSON originals live at `outputs/stage2/` in AWS). Numeric values retained from that output; if differences are found, use the on-instance original JSON as source of record.
- `Pasted text(1).txt` and `Pasted text(2).txt` contain user-shared Stage2 smoke and full-train logs.
- Other uploaded PNGs are original screenshots, visual QA figures and crop examples; they do not demonstrate ground truth for uploaded parrot image.
- `docs/experiment_results.json` is the curated cross-experiment summary, distinct from raw experiment JSON.
- **Raw AWS experiment outputs, trained weights, ClearML snapshots and Qwen weights are not transferred to this chat archive**. Use `scripts/archive_report_evidence.py` on AWS to capture small authoritative outputs; back up weights and datasets separately.
