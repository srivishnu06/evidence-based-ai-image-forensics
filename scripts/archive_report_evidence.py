#!/usr/bin/env python3
"""Archive small research evidence from THIS deployed SageMaker repository.

No credentials or checkpoint weight binaries are copied. Produces an archive + SHA256 manifest.
Usage from repo root: python scripts/archive_report_evidence.py
"""
from __future__ import annotations
import hashlib
import json
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'report_archive'
DEST.mkdir(parents=True, exist_ok=True)

# Allowlist only. Do not recursively copy arbitrary data or user uploads.
PATTERNS = [
    'outputs/stage1/*_full/summary.json',
    'outputs/stage1_evaluation/*/per_generator_metrics.csv',
    'outputs/stage2/*_full/summary.json',
    'outputs/stage2/deit_spatial_balanced_8ep/summary.json',
    'outputs/stage2/region_checkpoint_comparison.json',
    'outputs/stage2/region_per_category_validation.json',
    'outputs/stage2/region_threshold_calibration.json',
    'outputs/stage2/spatial_decoder_validation_comparison.json',
    'outputs/stage2/spatial_decoder_per_category_validation.json',
    'outputs/stage2/region_visual_qa_7_categories.png',
    'outputs/stage2_evaluation_original_bce/*.json',
    'outputs/stage2_evaluation_original_bce/*.csv',
    'outputs/vlm/step71_live/step71_summary.json',
    'outputs/vlm/step75_category_review/results.json',
    'outputs/vlm/step75_category_review/manual_review.csv',
    'outputs/vlm/step78_live_crops/summary.json',
    'configs/default.yaml',
    'requirements.txt',
    'README.md',
    'docs/REPORT_RESULTS.md',
    'docs/experiment_results.json',
    'docs/REPORT_EVIDENCE_MANIFEST.md',
    'docs/IMPROVEMENT_PLAN.md',
]
CHECKPOINTS = [
    'checkpoints/deit_tiny_detector.pt',
    'checkpoints/deit_region_calibrated.pt',
    'checkpoints/deit_region_full_best_spatial_ap.pt',
    'checkpoints/deit_forensics.pt',
    'checkpoints/deit_spatial_decoder_full.pt',
]

def sha256(path: Path):
    dig = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            dig.update(block)
    return dig.hexdigest()

files = set()
for pattern in PATTERNS:
    files.update(p for p in ROOT.glob(pattern) if p.is_file())

manifest = {'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'repo_root': str(ROOT), 'note':'Checkpoint hashes are recorded but checkpoint contents are NOT included.',
            'archive_files': [], 'checkpoints_not_copied': [], 'missing': []}
for pattern in PATTERNS:
    if not any(p.is_file() for p in ROOT.glob(pattern)):
        manifest['missing'].append(pattern)
for rel in CHECKPOINTS:
    p = ROOT / rel
    if p.is_file():
        manifest['checkpoints_not_copied'].append({'path':rel, 'size_bytes':p.stat().st_size, 'sha256':sha256(p)})
    else:
        manifest['missing'].append(rel)

stamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
archive = DEST / f'forensics_report_archive_{stamp}.zip'
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for p in sorted(files):
        rel = p.relative_to(ROOT).as_posix()
        size = p.stat().st_size
        if size > 15_000_000:
            print('SKIPPED oversized file:', rel)
            manifest['missing'].append(f'OVERSIZED: {rel}')
            continue
        z.write(p, rel)
        manifest['archive_files'].append({'path':rel, 'size_bytes':size, 'sha256':sha256(p)})
    z.writestr('ARCHIVE_MANIFEST.json', json.dumps(manifest, indent=2))

print('ARCHIVE:', archive)
print('ARCHIVE SIZE MiB:', round(archive.stat().st_size/2**20,2))
print('INCLUDED FILES:', len(manifest['archive_files']))
print('MISSING/EXCLUDED:', len(manifest['missing']))
for s in manifest['missing']:
    print(' NOT FOUND:', s)
print('CHECKPOINT HASHES (binary weights NOT included):', len(manifest['checkpoints_not_copied']))
print('RESULT: REPORT ARCHIVE CREATED — DOWNLOAD ZIP AND BACK UP WEIGHTS SEPARATELY')
