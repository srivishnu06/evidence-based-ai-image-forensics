"""Apply guarded, reversible React metadata and evidence-view improvements.

Run from ai_image_forensics_improved project root. This script patches ONLY:
frontend/index.html, frontend/src/App.tsx, frontend/src/index.css,
api_server.py, frontend/public/forensicslab-icon.svg.
It never changes research checkpoints, datasets, CV pipeline or VLM sources.
"""
from pathlib import Path
import ast
import re
import shutil
import sys

ROOT = Path.cwd()
FRONTEND = ROOT / 'frontend'
assert (FRONTEND / 'src' / 'App.tsx').is_file(), 'Frontend source missing'
assert (ROOT / 'api_server.py').is_file(), 'FastAPI source missing'


def replace_once(s, old, new, label):
    n = s.count(old)
    if n != 1:
        raise RuntimeError(f'{label}: expected exactly one anchor, found {n}: {old[:100]!r}')
    return s.replace(old, new, 1)

api_file = ROOT / 'api_server.py'
app_file = FRONTEND / 'src' / 'App.tsx'
css_file = FRONTEND / 'src' / 'index.css'
index_file = FRONTEND / 'index.html'
icon_file = FRONTEND / 'public' / 'forensicslab-icon.svg'

api = api_file.read_text()
app = app_file.read_text()
css = css_file.read_text()
index = index_file.read_text()

if 'def build_visible_evidence_views(' in api:
    raise SystemExit('Visual patch appears to be already installed; no changes made.')

api = replace_once(
    api,
    'import torch\n',
    'import torch\nimport numpy as np\nimport cv2\n',
    'FastAPI imports',
)

helper = '''
# The probability map uses a FIXED 0..1 colour scale for every image.
# This never rescales individual images to exaggerate weak patch scores.
def build_visible_evidence_views(image, cv):
    names = [ARTIFACT_DISPLAY_NAMES[label] for label in ARTIFACT_LABELS]
    selected_idx = names.index(cv["strongest_artifact"])
    heat = np.asarray(cv["patch_probabilities"][selected_idx], dtype=np.float32)
    if heat.shape != (14, 14) or not np.isfinite(heat).all():
        raise ValueError("Unexpected or non-finite CV spatial probabilities")
    heat = np.clip(heat, 0, 1)

    # Fixed colour stops from low (navy) to high (red), no per-image scaling.
    levels = np.array([0, .25, .50, .75, 1], dtype=np.float32)
    colours = np.array([
        [19, 43, 82], [46, 136, 183], [99, 197, 191],
        [231, 181, 82], [215, 74, 82],
    ], dtype=np.float32)
    rendered = np.stack([
        np.interp(heat, levels, colours[:, channel])
        for channel in range(3)
    ], axis=-1).astype(np.uint8)
    heat_only = Image.fromarray(rendered, "RGB").resize(
        image.size, Image.Resampling.NEAREST
    )

    # Clear boundaries display patches ABOVE the calibrated spatial threshold.
    # These are probability-map regions, not verified defects.
    cutoff = float(cv["localization_threshold"])
    mask_small = Image.fromarray(
        (heat >= cutoff).astype(np.uint8) * 255, mode="L"
    )
    mask_large = np.asarray(
        mask_small.resize(image.size, Image.Resampling.NEAREST)
    ).copy()
    contours, _ = cv2.findContours(
        mask_large, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    boundaries_bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    thickness = max(2, min(image.size) // 150)
    if contours:
        cv2.drawContours(boundaries_bgr, contours, -1, (15, 23, 40), thickness + 3)
        cv2.drawContours(boundaries_bgr, contours, -1, (65, 230, 255), thickness)
    boundaries = Image.fromarray(
        cv2.cvtColor(boundaries_bgr, cv2.COLOR_BGR2RGB), "RGB"
    )
    return heat_only, boundaries, float(heat.max()), cutoff


'''
api = replace_once(api, '\ndef prepare_image(data):\n', '\n' + helper + 'def prepare_image(data):\n', 'Backend helper injection')
api = replace_once(
    api,
    '        return {\n            "verdict": report["verdict"],',
    '        probability_map, region_boundaries, peak_probability, spatial_cutoff = (\n'
    '            build_visible_evidence_views(image, cv)\n'
    '        )\n\n'
    '        return {\n            "verdict": report["verdict"],',
    'Backend output preparation',
)
api = replace_once(
    api,
    '            "evidence_image": image_data_uri(cv["evidence_overlay"]),',
    '            "evidence_image": image_data_uri(cv["evidence_overlay"]),\n'
    '            "evidence_probability_map": image_data_uri(probability_map),\n'
    '            "evidence_boundaries": image_data_uri(region_boundaries),\n'
    '            "evidence_peak_probability": peak_probability,\n'
    '            "localization_threshold": spatial_cutoff,',
    'Backend response fields',
)
api = replace_once(
    api,
    'if (DIST / "assets").is_dir():',
    '@app.get("/forensicslab-icon.svg", include_in_schema=False)\n'
    'def custom_favicon():\n'
    '    path = DIST / "forensicslab-icon.svg"\n'
    '    if not path.is_file():\n'
    '        raise HTTPException(status_code=404, detail="Favicon not built")\n'
    '    return FileResponse(path, media_type="image/svg+xml")\n\n\n'
    'if (DIST / "assets").is_dir():',
    'Favicon route',
)

app = replace_once(
    app,
    '  evidence_image: string\n',
    '  evidence_image: string\n'
    '  evidence_probability_map: string\n'
    '  evidence_boundaries: string\n'
    '  evidence_peak_probability: number\n'
    '  localization_threshold: number\n',
    'React interface',
)
app = replace_once(
    app,
    "useState<'overlay' | 'crop'>('overlay')",
    "useState<'overlay' | 'probability' | 'boundaries' | 'crop'>('overlay')",
    'React evidence-state union',
)
app = replace_once(
    app,
    "  const visual = selectedView === 'crop' && result?.evidence_crop\n    ? result.evidence_crop\n    : result?.evidence_image ?? null",
    "  const visual = !result ? null\n"
    "    : selectedView === 'crop' && result.evidence_crop ? result.evidence_crop\n"
    "    : selectedView === 'probability' ? result.evidence_probability_map\n"
    "    : selectedView === 'boundaries' ? result.evidence_boundaries\n"
    "    : result.evidence_image",
    'React selected visual',
)
app = replace_once(
    app,
    '<span className="header-badge">CV + VLM</span>',
    '<span className="header-badge">{result?.vlm_invoked ? "CV + VLM" : "CV only"}</span>',
    'Header mode indicator',
)
app = replace_once(
    app,
    "selectedView === 'crop' ? 'Evidence crop' : 'Evidence overlay'",
    "selectedView === 'crop' ? 'Evidence crop' : selectedView === 'probability' ? 'Spatial probability map' : selectedView === 'boundaries' ? 'Thresholded evidence boundaries' : 'Evidence overlay'",
    'Right panel title',
)
old_selector = '''<button type="button" className={selectedView === 'overlay' ? 'current' : ''} onClick={() => setSelectedView('overlay')}>Heatmap overlay</button><button type="button" className={selectedView === 'crop' ? 'current' : ''} onClick={() => setSelectedView('crop')}>Unaltered crop</button>'''
new_selector = '''<button type="button" className={selectedView === 'overlay' ? 'current' : ''} onClick={() => setSelectedView('overlay')}>Overlay</button><button type="button" className={selectedView === 'probability' ? 'current' : ''} onClick={() => setSelectedView('probability')}>Probability map</button><button type="button" className={selectedView === 'boundaries' ? 'current' : ''} onClick={() => setSelectedView('boundaries')}>Region outlines</button><button type="button" className={selectedView === 'crop' ? 'current' : ''} onClick={() => setSelectedView('crop')}>Unaltered crop</button>'''
app = replace_once(app, old_selector, new_selector, 'React view selectors')
app = replace_once(
    app,
    '            <div className="evidence-note"><CircleHelp size={16} />',
    '''            {result && selectedView === 'probability' && !compare && <div className="probability-scale"><span>Fixed spatial probability scale</span><div className="probability-gradient" /><span>0</span><span>1.0</span><strong>Peak: {result.evidence_peak_probability.toFixed(3)}</strong></div>}
            {result && selectedView === 'boundaries' && !compare && <div className="boundary-caption">Yellow outlines mark 14×14 grid cells at or above localisation threshold {result.localization_threshold.toFixed(2)}. These are unverified candidate areas, not exact object boundaries.</div>}
            <div className="evidence-note"><CircleHelp size={16} />''',
    'React scale legend',
)

css += '''
/* Evidence visualisation controls: fixed map scale and transparent semantics. */
.probability-scale{display:flex;align-items:center;gap:11px;margin-top:11px;padding:10px 12px;border:1px solid #e0e8f4;border-radius:10px;background:white;font-size:11px;color:#62758e;flex-wrap:wrap}
.probability-scale>span:first-child{font-weight:700}
.probability-scale strong{font-size:11px;color:#253f65;margin-left:auto}
.probability-gradient{height:11px;min-width:115px;max-width:230px;flex:1;border-radius:6px;background:linear-gradient(90deg,#132b52 0%,#2e88b7 25%,#63c5bf 50%,#e7b552 75%,#d74a52 100%)}
.boundary-caption{margin-top:11px;padding:11px 13px;border-radius:10px;background:#fff9e8;border:1px solid #f3e2a7;font-size:11px;line-height:1.65;color:#705c32}
@media(max-width:650px){.selector{flex-wrap:wrap}.probability-scale strong{margin-left:0}}
'''

# Replace title and favicon only, preserve all other Vite HTML including root element.
index, n_title = re.subn(r'<title>[^<]*</title>', '<title>ForensicsLab | AI Image Forensics</title>', index, count=1)
if n_title != 1:
    raise RuntimeError('Expected a single HTML title element')
index, n_icon = re.subn(
    r'<link\s+rel="icon"[^>]*>',
    '<link rel="icon" type="image/svg+xml" href="./forensicslab-icon.svg" />',
    index, count=1,
)
if n_icon != 1:
    raise RuntimeError('Expected a single Vite favicon link')

favicon = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" role="img" aria-label="ForensicsLab">
<rect width="64" height="64" rx="14" fill="#172944"/>
<path d="M13 26V18q0-5 5-5h8M38 13h8q5 0 5 5v8M51 38v8q0 5-5 5h-8M26 51h-8q-5 0-5-5v-8" fill="none" stroke="#A3C3FF" stroke-width="4" stroke-linecap="round"/>
<circle cx="32" cy="32" r="8" fill="none" stroke="#fff" stroke-width="3"/>
<circle cx="32" cy="32" r="2.5" fill="#F1BA68"/>
</svg>
'''

# Validate source before changing any files.
ast.parse(api, filename=str(api_file))
assert 'evidence_probability_map' in app
assert 'ForensicsLab | AI Image Forensics' in index

updates = [
    (api_file, api),
    (app_file, app),
    (css_file, css),
    (index_file, index),
]
for path, _ in updates:
    backup = path.with_name(path.name + '.before_visual_v1.bak')
    if backup.exists():
        raise SystemExit(f'Existing backup found; refusing to overwrite: {backup}')
if icon_file.exists():
    raise SystemExit(f'Favicon already exists; refusing to overwrite: {icon_file}')

for path, content in updates:
    backup = path.with_name(path.name + '.before_visual_v1.bak')
    shutil.copy2(path, backup)
    path.write_text(content)
    print('UPDATED', path.relative_to(ROOT))

icon_file.parent.mkdir(parents=True, exist_ok=True)
icon_file.write_text(favicon)
print('CREATED', icon_file.relative_to(ROOT))
print('Models/checkpoints/datasets were not modified.')
print('Visual patch applied. Next: rebuild React and restart FastAPI server.')
