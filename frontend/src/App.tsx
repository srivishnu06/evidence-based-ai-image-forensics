import { useEffect, useRef, useState } from 'react'
import type { ChangeEvent, DragEvent } from 'react'
import {
  Activity, ArrowDownToLine, ArrowRight, Check,
  CircleHelp, FileImage, FileText, Focus,
  Layers2, LockKeyhole, Menu, MoveHorizontal, ScanLine,
  ShieldCheck, SlidersHorizontal, Sparkles, UploadCloud, X,
} from 'lucide-react'

type ArtifactScore = {
  name: string
  score: number
  threshold: number
  flagged: boolean
}

type AnalysisResult = {
  verdict: string
  ai_probability: number
  auth_threshold: number
  artifact_category: string
  artifact_score: number
  artifact_flagged: boolean
  evidence_regions: number
  artifact_scores: ArtifactScore[]
  evidence_image: string
  evidence_probability_map: string
  evidence_boundaries: string
  evidence_peak_probability: number
  localization_threshold: number
  evidence_crop: string | null
  report: string
  vlm_observation: string | null
  vlm_invoked: boolean
  inference_seconds?: number
}

const CATEGORY_NAMES = [
  'Edge / shape inconsistency',
  'Texture inconsistency',
  'Colour inconsistency',
  'Symbol / text inconsistency',
  'Semantic inconsistency',
  'Commonsense violation',
  'Physics violation',
]

function readableCategory(name: string) {
  return name.replace(/ inconsistency$/i, '')
}
function pct(value: number) { return `${(value * 100).toFixed(1)}%` }
function safeWidth(value: number) { return `${Math.max(0, Math.min(100, value * 100))}%` }

export default function App() {
  const fileInput = useRef<HTMLInputElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [vlmEnabled, setVlmEnabled] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [drag, setDrag] = useState(false)
  const [compare, setCompare] = useState(false)
  const [split, setSplit] = useState(50)
  const [selectedView, setSelectedView] = useState<'overlay' | 'probability' | 'boundaries' | 'crop'>('overlay')
  const [showMobileMenu, setShowMobileMenu] = useState(false)

  useEffect(() => {
    if (!file) { setPreview(null); return }
    const url = URL.createObjectURL(file)
    setPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  function acceptFile(candidate?: File) {
    if (!candidate) return
    if (!['image/png', 'image/jpeg', 'image/webp'].includes(candidate.type)) {
      setError('Choose a PNG, JPG or WebP image.'); return
    }
    if (candidate.size > 20 * 1024 * 1024) {
      setError('Image must be smaller than 20 MB.'); return
    }
    setFile(candidate)
    setResult(null)
    setError('')
    setCompare(false)
    setSelectedView('overlay')
  }
  function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    acceptFile(event.target.files?.[0])
    event.target.value = ''
  }
  function onDrop(event: DragEvent<HTMLButtonElement>) {
    event.preventDefault()
    setDrag(false)
    acceptFile(event.dataTransfer.files[0])
  }
  function clear() {
    setFile(null)
    setResult(null)
    setError('')
    setCompare(false)
  }

  async function analyse() {
    if (!file || busy) return
    setBusy(true)
    setError('')
    setResult(null)
    try {
      const form = new FormData()
      form.append('file', file)
      form.append('include_vlm', String(vlmEnabled))
      // Relative URL works when FastAPI serves React under a SageMaker proxy prefix.
      const response = await fetch('api/analyze', { method: 'POST', body: form })
      if (!response.ok) {
        let detail = `HTTP ${response.status}`
        try {
          const parsed = await response.json()
          if (typeof parsed.detail === 'string') detail = parsed.detail
        } catch { /* Server response may not be JSON. */ }
        throw new Error(detail)
      }
      const data: AnalysisResult = await response.json()
      if (
        typeof data.verdict !== 'string' ||
        !Array.isArray(data.artifact_scores) ||
        typeof data.evidence_image !== 'string' ||
        typeof data.report !== 'string'
      ) throw new Error('The inference API returned an incomplete response.')
      setResult(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Analysis failed.')
    } finally { setBusy(false) }
  }

  function downloadReport() {
    if (!result) return
    const url = URL.createObjectURL(new Blob([result.report], { type: 'text/plain;charset=utf-8' }))
    const link = document.createElement('a')
    link.href = url
    link.download = 'forensic_analysis_report.txt'
    link.click()
    URL.revokeObjectURL(url)
  }

  const scoreRows = CATEGORY_NAMES.map(name =>
    result?.artifact_scores.find(item => item.name === name) ??
    { name, score: 0, threshold: 1, flagged: false }
  )
  const titleVerdict = result?.verdict ?? 'Awaiting analysis'
  const flaggedTotal = scoreRows.filter(x => x.flagged).length
  const visual = !result ? null
    : selectedView === 'crop' && result.evidence_crop ? result.evidence_crop
    : selectedView === 'probability' ? result.evidence_probability_map
    : selectedView === 'boundaries' ? result.evidence_boundaries
    : result.evidence_image

  return (
    <div className="app-shell">
      <aside className={`sidebar ${showMobileMenu ? 'sidebar-open' : ''}`}>
        <div className="brand">
          <div className="brand-symbol"><ScanLine size={23} strokeWidth={2.4} /></div>
          <div className="brand-text"><b>FORENSICS<span>LAB</span></b><small>Vision intelligence workspace</small></div>
        </div>
        <div className="nav-group">
          <div className="nav-section">WORKSPACE</div>
          <a href="#workspace" className="nav-link nav-active" onClick={() => setShowMobileMenu(false)}><Activity size={18} /> Image analysis <span className="nav-pill">01</span></a>
          <div className="nav-section secondary-nav">RESEARCH PIPELINE</div>
          <div className="pipeline-item"><span className="pipeline-number">01</span><div><b>Authenticity detector</b><small>DeiT-Tiny · Stage 1</small></div><Check size={14} /></div>
          <div className="pipeline-item"><span className="pipeline-number">02</span><div><b>Artifact localisation</b><small>Spatial DeiT · Stage 2</small></div><Check size={14} /></div>
          <div className="pipeline-item"><span className="pipeline-number">03</span><div><b>Visual observation</b><small>Qwen2.5-VL · Experimental</small></div><Sparkles size={14} /></div>
        </div>
        <div className="side-bottom">
          <div className="research-note"><ShieldCheck size={18} /><div><strong>Research prototype</strong><p>Predictions and highlighted regions do not establish image origin.</p></div></div>
          <p>Evidence-based AI image forensics</p>
        </div>
      </aside>

      <div className="workspace" id="workspace">
        <header className="topbar">
          <div className="crumb"><button aria-label="Toggle menu" className="mobile-menu" onClick={() => setShowMobileMenu(v => !v)}><Menu size={20} /></button>Workspace <span>/</span> <b>Image analysis</b></div>
          <div className="topbar-right"><span className="status-dot" /> {result ? 'Analysis completed' : 'Research environment'} <span className="header-badge">{result?.vlm_invoked ? "CV + VLM" : "CV only"}</span></div>
        </header>
        <main className="main-content">
          <div className="hero"><div><div className="overline"><span className="overline-dot" /> COMPUTER VISION RESEARCH</div><h1>AI Image Forensics</h1><p>Assess image authenticity, investigate candidate visual artifacts and inspect the evidence behind model predictions.</p></div><div className="hero-mark"><ScanLine size={32} /></div></div>

          <section className="section" id="upload"><div className="section-title"><div><span className="section-kicker">01 / INPUT</span><h2>New image analysis</h2></div><span className="section-hint">PNG, JPG, WebP · up to 20 MB</span></div>
            <div className="surface upload-box"><input type="file" accept="image/png,image/jpeg,image/webp" ref={fileInput} onChange={onFileChange} hidden />
              <button type="button" className={`dropzone ${drag ? 'dragging' : ''}`} onClick={() => fileInput.current?.click()} onDragOver={e => { e.preventDefault(); setDrag(true) }} onDragLeave={() => setDrag(false)} onDrop={onDrop}>
                <div className="upload-symbol"><UploadCloud size={24} /></div><div className="upload-description"><strong>{file ? file.name : 'Drop your image here, or browse files'}</strong><span>{file ? `${(file.size / 1024 / 1024).toFixed(2)} MB · ready for analysis` : 'Original images stay in the research environment'}</span></div><span className="browse-label">{file ? 'Replace' : 'Browse files'} <ArrowRight size={15} /></span>
              </button>
              <div className="upload-footer"><label className="vlm-toggle"><input type="checkbox" checked={vlmEnabled} onChange={e => setVlmEnabled(e.target.checked)} /><span className="toggle-copy"><b>Experimental VLM observation</b><small>Off by default · interpretation may be unreliable</small></span></label><div className="button-row">{file && <button className="quiet-button" type="button" onClick={clear}><X size={15} /> Clear</button>}<button className="primary-button" type="button" onClick={analyse} disabled={!file || busy}>{busy ? <><span className="spinner" /> Analysing...</> : <><ScanLine size={17} /> Analyse image <ArrowRight size={16} /></>}</button></div></div>
            </div>{error && <div className="error-message" role="alert"><CircleHelp size={18} /><span>{error}</span></div>}
          </section>

          <section className="section" id="results"><div className="section-title"><div><span className="section-kicker">02 / PREDICTIONS</span><h2>Analysis overview</h2></div><span className="section-hint">{result ? 'Based on independent CV model outputs' : 'Awaiting input'}</span></div>
            <div className="metric-grid">
              <div className="surface metric metric-primary"><div className="metric-eyebrow">AUTHENTICITY PREDICTION <ShieldCheck size={17} /></div><div className="metric-main">{titleVerdict}</div><div className="metric-caption">{result ? `AI-generated model score: ${pct(result.ai_probability)}` : 'Upload an image to obtain a prediction'}</div><div className="score-track"><div className="score-fill cobalt" style={{ width: safeWidth(result?.ai_probability ?? 0) }} /></div><div className="metric-bottom">{result ? `Decision threshold ${pct(result.auth_threshold)}` : 'Independent authenticity classifier'}</div></div>
              <div className="surface metric"><div className="metric-eyebrow">HIGHEST ARTIFACT SIGNAL <Layers2 size={17} /></div><div className="metric-main category-metric">{result ? readableCategory(result.artifact_category) : 'Not analysed'}</div><div className="metric-caption">{result ? `Artifact score: ${result.artifact_score.toFixed(4)}` : 'Seven visual artifact categories'}</div><div className="score-track"><div className="score-fill gold" style={{ width: safeWidth(result?.artifact_score ?? 0) }} /></div><div className="metric-bottom">{result ? (result.artifact_flagged ? 'Above its category threshold' : 'Below its category threshold') : 'Category scores are independent'}</div></div>
              <div className="surface metric"><div className="metric-eyebrow">LOCALISED EVIDENCE <Focus size={17} /></div><div className="metric-main">{result ? result.evidence_regions : '—'} {result && <span className="unit">region{result.evidence_regions === 1 ? '' : 's'}</span>}</div><div className="metric-caption">{result ? `${flaggedTotal} categor${flaggedTotal === 1 ? 'y' : 'ies'} flagged` : 'Candidate spatial evidence'}</div><div className="no-verified"><CircleHelp size={15} /> Candidate anomalies, not confirmed defects</div></div>
            </div>
            <p className="small-disclaimer"><CircleHelp size={15} /> Model scores are not independently calibrated probabilities that an image is genuinely AI-generated.</p>
          </section>

          <section className="section" id="viewer"><div className="section-title"><div><span className="section-kicker">03 / VISUAL EVIDENCE</span><h2>Image inspection</h2></div><div className="view-switch"><button type="button" className={!compare ? 'active' : ''} onClick={() => setCompare(false)}><Layers2 size={15} /> Side by side</button><button type="button" className={compare ? 'active' : ''} onClick={() => setCompare(true)} disabled={!result || !preview}><MoveHorizontal size={15} /> Compare</button></div></div>
            {!compare || !result || !preview ? <div className="viewer-grid"><div className="surface image-panel"><div className="panel-header"><span>Original image</span><small>UPLOADED INPUT</small></div><div className="image-well">{preview ? <img src={preview} alt="Original uploaded image" /> : <div className="viewer-empty"><FileImage size={32} /><b>No image selected</b><small>Your uploaded image appears here</small></div>}</div></div><div className="surface image-panel"><div className="panel-header"><span>{selectedView === 'crop' ? 'Evidence crop' : selectedView === 'probability' ? 'Spatial probability map' : selectedView === 'boundaries' ? 'Thresholded evidence boundaries' : 'Evidence overlay'}</span><small>MODEL OUTPUT</small></div><div className="image-well">{visual ? <img src={visual} alt={selectedView === 'crop' ? 'Unaltered evidence crop' : 'Model-selected evidence overlay'} /> : <div className="viewer-empty"><Focus size={32} /><b>Awaiting evidence</b><small>Evidence will appear after analysis</small></div>}</div></div></div> : <div className="surface comparison-panel"><div className="panel-header"><span>Original ↔ evidence overlay</span><small>MOVE SLIDER TO COMPARE</small></div><div className="compare-well"><img src={preview} alt="Original image" /><img src={result.evidence_image} alt="Evidence overlay" className="compared-image" style={{ clipPath: `inset(0 ${100 - split}% 0 0)` }} /><div className="compare-divider" style={{ left: `${split}%` }}><span>↔</span></div></div><input aria-label="Image comparison divider" className="compare-input" type="range" min="0" max="100" value={split} onChange={e => setSplit(Number(e.target.value))} /><div className="compare-labels"><span>Evidence overlay</span><span>Original</span></div></div>}
            {result?.evidence_crop && !compare && <div className="viewer-options"><span><SlidersHorizontal size={15} /> EVIDENCE VIEW</span><div className="selector"><button type="button" className={selectedView === 'overlay' ? 'current' : ''} onClick={() => setSelectedView('overlay')}>Overlay</button><button type="button" className={selectedView === 'probability' ? 'current' : ''} onClick={() => setSelectedView('probability')}>Probability map</button><button type="button" className={selectedView === 'boundaries' ? 'current' : ''} onClick={() => setSelectedView('boundaries')}>Region outlines</button><button type="button" className={selectedView === 'crop' ? 'current' : ''} onClick={() => setSelectedView('crop')}>Unaltered crop</button></div></div>}
            {result && selectedView === 'probability' && !compare && <div className="probability-scale"><span>Fixed spatial probability scale</span><div className="probability-gradient" /><span>0</span><span>1.0</span><strong>Peak: {result.evidence_peak_probability.toFixed(3)}</strong></div>}
            {result && selectedView === 'boundaries' && !compare && <div className="boundary-caption">Yellow outlines mark 14×14 grid cells at or above localisation threshold {result.localization_threshold.toFixed(2)}. These are unverified candidate areas, not exact object boundaries.</div>}
            <div className="evidence-note"><CircleHelp size={16} /> Highlighted regions are estimates from a spatial model and may not correspond to a real visual defect. Enlarging a crop cannot restore lost image detail.</div>
          </section>

          <div className="deep-grid"><section className="surface deep-card" id="artifacts"><div className="inner-kicker">04 / ARTIFACT ANALYSIS</div><div className="deep-card-title"><h2>Artifact breakdown</h2><SlidersHorizontal size={18} /></div><p className="deep-subtitle">Category-specific model scores. Blue bars indicate scores above their separate calibrated thresholds.</p><div className="artifact-list">{scoreRows.map(item => <div className="artifact-row" key={item.name}><div className="artifact-heading"><span>{readableCategory(item.name)}</span><b>{result ? item.score.toFixed(3) : '—'}</b></div><div className="artifact-track"><div className={`artifact-fill ${item.flagged ? 'is-flagged' : ''}`} style={{ width: safeWidth(item.score) }} />{result && <div className="threshold-tick" style={{ left: safeWidth(item.threshold) }} title={`Threshold ${item.threshold.toFixed(2)}`} />}</div></div>)}</div><div className="legend"><span><i className="blue-key" /> Flagged</span><span><i className="gray-key" /> Below threshold</span><span className="threshold-help"><i className="tick-key" /> Threshold</span></div></section>
          <section className="surface deep-card" id="report"><div className="inner-kicker">05 / STRUCTURED FINDINGS</div><div className="deep-card-title"><h2>Forensic report</h2><FileText size={19} /></div>{result ? <><div className="report-body">{result.report.split(/\n\s*\n/).filter(Boolean).map((section, index) => { const newline = section.indexOf('\n'); return <div className="report-paragraph" key={index}><h3>{newline >= 0 ? section.slice(0, newline) : 'Finding'}</h3><p>{newline >= 0 ? section.slice(newline + 1) : section}</p></div> })}</div><button className="download-button" type="button" onClick={downloadReport}><ArrowDownToLine size={17} /> Download full report <ArrowRight size={15} /></button></> : <div className="report-placeholder"><FileText size={34} /><b>No report generated yet</b><p>Run image analysis to view predictions, evidence signals, and limitations in one place.</p></div>}</section></div>

          <section className="surface vlm-card" id="vlm"><div className="vlm-icon"><Sparkles size={20} /></div><div className="vlm-content"><div className="vlm-title"><h2>Vision-language observation</h2><span>EXPERIMENTAL</span></div><p className="vlm-subtitle">Qwen2.5-VL examines an unaltered crop chosen by the evidence model. The output may be inaccurate.</p><div className="vlm-answer">{result?.vlm_invoked ? (result.vlm_observation || 'The VLM did not return a usable observation.') : (vlmEnabled && result ? 'VLM not invoked: insufficient localised evidence, or generation unavailable.' : 'Enable experimental observation before analysis to request a VLM description.')}</div><div className="vlm-caution"><LockKeyhole size={14} /> The CV authenticity prediction is never overridden by Qwen.</div></div></section>
          <footer><span>Evidence-Based AI Image Forensics</span><span><ShieldCheck size={14} /> Research only · Not a forensic certification service</span></footer>
        </main>
      </div>
    </div>
  )
}
