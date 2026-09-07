import { useCallback, useEffect, useRef, useState } from 'react'
import Viewer from './Viewer.jsx'
import Dropzone from './components/Dropzone.jsx'
import RunProgress from './components/RunProgress.jsx'
import Results from './components/Results.jsx'
import Validation from './components/Validation.jsx'
import { createJob, pollJob, fileUrl, validateJob } from './api.js'

/* known_height_m starts EMPTY on purpose. alpha is metres per model unit and
 * this one number sets it for the whole scene, so a georeferenced image is
 * refused rather than silently anchored to a default. See server.py. */
const DEFAULTS = {
  style: 'city',
  gsd_m: 0.5,
  known_height_m: '',
  scale_source: 'known_height',
  sun_azimuth: 145,
  sun_elevation: 52,
  use_dem: true,
  alpha_gain: 1.0,
  flatten: 0.8,
  sharpen: 0.4,
  z_exaggeration: 1.5,
  target_grid: 256
}

const STYLES = [
  ['city', 'City', 'Footprints extruded as separate prisms'],
  ['stepped', 'Stepped', 'One surface with real vertical walls'],
  ['smooth', 'Smooth', 'Plain heightfield grid']
]

function Field ({ label, value, hint, required, children }) {
  return (
    <label className={`field${required ? ' required' : ''}`}>
      <span className="label">
        <span>{label}</span>
        {value != null && <span className="val">{value}</span>}
      </span>
      {children}
      {hint && <span className="hint">{hint}</span>}
    </label>
  )
}

export default function App () {
  const [file, setFile] = useState(null)
  const [params, setParams] = useState(DEFAULTS)
  const [job, setJob] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [report, setReport] = useState(null)
  const [validating, setValidating] = useState(false)
  const [valError, setValError] = useState('')
  const [tab, setTab] = useState('3d')
  const stopRef = useRef(null)

  const set = (k) => (e) => {
    const t = e.target
    const v = t.type === 'checkbox' ? t.checked
      : (t.type === 'number' || t.type === 'range')
          ? (t.value === '' ? '' : Number(t.value))
          : t.value
    setParams((p) => ({ ...p, [k]: v }))
  }

  useEffect(() => () => { if (stopRef.current) stopRef.current() }, [])

  const run = useCallback(async () => {
    if (!file) { setError('Choose an image first.'); return }
    setError(''); setReport(null); setValError(''); setBusy(true); setJob(null); setTab('3d')
    try {
      const { job_id } = await createJob(file, params)
      stopRef.current = pollJob(job_id, (j) => {
        setJob(j)
        if (j.status === 'done' || j.status === 'error') {
          setBusy(false)
          if (j.status === 'error') setError(j.error || 'The run failed.')
        }
      })
    } catch (e) {
      setError(String(e.message || e)); setBusy(false)
    }
  }, [file, params])

  const onReference = async (ref) => {
    if (!ref || !job?.id) return
    setValidating(true); setValError('')
    try { setReport(await validateJob(job.id, ref)) }
    catch (e) { setValError(`Validation failed: ${e.message || e}`) }
    finally { setValidating(false) }
  }

  const res = job?.status === 'done' ? job.result : null
  const glb = res ? fileUrl(job.id, 'terrain.glb') : null
  const heightMap = res ? fileUrl(job.id, 'height16.png') : null

  const chip = !res
    ? null
    : res.datum === 'sea level' ? { cls: 'on', text: 'absolute · sea level' }
    : res.datum === 'local ground' ? { cls: 'warn', text: 'absolute · local ground' }
    : { cls: '', text: 'relative surface' }

  return (
    <div className="app">
      <div className="topbar">
        <div className="brand">
          <h1>DepthWizard</h1>
          <span className="tagline">one optical image in — a measurable, navigable 3D city out</span>
        </div>
        <span className="spacer" />
        {chip && <span className={`mode-chip ${chip.cls}`}>{chip.text}</span>}
      </div>

      <div className="layout">
        {/* ------------------------------------------------------- controls */}
        <aside>
          <div className="panel">
            <p className="section-label">Source image</p>
            <Dropzone file={file} disabled={busy}
                      onFile={(f) => { setFile(f); setReport(null); setError('') }} />

            <p className="section-label" style={{ marginTop: 24 }}>Scale</p>
            <Field label="Tallest structure (m)" required
                   hint="The tallest building you can identify, not a typical one. This one number sets the scale for the whole scene, so a GeoTIFF will not run without it. A PNG or JPG runs fine without.">
              <input type="number" step="1" min="1" placeholder="required for GeoTIFF"
                     value={params.known_height_m} onChange={set('known_height_m')} />
            </Field>
            <Field label="Ground sample distance" value={`${params.gsd_m} m/px`}
                   hint="Only used when the file carries no coordinates.">
              <input type="number" step="0.05" min="0.05"
                     value={params.gsd_m} onChange={set('gsd_m')} />
            </Field>

            <p className="section-label" style={{ marginTop: 24 }}>Geometry</p>
            <Field label="Style" hint={STYLES.find((s) => s[0] === params.style)?.[2]}>
              <div className="seg" role="group" aria-label="Mesh style">
                {STYLES.map(([id, label]) => (
                  <button key={id} type="button" aria-pressed={params.style === id}
                          onClick={() => setParams((p) => ({ ...p, style: id }))}>{label}</button>
                ))}
              </div>
            </Field>
            <Field label="Vertical exaggeration" value={`${params.z_exaggeration}×`}
                   hint="Display only — the readouts divide it back out.">
              <input type="range" min="1" max="5" step="0.5"
                     value={params.z_exaggeration} onChange={set('z_exaggeration')} />
            </Field>

            <details className="adv">
              <summary>Advanced</summary>
              <div className="body">
                <Field label="Mesh detail" value={params.target_grid}
                       hint="Higher is sharper and heavier. Stepped geometry costs about 4× the vertices.">
                  <input type="range" min="128" max="512" step="32"
                         value={params.target_grid} onChange={set('target_grid')} />
                </Field>
                <Field label="Flatten structures" value={params.flatten}
                       hint="Fits a plane per building. Turn down over forest.">
                  <input type="range" min="0" max="1" step="0.1"
                         value={params.flatten} onChange={set('flatten')} />
                </Field>
                <Field label="Edge sharpening" value={params.sharpen}
                       hint="Applied to the object band only, never to flat ground.">
                  <input type="range" min="0" max="1" step="0.1"
                         value={params.sharpen} onChange={set('sharpen')} />
                </Field>
                <Field label="Scale correction" value={`×${params.alpha_gain}`}
                       hint="Feed back the gain the validation report suggests. Leave at 1 unless you have measured otherwise.">
                  <input type="range" min="0.4" max="2.5" step="0.02"
                         value={params.alpha_gain} onChange={set('alpha_gain')} />
                </Field>
              </div>
            </details>

            <button className="btn" style={{ marginTop: 20 }}
                    onClick={run} disabled={busy || !file}>
              {busy && <span className="spinner" />}
              {busy ? 'Working…' : 'Generate 3D terrain'}
            </button>
            {error && <p className="alert">{error}</p>}
          </div>
        </aside>

        {/* --------------------------------------------------------- output */}
        <main>
          <div className="tabs" role="tablist">
            <button role="tab" aria-selected={tab === '3d'}
                    onClick={() => setTab('3d')}>3D flythrough</button>
            <button role="tab" aria-selected={tab === 'map'} disabled={!heightMap}
                    onClick={() => setTab('map')}>Elevation map</button>
          </div>

          <div className="stage">
            {tab === 'map' && heightMap
              ? <div className="flat-view"><img src={heightMap} alt="Elevation map" /></div>
              : glb
                ? <Viewer url={glb} units={res.datum === 'relative' ? 'm*' : 'm'}
                          exaggeration={res.z_exaggeration || 1} baseM={res.base_m || 0} />
                : (
                  <div className="empty">
                    <p className="headline">
                      A depth model can only rank heights. Everything here exists to turn
                      that ranking into a measurement you can walk through.
                    </p>
                    <div className="steps">
                      <div className="step"><span className="n">1</span>
                        <span className="t">Drop an aerial or satellite image</span></div>
                      <span className="arrow">→</span>
                      <div className="step"><span className="n">2</span>
                        <span className="t">Name the tallest structure you can see</span></div>
                      <span className="arrow">→</span>
                      <div className="step"><span className="n">3</span>
                        <span className="t">Fly through the result and measure it</span></div>
                    </div>
                  </div>
                )}
          </div>

          {job && (
            <div className="panel" style={{ marginTop: 16 }}>
              <RunProgress job={job} />
            </div>
          )}

          {res && (
            <div className="panel">
              <p className="section-label">Result</p>
              <Results res={res} job={job} fileUrl={fileUrl} />
            </div>
          )}

          {res && (
            <Validation job={job} report={report} busy={validating} error={valError}
                        onFile={onReference} fileUrl={fileUrl} />
          )}
        </main>
      </div>
    </div>
  )
}
