/* ===========================================================================
 * App.jsx - the whole interface. Left panel sets up a run, right panel shows it.
 * ===========================================================================
 *
 * READ THIS FIRST. There is one piece of state that matters, `job`, and one
 * path through the file:
 *
 *   1. <Dropzone>          the user picks an image            -> file
 *   2. the control panel   scale source, geometry, advanced   -> params
 *   3. run()               POST /api/jobs, then pollJob() asks the server for
 *                          status about once a second until it says done
 *   4. job.result arrives  -> <Viewer> loads terrain.glb and you can fly
 *                          -> <Results> lists the numbers and the downloads
 *                          -> <Validation> optionally scores it against LiDAR
 *
 * Everything that talks to the backend is in api.js. Everything that draws 3D
 * is in Viewer.jsx. This file is the wiring between them.
 *
 * THE ONE RULE THE UI ENFORCES: a georeferenced image cannot run without a
 * scale source, because alpha (metres per model unit) is one number that sets
 * every elevation in the scene. See DEFAULTS below and server.py.
 */
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

/* All three of these are in the backend and in run_geotiff.py; only the first
 * used to be reachable from here, which made the other two invisible to anyone
 * who never opened the CLI. They are independent of each other - shadow length
 * is photogrammetry, control points are survey, a landmark height is human
 * knowledge - and the run cross-checks whichever ones it can. */
const SCALE_SOURCES = [
  ['known_height', 'Landmark', 'One number: how tall the tallest structure you can identify is'],
  ['gcps', 'Control points', 'Two or more pixels whose height above ground you know'],
  ['sun', 'Shadows', 'Sun angles — read from the GeoTIFF tags when the file carries them']
]

const VALIDATION_ARTEFACTS = ['validation.md', 'validation.json',
                              'error_map.png', 'scatter.png', 'stability.png']

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

function GcpEditor ({ gcps, onChange, disabled }) {
  const set = (i, k) => (e) => {
    const v = e.target.value
    onChange(gcps.map((g, j) => (j === i ? { ...g, [k]: v } : g)))
  }
  return (
    <div className="gcps">
      <div className="gcp-head"><span>row</span><span>col</span><span>height m</span><span /></div>
      {gcps.map((g, i) => (
        <div className="gcp-row" key={i}>
          <input type="number" step="1" min="0" value={g.row} onChange={set(i, 'row')}
                 disabled={disabled} aria-label={`point ${i + 1} row`} />
          <input type="number" step="1" min="0" value={g.col} onChange={set(i, 'col')}
                 disabled={disabled} aria-label={`point ${i + 1} column`} />
          <input type="number" step="0.5" value={g.height_m} onChange={set(i, 'height_m')}
                 disabled={disabled} aria-label={`point ${i + 1} height`} />
          <button type="button" className="x" disabled={disabled} aria-label="remove point"
                  onClick={() => onChange(gcps.filter((_, j) => j !== i))}>×</button>
        </div>
      ))}
      <button type="button" className="btn ghost" disabled={disabled}
              onClick={() => onChange([...gcps, { row: '', col: '', height_m: '' }])}>
        + Add control point
      </button>
    </div>
  )
}

export default function App () {
  const [file, setFile] = useState(null)
  const [params, setParams] = useState(DEFAULTS)
  const [gcps, setGcps] = useState([{ row: '', col: '', height_m: '' },
                                    { row: '', col: '', height_m: '' }])
  const [job, setJob] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [report, setReport] = useState(null)
  const [validating, setValidating] = useState(false)
  const [valError, setValError] = useState('')
  const [tab, setTab] = useState('3d')
  const [stamp, setStamp] = useState(0)
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
    const clean = gcps
      .filter((g) => g.row !== '' && g.col !== '' && g.height_m !== '')
      .map((g) => [Number(g.row), Number(g.col), Number(g.height_m)])
    if (params.scale_source === 'gcps' && clean.length < 2) {
      setError('Ground control points need at least two complete rows — ' +
               'pixel row, pixel column, and height above ground in metres.')
      return
    }
    setError(''); setReport(null); setValError(''); setBusy(true); setJob(null); setTab('3d')
    try {
      const { job_id } = await createJob(file, { ...params, gcps: clean })
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
  }, [file, params, gcps])

  const onReference = async (ref) => {
    if (!ref || !job?.id) return
    setValidating(true); setValError('')
    try {
      setReport(await validateJob(job.id, ref))
      // The validator writes five more files into the job folder, but the file
      // list came from result.json, which was written before they existed - so
      // the report and its figures were undownloadable until a page reload.
      // The same URLs are also re-used on a second validation, so they carry a
      // version stamp or the browser serves the first run's figures.
      setStamp(Date.now())
    } catch (e) { setValError(`Validation failed: ${e.message || e}`) }
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
            <Field label="Where metres come from"
                   hint={SCALE_SOURCES.find((s) => s[0] === params.scale_source)?.[2]}>
              <div className="seg" role="group" aria-label="Scale source">
                {SCALE_SOURCES.map(([id, label]) => (
                  <button key={id} type="button" aria-pressed={params.scale_source === id}
                          onClick={() => setParams((p) => ({ ...p, scale_source: id }))}>
                    {label}
                  </button>
                ))}
              </div>
            </Field>

            {params.scale_source === 'known_height' && (
              <Field label="Tallest structure (m)" required
                     hint="The tallest building you can identify, not a typical one. This one number sets the scale for the whole scene, so a GeoTIFF will not run without it. A PNG or JPG runs fine without.">
                <input type="number" step="1" min="1" placeholder="required for GeoTIFF"
                       value={params.known_height_m} onChange={set('known_height_m')} />
              </Field>
            )}

            {params.scale_source === 'gcps' && (
              <Field label="Ground control points" required
                     hint="Pixel row and column in the source image, and that point's height above the ground beside it. Two is enough to fix the multiplier; more make it steadier.">
                <GcpEditor gcps={gcps} onChange={setGcps} disabled={busy} />
              </Field>
            )}

            {params.scale_source === 'sun' && (
              <>
                <Field label="Sun azimuth" value={`${params.sun_azimuth}°`}
                       hint="Degrees clockwise from north. Leave both at the file's own values if it carries sun tags — Landsat, Sentinel and most commercial products do.">
                  <input type="number" step="1" min="0" max="360"
                         value={params.sun_azimuth} onChange={set('sun_azimuth')} />
                </Field>
                <Field label="Sun elevation" value={`${params.sun_elevation}°`}
                       hint="Degrees above the horizon. Shadow length × tan(elevation) is the height of whatever cast it.">
                  <input type="number" step="1" min="1" max="89"
                         value={params.sun_elevation} onChange={set('sun_elevation')} />
                </Field>
              </>
            )}

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
                <label className="check">
                  <input type="checkbox" checked={params.use_dem} onChange={set('use_dem')} />
                  <span>
                    <b>Sea-level terrain baseline (COP30)</b>
                    <span className="hint">
                      On, the coarse DEM supplies the terrain and the output is
                      metres above sea level. Off — or when the download fails —
                      the output is height above local ground, and the results
                      panel says so.
                    </span>
                  </span>
                </label>
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
              <Results res={res} job={job} fileUrl={fileUrl}
                       extraFiles={report ? VALIDATION_ARTEFACTS : []} />
            </div>
          )}

          {res && (
            <Validation job={job} report={report} busy={validating} error={valError}
                        onFile={onReference} fileUrl={fileUrl} datum={res.datum}
                        stamp={stamp} />
          )}
        </main>
      </div>
    </div>
  )
}
