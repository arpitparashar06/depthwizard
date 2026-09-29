import { useCallback, useEffect, useRef, useState } from 'react'
import Viewer from './Viewer.jsx'
import Dropzone from './components/Dropzone.jsx'
import RunProgress from './components/RunProgress.jsx'
import Results from './components/Results.jsx'
import Validation from './components/Validation.jsx'
import AmbientBackground from './components/AmbientBackground.jsx'
import Profile from './components/Profile.jsx'
import { createJob, pollJob, fileUrl, validateJob, getLayers, getProfile, getProbe } from './api.js'

const DEFAULTS = {
  style: 'city',
  gsd_m: 0.5,
  known_height_m: '',
  auto_prior: true,
  scale_source: 'known_height',
  sun_azimuth: 145,
  sun_elevation: 52,
  use_dem: true,
  fuse_scale: false,
  alpha_gain: 1.0,
  flatten: 0.8,
  sharpen: 0.4,
  z_exaggeration: 1.5,
  target_grid: 256
}

// [id, button label, the hint shown under the segmented control]
const STYLES = [
  ['city', 'City', 'Footprints extruded as separate prisms'],
  ['stepped', 'Stepped', 'One surface with real vertical walls'],
  ['smooth', 'Smooth', 'Plain heightfield grid']
]

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

// The ground-control-point table.
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
  const [file, setFile] = useState(null)       // the image the user picked
  const [params, setParams] = useState(DEFAULTS)
  const [gcps, setGcps] = useState([{ row: '', col: '', height_m: '' },
                                    { row: '', col: '', height_m: '' }])
  const [job, setJob] = useState(null)
  const [busy, setBusy] = useState(false)      // a run is in flight
  const [error, setError] = useState('')
  const [report, setReport] = useState(null)   // validation result, if scored
  const [validating, setValidating] = useState(false)
  const [valError, setValError] = useState('')
  const [tab, setTab] = useState('3d')
  const [stamp, setStamp] = useState(0)        // cache-buster for re-validation
  const [scrolled, setScrolled] = useState(false)  // has the page moved at all?
  const stopRef = useRef(null)                 // cancels the running poll loop
  const [layers, setLayers] = useState([])     // drapeable analysis layers
  const [layerId, setLayerId] = useState('photo')
  const [pins, setPins] = useState([])         // probe pins, in source pixels
  const [probes, setProbes] = useState(null)   // exact values at each pin
  const [profile, setProfile] = useState(null) // exact values along A -> B
  const [hoverIdx, setHoverIdx] = useState(null)
  const [clearPins, setClearPins] = useState(0)
  const layersRef = useRef([])

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8)
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  // One change handler for every input.
  const set = (k) => (e) => {
    const t = e.target
    const v = t.type === 'checkbox' ? t.checked
      : (t.type === 'number' || t.type === 'range')
          ? (t.value === '' ? '' : Number(t.value))
          : t.value
    setParams((p) => ({ ...p, [k]: v }))
  }

  // Stop polling if the component goes away mid-run
  useEffect(() => () => { if (stopRef.current) stopRef.current() }, [])

  // Which engine the server will use.
  const [engine, setEngine] = useState(null)
  useEffect(() => {
    fetch('/api/health').then((r) => r.json()).then((h) => setEngine(h.engine || null))
      .catch(() => setEngine(null))
  }, [])

  // ?job=<id> reopens a run.
  useEffect(() => {
    const id = new URLSearchParams(window.location.search).get('job')
    if (!id || !/^[A-Za-z0-9_-]+$/.test(id)) return
    setBusy(true)
    stopRef.current = pollJob(id, (j) => {
      setJob(j)
      if (j.status === 'done' || j.status === 'error') {
        setBusy(false)
        if (j.status === 'error') setError(j.error || 'The run failed.')
      }
    })
  }, [])

  const run = useCallback(async () => {
    if (!file) { setError('Choose an image first.'); return }
    // Strings -> numbers, and half-filled rows are dropped rather than sent
    const clean = gcps
      .filter((g) => g.row !== '' && g.col !== '' && g.height_m !== '')
      .map((g) => [Number(g.row), Number(g.col), Number(g.height_m)])
    if (params.scale_source === 'gcps' && clean.length < 2) {
      setError('Ground control points need at least two complete rows — ' +
               'pixel row, pixel column, and height above ground in metres.')
      return
    }
    setError(''); setReport(null); setValError(''); setBusy(true); setJob(null); setTab('3d')
    setLayers([]); setLayerId('photo'); setPins([]); setProfile(null); setProbes(null)
    try {
      const { job_id } = await createJob(file, { ...params, gcps: clean })
      try {
        const u = new URL(window.location.href)
        u.searchParams.set('job', job_id)
        window.history.replaceState(null, '', u)
      } catch { /* an address bar we cannot write is not worth failing a run for */ }
      // PollJob calls back on every status change and returns its own stopper
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
      setStamp(Date.now())
    } catch (e) { setValError(`Validation failed: ${e.message || e}`) }
    finally { setValidating(false) }
  }

  const res = job?.status === 'done' ? job.result : null
  const jobId = res ? job.id : null
  const glb = res ? fileUrl(job.id, 'terrain.glb') : null

  // The layer list.
  useEffect(() => {
    if (!jobId) return
    let live = true
    getLayers(jobId).then((d) => { if (live) setLayers(d.layers || []) }).catch(() => {})
    return () => { live = false }
  }, [jobId, stamp])

  // Pins -> exact values.
  useEffect(() => {
    if (!jobId || !pins.length) { setProbes(null); setProfile(null); return }
    let live = true
    Promise.all(pins.map((p) => getProbe(jobId, p).catch(() => null)))
      .then((v) => { if (live) setProbes(v) })
    if (pins.length === 2) {
      getProfile(jobId, pins[0], pins[1])
        .then((d) => { if (live) { setProfile(d); setHoverIdx(null) } })
        .catch(() => { if (live) setProfile(null) })
    } else {
      setProfile(null)
    }
    return () => { live = false }
  }, [jobId, pins, stamp])

  const onPins = useCallback((p) => setPins(p || []), [])
  layersRef.current = layers
  const cycleLayer = useCallback(() => {
    const ls = layersRef.current
    if (!ls.length) return
    setLayerId((cur) => ls[(ls.findIndex((l) => l.id === cur) + 1) % ls.length].id)
  }, [])
  const layer = layers.find((l) => l.id === layerId)
  const layerUrl = jobId && layer && layer.id !== 'photo'
    ? fileUrl(jobId, layer.file, layer.id === 'error' ? stamp : undefined) : null
  const legend = layer && layer.id !== 'photo' ? layer : null
  const heightMap = res ? fileUrl(job.id, 'height16.png') : null

  // The datum badge.
  const chip = !res
    ? null
    : res.datum === 'sea level' ? { cls: 'on', text: 'absolute · sea level' }
    : res.datum === 'local ground' ? { cls: 'warn', text: 'absolute · local ground' }
    : { cls: '', text: 'relative surface' }

  return (
    <div className="app">
      <AmbientBackground />

      <nav className="site-nav">
        <div className={`nav-bar${scrolled ? ' is-stuck' : ''}`}>
          <a href="#" className="nav-brand">
            <span className="nav-mark" aria-hidden="true" />
            <span>DepthWizard</span>
          </a>
          <div className="nav-meta">
            <span className="nav-note">single-view elevation</span>
            <span className="nav-tag">SIH · ISRO</span>
            {chip && (
              <span className={`nav-chip ${chip.cls}`}>
                <span className="dot-live" />
                <span>{chip.text}</span>
              </span>
            )}
          </div>
        </div>
      </nav>

      <header className="hero">
        <h1 className="hero-title">
          Single-View Height Estimation<br />
          <span className="muted">&amp; 3D Flythrough</span>
        </h1>
        <p className="hero-subtitle">
          One optical image in. A Digital Surface Model in real metres out — plus a
          city you can fly through and measure, from a single frame with no stereo
          pair, no LiDAR and no radar.
        </p>
      </header>

      <div className="layout">
        {/* controls */}
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
              <Field label="Tallest structure (m)"
                     hint={engine === 'hybrid' || engine === 'metric'
                       ? "Optional. The fine-tuned model sets the scale itself: its metric head measures this scene and calibrates the relative depth, then the two are fused. A number typed here is used instead of that learned prior, and the run reports whether they agree."
                       : "The tallest building you can identify, not a typical one. This one number sets the scale for the whole scene. Leave it empty and a GeoTIFF uses the GHSL building height for its location instead — a prior, and labelled as one. A PNG or JPG runs fine without."}>
                <input type="number" step="1" min="1"
                       placeholder={engine === 'hybrid' || engine === 'metric' ? 'optional - the model measures heights' : 'empty = GHSL building height'}
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

            {/* geometry */}
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
                {/* Inverse-variance fusion, inference.py::fuse_scale_estimates */}
                <label className="check">
                  <input type="checkbox" checked={params.fuse_scale}
                         onChange={set('fuse_scale')} />
                  <span>
                    <b>Combine every scale source</b>
                    <span className="hint">
                      Off, the first calibrator that answers sets the scale
                      (shadows, then control points, then the landmark). On,
                      every one that answers is combined by how well it knows
                      its own error — and it refuses to combine sources that
                      disagree by more than their error bars allow. Needs at
                      least two sources to do anything.
                    </span>
                  </span>
                </label>
                {/* This checkbox is what decides the datum badge at the top */}
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

        <main>
          <div className="stage-bar">
            <div className="tabs" role="tablist">
              <button role="tab" aria-selected={tab === '3d'}
                      onClick={() => setTab('3d')}>3D flythrough</button>
              <button role="tab" aria-selected={tab === 'map'} disabled={!heightMap}
                      onClick={() => setTab('map')}>Elevation map</button>
            </div>
            {/* What is draped on the terrain. */}
            {glb && tab === '3d' && layers.length > 1 && (
              <div className="layer-seg">
                <span className="lbl">Layer</span>
                <div className="seg layers" role="group" aria-label="Draped layer">
                  {layers.map((l) => (
                    <button key={l.id} type="button" aria-pressed={layerId === l.id}
                            onClick={() => setLayerId(l.id)}>{l.label}</button>
                  ))}
                </div>
              </div>
            )}
          </div>

          <div className="stage">
            {tab === 'map' && heightMap
              ? <div className="flat-view"><img src={heightMap} alt="Elevation map" /></div>
              : glb
                ? <Viewer url={glb} clearSignal={clearPins}
                          units={res.datum === 'relative' ? 'm*' : 'm'}
                          exaggeration={res.z_exaggeration || 1} baseM={res.base_m || 0}
                          pxSize={res.px_size_m || 1} layerUrl={layerUrl} legend={legend}
                          exact={probes} profile={profile} profileHover={hoverIdx}
                          onPins={onPins} onCycleLayer={cycleLayer} />
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
                        <span className="t">Optionally, name the tallest structure you can see</span></div>
                      <span className="arrow">→</span>
                      <div className="step"><span className="n">3</span>
                        <span className="t">Fly through the result and measure it</span></div>
                    </div>
                  </div>
                )}
          </div>

          {res && tab === '3d' && (
            profile
              ? (
                <div className="panel" style={{ marginTop: 16 }}>
                  <Profile profile={profile} units={res.datum === 'relative' ? 'm*' : 'm'}
                           hover={hoverIdx} onHover={setHoverIdx}
                           onClear={() => { setPins([]); setClearPins((k) => k + 1) }} />
                </div>
                )
              : (
                <p className="note" style={{ marginTop: 10 }}>
                  Click the view to fly, then click two points on the terrain: the
                  elevation profile between them appears here, read from the GeoTIFFs.
                </p>
                )
          )}

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
                        stamp={stamp}
                        onShowError={() => {
                          setTab('3d'); setLayerId('error')
                          document.querySelector('.stage')?.scrollIntoView({ behavior: 'smooth', block: 'center' })
                        }} />
          )}
        </main>
      </div>
    </div>
  )
}
