import { useCallback, useEffect, useRef, useState } from 'react'
import Viewer from './Viewer.jsx'
import { createJob, pollJob, fileUrl, validateJob } from './api.js'

const DEFAULTS = {
  style: 'city',
  gsd_m: 0.5,
  known_height_m: 40,
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

function Field ({ label, hint, children }) {
  return (
    <label className="field">
      <span className="label">{label}</span>
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
  const stopRef = useRef(null)
  const logRef = useRef(null)

  const set = (k) => (e) => {
    const v = e.target.type === 'checkbox' ? e.target.checked
      : e.target.type === 'number' || e.target.type === 'range' ? Number(e.target.value)
        : e.target.value
    setParams((p) => ({ ...p, [k]: v }))
  }

  useEffect(() => () => { if (stopRef.current) stopRef.current() }, [])
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [job?.log?.length])

  const run = useCallback(async () => {
    if (!file) { setError('Choose an image first.'); return }
    setError(''); setReport(null); setBusy(true); setJob(null)
    try {
      const { job_id } = await createJob(file, params)
      stopRef.current = pollJob(job_id, (j) => {
        setJob(j)
        if (j.status === 'done' || j.status === 'error') {
          setBusy(false)
          if (j.status === 'error') setError(j.error || 'job failed')
        }
      })
    } catch (e) {
      setError(String(e.message || e)); setBusy(false)
    }
  }, [file, params])

  const onValidate = async (e) => {
    const ref = e.target.files?.[0]
    if (!ref || !job?.id) return
    setValidating(true); setError('')
    try {
      setReport(await validateJob(job.id, ref))
    } catch (err) {
      setError(`Validation failed: ${err.message || err}`)
    } finally {
      setValidating(false)
      e.target.value = ''
    }
  }

  const res = job?.status === 'done' ? job.result : null
  const isRelative = res?.mode === 'relative'
  const units = res ? (isRelative ? 'm*' : 'm') : 'm'
  const glb = res ? fileUrl(job.id, 'terrain.glb') : null

  return (
    <div className="app">
      <header>
        <h1>DepthWizard</h1>
        <p>One optical image in — a measurable, navigable 3D city out.
          <span className="dim"> PNG/JPG gives a relative surface; a GeoTIFF with a CRS gives metres.</span>
        </p>
      </header>

      <div className="grid">
        <aside className="panel">
          <Field label="Satellite / aerial image">
            <input type="file" accept=".png,.jpg,.jpeg,.tif,.tiff"
                   onChange={(e) => { setFile(e.target.files?.[0] || null); setReport(null) }} />
          </Field>
          {file && <p className="dim small">{file.name} · {(file.size / 1e6).toFixed(1)} MB</p>}

          <h3>Scale</h3>
          <Field label="Ground sample distance (m/px)"
                 hint="Only used when the file has no coordinates. Sets the scale of everything.">
            <input type="number" step="0.05" min="0.05" value={params.gsd_m}
                   onChange={set('gsd_m')} />
          </Field>
          <Field label="Tallest structure (m)"
                 hint="Anchors a relative scene, and calibrates a georeferenced one.">
            <input type="number" step="1" min="1" value={params.known_height_m}
                   onChange={set('known_height_m')} />
          </Field>

          <h3>Geometry</h3>
          <Field label="Style"
                 hint="City extrudes building footprints as separate prisms. Stepped is one surface with vertical walls. Smooth is a plain grid.">
            <select value={params.style} onChange={set('style')}>
              <option value="city">city</option>
              <option value="stepped">stepped</option>
              <option value="smooth">smooth</option>
            </select>
          </Field>
          <Field label={`Vertical exaggeration · ${params.z_exaggeration}×`}>
            <input type="range" min="1" max="5" step="0.5"
                   value={params.z_exaggeration} onChange={set('z_exaggeration')} />
          </Field>
          <Field label={`Mesh detail · ${params.target_grid}`}>
            <input type="range" min="128" max="512" step="32"
                   value={params.target_grid} onChange={set('target_grid')} />
          </Field>

          <h3>Refinement</h3>
          <Field label={`Flatten structures · ${params.flatten}`}
                 hint="Fits a plane per building. Turn down over forest.">
            <input type="range" min="0" max="1" step="0.1"
                   value={params.flatten} onChange={set('flatten')} />
          </Field>
          <Field label={`Edge sharpening · ${params.sharpen}`}>
            <input type="range" min="0" max="1" step="0.1"
                   value={params.sharpen} onChange={set('sharpen')} />
          </Field>

          <button className="primary" onClick={run} disabled={busy || !file}>
            {busy ? 'Working…' : 'Generate 3D terrain'}
          </button>
          {error && <p className="error">{error}</p>}
        </aside>

        <main>
          <Viewer url={glb} units={units}
                  exaggeration={res?.z_exaggeration || 1}
                  baseM={res?.base_m || 0} />

          {res && (
            <section className="panel">
              <h3>Elevation Map</h3>
              <img src={fileUrl(job.id, 'height16.png')} alt="Elevation Map" style={{ width: '100%', height: 'auto', borderRadius: '4px' }} />
            </section>
          )}

          {job && (
            <section className="panel run">
              <div className="bar"><div style={{ width: `${(job.progress || 0) * 100}%` }} /></div>
              <pre className="log" ref={logRef}>{(job.log || []).join('\n')}</pre>
            </section>
          )}

          {res && (
            <section className="panel">
              <h3>{res.mode.toUpperCase()} mode</h3>
              <table className="kv">
                <tbody>
                  <tr><td>Size</td><td>{res.width} × {res.height} px @ {res.px_size_m.toFixed(2)} m/px</td></tr>
                  <tr><td>Height range</td><td>{res.min_m.toFixed(1)} – {res.max_m.toFixed(1)} {units}</td></tr>
                  <tr><td>Relief</td><td>{res.relief_m.toFixed(1)} {units}</td></tr>
                  <tr><td>Median slope</td><td>{res.median_slope_deg.toFixed(1)}° (99th {res.p99_slope_deg.toFixed(1)}°)</td></tr>
                  <tr><td>Mesh</td><td>{res.triangles.toLocaleString()} triangles ({res.style})</td></tr>
                  {res.info?.n != null && (
                    <tr><td>Buildings</td><td>
                      {res.info.n} extruded, tallest {Number(res.info.height_max_m || 0).toFixed(0)} m
                    </td></tr>
                  )}
                  {res.info?.self_check?.rmse_m != null && (
                    <tr><td>Self-check RMSE</td><td>
                      {Number(res.info.self_check.rmse_m).toFixed(2)} m over{' '}
                      {res.info.self_check.n} held-out shadow points
                    </td></tr>
                  )}
                </tbody>
              </table>
              {isRelative && (
                <p className="note">
                  Relative mode: heights are scaled so the scene spans{' '}
                  {Number(res.info?.relative_full_scale_m || 60).toFixed(0)} m.
                  Upload a GeoTIFF with a CRS for true metres.
                </p>
              )}
              <div className="downloads">
                {res.files.filter((f) => !f.startsWith('source')).map((f) => (
                  <a key={f} href={fileUrl(job.id, f)} download>{f}</a>
                ))}
              </div>
            </section>
          )}

          {res && (
            <section className="panel">
              <h3>Validate against reference LiDAR</h3>
              <p className="dim small">
                Metrics are reported under three alignments — raw, median datum shift
                and robust affine — because a no-DEM estimate is height above local
                ground, and scoring it raw against a sea-level reference measures the
                datum, not the model.
              </p>
              <input type="file" accept=".tif,.tiff,.png" onChange={onValidate}
                     disabled={validating} />
              {validating && <p className="dim">Scoring…</p>}
              {report && (
                <>
                  <table className="kv">
                    <tbody>
                      {['raw', 'shift', 'affine'].map((k) => (
                        <tr key={k}>
                          <td>{k}</td>
                          <td>
                            RMSE {report.alignment[k].rmse?.toFixed(2)} ·
                            MAE {report.alignment[k].mae?.toFixed(2)} ·
                            r {report.alignment[k].r?.toFixed(3)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {report.by_landscape && (
                    <table className="kv">
                      <tbody>
                        {Object.entries(report.by_landscape).map(([k, v]) => (
                          <tr key={k}>
                            <td>{k}</td>
                            <td>{v.share_pct?.toFixed(0)}% · RMSE {v.rmse?.toFixed(2)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                  <div className="figs">
                    {['error_map.png', 'scatter.png', 'stability.png'].map((f) => (
                      <img key={f} src={fileUrl(job.id, f)} alt={f} />
                    ))}
                  </div>
                </>
              )}
            </section>
          )}
        </main>
      </div>
    </div>
  )
}
