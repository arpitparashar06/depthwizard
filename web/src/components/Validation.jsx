import { useRef, useState } from 'react'

const ALIGN = {
  raw:    ['Raw', 'nothing removed — the honest number'],
  shift:  ['Datum shift', 'one constant offset removed'],
  affine: ['Robust affine', 'offset and scale removed']
}

export default function Validation ({ job, report, busy, error, onFile, fileUrl }) {
  const inputRef = useRef(null)
  const [over, setOver] = useState(false)
  if (!job) return null

  const classes = report?.by_landscape
    ? Object.entries(report.by_landscape).sort((a, b) => (b[1].rmse || 0) - (a[1].rmse || 0))
    : []
  const worst = classes.length ? Math.max(...classes.map(([, v]) => v.rmse || 0)) : 1

  return (
    <div className="panel">
      <p className="section-label">Validate against reference LiDAR</p>
      <p className="note" style={{ margin: '0 0 12px' }}>
        Scored under three alignments, because a no-DEM estimate is height above local
        ground and scoring it raw against a sea-level reference measures the datum
        rather than the model.
      </p>

      <div className={`drop${over ? ' over' : ''}`} style={{ padding: '16px 12px' }}
           onDragOver={(e) => { e.preventDefault(); setOver(true) }}
           onDragLeave={() => setOver(false)}
           onDrop={(e) => { e.preventDefault(); setOver(false); onFile(e.dataTransfer.files?.[0]) }}
           onClick={() => inputRef.current?.click()}>
        <input ref={inputRef} type="file" accept=".tif,.tiff,.png" disabled={busy}
               onChange={(e) => { onFile(e.target.files?.[0]); e.target.value = '' }} />
        <p className="lead">
          {busy ? 'Scoring…' : <><b>Drop a reference raster</b> to score this run</>}
        </p>
        <p className="formats">GeoTIFF · reprojected onto this grid automatically</p>
      </div>

      {error && <p className="alert">{error}</p>}

      {report && (
        <>
          <table className="metrics">
            <thead>
              <tr><th>Alignment</th><th>RMSE</th><th>MAE</th><th>Bias</th><th>r</th></tr>
            </thead>
            <tbody>
              {['raw', 'shift', 'affine'].map((k) => {
                const m = report.alignment?.[k]
                if (!m) return null
                const [label, sub] = ALIGN[k]
                return (
                  <tr key={k} className={report.headline_alignment === k ? 'headline' : ''}>
                    <td>{label}<br /><span style={{ fontSize: 11, color: 'var(--faint)' }}>{sub}</span></td>
                    <td>{m.rmse?.toFixed(2)}</td>
                    <td>{m.mae?.toFixed(2)}</td>
                    <td>{m.bias >= 0 ? '+' : ''}{m.bias?.toFixed(2)}</td>
                    <td>{m.r?.toFixed(3)}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>

          {classes.length > 0 && (
            <>
              <p className="section-label" style={{ marginTop: 24 }}>Stability across landscapes</p>
              <div className="bars">
                {classes.map(([name, v]) => (
                  <div className="bar-row" key={name}>
                    <span className="lbl">{name}</span>
                    <span className="track2">
                      <div style={{ width: `${Math.max(3, 100 * (v.rmse || 0) / worst)}%` }} />
                    </span>
                    <span className="num">{v.rmse?.toFixed(2)} m</span>
                  </div>
                ))}
              </div>
              <p className="note">
                Share of scene: {classes.map(([n, v]) => `${n} ${v.share_pct?.toFixed(0)}%`).join(' · ')}
              </p>
            </>
          )}

          <div className="figs">
            {['error_map.png', 'scatter.png', 'stability.png'].map((f) => (
              <a key={f} href={fileUrl(job.id, f)} target="_blank" rel="noreferrer">
                <img src={fileUrl(job.id, f)} alt={f} loading="lazy" />
              </a>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
