import { useRef, useState } from 'react'

/* ===========================================================================
 * Validation.jsx - drop a reference raster, get scored against it.
 * ===========================================================================
 *
 * This panel only appears once a run has finished. Drop a LiDAR GeoTIFF on it
 * and the backend reprojects that raster onto our grid, compares the two
 * surfaces and returns the report this file renders. All of the maths is in
 * mathsandml/validate.py; nothing here computes an accuracy number.
 *
 * WHY EVERY SCORE IS QUOTED THREE TIMES. Our surface and a reference DSM are
 * often on different vertical datums - if the coarse DEM could not be fetched,
 * ours is height above LOCAL GROUND while theirs is metres above sea level.
 * Score those against each other raw and you measure the datum, not the model.
 * Quietly subtracting the difference flatters us. So all three are shown side
 * by side and the report says which one it used as the headline:
 *
 *   raw     nothing removed - the honest number
 *   shift   one constant offset removed - normal practice between datums
 *   affine  offset AND scale removed - diagnostic only, it can hide a bad alpha
 *
 * Rendering all three is the point. Do not "simplify" this to one row.
 */
const ALIGN = {
  raw:    ['Raw', 'nothing removed — the honest number'],
  shift:  ['Datum shift', 'one constant offset removed'],
  affine: ['Robust affine', 'offset and scale removed']
}

/* `stamp` is a cache-buster, not decoration: a second validation overwrites the
 * same five filenames in the job folder, so without a version on the URL the
 * browser serves the FIRST run's figures. See App.jsx::onReference. */
export default function Validation ({ job, report, busy, error, onFile, fileUrl,
                                      datum, stamp }) {
  const inputRef = useRef(null)
  const [over, setOver] = useState(false)   // a file is being dragged over the well
  if (!job) return null

  /* Worst landscape first. "Stability across landscapes" is half of what the
   * problem statement marks, so the band we do worst on is the one a judge
   * should see at the top - not the one that happens to sort first by name. */
  const classes = report?.by_landscape
    ? Object.entries(report.by_landscape).sort((a, b) => (b[1].rmse || 0) - (a[1].rmse || 0))
    : []
  // bars are scaled to the worst band, so the shape of the list IS the answer
  // to "is it stable?" - a flat list of similar bars means yes.
  const worst = classes.length ? Math.max(...classes.map(([, v]) => v.rmse || 0)) : 1

  return (
    <div className="panel">
      <p className="section-label">Validate against reference LiDAR</p>
      <p className="note" style={{ margin: '0 0 12px' }}>
        Scored under three alignments, because a no-DEM estimate is height above local
        ground and scoring it raw against a sea-level reference measures the datum
        rather than the model.
      </p>

      {/* The single most important warning in the app. A local-ground surface
          scored raw against a sea-level DSM produces an RMSE of hundreds of
          metres that says nothing about the model, and someone WILL quote it.
          So the panel says so before the numbers appear, not after. */}
      {datum === 'local ground' && (
        <p className="datum risk" style={{ marginTop: 0 }}>
          <span>
            <b>Raw RMSE will be meaningless for this run.</b> No coarse DEM was
            available, so this surface is height above local ground while a reference
            DSM is metres above sea level. The gap between them is a constant of
            hundreds of metres, and it will swamp the raw score. Read the{' '}
            <b>datum shift</b> row — that removes exactly the offset you are missing —
            or score against an nDSM instead.
          </span>
        </p>
      )}

      {/* Same .drop styling as the source-image well, deliberately: it is the
          same gesture, so it should look like the same control. */}
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
              {/* Fixed order, honest first. The headline row is the one
                  validate.py chose, and it is marked rather than reordered so
                  you can still see what the other two said. */}
              {['raw', 'shift', 'affine'].map((k) => {
                const m = report.alignment?.[k]
                if (!m) return null          // the backend may omit an alignment
                const [label, sub] = ALIGN[k]
                return (
                  <tr key={k} className={report.headline_alignment === k ? 'headline' : ''}>
                    {/* on a local-ground run the raw row's caption is
                        rewritten, because there its number is the datum gap
                        and calling it "the honest number" would mislead */}
                    <td>{label}<br /><span style={{ fontSize: 11, color: 'var(--faint)' }}>
                      {k === 'raw' && datum === 'local ground' ? 'datum offset, not model error' : sub}
                    </span></td>
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

          {/* the three figures validate.py wrote, each linking to itself at
              full size. lazy, because they are large PNGs below the fold. */}
          <div className="figs">
            {['error_map.png', 'scatter.png', 'stability.png'].map((f) => (
              <a key={f} href={fileUrl(job.id, f, stamp)} target="_blank" rel="noreferrer">
                <img src={fileUrl(job.id, f, stamp)} alt={f} loading="lazy" />
              </a>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
