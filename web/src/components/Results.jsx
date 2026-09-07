/* Friendly names for the products. The raw filenames are correct but opaque -
 * "ndsm.tif" tells a judge nothing, "height above ground" tells them what the
 * layer is for. The filename stays visible underneath. */
const FILES = {
  'dsm.tif':         ['Surface model', 'top of everything · DSM'],
  'ndsm.tif':        ['Height above ground', 'buildings and canopy · nDSM'],
  'dtm.tif':         ['Bare terrain', 'ground under the structures · DTM'],
  'uncertainty.tif': ['Confidence map', 'disagreement between passes'],
  'dem_coarse.tif':  ['Coarse DEM', 'COP30 terrain baseline'],
  'terrain.glb':     ['3D model', 'glTF, opens in any viewer'],
  'height16.png':    ['Elevation map', '16-bit greyscale'],
  'texture.png':     ['Texture', 'source image, resampled'],
  'meta.json':       ['Run metadata', 'every parameter and result'],
  'validation.md':   ['Validation report', 'readable summary'],
  'validation.json': ['Validation data', 'machine readable'],
  'error_map.png':   ['Error map', 'figure'],
  'scatter.png':     ['Scatter', 'figure'],
  'stability.png':   ['Stability', 'figure']
}

function Datum ({ res }) {
  if (res.datum === 'sea level') {
    return (
      <p className="datum ok">
        <span>
          <b>Measured from sea level.</b> A coarse DEM supplied the terrain baseline,
          so these are true absolute elevations.
        </span>
      </p>
    )
  }
  if (res.datum === 'local ground') {
    return (
      <p className="datum risk">
        <span>
          <b>Measured from LOCAL GROUND.</b> No coarse DEM was available, so these are
          not sea-level elevations even though the GeoTIFF is tagged absolute.
          Score this against an nDSM, never against an absolute DSM.
        </span>
      </p>
    )
  }
  return (
    <p className="datum rel">
      <span>
        <b>Relative surface.</b> This image carries no coordinate system, so heights
        are scaled to a plausible range rather than measured. Upload a GeoTIFF with a
        CRS for true metres.
      </span>
    </p>
  )
}

export default function Results ({ res, job, fileUrl }) {
  if (!res) return null
  const u = res.datum === 'relative' ? 'm*' : 'm'
  const files = (res.files || []).filter((f) => !f.startsWith('source'))

  return (
    <>
      <div className="stats">
        <div className="stat">
          <span className="k">Relief</span>
          <span className="v">{res.relief_m.toFixed(1)} <small>{u}</small></span>
        </div>
        <div className="stat">
          <span className="k">Median slope</span>
          <span className="v">{res.median_slope_deg.toFixed(1)}<small>°</small></span>
        </div>
        <div className="stat">
          <span className="k">Triangles</span>
          <span className="v">{(res.triangles / 1000).toFixed(0)}<small>k</small></span>
        </div>
        {res.info?.n != null && (
          <div className="stat">
            <span className="k">Buildings</span>
            <span className="v">{res.info.n}</span>
          </div>
        )}
      </div>

      <Datum res={res} />

      <table className="rows">
        <tbody>
          <tr>
            <td>Extent</td>
            <td className="mono">{res.width} × {res.height} px @ {res.px_size_m.toFixed(2)} m/px</td>
          </tr>
          <tr>
            <td>Height range</td>
            <td className="mono">{res.min_m.toFixed(1)} – {res.max_m.toFixed(1)} {u}</td>
          </tr>
          {res.info?.alpha != null && (
            <tr>
              <td>Scale</td>
              <td className="mono">alpha {Number(res.info.alpha).toFixed(3)} m per model unit</td>
            </tr>
          )}
          {res.info?.sigma_m != null && (
            <tr>
              <td>Object / terrain split</td>
              <td className="mono">{Number(res.info.sigma_m).toFixed(0)} m
                <span style={{ color: 'var(--faint)' }}> · from scene structures</span></td>
            </tr>
          )}
          {res.info?.n != null && (
            <tr>
              <td>Tallest extruded</td>
              <td className="mono">{Number(res.info.height_max_m || 0).toFixed(0)} m</td>
            </tr>
          )}
          {res.info?.self_check?.rmse_m != null && (
            <tr>
              <td>Shadow self-check</td>
              <td className="mono">{Number(res.info.self_check.rmse_m).toFixed(2)} m RMSE
                <span style={{ color: 'var(--faint)' }}> · {res.info.self_check.n} held-out points</span></td>
            </tr>
          )}
        </tbody>
      </table>

      <p className="section-label" style={{ marginTop: 24 }}>Download</p>
      <div className="files">
        {files.map((f) => {
          const [name, desc] = FILES[f] || [f, '']
          const ext = (f.split('.').pop() || '').toLowerCase()
          return (
            <a key={f} className="file" href={fileUrl(job.id, f)} download title={f}>
              <span className="ext">{ext}</span>
              <span className="txt">
                <span className="n">{name}</span>
                <span className="d">{desc || f}</span>
              </span>
            </a>
          )
        })}
      </div>
    </>
  )
}
