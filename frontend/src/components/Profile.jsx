import { useEffect, useMemo, useRef, useState } from 'react'

const M = { l: 52, r: 76, t: 14, b: 30 }
const H = 230
const SERIES = { est: '#2a78d6', ref: '#eb6834' }

function ticks (lo, hi, n = 5) {
  const span = Math.max(hi - lo, 1e-9)
  const raw = span / n
  const mag = Math.pow(10, Math.floor(Math.log10(raw)))
  const step = [1, 2, 2.5, 5, 10].map((k) => k * mag).find((s) => s >= raw) || raw
  const out = []
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(6))
  return out
}
const f1 = (v) => (v == null || !isFinite(v) ? '—' : v.toFixed(1))
const f2 = (v) => (v == null || !isFinite(v) ? '—' : v.toFixed(2))

export default function Profile ({ profile, units = 'm', hover, onHover, onClear }) {
  const box = useRef(null)
  const [w, setW] = useState(640)
  useEffect(() => {
    if (!box.current) return
    const ro = new ResizeObserver(() => setW(box.current?.clientWidth || 640))
    ro.observe(box.current)
    setW(box.current.clientWidth || 640)
    return () => ro.disconnect()
  }, [])

  const d = profile
  const n = d?.dsm?.length || 0
  const hasRef = !!d?.reference && d.reference.some((v) => v != null)
  const hasUnc = !!d?.uncertainty && d.uncertainty.some((v) => v != null && v > 0)

  const geo = useMemo(() => {
    if (!n) return null
    const xs = d.distance_m
    const vals = []
    for (let i = 0; i < n; i++) {
      const z = d.dsm[i]
      if (z != null) {
        vals.push(z)
        if (hasUnc && d.uncertainty[i] != null) vals.push(z - d.uncertainty[i], z + d.uncertainty[i])
      }
      if (hasRef && d.reference[i] != null) vals.push(d.reference[i])
    }
    let lo = Math.min(...vals), hi = Math.max(...vals)
    const pad = Math.max((hi - lo) * 0.08, 0.5)
    lo -= pad; hi += pad
    const pw = Math.max(w - M.l - M.r, 60)
    const ph = H - M.t - M.b
    const X = (v) => M.l + (v / Math.max(d.length_m, 1e-9)) * pw
    const Y = (v) => M.t + (1 - (v - lo) / (hi - lo)) * ph
    const path = (arr) => {
      let s = '', pen = false
      for (let i = 0; i < n; i++) {
        const v = arr[i]
        if (v == null) { pen = false; continue }
        s += `${pen ? 'L' : 'M'}${X(xs[i]).toFixed(1)},${Y(v).toFixed(1)}`
        pen = true
      }
      return s
    }
    let band = ''
    if (hasUnc) {
      const up = [], dn = []
      for (let i = 0; i < n; i++) {
        const z = d.dsm[i], u = d.uncertainty[i]
        if (z == null || u == null) continue
        up.push(`${X(xs[i]).toFixed(1)},${Y(z + u).toFixed(1)}`)
        dn.push(`${X(xs[i]).toFixed(1)},${Y(z - u).toFixed(1)}`)
      }
      if (up.length > 1) band = `M${up.join('L')}L${dn.reverse().join('L')}Z`
    }
    const last = (arr) => { for (let i = n - 1; i >= 0; i--) if (arr[i] != null) return arr[i]; return null }
    return { X, Y, lo, hi, pw, ph, xs, band,
             est: path(d.dsm), ref: hasRef ? path(d.reference) : '',
             yt: ticks(lo, hi, 4), xt: ticks(0, d.length_m, Math.max(3, Math.floor(pw / 90))),
             lastEst: last(d.dsm), lastRef: hasRef ? last(d.reference) : null }
  }, [d, n, w, hasRef, hasUnc])

  const csv = useMemo(() => {
    if (!n) return null
    const cols = ['distance_m', 'dsm', 'uncertainty', 'ndsm', 'dtm', 'reference'].filter((k) => d[k])
    const rows = [cols.join(',')]
    for (let i = 0; i < n; i++) rows.push(cols.map((k) => (d[k][i] ?? '')).join(','))
    return URL.createObjectURL(new Blob([rows.join('\n')], { type: 'text/csv' }))
  }, [d, n])
  useEffect(() => () => { if (csv) URL.revokeObjectURL(csv) }, [csv])

  if (!d || !geo) return null

  const pick = (clientX) => {
    const r = box.current.querySelector('svg').getBoundingClientRect()
    const x = clientX - r.left
    const t = (x - M.l) / geo.pw
    const i = Math.round(Math.min(1, Math.max(0, t)) * (n - 1))
    onHover?.(i)
  }
  const onKey = (e) => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End', 'Escape'].includes(e.key)) return
    e.preventDefault()
    const cur = hover ?? 0
    const step = e.shiftKey ? 10 : 1
    const i = e.key === 'ArrowLeft' ? cur - step : e.key === 'ArrowRight' ? cur + step
      : e.key === 'Home' ? 0 : e.key === 'End' ? n - 1 : null
    onHover?.(i == null ? null : Math.min(n - 1, Math.max(0, i)))
  }

  const hi = hover != null && hover >= 0 && hover < n ? hover : null
  const hx = hi != null ? geo.X(geo.xs[hi]) : null
  const ez = hi != null ? d.dsm[hi] : null
  const rz = hi != null && hasRef ? d.reference[hi] : null
  const vr = d.vs_reference

  // Direct labels at the right end, nudged apart when the lines finish close
  let yEst = geo.lastEst != null ? geo.Y(geo.lastEst) : null
  let yRef = geo.lastRef != null ? geo.Y(geo.lastRef) : null
  if (yEst != null && yRef != null && Math.abs(yEst - yRef) < 14) {
    const mid = (yEst + yRef) / 2
    if (yEst <= yRef) { yEst = mid - 7; yRef = mid + 7 } else { yEst = mid + 7; yRef = mid - 7 }
  }

  return (
    <div className="profile">
      <div className="profile-head">
        <p className="section-label" style={{ margin: 0 }}>Elevation profile</p>
        <div className="profile-actions">
          {csv && <a className="btn ghost sm" href={csv} download="profile.csv">Table (CSV)</a>}
          {onClear && <button type="button" className="btn ghost sm" onClick={onClear}>Clear</button>}
        </div>
      </div>

      <div className="stats" style={{ marginTop: 12 }}>
        <div className="stat"><span className="k">Length</span>
          <span className="v">{f1(d.length_m)} <small>m</small></span></div>
        <div className="stat"><span className="k">Δh A→B</span>
          <span className="v">{d.dh_m != null ? (d.dh_m > 0 ? '+' : '') + f2(d.dh_m) : '—'} <small>{units}</small></span></div>
        <div className="stat"><span className="k">Grade</span>
          <span className="v">{f1(d.grade_deg)}<small>°</small></span></div>
        <div className="stat"><span className="k">Steepest 2 m</span>
          <span className="v">{f1(d.max_slope_deg)}<small>°</small></span></div>
        {vr && (
          <div className="stat" title={`median offset ${f2(vr.median_offset_m)} m removed for the shifted figure`}>
            <span className="k">RMSE vs LiDAR</span>
            <span className="v">{f2(vr.rmse_m)} <small>m raw</small></span></div>
        )}
        {vr && (
          <div className="stat" title="after removing the median vertical offset along this line">
            <span className="k">…datum-shifted</span>
            <span className="v">{f2(vr.rmse_shift_m)} <small>m</small></span></div>
        )}
      </div>

      <div className="legend-row" aria-hidden="true">
        <span><i style={{ background: SERIES.est }} />Estimate (dsm.tif)</span>
        {hasUnc && <span><i className="band" style={{ background: SERIES.est }} />± ensemble uncertainty</span>}
        {hasRef && <span><i style={{ background: SERIES.ref }} />Reference (validated LiDAR)</span>}
      </div>

      <div className="chart" ref={box}>
        <svg width={w} height={H} role="img" tabIndex={0} onKeyDown={onKey}
             aria-label={`Elevation profile, ${f1(d.length_m)} m long, height change ${f2(d.dh_m)} ${units}. Arrow keys move the readout.`}
             onPointerMove={(e) => pick(e.clientX)} onPointerLeave={() => onHover?.(null)}
             onBlur={() => onHover?.(null)}>
          {geo.yt.map((v) => (
            <g key={`y${v}`}>
              <line x1={M.l} x2={M.l + geo.pw} y1={geo.Y(v)} y2={geo.Y(v)} className="grid" />
              <text x={M.l - 8} y={geo.Y(v)} className="tick" textAnchor="end" dominantBaseline="middle">{v}</text>
            </g>
          ))}
          {geo.xt.map((v) => (
            <text key={`x${v}`} x={geo.X(v)} y={H - M.b + 18} className="tick" textAnchor="middle">{v}</text>
          ))}
          <line x1={M.l} x2={M.l + geo.pw} y1={M.t + geo.ph} y2={M.t + geo.ph} className="axis" />
          <text x={M.l + geo.pw} y={H - 2} className="tick" textAnchor="end">distance from A (m)</text>
          <text x={12} y={M.t + geo.ph / 2} className="tick" textAnchor="middle"
                transform={`rotate(-90 12 ${M.t + geo.ph / 2})`}>{units === 'm*' ? 'height (nominal m)' : 'elevation (m)'}</text>

          {geo.band && <path d={geo.band} fill={SERIES.est} opacity="0.16" />}
          {geo.ref && <path d={geo.ref} fill="none" stroke={SERIES.ref} strokeWidth="2" strokeLinejoin="round" />}
          <path d={geo.est} fill="none" stroke={SERIES.est} strokeWidth="2" strokeLinejoin="round" />

          {yEst != null && <text x={M.l + geo.pw + 8} y={yEst} className="dlabel" dominantBaseline="middle">Estimate</text>}
          {yRef != null && <text x={M.l + geo.pw + 8} y={yRef} className="dlabel" dominantBaseline="middle">LiDAR</text>}

          <text x={M.l + 4} y={M.t + 10} className="ab">A</text>
          <text x={M.l + geo.pw - 4} y={M.t + 10} className="ab" textAnchor="end">B</text>

          {hx != null && (
            <g pointerEvents="none">
              <line x1={hx} x2={hx} y1={M.t} y2={M.t + geo.ph} className="cross-line" />
              {ez != null && <circle cx={hx} cy={geo.Y(ez)} r="4.5" fill={SERIES.est} stroke="#fff" strokeWidth="2" />}
              {rz != null && <circle cx={hx} cy={geo.Y(rz)} r="4.5" fill={SERIES.ref} stroke="#fff" strokeWidth="2" />}
            </g>
          )}
        </svg>

        {hi != null && (
          <div className="tip" style={{ left: Math.min(Math.max(hx + 12, 8), w - 190), top: 8 }}>
            <div className="tip-k">{f1(geo.xs[hi])} m from A</div>
            <div className="tip-row"><i style={{ background: SERIES.est }} />
              <b>{f2(ez)} {units}</b>{hasUnc && d.uncertainty[hi] != null && <> ±{f2(d.uncertainty[hi])}</>}<span>estimate</span></div>
            {d.ndsm && d.ndsm[hi] != null && (
              <div className="tip-row"><i className="none" /><b>{f1(d.ndsm[hi])} {units}</b><span>above ground</span></div>
            )}
            {hasRef && (
              <div className="tip-row"><i style={{ background: SERIES.ref }} />
                <b>{f2(rz)} m</b><span>LiDAR{ez != null && rz != null ? ` · Δ ${(ez - rz > 0 ? '+' : '') + f2(ez - rz)}` : ''}</span></div>
            )}
          </div>
        )}
      </div>
      <p className="note">
        Sampled from the GeoTIFFs along the straight line between the two probe pins,
        not read off the display mesh. Hover or use the arrow keys to move the marker on
        the 3D line; the CSV is the same data as a table.
      </p>
    </div>
  )
}
