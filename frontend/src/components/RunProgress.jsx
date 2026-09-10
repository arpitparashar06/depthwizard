import { useEffect, useRef } from 'react'

/* The backend already reports honest progress: _set() moves the number at real
 * checkpoints and _log() names what it is doing. So the stepper reads that,
 * rather than animating a guess. Each stage is "live" when the run has reached
 * it and "done" once the run has moved past it. */
const STAGES = [
  { key: 'read',  label: 'Read',      at: 0.05 },
  { key: 'depth', label: 'Depth',     at: 0.15 },
  { key: 'scale', label: 'Calibrate', at: 0.45 },
  { key: 'refine', label: 'Refine',   at: 0.60 },
  { key: 'mesh',  label: 'Mesh',      at: 0.75 },
  { key: 'done',  label: 'Done',      at: 1.00 }
]

export default function RunProgress ({ job }) {
  const logRef = useRef(null)
  const lines = job?.log || []

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [lines.length])

  if (!job) return null
  const p = job.progress || 0
  const finished = job.status === 'done'
  const failed = job.status === 'error'

  const state = (i) => {
    if (failed) return i === 0 ? 'done' : ''
    if (finished) return 'done'
    const reached = p >= STAGES[i].at
    const next = STAGES[i + 1]
    if (!reached) return ''
    return next && p >= next.at ? 'done' : 'live'
  }

  return (
    <div className="run">
      <div className="track"><div style={{ width: `${Math.round(p * 100)}%` }} /></div>
      <div className="stages">
        {STAGES.map((s, i) => (
          <span key={s.key} className={`st ${state(i)}`}>
            <span className="dot" />{s.label}
          </span>
        ))}
      </div>
      {lines.length > 0 && (
        <pre className="log" ref={logRef}>{lines.join('\n')}</pre>
      )}
    </div>
  )
}
