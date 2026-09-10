// Thin wrapper over the FastAPI backend. Vite proxies /api to :8000 in dev,
// and in production server.py serves this bundle from the same origin, so the
// relative paths work in both without a base-URL switch.

export async function createJob (file, params) {
  const fd = new FormData()
  fd.append('file', file)
  fd.append('params', JSON.stringify(params))
  const r = await fetch('/api/jobs', { method: 'POST', body: fd })
  if (!r.ok) throw new Error(await r.text())
  return r.json()
}

export async function getJob (id) {
  const r = await fetch(`/api/jobs/${id}`)
  if (!r.ok) {
    let detail = ''
    try { detail = (await r.json()).detail || '' } catch { /* not JSON */ }
    const e = new Error(detail || `job ${id}: ${r.status}`)
    e.status = r.status
    throw e
  }
  return r.json()
}

// `v` is a cache-buster. The validator overwrites error_map.png / scatter.png /
// stability.png at the SAME url on every run, so without it a second reference
// raster shows the first one's figures.
export function fileUrl (id, name, v) {
  return `/api/jobs/${id}/files/${name}${v ? `?v=${v}` : ''}`
}

export async function validateJob (id, referenceFile, sigmaM = 15) {
  const fd = new FormData()
  fd.append('reference', referenceFile)
  fd.append('sigma_m', String(sigmaM))
  const r = await fetch(`/api/jobs/${id}/validate`, { method: 'POST', body: fd })
  if (!r.ok) throw new Error(await r.text())
  return r.json()
}

// Poll until the job leaves the running state. The Large backbone on CPU takes
// minutes, so a single blocking request would time out in the browser.
//
// Not every failed poll is the same. A dropped connection or a server still
// starting up is worth retrying. A 404 or 410 is permanent - the job is gone,
// usually because the server was restarted - and retrying that forever leaves
// the user watching a spinner with no explanation, which is what used to
// happen. Give up on those and say why.
const PERMANENT = new Set([404, 410])

export function pollJob (id, onUpdate, intervalMs = 900) {
  let stop = false
  ;(async () => {
    let transient = 0
    while (!stop) {
      let j
      try {
        j = await getJob(id)
        transient = 0
      } catch (e) {
        if (PERMANENT.has(e.status)) {
          onUpdate({ id, status: 'error', progress: 0, log: [], result: null,
                     error: e.message })
          return
        }
        // give a flaky connection a fair number of tries before quitting
        if (++transient > 20) {
          onUpdate({ id, status: 'error', progress: 0, log: [], result: null,
                     error: 'Lost contact with the server. Is it still running?' })
          return
        }
        await sleep(intervalMs)
        continue
      }
      onUpdate(j)
      if (j.status === 'done' || j.status === 'error') return
      await sleep(intervalMs)
    }
  })()
  return () => { stop = true }
}

const sleep = (ms) => new Promise((res) => setTimeout(res, ms))
