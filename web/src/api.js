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
  if (!r.ok) throw new Error(`job ${id}: ${r.status}`)
  return r.json()
}

export function fileUrl (id, name) {
  return `/api/jobs/${id}/files/${name}`
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
export function pollJob (id, onUpdate, intervalMs = 900) {
  let stop = false
  ;(async () => {
    while (!stop) {
      let j
      try { j = await getJob(id) } catch { await sleep(intervalMs); continue }
      onUpdate(j)
      if (j.status === 'done' || j.status === 'error') return
      await sleep(intervalMs)
    }
  })()
  return () => { stop = true }
}

const sleep = (ms) => new Promise((res) => setTimeout(res, ms))
