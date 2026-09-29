
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

// `v` is a cache-buster.
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

async function getJSON (url) {
  const r = await fetch(url)
  if (!r.ok) {
    let detail = ''
    try { detail = (await r.json()).detail || '' } catch { /* not JSON */ }
    throw new Error(detail || `${url}: ${r.status}`)
  }
  return r.json()
}

export const getLayers = (id) => getJSON(`/api/jobs/${id}/layers`)

// A and b are {row, col} in source-image pixels
export const getProfile = (id, a, b, n = 256) =>
  getJSON(`/api/jobs/${id}/profile?r0=${a.row}&c0=${a.col}&r1=${b.row}&c1=${b.col}&n=${n}`)

export const getProbe = (id, p) => getJSON(`/api/jobs/${id}/probe?r=${p.row}&c=${p.col}`)

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
        // Give a flaky connection a fair number of tries before quitting
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
