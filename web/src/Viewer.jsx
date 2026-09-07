import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { PointerLockControls } from 'three/examples/jsm/controls/PointerLockControls.js'

/*
 * The mesh is vertically exaggerated and its base is offset, so raw world Y is
 * not metres. Every number on screen is put back through
 *     true = y / exaggeration + base
 * and slope needs a different inverse:
 *     tan(true) = tan(apparent) / exaggeration
 * Getting this wrong is not cosmetic — reading heights off the model is a
 * deliverable, and at the default 1.5x every measurement would be 50% high.
 */
const FLIGHT_KEYS = new Set([
  'KeyW', 'KeyA', 'KeyS', 'KeyD',   // move
  'Space', 'KeyC',                  // altitude
  'ShiftLeft', 'ShiftRight',        // boost
  'KeyR'                            // clear probes
])

export default function Viewer({ url, exaggeration = 1, baseM = 0, units = 'm' }) {
  const mount = useRef(null)
  const lockRef = useRef(null)
  const [hud, setHud] = useState({ alt: '—', ht: '—', slope: '', probe: 'click terrain' })
  const [locked, setLocked] = useState(false)
  const [err, setErr] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!url || !mount.current) return
    const host = mount.current
    const toReal = (y) => y / (exaggeration || 1) + baseM

    const keys = {}
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x07090c)
    const camera = new THREE.PerspectiveCamera(70, 1, 0.5, 60000)
    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2))
    host.appendChild(renderer.domElement)

    scene.add(new THREE.HemisphereLight(0xdfefff, 0x2a2620, 2.0))
    const sun = new THREE.DirectionalLight(0xffffff, 1.4)
    sun.position.set(1, 2, 1.5)
    scene.add(sun)

    const controls = new PointerLockControls(camera, renderer.domElement)
    scene.add(controls.getObject())
    controls.addEventListener('lock', () => setLocked(true))
    controls.addEventListener('unlock', () => {
      setLocked(false)
      // Release every held key. Pressing Esc mid-flight otherwise leaves the
      // key latched true, and the camera drifts the moment you lock back in.
      for (const k of Object.keys(keys)) keys[k] = false
    })
    lockRef.current = () => controls.lock()

    let terrain = null
    setLoading(true)
    new GLTFLoader().load(
      url,
      (gltf) => {
        terrain = gltf.scene
        terrain.rotation.x = -Math.PI / 2   // glTF is Z-up from a raster grid
        scene.add(terrain)
        const box = new THREE.Box3().setFromObject(terrain)
        const size = box.getSize(new THREE.Vector3())
        const mid = box.getCenter(new THREE.Vector3())
        scene.fog = new THREE.Fog(0x07090c, size.length() * 0.3, size.length() * 1.8)
        camera.position.set(mid.x - size.x * 0.55,
                            box.max.y + size.y * 0.8 + size.z * 0.25,
                            mid.z + size.z * 0.75)
        camera.lookAt(mid.x, box.min.y, mid.z)
        // PointerLockControls drives camera.rotation, so hand lookAt's result
        // over as euler angles or the first mouse move snaps the view
        const e = new THREE.Euler().setFromQuaternion(camera.quaternion, 'YXZ')
        camera.rotation.set(e.x, e.y, 0, 'YXZ')
        setLoading(false)
      },
      undefined,
      () => { setErr('Could not load the mesh'); setLoading(false) }
    )

    // Space scrolls the page and W/A/S/D type into whatever is focused, so the
    // flight keys have to be claimed with preventDefault. But this listener is
    // on window, so claiming them unconditionally would swallow keystrokes
    // while someone is filling in the form. Only take them while the pointer
    // is actually locked - which is exactly when the viewer owns the input.
    // Two sources of truth for "the viewer owns the keyboard": three.js's own
    // flag, and the browser's. They agree in practice, but the browser's is
    // authoritative and costs nothing to check, so a lag in the three.js
    // listener can never let Space through to scroll the page.
    const owned = () => controls.isLocked ||
      document.pointerLockElement === renderer.domElement

    const onDown = (e) => {
      if (!owned() || !FLIGHT_KEYS.has(e.code)) return
      e.preventDefault()
      keys[e.code] = true
      if (e.code === 'KeyR') clearPins()
    }
    // Key-up always clears, locked or not: if the lock drops between press and
    // release the key would otherwise stay down forever.
    const onUp = (e) => {
      if (!FLIGHT_KEYS.has(e.code)) return
      if (owned()) e.preventDefault()
      keys[e.code] = false
    }
    addEventListener('keydown', onDown)
    addEventListener('keyup', onUp)

    const resize = () => {
      const w = host.clientWidth, h = host.clientHeight
      if (!w || !h) return
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      renderer.setSize(w, h, false)
    }
    resize()
    const ro = new ResizeObserver(resize)
    ro.observe(host)

    const ray = new THREE.Raycaster()
    const centre = new THREE.Vector2(0, 0)
    const nrm = new THREE.Matrix3()
    const n = new THREE.Vector3()
    const cast = () => {
      if (!terrain) return null
      ray.setFromCamera(centre, camera)
      return ray.intersectObject(terrain, true)[0] || null
    }
    const slopeOf = (hit) => {
      if (!hit || !hit.face) return null
      nrm.getNormalMatrix(hit.object.matrixWorld)
      n.copy(hit.face.normal).applyMatrix3(nrm).normalize()
      const apparent = Math.acos(Math.min(1, Math.abs(n.y)))
      return THREE.MathUtils.radToDeg(
        Math.atan(Math.tan(apparent) / (exaggeration || 1)))
    }

    const pins = []
    const pinGroup = new THREE.Group()
    scene.add(pinGroup)
    function clearPins () {
      pins.length = 0
      pinGroup.clear()
      setHud((h) => ({ ...h, probe: 'click terrain' }))
    }
    const onClick = () => {
      if (!controls.isLocked) return
      const hit = cast()
      if (!hit) { setHud((h) => ({ ...h, probe: 'no surface under crosshair' })); return }
      const m = new THREE.Mesh(
        new THREE.SphereGeometry(Math.max(1, hit.distance * 0.006), 12, 8),
        new THREE.MeshBasicMaterial({ color: 0x5fd3f5 }))
      m.position.copy(hit.point)
      pinGroup.add(m)
      pins.push(hit.point.clone())
      if (pins.length > 2) { pins.shift(); pinGroup.remove(pinGroup.children[0]) }
      if (pins.length === 2) {
        const [a, b] = pins
        const ground = Math.hypot(b.x - a.x, b.z - a.z)
        const dz = toReal(b.y) - toReal(a.y)
        const grade = THREE.MathUtils.radToDeg(Math.atan2(dz, ground))
        setHud((h) => ({ ...h,
          probe: `Δh ${dz.toFixed(1)} ${units} over ${ground.toFixed(1)} m (${grade.toFixed(1)}°)` }))
      } else {
        setHud((h) => ({ ...h,
          probe: `pin at ${toReal(hit.point.y).toFixed(1)} ${units} — click again to measure` }))
      }
    }
    renderer.domElement.addEventListener('mousedown', onClick)

    let last = performance.now(), probeAt = 0, raf = 0
    const vel = new THREE.Vector3()
    const loop = () => {
      raf = requestAnimationFrame(loop)
      const now = performance.now()
      const dt = Math.min((now - last) / 1000, 0.1)
      last = now
      const speed = (keys.ShiftLeft || keys.ShiftRight) ? 260 : 70
      vel.set(0, 0, 0)
      if (keys.KeyW) vel.z += 1
      if (keys.KeyS) vel.z -= 1
      if (keys.KeyA) vel.x -= 1
      if (keys.KeyD) vel.x += 1
      if (vel.lengthSq() > 0) vel.normalize()
      controls.moveForward(vel.z * speed * dt)
      controls.moveRight(vel.x * speed * dt)
      if (keys.Space) camera.position.y += speed * dt
      if (keys.KeyC) camera.position.y -= speed * dt

      // raycasting is linear in triangle count, so throttle rather than run it
      // every frame on a 200k-triangle city
      if (controls.isLocked && now - probeAt > 90) {
        probeAt = now
        const hit = cast()
        const s = slopeOf(hit)
        setHud((h) => ({ ...h,
          alt: toReal(camera.position.y).toFixed(0),
          ht: hit ? toReal(hit.point.y).toFixed(1) : '—',
          slope: s === null ? '' : `slope ${s.toFixed(0)}°` }))
      }
      renderer.render(scene, camera)
    }
    loop()

    return () => {
      lockRef.current = null
      cancelAnimationFrame(raf)
      ro.disconnect()
      removeEventListener('keydown', onDown)
      removeEventListener('keyup', onUp)
      renderer.domElement.removeEventListener('mousedown', onClick)
      controls.dispose()
      renderer.dispose()
      scene.traverse((o) => {
        if (o.geometry) o.geometry.dispose()
        if (o.material) {
          const mats = Array.isArray(o.material) ? o.material : [o.material]
          mats.forEach((mt) => { if (mt.map) mt.map.dispose(); mt.dispose() })
        }
      })
      if (renderer.domElement.parentNode === host) host.removeChild(renderer.domElement)
    }
  }, [url, exaggeration, baseM, units])

  if (!url) return null

  return (
    <div style={{ position: 'absolute', inset: 0 }} ref={mount}>
      <div className="hud">
        <div className="k">altitude</div>
        <div className="v">{hud.alt} <small>{units}</small></div>
        <hr />
        <div className="k">crosshair</div>
        <div className="v">{hud.ht} <small>{units}{hud.slope ? ` · ${hud.slope}` : ''}</small></div>
        <hr />
        <div className="k">probe</div>
        <div className="probe">{hud.probe}</div>
      </div>

      {locked && <div className="cross" />}

      {locked && (
        <p className="viewer-note">
          Readouts are true metres — the vertical exaggeration is divided back out.
        </p>
      )}

      {!locked && (
        <div className="overlay" onClick={() => lockRef.current?.()}>
          <span className="cta">
            {loading ? 'Loading mesh…' : err || 'Click to fly'}
          </span>
          <div className="keys">
            <span><kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> move</span>
            <span>mouse look</span>
            <span><kbd>Shift</kbd> boost</span>
            <span><kbd>Space</kbd><kbd>C</kbd> altitude</span>
          </div>
          <div className="keys">
            <span><kbd>click</kbd> drop a probe</span>
            <span><kbd>R</kbd> clear probes</span>
            <span><kbd>Esc</kbd> release cursor</span>
          </div>
        </div>
      )}
    </div>
  )
}
