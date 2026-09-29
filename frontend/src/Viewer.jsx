import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { PointerLockControls } from 'three/examples/jsm/controls/PointerLockControls.js'

const FLIGHT_KEYS = new Set([
  'KeyW', 'KeyA', 'KeyS', 'KeyD',   // move
  'Space', 'KeyC',                  // altitude (fly mode only)
  'ShiftLeft', 'ShiftRight',        // boost
  'KeyR',                           // clear probes
  'KeyF',                           // toggle fly / walk
  'KeyL'                            // cycle analysis layers
])

const EYE_HEIGHT_M = 1.7      // average standing eye height
const XR_STICK_DEADZONE = 0.15
const XR_WALK_SPEED = 1.6     // m/s from the thumbstick
const WALK_SPEED = 1.5        // m/s, an unhurried walk
const RUN_SPEED = 5.0         // m/s with Shift held
const FLY_SPEED = 70
const FLY_BOOST = 260

export default function Viewer({
  url, exaggeration = 1, baseM = 0, units = 'm', pxSize = 1,
  layerUrl = null, legend = null, exact = null, profile = null, profileHover = null,
  onPins, onCycleLayer, clearSignal = 0
}) {
  const mount = useRef(null)
  // Props the long-lived three.js effect reads.
  const onPinsRef = useRef(onPins)
  const onCycleRef = useRef(onCycleLayer)
  const layerUrlRef = useRef(layerUrl)
  const profileRef = useRef(profile)
  const hoverRef = useRef(profileHover)
  const applyLayerRef = useRef(null)
  const drawProfileRef = useRef(null)
  const moveMarkerRef = useRef(null)
  const clearPinsRef = useRef(null)
  onPinsRef.current = onPins
  onCycleRef.current = onCycleLayer
  const lockRef = useRef(null)
  const modeRef = useRef('fly')          // read by the loop every frame
  const setModeRef = useRef(null)
  const [mode, setMode] = useState('fly')
  const [hud, setHud] = useState({ alt: '—', ht: '—', slope: '', probe: 'click terrain' })
  const [locked, setLocked] = useState(false)
  const [xrOk, setXrOk] = useState(false)      // a headset is actually present
  const [inXR, setInXR] = useState(false)
  const enterXRRef = useRef(null)
  const [err, setErr] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!url || !mount.current) return
    const host = mount.current

    // The vertical scale currently applied to the world.
    let vScale = exaggeration || 1
    const toReal = (y) => y / vScale + baseM
    // Metres -> world units, vertically.
    const toWorldY = (m) => (m - baseM) * vScale

    const keys = {}
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x010110)
    // Near plane 0.1 so walls do not clip when you stand against one
    const camera = new THREE.PerspectiveCamera(70, 1, 0.1, 60000)
    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2))
    renderer.xr.enabled = true
    renderer.xr.setReferenceSpaceType('local-floor')
    host.appendChild(renderer.domElement)

    scene.add(new THREE.HemisphereLight(0xdfe6ff, 0x1a1a2a, 2.0))
    const sun = new THREE.DirectionalLight(0xffffff, 1.4)
    sun.position.set(1, 2, 1.5)
    scene.add(sun)

    // The camera lives inside a rig.
    const player = new THREE.Group()
    scene.add(player)

    const controls = new PointerLockControls(camera, renderer.domElement)
    player.add(controls.getObject())
    controls.addEventListener('lock', () => setLocked(true))
    controls.addEventListener('unlock', () => {
      setLocked(false)
      // Release every held key.
      for (const k of Object.keys(keys)) keys[k] = false
    })
    lockRef.current = () => controls.lock()

    const worldGroup = new THREE.Group()
    scene.add(worldGroup)

    let terrain = null
    let bounds = null
    const draped = []

    const texCache = new Map()
    const texLoader = new THREE.TextureLoader()
    const applyLayer = (u) => {
      layerUrlRef.current = u
      if (!draped.length) return
      const put = (tex) => {
        if (layerUrlRef.current !== u) return       // a newer choice won the race
        for (const d of draped) { d.mat.map = tex || d.photo; d.mat.needsUpdate = true }
      }
      if (!u) return put(null)
      if (texCache.has(u)) return put(texCache.get(u))
      texLoader.load(u, (tex) => {
        tex.flipY = false
        tex.colorSpace = THREE.SRGBColorSpace
        tex.anisotropy = renderer.capabilities.getMaxAnisotropy()
        texCache.set(u, tex)
        put(tex)
      })
    }
    applyLayerRef.current = applyLayer

    const profGroup = new THREE.Group()
    scene.add(profGroup)
    const marker = new THREE.Mesh(new THREE.SphereGeometry(1, 16, 12),
      new THREE.MeshBasicMaterial({ color: 0xffffff, depthTest: false }))
    marker.renderOrder = 10
    marker.visible = false
    scene.add(marker)
    let profPts = []
    const clearProfile = () => {
      for (const c of profGroup.children) { c.geometry.dispose(); c.material.dispose() }
      profGroup.clear()
      profPts = []
      marker.visible = false
    }
    const drawProfile = () => {
      clearProfile()
      const pr = profileRef.current
      if (!bounds || !pr || !pr.dsm || pr.dsm.length < 2) return
      const [r0, c0] = pr.start
      const [r1, c1] = pr.end
      const n = pr.dsm.length
      const lift = Math.max(0.3, bounds.size.x * 0.0015)
      for (let i = 0; i < n; i++) {
        const z = pr.dsm[i]
        if (z == null) continue
        const t = n > 1 ? i / (n - 1) : 0
        profPts.push(new THREE.Vector3(
          bounds.box0.min.x + (c0 + t * (c1 - c0)) * pxSize,
          toWorldY(z) + lift,
          bounds.box0.min.z + (r0 + t * (r1 - r0)) * pxSize))
      }
      if (profPts.length < 2) return
      const line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(profPts),
        new THREE.LineBasicMaterial({ color: 0xffffff, depthTest: false, transparent: true, opacity: 0.95 }))
      line.renderOrder = 9
      profGroup.add(line)
      moveMarker(hoverRef.current)
    }
    const moveMarker = (i) => {
      hoverRef.current = i
      const p = (i == null) ? null : profPts[Math.min(profPts.length - 1, Math.max(0, i))]
      marker.visible = !!p
      if (p) {
        marker.position.copy(p)
        marker.scale.setScalar(Math.max(0.4, camera.position.distanceTo(p) * 0.008))
      }
    }
    drawProfileRef.current = drawProfile
    moveMarkerRef.current = moveMarker

    setLoading(true)
    new GLTFLoader().load(
      url,
      (gltf) => {
        terrain = gltf.scene
        terrain.rotation.x = -Math.PI / 2
        worldGroup.add(terrain)
        const box = new THREE.Box3().setFromObject(worldGroup)
        const size = box.getSize(new THREE.Vector3())
        const mid = box.getCenter(new THREE.Vector3())
        bounds = { box, size, mid, box0: box.clone() }
        // Everything that wears the photo: the ground and the roof caps.
        terrain.traverse((o) => {
          if (!o.isMesh || /wall/i.test(o.name)) return
          const mats = Array.isArray(o.material) ? o.material : [o.material]
          for (const mt of mats) if (mt.map) draped.push({ mat: mt, photo: mt.map })
        })
        applyLayer(layerUrlRef.current)
        drawProfile()
        scene.fog = new THREE.Fog(0x010110, size.length() * 0.3, size.length() * 1.8)
        camera.position.set(mid.x - size.x * 0.55,
                            box.max.y + size.y * 0.8 + size.z * 0.25,
                            mid.z + size.z * 0.75)
        camera.lookAt(mid.x, box.min.y, mid.z)
        const e = new THREE.Euler().setFromQuaternion(camera.quaternion, 'YXZ')
        camera.rotation.set(e.x, e.y, 0, 'YXZ')
        setLoading(false)
      },
      undefined,
      () => { setErr('Could not load the mesh'); setLoading(false) }
    )

    const downRay = new THREE.Raycaster()
    const DOWN = new THREE.Vector3(0, -1, 0)
    const probeFrom = new THREE.Vector3()
    const groundUnder = (x, z, startY) => {
      if (!terrain) return null
      probeFrom.set(x, startY, z)
      downRay.set(probeFrom, DOWN)
      const hit = downRay.intersectObject(worldGroup, true)[0]
      return hit ? hit.point.y : null
    }

    // mode switching
    const applyMode = (next) => {
      if (!bounds || next === modeRef.current) return
      modeRef.current = next
      setMode(next)

      const prevScale = vScale
      vScale = next === 'walk' ? 1 : (exaggeration || 1)
      worldGroup.scale.y = vScale / (exaggeration || 1)
      worldGroup.updateMatrixWorld(true)

      // Keep the pins sitting on the surface after a rescale
      for (const p of pinGroup.children) {
        p.position.y = (p.position.y) * (vScale / prevScale)
      }
      for (const v of pins) v.y = v.y * (vScale / prevScale)
      drawProfile()

      if (next === 'walk') {
        // Drop the viewer into the middle of the scene, on the ground
        const x = bounds.mid.x
        const z = bounds.mid.z
        const ceiling = bounds.box.max.y * (vScale / (exaggeration || 1)) + 1000
        const g = groundUnder(x, z, ceiling)
        camera.position.set(x, (g ?? 0) + EYE_HEIGHT_M * vScale, z)
        // Look at the horizon rather than at your feet
        camera.rotation.set(0, camera.rotation.y, 0, 'YXZ')
      } else {
        // Back up to a sensible flying altitude above where you were standing
        camera.position.y = camera.position.y * (vScale / prevScale)
                          + bounds.size.y * 0.35
      }
    }
    setModeRef.current = applyMode

    if (navigator.xr?.isSessionSupported) {
      navigator.xr.isSessionSupported('immersive-vr')
        .then((ok) => setXrOk(!!ok))
        .catch(() => setXrOk(false))
    }

    const enterXR = async () => {
      if (!navigator.xr) return false
      let session
      try {
        session = await navigator.xr.requestSession('immersive-vr', {
          optionalFeatures: ['local-floor', 'bounded-floor']
        })
      } catch (e) {
        return false
      }
      // 1:1 scale is not optional in a headset.
      applyMode('walk')
      player.position.set(camera.position.x,
                          camera.position.y - EYE_HEIGHT_M * vScale,
                          camera.position.z)
      camera.position.set(0, 0, 0)
      session.addEventListener('end', () => {
        setInXR(false)
        // Hand the pose back to the desktop camera on the way out
        camera.position.set(player.position.x,
                            player.position.y + EYE_HEIGHT_M * vScale,
                            player.position.z)
        player.position.set(0, 0, 0)
      })
      await renderer.xr.setSession(session)
      setInXR(true)
      return true
    }
    enterXRRef.current = enterXR

    // Thumbstick locomotion.
    const xrMove = new THREE.Vector3()
    const xrFwd = new THREE.Vector3()
    const stepXR = (dt) => {
      const session = renderer.xr.getSession()
      if (!session) return
      let ax = 0, ay = 0
      for (const src of session.inputSources) {
        const gp = src.gamepad
        if (!gp || !gp.axes) continue
        const [a0 = 0, a1 = 0, a2 = 0, a3 = 0] = gp.axes
        const x = Math.abs(a2) > Math.abs(a0) ? a2 : a0
        const y = Math.abs(a3) > Math.abs(a1) ? a3 : a1
        if (Math.abs(x) > Math.abs(ax)) ax = x
        if (Math.abs(y) > Math.abs(ay)) ay = y
      }
      if (Math.abs(ax) < XR_STICK_DEADZONE) ax = 0
      if (Math.abs(ay) < XR_STICK_DEADZONE) ay = 0
      if (!ax && !ay) return

      const xrCam = renderer.xr.getCamera()
      xrCam.getWorldDirection(xrFwd)
      xrFwd.y = 0
      if (xrFwd.lengthSq() < 1e-9) return
      xrFwd.normalize()
      xrMove.set(-xrFwd.z, 0, xrFwd.x)            // strafe = forward rotated 90
      player.position.addScaledVector(xrFwd, -ay * XR_WALK_SPEED * dt)
      player.position.addScaledVector(xrMove, ax * XR_WALK_SPEED * dt)

      const g = groundUnder(player.position.x, player.position.z,
                            player.position.y + 200)
      if (g !== null) player.position.y += (g - player.position.y) * Math.min(1, dt * 12)
    }

    const owned = () => controls.isLocked ||
      document.pointerLockElement === renderer.domElement

    const onDown = (e) => {
      if (!owned() || !FLIGHT_KEYS.has(e.code)) return
      e.preventDefault()
      keys[e.code] = true
      if (e.code === 'KeyR') clearPins()
      if (e.code === 'KeyF') applyMode(modeRef.current === 'fly' ? 'walk' : 'fly')
      if (e.code === 'KeyL') onCycleRef.current?.()
    }
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
      return ray.intersectObject(worldGroup, true)[0] || null
    }
    const slopeOf = (hit) => {
      if (!hit || !hit.face) return null
      nrm.getNormalMatrix(hit.object.matrixWorld)
      n.copy(hit.face.normal).applyMatrix3(nrm).normalize()
      const apparent = Math.acos(Math.min(1, Math.abs(n.y)))
      // vScale, not exaggeration: in walk mode the world is already 1:1
      return THREE.MathUtils.radToDeg(
        Math.atan(Math.tan(apparent) / vScale))
    }

    const pins = []
    const pinGroup = new THREE.Group()
    scene.add(pinGroup)
    // World metres -> source-image pixels.
    const toPx = (p) => ({
      row: (p.z - bounds.box0.min.z) / pxSize,
      col: (p.x - bounds.box0.min.x) / pxSize
    })
    const reportPins = () => onPinsRef.current?.(bounds ? pins.map(toPx) : [])
    function clearPins () {
      pins.length = 0
      pinGroup.clear()
      setHud((h) => ({ ...h, probe: 'click terrain' }))
      reportPins()
    }
    const onClick = () => {
      if (!controls.isLocked) return
      const hit = cast()
      if (!hit) { setHud((h) => ({ ...h, probe: 'no surface under crosshair' })); return }
      const r = THREE.MathUtils.clamp(hit.distance * 0.006, 0.15, 30)
      const m = new THREE.Mesh(
        new THREE.SphereGeometry(r, 12, 8),
        new THREE.MeshBasicMaterial({ color: 0x5b4fff }))   // --accent
      m.position.copy(hit.point)
      pinGroup.add(m)
      pins.push(hit.point.clone())
      if (pins.length > 2) { pins.shift(); pinGroup.remove(pinGroup.children[0]) }
      reportPins()
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
    clearPinsRef.current = clearPins

    let last = performance.now(), probeAt = 0
    const vel = new THREE.Vector3()
    const loop = () => {
      const now = performance.now()
      const dt = Math.min((now - last) / 1000, 0.1)
      last = now

      if (renderer.xr.isPresenting) {
        stepXR(dt)
        renderer.render(scene, camera)
        return
      }

      const walking = modeRef.current === 'walk'
      const boost = keys.ShiftLeft || keys.ShiftRight
      const speed = walking ? (boost ? RUN_SPEED : WALK_SPEED)
                            : (boost ? FLY_BOOST : FLY_SPEED)
      vel.set(0, 0, 0)
      if (keys.KeyW) vel.z += 1
      if (keys.KeyS) vel.z -= 1
      if (keys.KeyA) vel.x -= 1
      if (keys.KeyD) vel.x += 1
      if (vel.lengthSq() > 0) vel.normalize()
      controls.moveForward(vel.z * speed * dt)
      controls.moveRight(vel.x * speed * dt)

      if (walking) {
        // Glue the eye to the surface underfoot.
        const g = groundUnder(camera.position.x, camera.position.z,
                              camera.position.y + 200)
        if (g !== null) {
          const target = g + EYE_HEIGHT_M * vScale
          // Ease rather than snap, so a kerb does not jolt the view
          camera.position.y += (target - camera.position.y) *
                               Math.min(1, dt * 12)
        }
      } else {
        if (keys.Space) camera.position.y += speed * dt
        if (keys.KeyC) camera.position.y -= speed * dt
      }

      if (controls.isLocked && now - probeAt > 90) {
        probeAt = now
        const hit = cast()
        const s = slopeOf(hit)
        setHud((h) => ({ ...h,
          alt: toReal(camera.position.y).toFixed(walking ? 1 : 0),
          ht: hit ? toReal(hit.point.y).toFixed(1) : '—',
          slope: s === null ? '' : `slope ${s.toFixed(0)}°` }))
      }
      renderer.render(scene, camera)
    }
    // Not requestAnimationFrame.
    renderer.setAnimationLoop(loop)

    return () => {
      lockRef.current = null
      setModeRef.current = null
      enterXRRef.current = null
      applyLayerRef.current = null
      drawProfileRef.current = null
      moveMarkerRef.current = null
      clearPinsRef.current = null
      clearProfile()
      for (const t of texCache.values()) t.dispose()
      renderer.setAnimationLoop(null)
      renderer.xr.getSession()?.end().catch(() => {})
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
  }, [url, exaggeration, baseM, units, pxSize])

  useEffect(() => { layerUrlRef.current = layerUrl; applyLayerRef.current?.(layerUrl) }, [layerUrl])
  useEffect(() => { profileRef.current = profile; drawProfileRef.current?.() }, [profile])
  useEffect(() => { hoverRef.current = profileHover; moveMarkerRef.current?.(profileHover) }, [profileHover])
  useEffect(() => { if (clearSignal) clearPinsRef.current?.() }, [clearSignal])

  if (!url) return null

  const walking = mode === 'walk'

  return (
    <div style={{ position: 'absolute', inset: 0 }} ref={mount}>
      <div className="hud">
        <div className="k">{walking ? 'eye height' : 'altitude'}</div>
        <div className="v">{hud.alt} <small>{units}</small></div>
        <hr />
        <div className="k">crosshair</div>
        <div className="v">{hud.ht} <small>{units}{hud.slope ? ` · ${hud.slope}` : ''}</small></div>
        <hr />
        <div className="k">probe</div>
        <div className="probe">{hud.probe}</div>
        {exact && exact.length > 0 && (
          <>
            <hr />
            <div className="k">from the GeoTIFF</div>
            {exact.map((e, i) => e && (
              <div className="probe exact" key={i}>
                <b>{exact.length > 1 ? (i ? 'B ' : 'A ') : ''}{e.dsm != null ? e.dsm.toFixed(2) : '—'} {units}</b>
                {e.uncertainty != null && <> ±{e.uncertainty.toFixed(2)}</>}
                {e.ndsm != null && <> · {e.ndsm.toFixed(1)} above ground</>}
                {e.reference != null && <> · LiDAR {e.reference.toFixed(2)}</>}
              </div>
            ))}
          </>
        )}
      </div>

      {legend && (
        <div className="legend" aria-label={`${legend.label} layer legend`}>
          <div className="k">{legend.label}</div>
          <div className="ramp" style={{ background: `linear-gradient(90deg, ${legend.stops.join(', ')})` }} />
          <div className="ends">
            <span>{fmtLegend(legend.vmin)}</span>
            {legend.kind === 'diverging' && <span>0</span>}
            <span>{fmtLegend(legend.vmax)} {legend.unit}</span>
          </div>
          {legend.note && <div className="note2">{legend.note}</div>}
        </div>
      )}

      {inXR && (
        <p className="viewer-note">
          In the headset. Thumbstick to walk, headset menu to exit.
        </p>
      )}

      {locked && <div className="cross" />}

      {locked && (
        <p className="viewer-note">
          {walking
            ? 'Standing at 1.7 m, 1:1 scale — no vertical exaggeration. Press F to fly.'
            : 'Readouts are true metres — the vertical exaggeration is divided back out.'}
        </p>
      )}

      {!locked && (
        <div className="overlay" onClick={() => lockRef.current?.()}>
          <span className="cta">
            {loading ? 'Loading mesh…' : err || (walking ? 'Click to walk' : 'Click to fly')}
          </span>

          <div className="keys">
            <span><kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> move</span>
            <span>mouse look</span>
            <span><kbd>Shift</kbd> {walking ? 'run' : 'boost'}</span>
            {!walking && <span><kbd>Space</kbd><kbd>C</kbd> altitude</span>}
          </div>
          <div className="keys">
            <span><kbd>click</kbd> drop a probe · two draw a profile</span>
            <span><kbd>R</kbd> clear probes</span>
            <span><kbd>F</kbd> fly / walk</span>
            <span><kbd>L</kbd> layers</span>
            <span><kbd>Esc</kbd> release cursor</span>
          </div>

          <button
            type="button"
            className="btn ghost mode-toggle"
            onClick={async (e) => {
              e.stopPropagation()
              if (walking) { setModeRef.current?.('fly'); return }
              // Try the headset first.
              const entered = xrOk ? await enterXRRef.current?.() : false
              if (!entered) setModeRef.current?.('walk')   // desktop fallback
            }}
            disabled={loading || !!err}
          >
            {walking ? 'Back to flythrough' : 'Enter VR mode'}
          </button>

          {!walking && (
            <p className="viewer-note" style={{ marginTop: 8 }}>
              {xrOk
                ? 'Headset detected — 1:1 scale, thumbstick to walk.'
                : 'No headset here — opens at 1.7 m eye height, 1:1 scale, mouse and WASD.'}
            </p>
          )}

        </div>
      )}
    </div>
  )
}

function fmtLegend (v) {
  if (v == null || !isFinite(v)) return '—'
  const a = Math.abs(v)
  return a >= 100 ? v.toFixed(0) : a >= 10 ? v.toFixed(1) : v.toFixed(2)
}
