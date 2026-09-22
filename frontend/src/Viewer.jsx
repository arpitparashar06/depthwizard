import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { PointerLockControls } from 'three/examples/jsm/controls/PointerLockControls.js'

/* ===========================================================================
 * Viewer.jsx - the 3D flythrough. Loads terrain.glb and lets you walk it.
 * ===========================================================================
 *
 * READ THIS FIRST. It is one useEffect that builds a three.js world and one
 * loop() that runs every frame:
 *
 *   setup     scene, camera, renderer, two lights, PointerLockControls
 *   load      GLTFLoader pulls terrain.glb, rotates it Z-up -> Y-up, and puts
 *             the camera above one corner looking at the middle
 *   input     WASD moves, mouse looks, Space/C change altitude, Shift boosts.
 *             The keys are only claimed while the pointer is locked, so they
 *             do not swallow typing in the form
 *   measure   a raycast down the crosshair every ~90 ms gives the height under
 *             the crosshair and the slope of the face there. Clicking drops a
 *             pin; two pins measure the drop and the grade between them
 *   loop      move, raycast, render
 *
 * THE ONE THING TO GET RIGHT. The mesh is vertically exaggerated and its base
 * is offset, so raw world Y is NOT metres. Every number on screen goes back
 * through
 *     true = y / exaggeration + base
 * and slope needs a different inverse:
 *     tan(true) = tan(apparent) / exaggeration
 * This is not cosmetic - reading heights off the model is a deliverable, and
 * at the default 1.5x every measurement would be 50% high.
 *
 * ---------------------------------------------------------------------------
 * TWO MODES.
 *
 *   FLY   the original. Camera free in space, 70 m/s, Space/C for altitude,
 *         terrain drawn at the exported exaggeration so relief reads clearly
 *         from above. Nothing about this mode has changed.
 *
 *   WALK  human scale. Eye at 1.7 m above whatever surface is underfoot,
 *         walking pace, no flying. THE EXAGGERATION IS DROPPED TO 1:1 HERE -
 *         standing next to a building that has been stretched 1.5x would
 *         report 34 m in the HUD while looking 51 m tall, which is exactly
 *         the kind of quiet lie the rest of this file exists to prevent.
 *
 * Switching modes rescales ONE group and swaps two constants. It never
 * reloads the mesh, which is why mode lives in a ref rather than in the
 * effect's dependency list - a dependency would tear down the scene and
 * re-download the .glb on every toggle.
 */
const FLIGHT_KEYS = new Set([
  'KeyW', 'KeyA', 'KeyS', 'KeyD',   // move
  'Space', 'KeyC',                  // altitude (fly mode only)
  'ShiftLeft', 'ShiftRight',        // boost
  'KeyR',                           // clear probes
  'KeyF'                            // toggle fly / walk
])

const EYE_HEIGHT_M = 1.7      // average standing eye height
const XR_STICK_DEADZONE = 0.15
const XR_WALK_SPEED = 1.6     // m/s from the thumbstick
const WALK_SPEED = 1.5        // m/s, an unhurried walk
const RUN_SPEED = 5.0         // m/s with Shift held
const FLY_SPEED = 70
const FLY_BOOST = 260

export default function Viewer({ url, exaggeration = 1, baseM = 0, units = 'm' }) {
  const mount = useRef(null)
  const lockRef = useRef(null)
  const modeRef = useRef('fly')          // read by the loop every frame
  const setModeRef = useRef(null)        // set by the effect, called by the UI
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

    // The vertical scale currently applied to the world. Fly mode keeps the
    // exported exaggeration; walk mode forces 1:1. Every readout divides by
    // this, so the HUD stays in true metres in both modes.
    let vScale = exaggeration || 1
    const toReal = (y) => y / vScale + baseM
    // metres -> world units, vertically. X and Z are already metres.
    const toWorldY = (m) => (m - baseM) * vScale

    const keys = {}
    const scene = new THREE.Scene()
    // matched to .stage's background in styles.css, so there is no seam
    // between the CSS box and the canvas while the .glb downloads
    scene.background = new THREE.Color(0x010110)
    // near plane 0.1 so walls do not clip when you stand against one
    const camera = new THREE.PerspectiveCamera(70, 1, 0.1, 60000)
    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2))
    // WebXR costs nothing when no headset is present - the session simply
    // never starts - so it is always on rather than behind a build flag.
    renderer.xr.enabled = true
    // 'local-floor' means the headset reports height above the REAL floor the
    // wearer is standing on. Combined with a 1:1 world that is what makes a
    // 30 m building read as 30 m to a human body rather than to a number.
    renderer.xr.setReferenceSpaceType('local-floor')
    host.appendChild(renderer.domElement)

    scene.add(new THREE.HemisphereLight(0xdfe6ff, 0x1a1a2a, 2.0))
    const sun = new THREE.DirectionalLight(0xffffff, 1.4)
    sun.position.set(1, 2, 1.5)
    scene.add(sun)

    // The camera lives inside a rig. On the desktop the rig sits at the origin
    // and the camera moves, exactly as before. In XR the headset owns the
    // camera's pose completely - you cannot move it - so the rig is what gets
    // moved instead, and the wearer's own head motion happens on top of it.
    const player = new THREE.Group()
    scene.add(player)

    const controls = new PointerLockControls(camera, renderer.domElement)
    player.add(controls.getObject())
    controls.addEventListener('lock', () => setLocked(true))
    controls.addEventListener('unlock', () => {
      setLocked(false)
      // Release every held key. Pressing Esc mid-flight otherwise leaves the
      // key latched true, and the camera drifts the moment you lock back in.
      for (const k of Object.keys(keys)) keys[k] = false
    })
    lockRef.current = () => controls.lock()

    // The .glb goes inside a plain, unrotated group so that changing the
    // vertical scale is one axis on one object. Scaling the mesh itself would
    // mean scaling its LOCAL Z (it is rotated -90 degrees about X), which is
    // the kind of thing that works until someone changes the rotation.
    const worldGroup = new THREE.Group()
    scene.add(worldGroup)

    let terrain = null
    let bounds = null
    setLoading(true)
    new GLTFLoader().load(
      url,
      (gltf) => {
        terrain = gltf.scene
        terrain.rotation.x = -Math.PI / 2   // glTF is Z-up from a raster grid
        worldGroup.add(terrain)
        const box = new THREE.Box3().setFromObject(worldGroup)
        const size = box.getSize(new THREE.Vector3())
        const mid = box.getCenter(new THREE.Vector3())
        bounds = { box, size, mid }
        scene.fog = new THREE.Fog(0x010110, size.length() * 0.3, size.length() * 1.8)
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

    // ---- ground following -------------------------------------------------
    // A ray fired straight down from well above the camera. Whatever it hits
    // first is the surface underfoot - terrain, a roof, a wall top. Cheaper
    // and far more predictable than physics, and it means you can walk up onto
    // a roof simply by walking onto it.
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

    // ---- mode switching ---------------------------------------------------
    const applyMode = (next) => {
      if (!bounds || next === modeRef.current) return
      modeRef.current = next
      setMode(next)

      const prevScale = vScale
      vScale = next === 'walk' ? 1 : (exaggeration || 1)
      // the .glb was exported WITH the exaggeration baked in, so the group has
      // to undo it to reach 1:1
      worldGroup.scale.y = vScale / (exaggeration || 1)
      worldGroup.updateMatrixWorld(true)

      // keep the pins sitting on the surface after a rescale
      for (const p of pinGroup.children) {
        p.position.y = (p.position.y) * (vScale / prevScale)
      }
      for (const v of pins) v.y = v.y * (vScale / prevScale)

      if (next === 'walk') {
        // drop the viewer into the middle of the scene, on the ground
        const x = bounds.mid.x
        const z = bounds.mid.z
        const ceiling = bounds.box.max.y * (vScale / (exaggeration || 1)) + 1000
        const g = groundUnder(x, z, ceiling)
        camera.position.set(x, (g ?? 0) + EYE_HEIGHT_M * vScale, z)
        // look at the horizon rather than at your feet
        camera.rotation.set(0, camera.rotation.y, 0, 'YXZ')
      } else {
        // back up to a sensible flying altitude above where you were standing
        camera.position.y = camera.position.y * (vScale / prevScale)
                          + bounds.size.y * 0.35
      }
    }
    setModeRef.current = applyMode

    // ---- WebXR ------------------------------------------------------------
    // Ask the browser whether a headset is actually attached. This resolves
    // false on every ordinary laptop, which is the point: the button stays a
    // desktop walk-through there instead of promising something the machine
    // cannot do.
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
        // user declined the headset prompt, or no device - fall back to the
        // desktop walk rather than leaving the button doing nothing
        return false
      }
      // 1:1 scale is not optional in a headset. A body standing next to a
      // building stretched 1.5x is the one place the exaggeration stops being
      // a presentation choice and becomes a false measurement.
      applyMode('walk')
      // In XR the headset writes camera.position every frame, so anything the
      // desktop path put there has to move up to the rig or it is applied
      // twice and the viewer starts 1.7 m underground.
      player.position.set(camera.position.x,
                          camera.position.y - EYE_HEIGHT_M * vScale,
                          camera.position.z)
      camera.position.set(0, 0, 0)
      session.addEventListener('end', () => {
        setInXR(false)
        // hand the pose back to the desktop camera on the way out
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

    // Thumbstick locomotion. A headset has no keyboard, so without this the
    // wearer can look but not walk - which is most of the point.
    const xrMove = new THREE.Vector3()
    const xrFwd = new THREE.Vector3()
    const stepXR = (dt) => {
      const session = renderer.xr.getSession()
      if (!session) return
      let ax = 0, ay = 0
      for (const src of session.inputSources) {
        const gp = src.gamepad
        if (!gp || !gp.axes) continue
        // axes 2/3 are the thumbstick on every mainstream controller profile;
        // 0/1 is the trackpad on the older ones. Take whichever is deflected.
        const [a0 = 0, a1 = 0, a2 = 0, a3 = 0] = gp.axes
        const x = Math.abs(a2) > Math.abs(a0) ? a2 : a0
        const y = Math.abs(a3) > Math.abs(a1) ? a3 : a1
        if (Math.abs(x) > Math.abs(ax)) ax = x
        if (Math.abs(y) > Math.abs(ay)) ay = y
      }
      if (Math.abs(ax) < XR_STICK_DEADZONE) ax = 0
      if (Math.abs(ay) < XR_STICK_DEADZONE) ay = 0
      if (!ax && !ay) return

      // Walk where the wearer is LOOKING, flattened to the ground plane -
      // tilting your head down must not drive you into the terrain.
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
      if (e.code === 'KeyF') applyMode(modeRef.current === 'fly' ? 'walk' : 'fly')
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
    function clearPins () {
      pins.length = 0
      pinGroup.clear()
      setHud((h) => ({ ...h, probe: 'click terrain' }))
    }
    const onClick = () => {
      if (!controls.isLocked) return
      const hit = cast()
      if (!hit) { setHud((h) => ({ ...h, probe: 'no surface under crosshair' })); return }
      // pin radius scales with distance so it stays visible from the air and
      // does not become a beach ball at arm's length
      const r = THREE.MathUtils.clamp(hit.distance * 0.006, 0.15, 30)
      const m = new THREE.Mesh(
        new THREE.SphereGeometry(r, 12, 8),
        new THREE.MeshBasicMaterial({ color: 0x5b4fff }))   // --accent
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

    let last = performance.now(), probeAt = 0
    const vel = new THREE.Vector3()
    const loop = () => {
      const now = performance.now()
      const dt = Math.min((now - last) / 1000, 0.1)
      last = now

      // In a headset the pose belongs to the wearer: PointerLockControls and
      // the WASD path must not touch it, or head motion and code fight for
      // the camera and the result is instant nausea.
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
        // glue the eye to the surface underfoot. Start the ray from above the
        // head, not from the eye, so walking into a building steps onto it
        // instead of the ray starting inside a wall.
        const g = groundUnder(camera.position.x, camera.position.z,
                              camera.position.y + 200)
        if (g !== null) {
          const target = g + EYE_HEIGHT_M * vScale
          // ease rather than snap, so a kerb does not jolt the view
          camera.position.y += (target - camera.position.y) *
                               Math.min(1, dt * 12)
        }
      } else {
        if (keys.Space) camera.position.y += speed * dt
        if (keys.KeyC) camera.position.y -= speed * dt
      }

      // raycasting is linear in triangle count, so throttle rather than run it
      // every frame on a 200k-triangle city
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
    // NOT requestAnimationFrame. Inside an XR session the browser drives
    // frames from the headset's own callback at its own rate, and a rAF loop
    // simply stops being called - the view freezes the moment you put the
    // headset on. setAnimationLoop is three.js's switch between the two.
    renderer.setAnimationLoop(loop)

    return () => {
      lockRef.current = null
      setModeRef.current = null
      enterXRRef.current = null
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
  }, [url, exaggeration, baseM, units])

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
      </div>

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
            <span><kbd>click</kbd> drop a probe</span>
            <span><kbd>R</kbd> clear probes</span>
            <span><kbd>F</kbd> fly / walk</span>
            <span><kbd>Esc</kbd> release cursor</span>
          </div>

          {/* stopPropagation: the overlay itself grabs the pointer lock, and a
              click that both switched mode AND locked would drop the user into
              the new mode before the camera had been moved. */}
          <button
            type="button"
            className="btn ghost mode-toggle"
            onClick={async (e) => {
              e.stopPropagation()
              if (walking) { setModeRef.current?.('fly'); return }
              // Try the headset first. requestSession has to be called from a
              // real user gesture, which is why this lives on the click and
              // not in an effect.
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
