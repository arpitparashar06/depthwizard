"""
viewer.py - writes a single-file three.js flythrough next to the .glb.

Why this exists: gr.Model3D only orbits. The brief asks for first-person
navigation and for reading structural heights and slopes from arbitrary aerial
perspectives, so this adds WASD + mouse-look, a live crosshair probe, and a
two-point measure. No build step and no npm - three.js comes from a CDN and the
page reads terrain.glb sitting beside it.

The one thing that is easy to get wrong: the mesh is vertically exaggerated, so
raw Y is NOT metres. Every number on screen is divided back down by the
exaggeration and offset by the mesh base before it is shown. Slope has to be
un-exaggerated too, and that is not the same operation - it is
tan(true) = tan(apparent) / exaggeration.

Serve it (file:// blocks GLTFLoader's fetch):
    cd outputs && python -m http.server 8000
    open http://localhost:8000/viewer.html
app.py does this for you on its own port.
"""

import os

_HTML = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Terrain flythrough</title>
<style>
  :root{--ink:#e6edf5;--dim:#8e9cad;--key:#7fd1ff;--panel:rgba(12,15,20,.84);--line:#28303a}
  *{box-sizing:border-box}
  body{margin:0;overflow:hidden;background:#0d0f12;
       font:13px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--ink)}
  #hud{position:fixed;top:14px;left:14px;background:var(--panel);border:1px solid var(--line);
       border-radius:10px;padding:12px 14px;min-width:210px;pointer-events:none;
       backdrop-filter:blur(6px)}
  #hud .k{color:var(--dim);font-size:11px;letter-spacing:.08em;text-transform:uppercase}
  #hud .v{color:var(--key);font-size:17px}
  #hud hr{border:0;border-top:1px solid var(--line);margin:9px 0}
  #cross{position:fixed;left:50%;top:50%;width:16px;height:16px;margin:-8px 0 0 -8px;
         pointer-events:none;opacity:.85}
  #cross:before,#cross:after{content:"";position:absolute;background:var(--key)}
  #cross:before{left:7px;top:0;width:2px;height:16px}
  #cross:after{top:7px;left:0;height:2px;width:16px}
  #start{position:fixed;inset:0;display:flex;align-items:center;justify-content:center;
         background:rgba(8,10,13,.92);z-index:10;cursor:pointer;text-align:center}
  #start b{color:var(--key);font-size:20px;letter-spacing:.04em}
  #start p{color:var(--dim);margin:14px 0 0;line-height:2}
  kbd{border:1px solid var(--line);border-bottom-width:2px;border-radius:4px;
      padding:1px 6px;color:var(--ink)}
  #err{position:fixed;bottom:14px;left:14px;color:#ff9a9a;max-width:60ch}
</style></head><body>
<div id="cross" style="display:none"></div>
<div id="start"><div><b>CLICK TO FLY</b>
  <p><kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> move &nbsp; mouse look &nbsp;
     <kbd>Shift</kbd> boost &nbsp; <kbd>Space</kbd>/<kbd>C</kbd> altitude<br>
     <kbd>click</kbd> drop a probe &nbsp; <kbd>R</kbd> clear &nbsp; <kbd>Esc</kbd> release</p></div></div>
<div id="hud">
  <div class="k">altitude</div><div><span class="v" id="alt">&mdash;</span> __UNITS__</div>
  <hr>
  <div class="k">crosshair</div><div><span class="v" id="ht">&mdash;</span> __UNITS__ &nbsp;
    <span id="sl" style="color:var(--dim)"></span></div>
  <hr>
  <div class="k">probe</div><div id="probe" style="color:var(--dim)">click terrain</div>
</div>
<div id="err"></div>

<script type="importmap">
{"imports":{
  "three":"__THREE__build/three.module.js",
  "three/addons/":"__THREE__examples/jsm/"
}}
</script>
<script type="module">
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { PointerLockControls } from 'three/addons/controls/PointerLockControls.js';

const MODEL = "__MODEL__";
const EXAG  = __EXAG__;          // vertical exaggeration baked into the mesh
const BASE  = __BASE__;          // metres subtracted before exaggerating
const UNITS = "__UNITS__";

// mesh Y -> real elevation. Undo the exaggeration, then put the base back.
const toReal = y => y / EXAG + BASE;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d0f12);

const camera = new THREE.PerspectiveCamera(70, innerWidth/innerHeight, 0.5, 40000);
const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
document.body.appendChild(renderer.domElement);

scene.add(new THREE.HemisphereLight(0xdfefff, 0x2a2620, 2.0));
const sun = new THREE.DirectionalLight(0xffffff, 1.4);
sun.position.set(1, 2, 1.5); scene.add(sun);

const controls = new PointerLockControls(camera, document.body);
const startEl = document.getElementById('start'), crossEl = document.getElementById('cross');
startEl.addEventListener('click', ()=>controls.lock());
controls.addEventListener('lock',   ()=>{startEl.style.display='none'; crossEl.style.display='block';});
controls.addEventListener('unlock', ()=>{startEl.style.display='flex'; crossEl.style.display='none';});
scene.add(controls.getObject());

let terrain=null;
new GLTFLoader().load(MODEL, gltf=>{
  terrain = gltf.scene;
  terrain.rotation.x = -Math.PI/2;      // GLB is Z-up from a raster grid, three.js is Y-up
  scene.add(terrain);

  const box  = new THREE.Box3().setFromObject(terrain);
  const size = box.getSize(new THREE.Vector3());
  const mid  = box.getCenter(new THREE.Vector3());
  scene.fog = new THREE.Fog(0x0d0f12, size.length()*0.25, size.length()*1.6);

  // start off one corner, high enough to read the whole scene
  camera.position.set(mid.x - size.x*0.55, box.max.y + size.y*0.8 + size.z*0.25,
                      mid.z + size.z*0.75);
  camera.lookAt(mid.x, box.min.y, mid.z);
  // PointerLockControls drives camera.rotation directly, so hand the lookAt
  // result over as euler angles or the first mouse move snaps the view.
  const e = new THREE.Euler().setFromQuaternion(camera.quaternion, 'YXZ');
  camera.rotation.set(e.x, e.y, 0, 'YXZ');
  document.getElementById('err').textContent = '';
}, undefined, ()=>{
  document.getElementById('err').textContent =
    'Could not load ' + MODEL + ' \u2014 serve this folder over http, e.g. python -m http.server 8000';
});

const keys = {};
addEventListener('keydown', e=>{keys[e.code]=true; if(e.code==='KeyR') clearProbes();});
addEventListener('keyup',   e=>keys[e.code]=false);
addEventListener('resize',  ()=>{
  camera.aspect = innerWidth/innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

// ---- crosshair probe -------------------------------------------------------
const ray = new THREE.Raycaster(), centre = new THREE.Vector2(0,0);
const nrm = new THREE.Matrix3(), n = new THREE.Vector3();
let lastHit = null;

function castCentre(){
  if(!terrain) return null;
  ray.setFromCamera(centre, camera);
  return ray.intersectObject(terrain, true)[0] || null;
}

function readSlope(hit){
  if(!hit || !hit.face) return null;
  nrm.getNormalMatrix(hit.object.matrixWorld);
  n.copy(hit.face.normal).applyMatrix3(nrm).normalize();
  const apparent = Math.acos(Math.min(1, Math.abs(n.y)));
  // undo the vertical stretch: tan(true) = tan(apparent) / exaggeration
  return THREE.MathUtils.radToDeg(Math.atan(Math.tan(apparent) / EXAG));
}

// ---- two-point measure -----------------------------------------------------
const pins = [], pinGroup = new THREE.Group(); scene.add(pinGroup);
function clearProbes(){
  pins.length = 0;
  pinGroup.clear();
  document.getElementById('probe').textContent = 'click terrain';
}
addEventListener('mousedown', ()=>{
  if(!controls.isLocked) return;
  const hit = castCentre();
  const el = document.getElementById('probe');
  if(!hit){ el.textContent = 'no surface under crosshair'; return; }

  const m = new THREE.Mesh(new THREE.SphereGeometry(Math.max(1, hit.distance*0.006), 12, 8),
                           new THREE.MeshBasicMaterial({color:0x7fd1ff}));
  m.position.copy(hit.point); pinGroup.add(m);
  pins.push(hit.point.clone());
  if(pins.length > 2){ pins.shift(); pinGroup.remove(pinGroup.children[0]); }

  if(pins.length === 2){
    const a = pins[0], b = pins[1];
    const ground = Math.hypot(b.x-a.x, b.z-a.z);
    const dz = toReal(b.y) - toReal(a.y);
    el.innerHTML = '\u0394h <b style="color:var(--key)">' + dz.toFixed(1) + '</b> ' + UNITS +
                   ' over <b style="color:var(--key)">' + ground.toFixed(1) + '</b> m' +
                   ' &nbsp;(' + THREE.MathUtils.radToDeg(Math.atan2(dz, ground)).toFixed(1) + '\u00b0)';
  } else {
    el.innerHTML = 'pin at <b style="color:var(--key)">' + toReal(hit.point.y).toFixed(1) +
                   '</b> ' + UNITS + ' \u2014 click again to measure between pins';
  }
});

// ---- loop ------------------------------------------------------------------
let last = performance.now(), probeAt = 0;
const vel = new THREE.Vector3();
(function loop(){
  requestAnimationFrame(loop);
  const now = performance.now(), dt = Math.min((now-last)/1000, 0.1); last = now;

  const speed = (keys['ShiftLeft']||keys['ShiftRight']) ? 260 : 70;
  vel.set(0,0,0);
  if(keys['KeyW']) vel.z += 1;
  if(keys['KeyS']) vel.z -= 1;
  if(keys['KeyA']) vel.x -= 1;
  if(keys['KeyD']) vel.x += 1;
  if(vel.lengthSq() > 0) vel.normalize();
  controls.moveForward(vel.z * speed * dt);
  controls.moveRight  (vel.x * speed * dt);
  if(keys['Space']) camera.position.y += speed*dt;
  if(keys['KeyC'])  camera.position.y -= speed*dt;

  document.getElementById('alt').textContent = toReal(camera.position.y).toFixed(0);

  // raycasting is linear in triangles, so throttle it rather than run it per frame
  if(controls.isLocked && now - probeAt > 90){
    probeAt = now;
    lastHit = castCentre();
    document.getElementById('ht').textContent =
      lastHit ? toReal(lastHit.point.y).toFixed(1) : '\u2014';
    const s = readSlope(lastHit);
    document.getElementById('sl').textContent = s === null ? '' : ('slope ' + s.toFixed(0) + '\u00b0');
  }
  renderer.render(scene, camera);
})();
</script></body></html>
"""


CDN_THREE = "https://unpkg.com/three@0.160.0/"


def resolve_three(outdir):
    """Prefer a vendored three.js over the CDN.

    The claim "runs with no internet" is only true if the renderer ships with
    the app. If ./three/build/three.module.js exists next to viewer.html we
    point at it; otherwise we fall back to the CDN so a plain checkout still
    works. The Dockerfile vendors it, so the container is genuinely offline.
    """
    local = os.path.join(outdir, "three", "build", "three.module.js")
    if os.path.exists(local):
        return "three/"
    return CDN_THREE


def write_viewer_html(out_path="outputs/viewer.html", model_file="terrain.glb",
                      z_exaggeration=1.0, base_m=0.0, units="m", three_base=None):
    """Write the flythrough page.

    z_exaggeration / base_m come from mesh.metadata, so the on-screen numbers
    are real elevations rather than stretched mesh coordinates.
    """
    outdir = os.path.dirname(os.path.abspath(out_path)) or "."
    if three_base is None:
        three_base = resolve_three(outdir)
    html = (_HTML.replace("__THREE__", three_base)
                 .replace("__MODEL__", model_file)
                 .replace("__EXAG__", repr(float(z_exaggeration) or 1.0))
                 .replace("__BASE__", repr(float(base_m)))
                 .replace("__UNITS__", units))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    src = "vendored (offline)" if three_base.startswith("three/") else "CDN"
    print(f"[viewer] {out_path}  three.js: {src}")
    return out_path


if __name__ == "__main__":
    write_viewer_html()
