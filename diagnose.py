"""
diagnose.py - why does the 3D view look flat?

Reads what the last run actually wrote and measures the relief at each stage,
so the answer comes from your files rather than from guessing.

    python diagnose.py

Three places the height can go missing, and this tells them apart:
  1. the DSM itself is flat        -> the depth/calibration stage
  2. the DSM has relief, mesh does not -> the mesh stage
  3. both fine, it just looks flat -> exaggeration vs. ground extent
"""

import os
import json
import warnings
import numpy as np

warnings.filterwarnings("ignore")

WORK = "outputs"


def _p(label, value):
    print(f"  {label:<34s} {value}")


def main():
    if not os.path.isdir(WORK):
        print(f"No '{WORK}' folder. Run a Generate first.")
        return

    print(f"\nFiles in {WORK}/")
    for f in sorted(os.listdir(WORK)):
        print(f"  {f:<20s} {os.path.getsize(os.path.join(WORK, f))/1e6:8.2f} MB")

    # ---- 1. metadata -------------------------------------------------------
    meta_path = os.path.join(WORK, "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        print("\nmeta.json")
        for k, v in meta.items():
            _p(k, v)

    # ---- 2. the DSM --------------------------------------------------------
    dsm_path = os.path.join(WORK, "dsm.tif")
    dsm_range = None
    if os.path.exists(dsm_path):
        import rasterio
        with rasterio.open(dsm_path) as ds:
            a = ds.read(1).astype(float)
            crs, px = ds.crs, abs(ds.transform.a)
        a = a[np.isfinite(a)]
        dsm_range = float(np.nanmax(a) - np.nanmin(a))
        print("\nDSM (dsm.tif)")
        _p("CRS", crs or "none (relative mode)")
        _p("pixel size", f"{px:.3f}")
        _p("min / max", f"{np.nanmin(a):.3f} / {np.nanmax(a):.3f}")
        _p("range", f"{dsm_range:.3f}")
        _p("p1 / p99", f"{np.percentile(a,1):.3f} / {np.percentile(a,99):.3f}")
        if dsm_range < 1e-3:
            print("  >> The DSM itself is flat. The problem is upstream of the mesh.")
        elif dsm_range <= 1.0 + 1e-6:
            print("  >> Relative mode: 0..1. build_mesh stretches this to 60 m.")

    # ---- 3. the mesh -------------------------------------------------------
    glb = os.path.join(WORK, "terrain.glb")
    if os.path.exists(glb):
        import trimesh
        scene = trimesh.load(glb)
        if isinstance(scene, trimesh.Scene):
            m = (scene.to_geometry() if hasattr(scene, "to_geometry")
                 else scene.dump(concatenate=True))
        else:
            m = scene
        v = m.vertices
        ext_x = float(v[:, 0].max() - v[:, 0].min())
        ext_y = float(v[:, 1].max() - v[:, 1].min())
        relief = float(v[:, 2].max() - v[:, 2].min())
        ground = max(ext_x, ext_y)
        ratio = 100 * relief / max(ground, 1e-9)

        n = m.face_normals
        ang = np.degrees(np.arccos(np.clip(np.abs(n[:, 2]), 0, 1)))

        print("\nMesh (terrain.glb)")
        _p("triangles", f"{len(m.faces):,}")
        _p("ground extent", f"{ext_x:.0f} x {ext_y:.0f} (model units)")
        _p("vertical relief", f"{relief:.2f}")
        _p("relief / ground extent", f"{ratio:.2f}%")
        _p("faces steeper than 80 deg", f"{int((ang>80).sum()):,}")
        _p("median face slope", f"{np.median(ang):.1f} deg")

        print()
        if relief < 2.0:
            print("  >> VERDICT: the mesh is flat.")
            if dsm_range and dsm_range > 1.0:
                print("     The DSM had relief, so it was lost in build_mesh.")
            elif dsm_range and dsm_range <= 1.0:
                print("     The DSM is 0..1 and the relative stretch did NOT fire.")
                print("     Check np.nanmax(height) in build_mesh - a value above")
                print("     1.0 by any amount skips the x60 stretch.")
        elif ratio < 2.0:
            print(f"  >> VERDICT: relief is real ({relief:.0f} units) but small next to")
            print(f"     {ground:.0f} units of ground - {ratio:.1f}%. That reads as flat")
            print("     from a distance. Raise vertical exaggeration, or fly lower.")
        elif (ang > 80).sum() < len(m.faces) * 0.01:
            print("  >> VERDICT: relief is fine but there are almost no vertical")
            print("     faces, so geometry is 'smooth'. Switch to stepped.")
        else:
            print("  >> VERDICT: the mesh is fine. If the view still looks flat, the")
            print("     camera is far away - check the Flythrough tab, not Orbit,")
            print("     click to lock the pointer, and fly down with C.")

    # ---- 4. the viewer -----------------------------------------------------
    vh = os.path.join(WORK, "viewer.html")
    if os.path.exists(vh):
        s = open(vh, encoding="utf-8").read()
        exag = [l for l in s.splitlines() if "const EXAG" in l]
        print("\nviewer.html")
        _p("exaggeration baked in", exag[0].split("=")[1].split(";")[0].strip() if exag else "?")
        _p("placeholders left unfilled", "yes - BUG" if "__" in s.split("<script")[0] else "no")
    print()


if __name__ == "__main__":
    main()
