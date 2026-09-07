#!/usr/bin/env python
"""run_geotiff.py - one image in, the whole pipeline out, no browser.

    python run_geotiff.py benchmark/scenes/ahn_rotterdam_centre/rgb.tif --tallest 95

inference.py's own __main__ stops at the rasters: no .env, no refine, no mesh.
This is the same sequence server.py runs for a job, so what you get here and
what the UI shows are the same surface.

Two things it will not do quietly:

  * A georeferenced image with no scale anchor stops with an error instead of
    defaulting. alpha is metres per model unit and one number sets it for the
    whole scene, so a guess returns confident metres anchored to nothing.
  * If the coarse DEM cannot be fetched the run still finishes, but it says so
    in capitals: the surface is then height above LOCAL GROUND, not above sea
    level, whatever the GeoTIFF tags claim.

Add --reference <lidar.tif> to score the result in the same command.
"""
import argparse
import os
import sys
import time

import numpy as np


def _load_env():
    """.env before inference is imported - it binds OPENTOPO_KEY at import."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, ".env")
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


_load_env()

from inference import (estimate_elevation, load_image, export_products,  # noqa: E402
                       clip_below_ground)
from mesh_builder import build_mesh, build_city, export_mesh, slope_map  # noqa: E402
from refine import refine                                              # noqa: E402

RELATIVE_FULL_SCALE_M = 60.0


def parse_gcps(items):
    """--gcp row,col,height  (repeatable, needs at least two)."""
    out = []
    for s in items or []:
        parts = s.split(",")
        if len(parts) != 3:
            raise SystemExit(f"--gcp wants row,col,height - got {s!r}")
        r, c, h = (float(x) for x in parts)
        out.append((int(r), int(c), h))
    return out


def main():
    ap = argparse.ArgumentParser(
        description="single-view optical image -> DSM + navigable mesh",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("image", help="PNG/JPG (relative) or GeoTIFF with a CRS (metres)")
    ap.add_argument("-o", "--out", default=None,
                    help="output directory (default: outputs/<image stem>)")

    g = ap.add_argument_group("scale (georeferenced input needs exactly one)")
    g.add_argument("--tallest", type=float, default=None, metavar="M",
                   help="height of the TALLEST structure you can identify, in metres")
    g.add_argument("--height-reference", default="tallest",
                   choices=("tallest", "tall", "typical"),
                   help="what --tallest refers to in the height distribution")
    g.add_argument("--gcp", action="append", metavar="ROW,COL,H",
                   help="ground control point, repeatable, at least two")
    g.add_argument("--sun", nargs=2, type=float, default=None,
                   metavar=("AZ", "ELEV"),
                   help="sun azimuth and elevation for shadow calibration; "
                        "read from the GeoTIFF tags automatically when present")
    g.add_argument("--alpha-gain", type=float, default=1.0,
                   help="multiplier on the fitted scale; feed back the value "
                        "the validation report suggests")

    t = ap.add_argument_group("terrain and geometry")
    t.add_argument("--no-dem", action="store_true",
                   help="skip the COP30 baseline; output becomes height above ground")
    t.add_argument("--gsd", type=float, default=0.5, metavar="M",
                   help="metres per pixel, used only when the file has no coordinates")
    t.add_argument("--style", default="city", choices=("city", "stepped", "smooth"))
    t.add_argument("--grid", type=int, default=256, help="mesh resolution")
    t.add_argument("--exaggeration", type=float, default=1.5)
    t.add_argument("--rotations", type=int, default=4,
                   help="ensemble passes; 4 cancels the frame ramp, 1 is fastest")
    t.add_argument("--flatten", type=float, default=0.8)
    t.add_argument("--sharpen", type=float, default=0.4)

    ap.add_argument("--reference", default=None, metavar="TIF",
                    help="reference LiDAR raster - scores the result when given")
    args = ap.parse_args()

    if not os.path.exists(args.image):
        raise SystemExit(f"no such file: {args.image}")

    out = args.out or os.path.join(
        "outputs", os.path.splitext(os.path.basename(args.image))[0])
    os.makedirs(out, exist_ok=True)

    t0 = time.time()
    rgb, meta = load_image(args.image)
    mode = meta["mode"]
    print(f"\n=== {os.path.basename(args.image)} ===")
    print(f"{rgb.shape[1]}x{rgb.shape[0]} px   mode={mode}"
          f"   crs={meta['crs']}   px={meta['px_size_m']}")

    gcps = parse_gcps(args.gcp)
    az = el = None
    if args.sun:
        az, el = args.sun

    if mode == "absolute":
        # The same refusal server.py makes. estimate_elevation would raise
        # anyway; catching it here says which flag fixes it.
        if not args.tallest and len(gcps) < 2 and not args.sun:
            if meta.get("sun_azimuth") is not None and meta.get("sun_elevation") is not None:
                print(f"[scale] no flag given - calibrating from the sun angles "
                      f"in the tags ({meta['sun_azimuth']}, {meta['sun_elevation']})")
            else:
                raise SystemExit(
                    "\nThis image is georeferenced, so the output is in metres -\n"
                    "and metres need a scale anchor. Pass one of:\n\n"
                    "  --tallest 95            height of the tallest structure, in metres\n"
                    "  --gcp r,c,h --gcp r,c,h at least two ground control points\n"
                    "  --sun 145 52            sun azimuth and elevation\n\n"
                    "There is no way to turn relative depth into metres without\n"
                    "one, and a guessed value rescales every elevation in the scene.")
    elif args.tallest:
        print(f"[scale] no coordinates in this file, so --tallest {args.tallest:g} "
              f"sets the full-scale height rather than calibrating metres")

    height, meta, info = estimate_elevation(
        args.image, known_height_m=args.tallest, gcps=gcps or None,
        sun_azimuth=az, sun_elevation=el,
        use_dem=not args.no_dem, alpha_gain=args.alpha_gain,
        rotations=args.rotations, height_reference=args.height_reference,
        outdir=out)

    px = meta.get("px_size_m") or args.gsd
    if mode == "relative":
        scale = args.tallest or RELATIVE_FULL_SCALE_M
        height = height * scale
        meta = dict(meta, px_size_m=px)
        info["relative_full_scale_m"] = scale
        if meta.get("_uncertainty") is not None:
            meta["_uncertainty"] = meta["_uncertainty"] * scale
        print(f"[relative] scaled to a {scale:.0f} m full range")

    if args.flatten or args.sharpen:
        # the split this scene actually used, not refine's 15 m default
        height, rinfo = refine(height, rgb, px_size_m=px,
                               object_sigma_m=float(info.get("sigma_m") or 15.0),
                               flatten=args.flatten, sharpen=args.sharpen)
        if rinfo.get("applied") is False:
            print("[refine] declined - it reduced edge crispness")
    else:
        rinfo = {}

    # refine overshoots at rooflines; re-apply the no-DEM clip to the surface
    # that is actually exported, meshed and scored
    height, info = clip_below_ground(height, info)

    # The surface moved after estimate_elevation exported it, so rewrite the
    # rasters. Otherwise dsm.tif and the mesh are two different surfaces and
    # the validator scores one nobody ever sees.
    info = export_products(height, rgb, meta, out, info,
                           uncertainty=meta.get("_uncertainty"))

    if args.style == "city":
        import buildings as B
        nd = B.normalised_height(height, px, max_building_m=90.0)
        mesh, cinfo = build_city(height, nd, rgb, px_size_m=px,
                                 target_grid=args.grid,
                                 z_exaggeration=args.exaggeration,
                                 is_relative=False)
        rinfo.update(cinfo)
        glb = os.path.join(out, "terrain.glb")
        mesh.export(glb)
        tris = cinfo["ground_tris"] + cinfo["roof_tris"] + cinfo["wall_tris"]
    else:
        mesh = build_mesh(height, rgb, px_size_m=px, target_grid=args.grid,
                          z_exaggeration=args.exaggeration, style=args.style,
                          is_relative=False)
        glb = export_mesh(mesh, os.path.join(out, "terrain.glb"))
        tris = len(mesh.faces)

    slope = slope_map(height, px)
    calib = info.get("calibration", "")
    datum = ("local ground" if "no DEM" in calib
             else "sea level" if mode == "absolute" else "relative")

    print("\n" + "-" * 66)
    if datum == "local ground":
        print("MEASURED FROM: LOCAL GROUND")
        print("  The coarse DEM was not available, so these are NOT sea-level")
        print("  elevations even though the GeoTIFF says MODE=absolute.")
        print("  Score this against an nDSM, never against an absolute DSM.")
        if not os.environ.get("OPENTOPO_KEY"):
            print("  (OPENTOPO_KEY is empty - a free key from"
                  " portal.opentopography.org fixes this.)")
    elif datum == "sea level":
        print("MEASURED FROM: sea level, COP30 supplied the terrain baseline")
        if info.get("dem_debias_m") is not None:
            print(f"  rooftop bias removed: terrain lowered "
                  f"{info['dem_debias_m']:.2f} m")
    else:
        print("MEASURED FROM: nothing - relative surface, no metric datum")

    if info.get("alpha") is not None:
        print(f"scale        alpha {info['alpha']:.3f} m per model unit")
    u = "m" if datum != "relative" else "m*"
    print(f"range        {np.nanmin(height):.1f} .. {np.nanmax(height):.1f} {u}"
          f"   (relief {np.nanmax(height) - np.nanmin(height):.1f} {u})")
    print(f"slope        median {np.nanmedian(slope):.1f} deg, "
          f"99th {np.nanpercentile(slope, 99):.1f} deg")
    if rinfo.get("n") is not None:
        print(f"buildings    {rinfo['n']} extruded, tallest "
              f"{float(rinfo.get('height_max_m', 0)):.0f} m")
    print(f"mesh         {tris:,} triangles ({args.style}) -> {glb}")
    print(f"elapsed      {time.time() - t0:.0f}s")
    print("-" * 66)

    if args.reference:
        import validate as V
        print("\nscoring against", args.reference)
        rep, refarr, rgbarr = V.validate_files(
            os.path.join(out, "dsm.tif"), args.reference, args.image, outdir=out)
        V.error_figures(rep, refarr, rgbarr, outdir=out)
        V.save_report(rep, out)
        h = rep["headline"]
        print(f"\nRMSE {h['rmse']:.2f} m   MAE {h['mae']:.2f} m   "
              f"r {h.get('r', float('nan')):.3f}   "
              f"(alignment: {rep.get('headline_alignment', '?')})")
        print(f"raw, no alignment: RMSE "
              f"{rep['alignment']['raw']['rmse']:.2f} m")
        print("full report ->", os.path.join(out, "validation.md"))

    print("\noutputs in", os.path.abspath(out))
    for f in sorted(os.listdir(out)):
        if not f.startswith("."):
            print("   ", f)


if __name__ == "__main__":
    sys.exit(main())
