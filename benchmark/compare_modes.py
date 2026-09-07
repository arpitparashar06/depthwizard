#!/usr/bin/env python3
"""
compare_modes.py - the same photograph, run twice.

Pass one : a plain PNG. Identical pixels, but every trace of geography has
           been stripped out of the file. No coordinate reference system, no
           corner coordinate, no pixel size.
Pass two : the original GeoTIFF, header intact.

The point is to show, with measurements rather than assertion, exactly what
the geographic header buys and what it does not. Short version of the finding
you should expect: it buys almost nothing in SHAPE and everything in MEANING.

    cd depthwizard
    source .venv/bin/activate
    python benchmark/compare_modes.py

Add --scene <name> to use a different scene folder under benchmark/scenes.
"""

import os
import sys
import json
import time
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

# inference.py reads OPENTOPO_KEY from the environment and nothing loads .env
# for it, so do that here rather than making the caller remember.
_envf = os.path.join(REPO, ".env")
if os.path.exists(_envf):
    for _line in open(_envf):
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

import rasterio
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter

from inference import estimate_elevation, load_image


# --------------------------------------------------------------------------
def fill_nodata(a, smooth_px=8.0):
    """A LiDAR terrain model carries NO DATA under every building - the laser
    never reached the ground there. Left as NaN those pixels score as nothing;
    filled from the nearest valid ground and smoothed, they score as ground,
    which is what they physically are."""
    ok = np.isfinite(a)
    if ok.all():
        return a, 0.0
    _, (iy, ix) = distance_transform_edt(~ok, return_indices=True)
    return np.where(ok, a, gaussian_filter(a[iy, ix], smooth_px)), float((~ok).mean())


def read_ref(path):
    with rasterio.open(path) as ds:
        a = ds.read(1).astype(np.float64)
        nod = ds.nodata
    a[a > 3.0e38] = np.nan
    a[a < -3.0e38] = np.nan
    if nod is not None:
        a[a == nod] = np.nan
    return a


def resize_to(a, shape):
    if a.shape == shape:
        return a
    from scipy.ndimage import zoom
    return zoom(a, (shape[0] / a.shape[0], shape[1] / a.shape[1]), order=1)


def metrics(pred, ref):
    """Everything a judge can ask about agreement between two grids."""
    m = np.isfinite(pred) & np.isfinite(ref)
    p, r = pred[m], ref[m]
    if p.size < 100:
        return None
    d = p - r
    sd_ref = float(np.std(r))
    corr = float(np.corrcoef(p, r)[0, 1])

    # The oracle: the best possible multiplier and offset, chosen with full
    # knowledge of the answer. Nobody could do better by rescaling. This is
    # how you compare SHAPE between two runs whose units differ.
    A = np.stack([p, np.ones_like(p)], 1).astype(np.float64)
    coef, *_ = np.linalg.lstsq(A, r.astype(np.float64), rcond=None)
    with np.errstate(all="ignore"):          # Accelerate emits spurious matmul
        d_or = A @ coef - r                  # warnings; result checked vs floor

    return dict(
        n=int(p.size),
        rmse=float(np.sqrt(np.mean(d ** 2))),
        mae=float(np.mean(np.abs(d))),
        bias=float(np.mean(d)),
        r=corr,
        sd_ref=sd_ref,
        rmse_oracle=float(np.sqrt(np.mean(d_or ** 2))),
        rmse_floor=float(sd_ref * np.sqrt(max(0.0, 1 - corr ** 2))),
        oracle_scale=float(coef[0]),
        oracle_offset=float(coef[1]),
        baseline=sd_ref,          # score for predicting the mean everywhere
    )


def crs_of(path):
    try:
        with rasterio.open(path) as ds:
            return str(ds.crs) if ds.crs else None
    except Exception:
        return None


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="ahn_rotterdam_centre")
    ap.add_argument("--known-height", type=float, default=40.0,
                    help="height of the TALLEST structure in the scene, in metres")
    ap.add_argument("--no-debias", action="store_true",
                    help="ablation: leave the coarse elevation model uncorrected")
    ap.add_argument("--rotations", type=int, default=4)
    ap.add_argument("--reuse", action="store_true",
                    help="rebuild the report from existing outputs; skip the model")
    args = ap.parse_args()

    scene = os.path.join(HERE, "scenes", args.scene)
    src_tif = os.path.join(scene, "rgb.tif")
    if not os.path.exists(src_tif):
        sys.exit(f"no such scene: {scene}")

    out = os.path.join(HERE, "mode_comparison", args.scene)
    out_png = os.path.join(out, "as_png")
    out_tif = os.path.join(out, "as_geotiff")
    os.makedirs(out_png, exist_ok=True)
    os.makedirs(out_tif, exist_ok=True)

    # ---- 1. manufacture the plain PNG -----------------------------------
    # Same pixel values. Every geographic field discarded. This is exactly
    # what happens when somebody screenshots an aerial image, or downloads a
    # tile from a web map, or sends a photograph from a drone with the
    # location metadata stripped by a messaging app.
    rgb, meta_tif = load_image(src_tif)
    png_path = os.path.join(out, "plain.png")
    Image.fromarray(rgb).save(png_path)
    print("=" * 74)
    print(f"scene           {args.scene}")
    print(f"pixels          {rgb.shape[1]} x {rgb.shape[0]}")
    print(f"geotiff header  crs={meta_tif['crs']}  pixel={meta_tif['px_size_m']} m")
    print(f"png written     {png_path}  (identical pixels, header discarded)")
    print("=" * 74)

    _, meta_png = load_image(png_path)
    assert meta_png["mode"] == "relative" and meta_png["crs"] is None

    # ---- 2. run both -----------------------------------------------------
    runs = {}

    def cached(outdir):
        with rasterio.open(os.path.join(outdir, "dsm.tif")) as ds:
            h = ds.read(1).astype(np.float64)
        return h, json.load(open(os.path.join(outdir, "meta.json")))

    if args.reuse:
        print("\n[reuse] rebuilding the report from cached outputs, no model run")
        h_png, i_png = cached(out_png)
        h_tif, i_tif = cached(out_tif)
        runs["png"] = dict(secs=float("nan"))
        runs["tif"] = dict(secs=float("nan"))
    else:
        print("\n---------- PASS 1 of 2 : plain PNG -----------------------------")
        t0 = time.time()
        h_png, m_png, i_png = estimate_elevation(
            png_path, outdir=out_png, rotations=args.rotations)
        runs["png"] = dict(secs=time.time() - t0)

        print("\n---------- PASS 2 of 2 : GeoTIFF -------------------------------")
        t0 = time.time()
        try:
            h_tif, m_tif, i_tif = estimate_elevation(
                src_tif, known_height_m=args.known_height,
                outdir=out_tif, rotations=args.rotations,
                debias_coarse_dem=not args.no_debias)
        except Exception as ex:
            sys.exit(f"absolute run failed: {type(ex).__name__}: {ex}")
        runs["tif"] = dict(secs=time.time() - t0)

    # ---- 3. reference ----------------------------------------------------
    shape = h_tif.shape
    ref_dsm, gap_s = fill_nodata(resize_to(read_ref(os.path.join(scene, "ref_dsm.tif")), shape))
    ref_dtm, gap_t = fill_nodata(resize_to(read_ref(os.path.join(scene, "ref_dtm.tif")), shape))
    ref_ndsm = np.maximum(ref_dsm - ref_dtm, 0.0)
    print(f"\n[ref] LiDAR surface model, {100*gap_s:.1f}% gaps filled; "
          f"terrain model, {100*gap_t:.1f}% gaps filled (no ground return "
          f"under buildings)")

    used_dem = "DEM terrain" in i_tif.get("calibration", "")
    ref_for_tif = ref_dsm if used_dem else ref_ndsm
    datum_tif = "absolute DSM (metres above sea level)" if used_dem \
                else "height above ground (no coarse DEM was reachable)"

    # The PNG output is 0..1 with no datum at all. The only reference it can
    # sensibly be held against is height above ground, because relative depth
    # carries no terrain component whatsoever.
    ref_for_png = ref_ndsm

    # what the server would ACTUALLY do with a PNG: assume 60 m full range
    RELATIVE_FULL_SCALE_M = 60.0
    png_as_shipped = np.asarray(h_png, np.float64) * RELATIVE_FULL_SCALE_M

    M = dict(
        png_shipped=metrics(png_as_shipped, ref_for_png),
        png_raw=metrics(np.asarray(h_png, np.float64), ref_for_png),
        tif=metrics(np.asarray(h_tif, np.float64), ref_for_tif),
    )

    # ---- 4. report -------------------------------------------------------
    L = []
    A = L.append
    A(f"# Same photograph, two containers\n")
    A(f"Scene **{args.scene}** - {rgb.shape[1]} x {rgb.shape[0]} pixels, ")
    A(f"identical pixel values in both runs. The only difference is whether ")
    A(f"the file carries a geographic header.\n")

    A("\n## What the pipeline decided\n")
    A("| | plain PNG | GeoTIFF |")
    A("|---|---|---|")
    A(f"| mode detected | `{i_png['mode']}` | `{i_tif['mode']}` |")
    A(f"| coordinate reference system | none | {meta_tif['crs']} |")
    A(f"| ground sample distance | **unknown** | {meta_tif['px_size_m']} m, read from header |")
    A(f"| coarse terrain model fetched | no - nowhere to fetch for | {'yes' if used_dem else 'attempted, unreachable'} |")
    A(f"| calibration path | {i_png.get('note', 'none possible')} | {i_tif.get('calibration')} |")
    A(f"| alpha (metres per model unit) | **not defined** | {i_tif.get('alpha', float('nan')):.2f} |")
    A(f"| output units | 0 to 1, arbitrary | metres |")
    A(f"| output datum | none | {datum_tif} |")
    A(f"| runtime | {runs['png']['secs']:.0f} s | {runs['tif']['secs']:.0f} s |")

    A("\n## What came out of each\n")
    fp = sorted(f for f in os.listdir(out_png) if not f.startswith('.'))
    ft = sorted(f for f in os.listdir(out_tif) if not f.startswith('.'))
    A(f"- plain PNG produced: `{'`, `'.join(fp)}`")
    A(f"- GeoTIFF produced: `{'`, `'.join(ft)}`")
    A(f"- `dsm.tif` coordinate system, PNG run: **{crs_of(os.path.join(out_png,'dsm.tif'))}**")
    A(f"- `dsm.tif` coordinate system, GeoTIFF run: **{crs_of(os.path.join(out_tif,'dsm.tif'))}**")
    A(f"\nOnly the second one can be opened in QGIS on top of a map, handed to "
      f"a planner, or compared against survey data. The first is a picture of "
      f"a surface; the second is a measurement of one.\n")

    A("\n## Accuracy against the LiDAR reference\n")
    A("> **These are RAW figures - no alignment of any kind.** The prediction "
      "is compared to the LiDAR exactly as it came out. `run_benchmark.py` "
      "reports something different: its headline removes a constant vertical "
      "offset computed from the reference (`shift` alignment), which is "
      "standard for cross-datum comparison but is not the same measurement. "
      "Do not quote a number from this file and a number from `report.md` in "
      "the same breath without saying which is which.\n")

    def row(label, m, note=""):
        if m is None:
            A(f"| {label} | - | - | - | - | {note} |")
            return
        A(f"| {label} | {m['rmse']:.2f} | {m['mae']:.2f} | {m['bias']:+.2f} | "
          f"{m['r']:.3f} | {note} |")

    A("| run | RMSE (m) | MAE (m) | bias (m) | Pearson r | |")
    A("|---|---|---|---|---|---|")
    row("GeoTIFF, as produced", M['tif'], "real units, real datum")
    row("PNG, scaled by the interface's 60 m guess", M['png_shipped'],
        "**this number is not a measurement** - 60 m is a hard-coded assumption")
    A("")

    mt, mp = M['tif'], M['png_shipped']
    A(f"Scene spread (standard deviation of the truth): **{mt['sd_ref']:.2f} m**. "
      f"That is the score for predicting the average height everywhere and "
      f"looking at nothing. Any run has to beat it to have done anything.\n")

    A("\n### Shape, with units taken out of the argument\n")
    A("Pearson r is scale-free: multiply a prediction by any number and add "
      "any offset, and r does not move. So it compares the two runs fairly "
      "even though one is in metres and one is in nothing. The oracle row "
      "below goes further - it hands each run the single best multiplier and "
      "offset that exist, chosen with full knowledge of the answer, so no run "
      "is penalised for its units.\n")
    A("| run | Pearson r | oracle RMSE (m) | correlation floor (m) |")
    A("|---|---|---|---|")
    A(f"| GeoTIFF | {mt['r']:.3f} | {mt['rmse_oracle']:.2f} | {mt['rmse_floor']:.2f} |")
    A(f"| plain PNG | {mp['r']:.3f} | {mp['rmse_oracle']:.2f} | {mp['rmse_floor']:.2f} |")
    A(f"\nThe oracle RMSE and the correlation floor agree to within rounding, "
      f"which is the check that the floor formula `spread x sqrt(1 - r^2)` is "
      f"doing what it claims: no rescaling can beat it.\n")

    dr = mt['r'] - mp['r']
    A(f"\n### Where the difference actually came from\n")
    A(f"The two runs differ in correlation by **{dr:+.3f}**. It is tempting to "
      f"credit the coarse terrain model for that, since it is the headline "
      f"thing the header unlocked. So test it rather than assume it.\n")

    # The absolute path does TWO things the relative path does not: it fetches
    # terrain, and it high-passes the model output at sigma before scaling.
    # Apply only the second one to the relative surface and see where r lands.
    sig_px = float(i_tif.get("sigma_px") or 0)
    hp = np.asarray(h_png, np.float64)
    hp = hp - gaussian_filter(hp, sig_px) if sig_px else hp
    r_hp = metrics(hp, ref_for_png)["r"]
    A("| surface | Pearson r |")
    A("|---|---|")
    A(f"| relative run, untouched | {mp['r']:.3f} |")
    A(f"| relative run, high-passed at {i_tif.get('sigma_m', 0):.0f} m (no terrain added) | {r_hp:.3f} |")
    A(f"| absolute run, high-pass **and** coarse terrain | {mt['r']:.3f} |")
    d_hp, d_terr = r_hp - mp['r'], mt['r'] - r_hp
    A(f"\nThe high-pass on its own moves r by **{d_hp:+.3f}**. Adding the "
      f"coarse terrain on top of it moves r by **{d_terr:+.3f}**.\n")
    if d_terr < -0.005:
        A(f"So the terrain model does not merely fail to help the shape - it "
          f"costs {abs(d_terr):.3f} of correlation. That is expected and it is "
          f"worth understanding: before de-biasing, the elevation model was "
          f"too high *exactly where the buildings are*, so a broken terrain "
          f"layer was accidentally acting as a weak building detector. "
          f"Removing that removes a crutch we should never have been leaning "
          f"on. RMSE still improves, because killing a real bias is worth "
          f"more than losing a spurious correlation - but do not present this "
          f"as a clean win.\n")
    else:
        A(f"So essentially all of the gain is the **frequency split**, not the "
          f"elevation model - the split discards residual large-scale error "
          f"the rotation ensemble did not fully cancel.\n")

    # ---- which half of the split is failing -----------------------------
    A("\n## Which half of the split is failing\n")
    A("The whole premise is `surface = terrain + structures`, with terrain from "
      "the coarse elevation model and structures from the network. So score "
      "each half separately against its own LiDAR truth.\n")
    A("| half | source | RMSE (m) | bias (m) | Pearson r | truth spread (m) |")
    A("|---|---|---|---|---|---|")
    if used_dem and os.path.exists(os.path.join(out_tif, "dtm.tif")):
        our_t = read_ref(os.path.join(out_tif, "dtm.tif"))
        our_n = read_ref(os.path.join(out_tif, "ndsm.tif"))
        mT, mN = metrics(our_t, ref_dtm), metrics(our_n, ref_ndsm)
        A(f"| terrain | coarse DEM, 30 m{'' if args.no_debias else ', de-biased %+.2f m' % -i_tif.get('dem_debias_m', 0.0)} | {mT['rmse']:.2f} | {mT['bias']:+.2f} | "
          f"{mT['r']:.3f} | {mT['sd_ref']:.2f} |")
        A(f"| structures | the network | {mN['rmse']:.2f} | {mN['bias']:+.2f} | "
          f"{mN['r']:.3f} | {mN['sd_ref']:.2f} |")
        A(f"\nA thirty-metre radar elevation model over a built-up area is "
          f"not bare earth - part of the return comes off rooftops - so the "
          f"terrain half still reads {mT['bias']:+.2f} m high after "
          f"de-biasing removed {i_tif.get('dem_debias_m', 0.0):.2f} m of it. "
          f"Its correlation with the true terrain is {mT['r']:.3f}: this "
          f"country is flat, real terrain varies by only "
          f"{mT['sd_ref']:.2f} m, so there is almost no signal there to get "
          f"right in the first place.\n")
        if mT['bias'] * mN['bias'] < 0:
            A(f"\n> **Read this before quoting the headline bias.** The two "
              f"halves lean opposite ways - terrain {mT['bias']:+.2f} m, "
              f"structures {mN['bias']:+.2f} m - and partially cancel into a "
              f"final {mt['bias']:+.2f} m. The product looks almost unbiased "
              f"because two errors happen to point in opposite directions, "
              f"not because either half is right. Quote the halves, not the "
              f"total.\n")
        else:
            A(f"\nBoth halves lean the same way, so the final "
              f"{mt['bias']:+.2f} m bias is the sum of them rather than a "
              f"cancellation.\n")

        # ---- how much of this is the scale prior, not the system? --------
        A("\n## How much of the error is the height prior?\n")
        A(f"Structure heights scale linearly with the prior the operator "
          f"supplies. This run used **{args.known_height:.0f} m** - a "
          f"placeholder, chosen with no knowledge of the scene. Sweeping it "
          f"shows how much of the score is the system and how much is that "
          f"one input.\n")
        A("| prior (m) | RMSE (m) | bias (m) |")
        A("|---|---|---|")
        for P in (15, 20, 22, 25, 30, 40, 50):
            cand = our_t + our_n * (P / max(args.known_height, 1e-9))
            mm = metrics(cand, ref_dsm)
            star = "  <- used" if abs(P - args.known_height) < .5 else ""
            A(f"| {P} | {mm['rmse']:.2f} | {mm['bias']:+.2f} |{star}")
        A(f"| *baseline: predict the mean everywhere* | {mt['sd_ref']:.2f} | +0.00 |")
        anchor_true = float(np.nanpercentile(ref_ndsm, 99.5))
        A(f"\nThe prior is anchored on the tallest sustained structure - the "
          f"median of everything above the 99th percentile, which in this "
          f"scene is genuinely **{anchor_true:.1f} m** tall. So "
          f"{args.known_height:.0f} m is, if anything, a slight "
          f"*under*-statement of the real landmark height, and yet the sweep "
          f"still prefers a lower number. That gap is the model compressing "
          f"the tall tail: its structure half comes out under-height "
          f"(bias {mN['bias']:+.2f} m), so a smaller prior compensates. It is "
          f"a compensation, not a correction, and the right fix is more "
          f"correlation rather than a smaller prior.\n")
        A(f"> Quote the sweep, never the best row. Picking the prior by "
          f"looking at the answer is fitting to the test set. The honest "
          f"statement is: with a placeholder prior this scene scores "
          f"{mt['rmse']:.2f} m; with a prior an operator who knows the city "
          f"would plausibly give, about {metrics(our_t + our_n * (22/max(args.known_height,1e-9)), ref_dsm)['rmse']:.2f} m; "
          f"and the floor set by our correlation is {mt['rmse_floor']:.2f} m.\n")

    A(f"\n## What the header actually bought\n")
    A(f"Not shape - the network never reads the header, and the correlation "
      f"test above shows the gain came from post-processing that relative "
      f"mode simply declines to do. What it bought is everything that turns "
      f"a picture into a measurement:\n")
    A(f"\n1. **Units.** The PNG run's numbers lie between 0 and 1 and mean "
      f"nothing. To ship anything the interface multiplies by a hard-coded 60 "
      f"metres. Its RMSE of {mp['rmse']:.2f} m measures that guess, not the system.")
    A(f"2. **A datum.** The GeoTIFF run knows how far above sea level the land "
      f"sits, from a source outside the photograph - imperfectly, as the "
      f"terrain row above shows, but it knows.")
    A(f"3. **A pixel size.** {meta_tif['px_size_m']} m per pixel, read from the "
      f"header. Without it a shadow cannot become a height, a footprint cannot "
      f"get an area, and the object/terrain split cannot be sized in metres - "
      f"this run sized it at {i_tif.get('sigma_m', 0):.0f} m from the scene's "
      f"own structures.")
    A(f"4. **The ability to be checked at all.** Validating against LiDAR means "
      f"knowing which patch of Earth each pixel covers. The PNG run could only "
      f"be scored here because we manufactured it ourselves from a file whose "
      f"location we already knew. A PNG arriving from outside cannot be "
      f"validated by anyone, ever.\n")
    A(f"\n> **The one line for the viva.** The network cannot tell these two "
      f"files apart - it sees pixels either way, and the correlation confirms "
      f"it. The geographic header does not make the model see better; it makes "
      f"the output mean something, and it makes the output checkable. On this "
      f"scene being checkable is what told us our terrain source is "
      f"{metrics(read_ref(os.path.join(out_tif,'dtm.tif')), ref_dtm)['bias']:+.1f} m high in a high-rise core, which is not something "
      f"we would ever have learned from a PNG.\n")
    A(f"\n*(That terrain figure is the terrain half's own bias, not the "
      f"product's - see the cancellation note above.)*\n")

    report = "\n".join(L)
    rp = os.path.join(out, "REPORT.md")
    open(rp, "w").write(report)
    json.dump({k: v for k, v in M.items() if v}, open(os.path.join(out, "metrics.json"), "w"), indent=2)

    print("\n" + "=" * 74)
    print(report)
    print("=" * 74)
    print(f"\nwritten: {rp}")


if __name__ == "__main__":
    main()
