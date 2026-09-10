"""
refine.py - make the height map look like buildings instead of hills.

===========================================================================
READ THIS FIRST
===========================================================================
One function matters: refine(). The rest are its three passes.

THE PROBLEM. Depth-Anything returns a smooth field. Nothing in it knows that a
roof is flat or that a wall is vertical, so a 30 m tower arrives as a 30 m
DOME and the mesh reads as terrain instead of a city.

    refine(height, rgb)
        1. guided_filter()       snap the height edges onto the PHOTO's edges.
                                 The depth map is blurry at a roofline; the
                                 photo is not
        2. flatten_structures()  find each structure and replace it with its
                                 own best-fit plane. Domes become roofs. This
                                 is the biggest visual change in the mesh
        3. sharpen_objects()     put back the contrast the first two shaved
                                 off - and only on the object band, so flat
                                 ground is never touched
        4. _crispness()          measure whether all that actually helped, and
                                 THROW THE RESULT AWAY IF IT DID NOT

Step 4 is the unusual one. On a dense downtown tile these settings took edge
definition DOWN by 33%, and that softer surface was what got meshed and
scored. So the chain now measures itself and declines when it loses.

NOTHING HERE INVENTS DETAIL. It redistributes height the depth model already
produced into shapes that match the image.

===========================================================================

Three passes fix most of it, in this order:

  1. guided filter   - snap height edges onto image edges. The depth map is
                       blurry at building boundaries; the photo is not. Using
                       the photo as a guide pulls the height discontinuity to
                       where the roofline actually is.
  2. flatten         - find connected structures, replace each with a fitted
                       plane. This is what turns domes into roofs. Optional and
                       strength-controlled, because it is a strong prior: it
                       helps enormously on buildings and hurts on tree canopy.
  3. unsharp on the  - restore the contrast that steps 1-2 shave off the
     object band       object band only, leaving terrain untouched.

None of this invents detail. It redistributes height that the depth model
already produced into shapes that match the image.
"""

import numpy as np
from scipy.ndimage import (uniform_filter, gaussian_filter, label,
                           binary_opening, binary_closing, find_objects)


# ---------------------------------------------------------------------------

def _lstsq(A, y):
    """Least squares that neither prints spurious warnings nor returns inf.

    Two separate problems, one wrapper. macOS/Accelerate raises floating-point
    flags for the unused lanes of its vectorised matmul, so a perfectly healthy
    solve prints "divide by zero", "overflow" and "invalid" all at once - noise
    that trains you to ignore the real thing. And a genuinely singular design
    matrix (a one-pixel-wide region, every pixel collinear, one distinct value)
    returns coefficients that are inf or nan, which would then propagate
    silently into the surface. Returns None in that case so the caller can fall
    back to something safe.
    """
    with np.errstate(all="ignore"):
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef if np.all(np.isfinite(coef)) else None


def _box(a, r):
    return uniform_filter(a, size=2 * int(r) + 1, mode="nearest")


def guided_filter(src, guide, radius=8, eps=1e-3):
    """He et al. edge-preserving filter, scalar guide.

    A bilateral filter smooths by height similarity, so it happily preserves a
    fake dome. A guided filter smooths by similarity in the GUIDE, so structure
    from the photo transfers into the height map. That is the difference that
    matters here.

    src   : height map (any units)
    guide : same shape, normalised 0..1 - use image luma
    eps   : smaller = sharper and noisier. 1e-3 on 0..1 luma is a good start.
    """
    src = np.asarray(src, np.float32)
    guide = np.asarray(guide, np.float32)

    mean_g = _box(guide, radius)
    mean_s = _box(src, radius)
    cov_gs = _box(guide * src, radius) - mean_g * mean_s
    var_g = _box(guide * guide, radius) - mean_g * mean_g

    a = cov_gs / (var_g + eps)
    b = mean_s - a * mean_g
    return _box(a, radius) * guide + _box(b, radius)


def luma(rgb):
    c = np.asarray(rgb, np.float32)
    if c.ndim == 2:
        g = c
    else:
        g = 0.299 * c[..., 0] + 0.587 * c[..., 1] + 0.114 * c[..., 2]
    g = g - g.min()
    return g / max(g.max(), 1e-9)


# ---------------------------------------------------------------------------
def object_band(height, sigma_px):
    """Height above local ground, NaN-safe."""
    a = np.asarray(height, np.float64)
    ok = np.isfinite(a)
    if not ok.all():
        a = np.where(ok, a, np.nanmedian(a[ok]) if ok.any() else 0.0)
    ground = gaussian_filter(a, sigma_px)
    return a - ground, ground


def flatten_structures(height, px_size_m=1.0, object_sigma_m=15.0,
                       min_height_m=2.5, min_area_m2=40.0, strength=0.8,
                       plane=True):
    """Turn each dome back into a flat roof.

    IN PLAIN ENGLISH: find every structure standing above the local ground, fit
    the best flat plane through it, and blend that plane back in. A roof IS
    planar; the blurry mound the depth model returned is not. Strength 0.8
    keeps a little of the original texture so the result does not look
    CAD-generated.

    It is a strong assumption, which is why it is a dial: buildings love it,
    tree canopy does not.

    ---------------------------------------------------------------------
    A roof is planar. A dome is not. Finding the structures and replacing each
    with its own best-fit plane is a much stronger statement than any amount of
    filtering, and it is the single biggest visual change in the mesh.

    strength : 0 = off, 1 = fully planar. 0.8 keeps a little texture so the
               result does not look CAD-generated.
    plane    : True fits a tilted plane (handles pitched roofs and sloped
               ground under a building). False uses a flat median, which is
               safer on noisy inputs but wrong on any pitched roof.
    """
    h = np.asarray(height, np.float64)
    sig = max(2.0, object_sigma_m / max(px_size_m, 1e-6))
    obj, ground = object_band(h, sig)

    mask = obj > min_height_m
    # clean up speckle: an opening removes stray pixels, a closing seals roofs
    st = np.ones((3, 3), bool)
    mask = binary_closing(binary_opening(mask, st), st)

    lab, n = label(mask)
    if n == 0:
        return h, dict(structures=0)

    min_px = max(9, int(min_area_m2 / max(px_size_m ** 2, 1e-9)))
    # Fit on ABSOLUTE height, not the object band. A Gaussian "ground" is
    # dragged upward under a large building, so flattening the object band and
    # then adding that ground back re-curves the roof you just flattened.
    out = h.copy()
    kept = 0

    for i, sl in enumerate(find_objects(lab), start=1):
        if sl is None:
            continue
        sub = (lab[sl] == i)
        if sub.sum() < min_px:
            continue
        vals = h[sl][sub]

        coef = None
        if plane and sub.sum() >= 30:
            yy, xx = np.nonzero(sub)
            A = np.c_[xx, yy, np.ones(xx.size)]
            coef = _lstsq(A, vals)
        if coef is not None:
            with np.errstate(all="ignore"):
                # one robust pass: drop the tails, refit. Chimneys and edge
                # pixels otherwise tilt the whole roof.
                r = vals - A @ coef
                keep = np.abs(r) < 2.5 * (1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6)
                if keep.sum() >= 20:
                    refit = _lstsq(A[keep], vals[keep])
                    if refit is not None:
                        coef = refit
                fit = A @ coef
            if not np.all(np.isfinite(fit)):
                fit = np.full(vals.size, np.median(vals))
        else:
            # no plane, or a degenerate one: a flat median is always safe
            fit = np.full(vals.size, np.median(vals))

        blk = out[sl]
        blk[sub] = (1 - strength) * vals + strength * fit
        out[sl] = blk
        kept += 1

    return out, dict(structures=kept, min_px=min_px)


# ---------------------------------------------------------------------------
def sharpen_objects(height, px_size_m=1.0, object_sigma_m=15.0, amount=0.5):
    """Unsharp mask on the object band only.

    Applied to the whole surface this would carve gullies into flat ground.
    Restricted to height-above-local-ground it only crisps the structures.
    """
    sig = max(2.0, object_sigma_m / max(px_size_m, 1e-6))
    obj, ground = object_band(height, sig)
    detail = obj - gaussian_filter(obj, max(1.0, sig / 8.0))
    return ground + obj + amount * detail


# ---------------------------------------------------------------------------
def refine(height, rgb, px_size_m=1.0, object_sigma_m=15.0,
           guided=True, guide_radius_m=4.0, guide_eps=1e-3,
           flatten=0.8, sharpen=0.4, verbose=True, min_gain=0.98):
    """The whole chain. Returns (height, info).

    Every stage is individually switchable because they are not equally safe.
    The guided filter is nearly free of risk. Flattening is a strong prior that
    assumes planar structures - turn it down over forest.

    The chain measures its own effect and will DECLINE to apply itself. On a
    dense downtown tile the settings that help a sparse scene took crispness
    from 0.200 to 0.134 - a 33% loss of edge definition - and that softened
    surface is what the mesh and the validator then used, while the exported
    raster still showed the sharp original. Measuring the result and throwing
    it away when it is worse costs one extra crispness evaluation and removes a
    whole class of silent quality regression. min_gain=0 forces it through.
    """
    h = np.asarray(height, np.float64)
    finite = np.isfinite(h)
    fill = np.nanmedian(h[finite]) if finite.any() else 0.0
    h = np.where(finite, h, fill)
    original = h.copy()
    info = {}

    before = _crispness(h, px_size_m, object_sigma_m)

    if guided:
        r = max(2, int(round(guide_radius_m / max(px_size_m, 1e-6))))
        # scale-invariant eps: the guide is 0..1 but the height is metres, so
        # eps has to be read relative to the height variance, not absolutely.
        h_lo, h_hi = float(np.nanmin(h)), float(np.nanmax(h))
        h_rng = max(h_hi - h_lo, 1e-9)
        h_norm = ((h - h_lo) / h_rng).astype(np.float32)
        h_norm = guided_filter(h_norm, luma(rgb), radius=r, eps=guide_eps).astype(np.float64)
        h = h_norm * h_rng + h_lo
        info["guide_radius_px"] = r

    if flatten and flatten > 0:
        h, fi = flatten_structures(h, px_size_m, object_sigma_m,
                                   strength=float(flatten))
        info.update(fi)

    if sharpen and sharpen > 0:
        h = sharpen_objects(h, px_size_m, object_sigma_m, amount=float(sharpen))

    after = _crispness(h, px_size_m, object_sigma_m)
    gain = after / max(before, 1e-9)
    applied = gain >= float(min_gain)
    if not applied:
        h = original                      # keep the sharper input
    info.update(crispness_before=before, crispness_after=after,
                crispness_gain=gain, applied=bool(applied),
                min_gain=float(min_gain))
    if verbose:
        if applied:
            print(f"[refine] structures={info.get('structures', 0)} "
                  f"crispness {before:.3f} -> {after:.3f} ({gain:.2f}x)")
        else:
            print(f"[refine] DECLINED: crispness {before:.3f} -> {after:.3f} "
                  f"({gain:.2f}x) is below min_gain={min_gain:.2f}. "
                  f"Keeping the unrefined surface.")

    return np.where(finite, h, np.nan), info


def _crispness(height, px_size_m, object_sigma_m):
    """Share of the total height gradient concentrated in the steepest 1% of pixels.

    A dome spreads its height change over many pixels; a building concentrates
    it at the roofline. My first attempt normalised by the object band's own
    std, which ranked a BLURRED surface above the ground truth - attenuating the
    buildings shrinks the denominator faster than it shrinks the edges. This
    version ranks truth 0.35, blurred 0.07, refined 0.36 on the synthetic city,
    which is the ordering you actually want.
    """
    sig = max(2.0, object_sigma_m / max(px_size_m, 1e-6))
    obj, _ = object_band(height, sig)
    gy, gx = np.gradient(obj, max(px_size_m, 1e-6))
    g = np.sort(np.hypot(gx, gy).ravel())[::-1]
    k = max(1, int(0.01 * g.size))
    return float(g[:k].sum() / max(g.sum(), 1e-9))


if __name__ == "__main__":
    # synthetic city: flat roofs, then blurred the way a depth model blurs them
    H = W = 300
    yy, xx = np.mgrid[0:H, 0:W]
    truth = 100 + 6 * np.sin(xx / 90.)
    for (r, c, hh) in [(40, 40, 30), (40, 160, 45), (160, 60, 18), (170, 180, 36)]:
        truth[r:r + 70, c:c + 80] += hh
    rgb = np.zeros((H, W, 3), np.uint8) + 120
    for (r, c, _) in [(40, 40, 0), (40, 160, 0), (160, 60, 0), (170, 180, 0)]:
        rgb[r:r + 70, c:c + 80] = 200

    blurred = gaussian_filter(truth, 6.0)          # what the model gives you
    out, info = refine(blurred, rgb, px_size_m=0.5)

    for name, a in (("truth", truth), ("blurred", blurred), ("refined", out)):
        print(f"{name:9s} crispness {_crispness(a, 0.5, 15.0):.3f}  "
              f"roof std {a[55:95, 55:105].std():.2f} m")
