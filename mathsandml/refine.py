"""Make the height map look like buildings instead of hills."""

import numpy as np
from scipy.ndimage import (uniform_filter, gaussian_filter, label,
                           binary_opening, binary_closing, find_objects)


def _lstsq(A, y):
    with np.errstate(all="ignore"):
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef if np.all(np.isfinite(coef)) else None


def _box(a, r):
    return uniform_filter(a, size=2 * int(r) + 1, mode="nearest")


def guided_filter(src, guide, radius=8, eps=1e-3):
    """He et al. edge-preserving filter, scalar guide."""
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


def object_band(height, sigma_px):
    """Height above local ground, NaN-safe."""
    a = np.asarray(height, np.float64)
    valid = np.isfinite(a)
    if not valid.all():
        a = np.where(valid, a, np.nanmedian(a[valid]) if valid.any() else 0.0)
    ground = gaussian_filter(a, sigma_px)
    return a - ground, ground


def flatten_structures(height, px_size_m=1.0, object_sigma_m=15.0,
                       min_height_m=2.5, min_area_m2=40.0, strength=0.8,
                       plane=True):
    """Turn each dome back into a flat roof."""
    h = np.asarray(height, np.float64)
    sigma = max(2.0, object_sigma_m / max(px_size_m, 1e-6))
    obj, ground = object_band(h, sigma)

    mask = obj > min_height_m
    # Clean up speckle: an opening removes stray pixels, a closing seals roofs
    kernel = np.ones((3, 3), bool)
    mask = binary_closing(binary_opening(mask, kernel), kernel)

    labels, n = label(mask)
    if n == 0:
        return h, dict(structures=0)

    min_px = max(9, int(min_area_m2 / max(px_size_m ** 2, 1e-9)))
    # Fit on absolute height, not the object band.
    out = h.copy()
    kept = 0

    for i, box in enumerate(find_objects(labels), start=1):
        if box is None:
            continue
        sub = (labels[box] == i)
        if sub.sum() < min_px:
            continue
        vals = h[box][sub]

        coef = None
        if plane and sub.sum() >= 30:
            yy, xx = np.nonzero(sub)
            A = np.c_[xx, yy, np.ones(xx.size)]
            coef = _lstsq(A, vals)
        if coef is not None:
            with np.errstate(all="ignore"):
                # One robust pass: drop the tails, refit.
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
            # No plane, or a degenerate one: a flat median is always safe
            fit = np.full(vals.size, np.median(vals))

        blk = out[box]
        blk[sub] = (1 - strength) * vals + strength * fit
        out[box] = blk
        kept += 1

    return out, dict(structures=kept, min_px=min_px)


def sharpen_objects(height, px_size_m=1.0, object_sigma_m=15.0, amount=0.5):
    """Unsharp mask on the object band only."""
    sigma = max(2.0, object_sigma_m / max(px_size_m, 1e-6))
    obj, ground = object_band(height, sigma)
    detail = obj - gaussian_filter(obj, max(1.0, sigma / 8.0))
    return ground + obj + amount * detail


def refine(height, rgb, px_size_m=1.0, object_sigma_m=15.0,
           guided=True, guide_radius_m=4.0, guide_eps=1e-3,
           flatten=0.8, sharpen=0.4, verbose=True, min_gain=0.98):
    """The whole chain."""
    h = np.asarray(height, np.float64)
    finite = np.isfinite(h)
    fill = np.nanmedian(h[finite]) if finite.any() else 0.0
    h = np.where(finite, h, fill)
    original = h.copy()
    info = {}

    before = _crispness(h, px_size_m, object_sigma_m)

    if guided:
        r = max(2, int(round(guide_radius_m / max(px_size_m, 1e-6))))
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
    sigma = max(2.0, object_sigma_m / max(px_size_m, 1e-6))
    obj, _ = object_band(height, sigma)
    gy, gx = np.gradient(obj, max(px_size_m, 1e-6))
    g = np.sort(np.hypot(gx, gy).ravel())[::-1]
    k = max(1, int(0.01 * g.size))
    return float(g[:k].sum() / max(g.sum(), 1e-9))


if __name__ == "__main__":
    # Synthetic city: flat roofs, then blurred the way a depth model blurs them
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
