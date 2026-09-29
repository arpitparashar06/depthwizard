"""Condition the image before the depth model sees it."""

import numpy as np

# Percentile clip for the stretch.
STRETCH_LO_PCT = 2.0
STRETCH_HI_PCT = 98.0

NARROW_RANGE_LEVELS = 90.0

# Clahe.
CLAHE_CLIP = 2.0
CLAHE_GRID = 8

CLOUD_K = 2.5              # IQRs above the median ...
CLOUD_SAT = 0.15           # ... and nearly colourless
SHADOW_K = 2.5             # IQRs below the median
MASK_MAX_FRAC = 0.35


# 1. band selection
def pick_rgb_bands(ds):
    """Which band indices are actually Red, Green, Blue?"""
    from rasterio.enums import ColorInterp

    n = ds.count
    if n == 1:
        return (1, 1, 1), "single band, replicated to grey"

    # (a) declared colour interpretation - the authoritative answer
    try:
        ci = list(ds.colorinterp)
        want = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
        if all(c in ci for c in want):
            idx = tuple(ci.index(c) + 1 for c in want)
            if idx != (1, 2, 3):
                return idx, f"bands reordered from colorinterp -> R{idx[0]} G{idx[1]} B{idx[2]}"
            return idx, None
    except Exception:
        pass

    # (b) band descriptions, e.g. ('nir', 'red', 'green')
    desc = [(d or "").strip().lower() for d in (ds.descriptions or [])]
    if desc and any(desc):
        def find(*names):
            for i, d in enumerate(desc):
                if any(d == nm or d.startswith(nm) for nm in names):
                    return i + 1
            return None
        r, g, b = find("red", "r"), find("green", "g"), find("blue", "b")
        if r and g and b:
            idx = (r, g, b)
            if idx != (1, 2, 3):
                return idx, f"bands reordered from descriptions -> R{r} G{g} B{b}"
            return idx, None
        # False-colour: NIR present, no blue.
        nir = find("nir", "near")
        if nir and r and g and not b:
            return (r, g, g), "false colour (NIR present, no blue) - using (R,G,G) proxy"

    # (c) give up, but say so
    return (1, 2, 3), "band roles undeclared - assuming first three are RGB"


# 2. valid mask
def valid_mask(rgb_raw, nodata_mask=None):
    """True where the pixel is a real surface we can trust."""
    f = rgb_raw.astype(np.float64)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b

    # Saturation is a ratio, so it is already free of the sensor's units.
    highest = np.maximum(np.maximum(r, g), b)
    lowest = np.minimum(np.minimum(r, g), b)
    sat = np.where(np.abs(highest) > 1e-9, (highest - lowest) / np.maximum(np.abs(highest), 1e-9), 0.0)

    finite = np.isfinite(luma)
    if not finite.any():
        return np.zeros(luma.shape, bool), dict(masked_frac=1.0,
                                                mask_rejected="no finite pixels")
    q25, med, q75 = np.percentile(luma[finite], [25, 50, 75])
    iqr = float(q75 - q25)
    if iqr < 1e-9:
        # A genuinely uniform frame - nothing stands out, so nothing is masked
        return np.ones(luma.shape, bool), dict(cloud_frac=0.0, shadow_frac=0.0,
                                               masked_frac=0.0,
                                               mask_note="flat histogram, no mask applied")

    cloud = (luma > med + CLOUD_K * iqr) & (sat < CLOUD_SAT)
    shadow = luma < med - SHADOW_K * iqr
    bad = (cloud | shadow) & finite
    bad |= ~finite
    if nodata_mask is not None:
        bad |= nodata_mask

    frac = float(bad.mean())
    report = dict(cloud_frac=float(cloud.mean()),
               shadow_frac=float(shadow.mean()),
               masked_frac=frac)

    if frac > MASK_MAX_FRAC:
        report["mask_rejected"] = (f"masked {100 * frac:.0f}% of the frame - "
                                f"treating the whole scene as valid instead")
        return np.ones(luma.shape, bool), report

    return ~bad, report


# 3. stretch
def stretch_per_band(arr, mask=None, lo_pct=STRETCH_LO_PCT, hi_pct=STRETCH_HI_PCT):
    """Percentile stretch each band independently, to 0..255 uint8."""
    out = np.empty(arr.shape[:2] + (3,), np.uint8)
    lohi = []
    for c in range(3):
        band = arr[..., c].astype(np.float64)
        sample = band[mask] if mask is not None else band.ravel()
        sample = sample[np.isfinite(sample)]
        if sample.size < 16:
            sample = band[np.isfinite(band)].ravel()
        if sample.size == 0:
            out[..., c] = 0
            lohi.append((0.0, 0.0))
            continue
        lo, hi = np.percentile(sample, [lo_pct, hi_pct])
        if hi - lo < 1e-9:
            hi = lo + 1e-9
        scaled = np.clip((band - lo) / (hi - lo), 0, 1)
        out[..., c] = (np.where(np.isfinite(scaled), scaled, 0.0) * 255).astype(np.uint8)
        lohi.append((float(lo), float(hi)))
    return out, lohi


def needs_stretch(arr):
    """Is this image worth stretching?"""
    if arr.dtype != np.uint8:
        return True, f"{arr.dtype} input - compressing to 8-bit"
    lo, hi = np.percentile(arr, [STRETCH_LO_PCT, STRETCH_HI_PCT])
    span = float(hi - lo)
    if span < 1.0:
        return False, None
    if span < NARROW_RANGE_LEVELS:
        return True, f"8-bit but only {span:.0f} levels of range - hazy or under-exposed"
    return False, None


# 4. local contrast
def clahe_luma(rgb_u8, clip=CLAHE_CLIP, grid=CLAHE_GRID):
    """Clahe on the L channel of LAB, leaving colour alone."""
    import cv2
    labelled = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)
    c = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(grid), int(grid)))
    labelled[..., 0] = c.apply(labelled[..., 0])
    return cv2.cvtColor(labelled, cv2.COLOR_LAB2RGB)


# the whole chain
def condition(arr, nodata_mask=None, do_stretch=True, do_clahe=True,
              do_mask=True):
    """Raw band stack (HxWx3, any dtype) -> (uint8 RGB, mask, report)."""
    report = {}
    a = np.asarray(arr)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)

    # The mask runs on the native values.
    mask = None
    if do_mask:
        mask, mrep = valid_mask(a, nodata_mask)
        report.update(mrep)

    want, why = needs_stretch(a)
    if do_stretch and want:
        u8, lohi = stretch_per_band(a, mask)
        report["stretch"] = why
        report["stretch_lohi"] = lohi
    else:
        u8 = a.astype(np.uint8) if a.dtype == np.uint8 else \
             np.clip(f01 * 255, 0, 255).astype(np.uint8)

    if do_clahe:
        try:
            u8 = clahe_luma(u8)
            report["clahe"] = f"clip={CLAHE_CLIP} grid={CLAHE_GRID}"
        except Exception as ex:
            # Never let a contrast tweak take the run down
            report["clahe_failed"] = str(ex)

    return np.ascontiguousarray(u8), mask, report


def summarise(rep):
    """One log line, or None if nothing interesting happened."""
    bits = []
    if rep.get("band_note"):
        bits.append(rep["band_note"])
    if rep.get("stretch"):
        bits.append(f"stretched ({rep['stretch']})")
    if rep.get("masked_frac", 0) > 0.001:
        bits.append(f"masked {100 * rep['masked_frac']:.1f}% "
                    f"(cloud {100 * rep.get('cloud_frac', 0):.1f}%, "
                    f"shadow {100 * rep.get('shadow_frac', 0):.1f}%)")
    if rep.get("mask_rejected"):
        bits.append(rep["mask_rejected"])
    if rep.get("clahe"):
        bits.append("CLAHE on luminance")
    return " | ".join(bits) if bits else None
