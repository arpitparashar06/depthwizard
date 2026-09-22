"""
preprocess.py - condition the image BEFORE the depth model sees it.
=============================================================================

WHY THIS EXISTS. load_image() used to hand the backbone whatever came out of
the file. That is fine for a clean 8-bit aerial crop and quietly wrong for
almost everything a real archive contains:

  * an 11-bit sensor value packed in a 16-bit container, so the whole scene
    occupies the bottom 3% of the range and reads as flat grey
  * a false-colour product whose first three bands are NIR, R, G - the
    backbone was trained on natural RGB, and near-infrared reflectance looks
    nothing like a red channel to a network that has never seen it labelled
    as one. Vegetation goes bright white, and the depth prior breaks
  * haze, which flattens local contrast across the whole frame
  * cloud and its shadow, which are not surfaces but which the model happily
    assigns a depth to, and which then skew every percentile the pipeline
    computes downstream

None of these are exotic. They are what Landsat, Sentinel and most commercial
deliveries actually look like.

WHAT IT DOES NOT DO. It does not sharpen, denoise or otherwise invent
structure. Everything here is a monotonic, per-band intensity remap plus a
mask - the geometry in the image is untouched, because the whole pipeline's
accuracy claim rests on that geometry being the sensor's, not ours.

ORDER MATTERS:
    1. pick the real RGB bands      (before anything reads pixel values)
    2. build the valid mask          (before the stretch, so cloud cannot
                                      skew the percentiles)
    3. per-band percentile stretch   (bit depth -> 0..255)
    4. CLAHE on luminance only       (local contrast, no colour shift)

Returns the conditioned uint8 RGB plus a report, so what was applied ends up
in the job log and in meta.json rather than only in someone's memory.
"""

import numpy as np

# Percentile clip for the stretch. 2/98 rather than min/max: one saturated
# pixel (a roof specular, a hot pixel) would otherwise crush everything else
# into a handful of grey levels.
STRETCH_LO_PCT = 2.0
STRETCH_HI_PCT = 98.0

# Below this, an 8-bit image is using so little of its range that it is worth
# stretching even though it is already uint8. A well-exposed photo spans
# 150-250 levels; 90 is a genuinely flat, hazy frame.
NARROW_RANGE_LEVELS = 90.0

# CLAHE. Deliberately mild. A high clip limit manufactures texture in flat
# regions, and the depth model will read manufactured texture as structure -
# which is the one failure this file must not cause.
CLAHE_CLIP = 2.0
CLAHE_GRID = 8

# Cloud/shadow thresholds, in ROBUST UNITS - multiples of the interquartile
# range away from the median luminance.
#
# Not absolute 0..1 thresholds, and not percentiles either. Both fail the
# same way: a cloud covering 3% of the frame sits inside the 99th percentile,
# so any normalisation anchored on the extremes gets dragged out to meet it,
# and then every ordinary surface looks black by comparison. The median and
# the IQR cannot be moved by anything smaller than a quarter of the image, so
# "much brighter than typical" stays meaningful no matter what the sensor's
# units are or how much cloud is in frame.
CLOUD_K = 2.5              # IQRs above the median ...
CLOUD_SAT = 0.15           # ... and nearly colourless
SHADOW_K = 2.5             # IQRs below the median
MASK_MAX_FRAC = 0.35       # if more than this is masked, distrust the mask


# ---------------------------------------------------------------------------
# 1. band selection
# ---------------------------------------------------------------------------
def pick_rgb_bands(ds):
    """Which band indices are actually Red, Green, Blue? (1-based, rasterio.)

    Taking the first three bands is the common shortcut and it is wrong on
    two very ordinary cases: a 4-band product ordered (B,G,R,NIR), and a
    false-colour aerial ordered (NIR,R,G). Both are widespread.

    rasterio exposes the answer when the file bothers to declare it, via
    colorinterp; we fall back to band descriptions, then to "first three"
    with a flag so the caller can say so in the log rather than silently
    feeding infrared to the backbone.
    """
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
        # false-colour: NIR present, no blue. Substitute green for blue -
        # a cheap natural-ish proxy. DAv2 cares about structure, and a
        # (R,G,G) stack keeps vegetation from blowing out to white.
        nir = find("nir", "near")
        if nir and r and g and not b:
            return (r, g, g), "false colour (NIR present, no blue) - using (R,G,G) proxy"

    # (c) give up, but say so
    return (1, 2, 3), "band roles undeclared - assuming first three are RGB"


# ---------------------------------------------------------------------------
# 2. valid mask
# ---------------------------------------------------------------------------
def valid_mask(rgb_raw, nodata_mask=None):
    """True where the pixel is a real surface we can trust.

    Cloud has no depth and the model will still give it one. Worse, cloud is
    bright and shadow is dark, so leaving them in skews every percentile the
    pipeline computes - the stretch, the structure-scale estimate, and the
    height reference the metre scale is fitted against.

    rgb_raw : the band stack in its NATIVE units. No normalisation is applied
        first, deliberately: every scheme for mapping the data to 0..1 is
        itself distorted by the outliers we are trying to find.

    Returns (mask, report).
    """
    f = rgb_raw.astype(np.float64)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b

    # Saturation is a ratio, so it is already free of the sensor's units.
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    sat = np.where(np.abs(mx) > 1e-9, (mx - mn) / np.maximum(np.abs(mx), 1e-9), 0.0)

    finite = np.isfinite(luma)
    if not finite.any():
        return np.zeros(luma.shape, bool), dict(masked_frac=1.0,
                                                mask_rejected="no finite pixels")
    q25, med, q75 = np.percentile(luma[finite], [25, 50, 75])
    iqr = float(q75 - q25)
    if iqr < 1e-9:
        # a genuinely uniform frame - nothing stands out, so nothing is masked
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
    rep = dict(cloud_frac=float(cloud.mean()),
               shadow_frac=float(shadow.mean()),
               masked_frac=frac)

    # A mask covering a third of the frame is more likely a thresholding
    # failure (a snowfield, a night scene, a desert) than genuine cloud.
    # Fall back to "everything valid" and say why, rather than throwing away
    # the scene.
    if frac > MASK_MAX_FRAC:
        rep["mask_rejected"] = (f"masked {100 * frac:.0f}% of the frame - "
                                f"treating the whole scene as valid instead")
        return np.ones(luma.shape, bool), rep

    return ~bad, rep


# ---------------------------------------------------------------------------
# 3. stretch
# ---------------------------------------------------------------------------
def stretch_per_band(arr, mask=None, lo_pct=STRETCH_LO_PCT, hi_pct=STRETCH_HI_PCT):
    """Percentile stretch EACH band independently, to 0..255 uint8.

    Per band, not joint. Atmospheric scattering is wavelength dependent - blue
    scatters most - so a hazy scene arrives with a colour cast and a joint
    stretch preserves it. Stretching each band to the same percentiles removes
    the cast, which is a crude but effective white balance.

    Percentiles are taken over `mask` only, so a cloud cannot define the
    bright end and a shadow cannot define the dark end.
    """
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
        # NaN survives clip() and casting it to uint8 is undefined - on this
        # machine it lands on 0, on another it could be anything. Send holes
        # to the dark end explicitly.
        scaled = np.clip((band - lo) / (hi - lo), 0, 1)
        out[..., c] = (np.where(np.isfinite(scaled), scaled, 0.0) * 255).astype(np.uint8)
        lohi.append((float(lo), float(hi)))
    return out, lohi


def needs_stretch(arr):
    """Is this image worth stretching? Returns (bool, reason)."""
    if arr.dtype != np.uint8:
        return True, f"{arr.dtype} input - compressing to 8-bit"
    lo, hi = np.percentile(arr, [STRETCH_LO_PCT, STRETCH_HI_PCT])
    span = float(hi - lo)
    if span < 1.0:
        # A uniform frame has nothing to stretch, and stretching it anyway
        # maps every pixel to the dark end - a synthetic all-grey test tile
        # came out black. Leave it exactly as it arrived.
        return False, None
    if span < NARROW_RANGE_LEVELS:
        return True, f"8-bit but only {span:.0f} levels of range - hazy or under-exposed"
    return False, None


# ---------------------------------------------------------------------------
# 4. local contrast
# ---------------------------------------------------------------------------
def clahe_luma(rgb_u8, clip=CLAHE_CLIP, grid=CLAHE_GRID):
    """CLAHE on the L channel of LAB, leaving colour alone.

    Applied per channel in RGB it shifts hue, which then breaks the green
    index buildings.py uses to reject vegetation. Applied to luminance only,
    it recovers local contrast in haze without touching the colour the rest
    of the pipeline reasons about.
    """
    import cv2
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)
    c = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(grid), int(grid)))
    lab[..., 0] = c.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# ---------------------------------------------------------------------------
# the whole chain
# ---------------------------------------------------------------------------
def condition(arr, nodata_mask=None, do_stretch=True, do_clahe=True,
              do_mask=True):
    """raw band stack (HxWx3, any dtype) -> (uint8 RGB, mask, report).

    `mask` is True where the pixel is trustworthy. It is returned rather than
    applied, because the right thing to do with a cloud differs by stage: the
    stretch excludes it from its percentiles, the calibrator should exclude it
    from its sample, and the texture drape should keep it so the render still
    looks like the input.
    """
    rep = {}
    a = np.asarray(arr)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)

    # The mask runs on the NATIVE values. It uses median/IQR, which need no
    # normalisation and cannot be dragged around by the very pixels they are
    # looking for.
    mask = None
    if do_mask:
        mask, mrep = valid_mask(a, nodata_mask)
        rep.update(mrep)

    want, why = needs_stretch(a)
    if do_stretch and want:
        u8, lohi = stretch_per_band(a, mask)
        rep["stretch"] = why
        rep["stretch_lohi"] = lohi
    else:
        u8 = a.astype(np.uint8) if a.dtype == np.uint8 else \
             np.clip(f01 * 255, 0, 255).astype(np.uint8)

    if do_clahe:
        try:
            u8 = clahe_luma(u8)
            rep["clahe"] = f"clip={CLAHE_CLIP} grid={CLAHE_GRID}"
        except Exception as ex:
            # never let a contrast tweak take the run down
            rep["clahe_failed"] = str(ex)

    return np.ascontiguousarray(u8), mask, rep


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
