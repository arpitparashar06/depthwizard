"""Find the buildings in a height map and turn them into solids."""

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_opening, binary_closing, gaussian_filter


def _lstsq(A, y):
    with np.errstate(all="ignore"):
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef if np.all(np.isfinite(coef)) else None


def ground_surface(dsm, px_size_m=0.5, max_building_m=45.0, smooth_m=8.0):
    """Where would the ground be if the buildings were not there?"""
    from scipy.ndimage import grey_opening, median_filter, gaussian_filter as gf
    a = np.asarray(dsm, np.float64)
    a = np.where(np.isfinite(a), a, np.nanmedian(a[np.isfinite(a)]) if np.isfinite(a).any() else 0.0)

    k = int(round(max_building_m / max(px_size_m, 1e-6)))
    k = int(np.clip(k, 3, min(a.shape) - 1 if min(a.shape) > 4 else 3))
    g = grey_opening(a, size=k)
    g = median_filter(g, size=max(3, k // 8))
    g = gf(g, max(1.0, smooth_m / max(px_size_m, 1e-6)))
    return np.minimum(g, a)


def normalised_height(dsm, px_size_m=0.5, max_building_m=45.0):
    """nDSM = surface minus bare ground."""
    return np.maximum(np.asarray(dsm, np.float64) - ground_surface(
        dsm, px_size_m, max_building_m), 0.0)


# 1. FOOTPRINTS
def _vegetation(rgb):
    c = np.asarray(rgb, np.float64)[..., :3]
    s = c.sum(2) + 1e-6
    return 2 * c[..., 1] / s - c[..., 0] / s - c[..., 2] / s


def _merge_touching(labels, nlab, n, roof_percentile, merge_tol_m):
    if nlab < 2:
        return labels, nlab

    hts = np.zeros(nlab + 1)
    for i in range(1, nlab + 1):
        m = labels == i
        if m.any():
            hts[i] = np.percentile(n[m], roof_percentile)

    parent = list(range(nlab + 1))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    # Adjacency = two different labels meeting across a one-pixel step
    for A, Bm in ((labels[:, :-1], labels[:, 1:]), (labels[:-1], labels[1:])):
        d = (A != Bm) & (A > 0) & (Bm > 0)
        if not d.any():
            continue
        for a, b in np.unique(np.c_[A[d], Bm[d]], axis=0):
            if abs(hts[a] - hts[b]) <= merge_tol_m:
                ra, rb = find(int(a)), find(int(b))
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)

    remap = np.zeros(nlab + 1, np.int32)
    next_idx = 0
    for i in range(1, nlab + 1):
        r = find(i)
        if remap[r] == 0:
            next_idx += 1
            remap[r] = next_idx
        remap[i] = remap[r]
    return remap[labels], next_idx


def _regularise_contour(contour, eps_px, rect_fill=0.72):
    import cv2
    area = cv2.contourArea(contour)
    if area <= 0:
        return None

    rect = cv2.minAreaRect(contour)
    rw, rh = rect[1]
    if rw > 0 and rh > 0 and area / (rw * rh) >= rect_fill:
        return cv2.boxPoints(rect).astype(np.float64)

    eps = max(float(eps_px), 0.02 * cv2.arcLength(contour, True))
    poly = cv2.approxPolyDP(contour, eps, True).reshape(-1, 2).astype(np.float64)
    return poly if poly.shape[0] >= 3 else None


def _segment_by_height(ndsm, min_height_m, step_m, px_size_m, min_px,
                       peak_window_m=15.0, merge_tol_m=2.5,
                       roof_percentile=75):
    from scipy.ndimage import median_filter, label, distance_transform_edt
    n = median_filter(np.asarray(ndsm, np.float64),
                      size=max(3, int(round(1.5 / max(px_size_m, 1e-6)))))
    band = np.floor(n / float(step_m)).astype(np.int32)
    band[n < min_height_m] = -1

    from scipy.ndimage import maximum_filter
    win = max(3, int(round(peak_window_m / max(px_size_m, 1e-6))))
    peak = maximum_filter(n, size=win)
    plateau = (n > peak - float(step_m)) & (n > min_height_m)

    markers, mid = label(plateau)
    if mid:
        sizes = np.bincount(markers.ravel())
        drop = np.nonzero(sizes < max(4, min_px // 6))[0]
        if drop.size:
            markers[np.isin(markers, drop)] = 0
        keep = [i for i in np.unique(markers) if i]
        remap = np.zeros(markers.max() + 1, np.int32)
        for j, i in enumerate(keep, start=1):
            remap[i] = j
        markers = remap[markers]
        mid = len(keep)
    _, (iy, ix) = distance_transform_edt(markers == 0, return_indices=True)
    grown = markers[iy, ix]
    grown[n <= min_height_m] = 0

    # Growing splits one roof across several markers; merge the pieces back
    if merge_tol_m:
        grown, mid = _merge_touching(grown, mid, n, roof_percentile, merge_tol_m)

    return [grown == i for i in range(1, mid + 1)
            if int((grown == i).sum()) >= min_px]


def extract_footprints(ndsm, rgb=None, px_size_m=0.5, min_height_m=2.5,
                       min_area_m2=25.0, simplify_m=2.0, veg_percentile=75,
                       roof_percentile=75, max_buildings=2000,
                       height_step_m=3.0, peak_window_m=15.0,
                       merge_tol_m=2.5, rect_fill=0.72):
    """Find building footprints in the nDSM."""
    import cv2

    n = np.asarray(ndsm, np.float64)
    n = np.where(np.isfinite(n), n, 0.0)

    veg = _vegetation(rgb) if rgb is not None else np.zeros_like(n)
    veg_ref = float(np.median(veg)) + 0.06

    min_px = max(9, int(min_area_m2 / max(px_size_m ** 2, 1e-9)))
    eps_px = max(1.0, simplify_m / max(px_size_m, 1e-6))
    kernel = np.ones((3, 3), bool)

    out, rejected = [], 0
    for region in _segment_by_height(n, min_height_m, height_step_m,
                                     px_size_m, min_px, peak_window_m,
                                     merge_tol_m, roof_percentile):
        region = binary_closing(binary_opening(region, kernel), kernel)
        if region.sum() < min_px:
            continue

        # Judge vegetation per region, not per pixel.
        ys, xs = np.nonzero(region)
        vals = n[region]
        A = np.c_[xs, ys, np.ones(xs.size)]
        coef = _lstsq(A, vals)
        with np.errstate(all="ignore"):
            planar_resid = (float(np.std(vals - A @ coef)) if coef is not None
                            else float(np.std(vals)))
        veg_mean = float(np.mean(veg[region]))
        if veg_mean > veg_ref and planar_resid > 1.0:
            rejected += 1
            continue

        cnts, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < min_px:
            continue
        poly = _regularise_contour(c, eps_px, rect_fill)
        if poly is None:
            continue

        h = float(np.percentile(vals, roof_percentile))
        if h < min_height_m:
            continue
        out.append(dict(contour=poly, height_m=h,
                        area_m2=float(cv2.contourArea(c)) * px_size_m ** 2,
                        planar_resid_m=planar_resid, veg=veg_mean))

    if rejected:
        print(f"[buildings] rejected {rejected} vegetated non-planar regions")
    out.sort(key=lambda b: -b["area_m2"])
    return out[:max_buildings]


def flatten_ground(surface, footprints, px_size_m=0.5, pad_px=1):
    """Remove the buildings from the ground surface."""
    import cv2
    out = np.array(surface, np.float64, copy=True)
    if not footprints:
        return out
    m = np.zeros(out.shape, np.uint8)
    for b in footprints:
        cv2.fillPoly(m, [b["contour"].astype(np.int32)], 1)
    if pad_px:
        m = cv2.dilate(m, np.ones((2 * pad_px + 1,) * 2, np.uint8))
    m = m.astype(bool)
    if not m.any():
        return out
    filled = np.where(m, np.nan, out)
    med = np.nanmedian(filled)
    g = gaussian_filter(np.where(np.isfinite(filled), filled, med),
                        max(4.0, 12.0 / max(px_size_m, 1e-6)))
    out[m] = g[m]
    return out


# 2. TRIANGULATION (ear clipping)
def _signed_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _point_in_tri(p, a, b, c):
    d = ((b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]))
    if abs(d) < 1e-12:
        return False
    u = ((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) / d
    v = ((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) / d
    return u >= -1e-9 and v >= -1e-9 and u + v <= 1 + 1e-9


def earclip(poly):
    """Triangulate a simple polygon."""
    p = np.asarray(poly, np.float64)
    n = p.shape[0]
    if n < 3:
        return []
    ccw = _signed_area(p) > 0
    idx = list(range(n)) if ccw else list(range(n))[::-1]

    triangles = []
    guard = 0
    while len(idx) > 3 and guard < 5000:
        guard += 1
        clipped = False
        for i in range(len(idx)):
            ia, ib, ic = idx[i - 1], idx[i], idx[(i + 1) % len(idx)]
            a, b, c = p[ia], p[ib], p[ic]
            # Convex corner?
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if cross <= 1e-12:
                continue
            if any(_point_in_tri(p[j], a, b, c)
                   for j in idx if j not in (ia, ib, ic)):
                continue
            triangles.append((ia, ib, ic))
            idx.pop(i)
            clipped = True
            break
        if not clipped:
            break
    if len(idx) == 3:
        triangles.append(tuple(idx))
    return triangles


# 3. PROCEDURAL FACADE
def facade_texture(size=512, floors=8, bays=6, seed=0,
                   wall=(196, 190, 180), trim=(150, 143, 132),
                   glass=(88, 112, 132)):
    """A tiling facade: floor bands, a window grid, lit windows scattered in."""
    rng = np.random.default_rng(seed)
    im = Image.new("RGB", (size, size), wall)
    d = ImageDraw.Draw(im)

    fh = size / floors
    bw = size / bays
    for f in range(floors):
        y0 = f * fh
        d.rectangle([0, y0, size, y0 + fh * 0.10], fill=trim)      # floor band
        for b in range(bays):
            x0 = b * bw
            wx0, wx1 = x0 + bw * 0.22, x0 + bw * 0.78
            wy0, wy1 = y0 + fh * 0.28, y0 + fh * 0.82
            lit = rng.random() < 0.18
            col = (glass if not lit
                   else (min(glass[0] + 90, 255), min(glass[1] + 80, 255), 170))
            d.rectangle([wx0, wy0, wx1, wy1], fill=col)
            d.rectangle([wx0, wy0, wx1, wy1], outline=trim, width=2)

    # Ground floor reads differently: taller openings, darker base
    d.rectangle([0, size - fh, size, size], fill=trim)
    for b in range(bays):
        x0 = b * bw
        d.rectangle([x0 + bw * 0.12, size - fh * 0.86, x0 + bw * 0.88, size - fh * 0.06],
                    fill=(70, 78, 88))

    a = np.asarray(im, np.float64)
    a *= (0.92 + 0.16 * rng.random(a.shape[:2]))[..., None]        # break flatness
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


# 4. GEOMETRY
def building_meshes(footprints, ground, px_size_m, y_top, z_base=0.0,
                    z_exaggeration=1.0, image_shape=None,
                    facade_tile_m=(12.0, 24.0), min_wall_m=2.0):
    """Extrude footprints into prisms in the ground mesh's coordinate frame."""
    px = float(px_size_m)
    tw, th = facade_tile_m
    H, W = image_shape if image_shape else (None, None)

    rv, rf, ruv = [], [], []
    wv, wf, wuv = [], [], []

    for b in footprints:
        poly = b["contour"]
        rows = np.clip(poly[:, 1].astype(int), 0, ground.shape[0] - 1)
        cols = np.clip(poly[:, 0].astype(int), 0, ground.shape[1] - 1)
        base = float(np.median(ground[rows, cols]))
        top = base + float(b["height_m"])
        if top - base < min_wall_m:
            continue

        def Z(v):
            return (v - z_base) * z_exaggeration

        zb, zt = Z(base), Z(top)
        x = poly[:, 0] * px
        y = y_top - poly[:, 1] * px

        # roof cap, textured from the photo
        triangles = earclip(np.c_[x, y])
        if not triangles:
            continue
        off = len(rv)
        for i in range(poly.shape[0]):
            rv.append((x[i], y[i], zt))
            if W:
                ruv.append((poly[i, 0] / W, 1.0 - poly[i, 1] / H))
            else:
                ruv.append((0.0, 0.0))
        for a, c, d_ in triangles:
            rf.append((off + a, off + c, off + d_))

        # walls, one quad per edge, UVs in metres
        n = poly.shape[0]
        run = 0.0
        for i in range(n):
            j = (i + 1) % n
            ex, ey = x[j] - x[i], y[j] - y[i]
            seg = float(np.hypot(ex, ey))
            if seg < 1e-6:
                continue
            u0, u1 = run / tw, (run + seg) / tw
            run += seg
            v1 = (top - base) / th
            k = len(wv)
            wv += [(x[i], y[i], zb), (x[j], y[j], zb),
                   (x[i], y[i], zt), (x[j], y[j], zt)]
            wuv += [(u0, 0.0), (u1, 0.0), (u0, v1), (u1, v1)]
            wf += [(k, k + 1, k + 3), (k, k + 3, k + 2)]

    return (np.array(rv, np.float64).reshape(-1, 3), np.array(rf, np.int64).reshape(-1, 3),
            np.array(ruv, np.float64).reshape(-1, 2),
            np.array(wv, np.float64).reshape(-1, 3), np.array(wf, np.int64).reshape(-1, 3),
            np.array(wuv, np.float64).reshape(-1, 2))


def summarise(footprints):
    if not footprints:
        return dict(n=0)
    h = np.array([b["height_m"] for b in footprints])
    a = np.array([b["area_m2"] for b in footprints])
    return dict(n=len(footprints), height_median_m=float(np.median(h)),
                height_p95_m=float(np.percentile(h, 95)),
                height_max_m=float(h.max()),
                area_median_m2=float(np.median(a)),
                total_footprint_m2=float(a.sum()))


if __name__ == "__main__":
    # Ear clipping on the shapes that actually break naive implementations
    sq = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    L = np.array([[0, 0], [10, 0], [10, 4], [4, 4], [4, 10], [0, 10]], float)
    U = np.array([[0, 0], [12, 0], [12, 10], [9, 10], [9, 3],
                  [3, 3], [3, 10], [0, 10]], float)
    for name, p in (("square", sq), ("L-shape", L), ("U-shape", U)):
        t = earclip(p)
        area = sum(abs(_signed_area(p[list(tri)])) for tri in t)
        print(f"{name:9s} {len(p)} verts -> {len(t)} tris "
              f"(expected {len(p)-2}) | area {area:.1f} vs {abs(_signed_area(p)):.1f}")
