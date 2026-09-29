"""Training tiles on disk, and the Dataset that feeds HeightNet."""

import json
import math
import os
import random

import cv2
import numpy as np
import torch

import heightnet as HN

NODATA_CM = 65535
cv2.setNumThreads(0)


# reading / writing tiles
def write_tile(root, split, tid, rgb, height_m, cls=None, jpeg_q=95):
    """Rgb uint8 HxWx3, height_m float (nan = no data), cls uint8 or None."""
    d = os.path.join(root, split)
    os.makedirs(d, exist_ok=True)
    rgb_path = os.path.join(split, f"{tid}_rgb.jpg")
    height_path = os.path.join(split, f"{tid}_h.png")
    cv2.imwrite(os.path.join(root, rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_q)])
    h = np.where(np.isfinite(height_m), np.clip(height_m, 0, 655.0) * 100.0, NODATA_CM)
    cv2.imwrite(os.path.join(root, height_path), np.round(h).astype(np.uint16),
                [cv2.IMWRITE_PNG_COMPRESSION, 6])
    cp = None
    if cls is not None:
        cp = os.path.join(split, f"{tid}_cls.png")
        cv2.imwrite(os.path.join(root, cp), cls.astype(np.uint8))
    return rgb_path, height_path, cp


def read_rgb(path):
    a = cv2.imread(path, cv2.IMREAD_COLOR)
    if a is None:
        raise IOError(f"cannot read {path}")
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def read_height(path):
    a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if a is None:
        raise IOError(f"cannot read {path}")
    h = a.astype(np.float32) / 100.0
    h[a == NODATA_CM] = np.nan
    return h


def read_cls(path, shape):
    if not path:
        return np.full(shape, HN.IGNORE_INDEX, np.uint8)
    a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return a if a is not None else np.full(shape, HN.IGNORE_INDEX, np.uint8)


def load_index(root):
    with open(os.path.join(root, "index.json")) as f:
        idx = json.load(f)
    tiles = []
    for t in idx["tiles"]:
        t = dict(t)
        t["root"] = root
        tiles.append(t)
    return tiles


def save_index(root, tiles, **extra):
    os.makedirs(root, exist_ok=True)
    clean = [{k: v for k, v in t.items() if k != "root"} for t in tiles]
    tmp = os.path.join(root, "index.json.tmp")
    with open(tmp, "w") as f:
        json.dump(dict(format="depthwizard-tiles-v1", tiles=clean, **extra), f)
    os.replace(tmp, os.path.join(root, "index.json"))


def gather(roots, splits):
    """All tiles from several data roots whose split is in `splits`."""
    out = []
    for r in roots:
        out += [t for t in load_index(r) if t["split"] in splits]
    return out


# augmentation
def resize_height(h, size):
    """Nodata-aware area/linear resize of a float height map to (w, h) size."""
    valid = np.isfinite(h).astype(np.float32)
    shrink = size[0] < h.shape[1]
    interp = cv2.INTER_AREA if shrink else cv2.INTER_LINEAR
    a = cv2.resize(np.nan_to_num(h) * valid, size, interpolation=interp)
    b = cv2.resize(valid, size, interpolation=interp)
    out = a / np.maximum(b, 1e-6)
    out[b < 0.5] = np.nan
    return out.astype(np.float32)


def photometric(img, rng):
    x = img.astype(np.float32)
    # Brightness / contrast
    a = rng.uniform(0.75, 1.25)
    b = rng.uniform(-20, 20)
    x = x * a + b + (x.mean() * (1 - a)) * rng.uniform(0, 1)
    # Saturation / hue
    if rng.random() < 0.8:
        hsv = cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1] *= rng.uniform(0.6, 1.3)
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(-6, 6)) % 180
        x = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    # Gamma
    if rng.random() < 0.5:
        g = rng.uniform(0.8, 1.25)
        x = 255.0 * (np.clip(x, 0, 255) / 255.0) ** g
    # Haze (atmosphere seen from orbit)
    if rng.random() < 0.25:
        t = rng.uniform(0.05, 0.35)
        x = x * (1 - t) + t * rng.uniform(170, 235)
    # Blur (lower effective resolution than the GSD claims)
    if rng.random() < 0.35:
        s = rng.uniform(0.3, 1.3)
        x = cv2.GaussianBlur(x, (0, 0), s)
    # Sensor noise
    if rng.random() < 0.3:
        x = x + rng.normal(0, rng.uniform(1, 6), x.shape)
    x = np.clip(x, 0, 255).astype(np.uint8)
    if rng.random() < 0.05:                         # panchromatic
        g = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
        x = np.repeat(g[..., None], 3, axis=2)
    if rng.random() < 0.3:                          # compression
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(x, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(40, 91))])
        if ok:
            x = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return x


def _condition(img):
    try:
        import preprocess as PRE
        out, _, _ = PRE.condition(img)
        return out
    except Exception:
        return img


def parse_mix(s):
    """'gamus=0.65,ahn=0.15,swisstopo=0.2' -> dict."""
    if not s:
        return {}
    out = {}
    for part in s.split(","):
        k, v = part.split("=")
        out[k.strip()] = float(v)
    return out


# datasets
class TrainTiles(torch.utils.data.Dataset):
    """Random crops, random GSD, heavy augmentation."""

    def __init__(self, tiles, length=10000, crop=HN.TILE, gsd_range=(HN.GSD_MIN, HN.GSD_MAX),
                 mix=None, condition_p=0.5, max_upsample=1.5, seed=0):
        if not tiles:
            raise ValueError("no training tiles")
        self.tiles, self.length, self.crop = tiles, int(length), int(crop)
        self.gsd_range, self.condition_p = gsd_range, condition_p
        self.max_upsample, self.seed = max_upsample, seed
        mix = mix or {}
        by_src = {}
        for i, t in enumerate(tiles):
            by_src.setdefault(t["source"], []).append(i)
        w = np.zeros(len(tiles))
        for src, ids in by_src.items():
            share = mix.get(src, 1.0 / len(by_src))
            by_land = {}
            for i in ids:
                by_land.setdefault(tiles[i].get("landscape") or "any", []).append(i)
            for land_ids in by_land.values():
                w[land_ids] = share / len(by_land) / len(land_ids)
        self.p = w / w.sum()
        self.sources = sorted(by_src)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        wi = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([self.seed, i, wi.id if wi else 0,
                                     int(torch.initial_seed() % (2 ** 31))])
        for _ in range(5):
            t = self.tiles[rng.choice(len(self.tiles), p=self.p)]
            try:
                return self._sample(t, rng)
            except Exception as e:
                print(f"[data] skipping {t.get('id')}: {e}")
        raise RuntimeError("five unreadable tiles in a row - check the data")

    def _sample(self, t, rng):
        root = t["root"]
        rgb = read_rgb(os.path.join(root, t["rgb"]))
        h = read_height(os.path.join(root, t["height"]))
        cls = read_cls(os.path.join(root, t["cls"]) if t.get("cls") else None, h.shape)
        native = float(t["gsd"])

        lo = max(self.gsd_range[0], native / self.max_upsample)
        hi = max(lo, self.gsd_range[1])
        gsd = float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
        s = native / gsd
        if abs(s - 1) > 0.02:
            size = (max(8, int(round(rgb.shape[1] * s))), max(8, int(round(rgb.shape[0] * s))))
            rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
            h = resize_height(h, size)
            cls = cv2.resize(cls, size, interpolation=cv2.INTER_NEAREST)

        c = self.crop
        H, W = h.shape
        if H < c or W < c:
            ph, pw = max(0, c - H), max(0, c - W)
            rgb = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="reflect")
            h = np.pad(h, ((0, ph), (0, pw)), constant_values=np.nan)
            cls = np.pad(cls, ((0, ph), (0, pw)), constant_values=HN.IGNORE_INDEX)
            H, W = h.shape
        y = int(rng.integers(0, H - c + 1))
        x = int(rng.integers(0, W - c + 1))
        rgb, h, cls = rgb[y:y + c, x:x + c], h[y:y + c, x:x + c], cls[y:y + c, x:x + c]

        k = int(rng.integers(0, 4))
        rgb, h, cls = np.rot90(rgb, k), np.rot90(h, k), np.rot90(cls, k)
        if rng.random() < 0.5:
            rgb, h, cls = rgb[:, ::-1], h[:, ::-1], cls[:, ::-1]
        rgb = photometric(np.ascontiguousarray(rgb), rng)
        if rng.random() < self.condition_p:
            rgb = _condition(rgb)
        return dict(image=HN.normalise(rgb)[0],
                    height=torch.from_numpy(np.ascontiguousarray(h, np.float32)),
                    cls=torch.from_numpy(np.ascontiguousarray(cls).astype(np.int64)),
                    gsd=torch.tensor(gsd, dtype=torch.float32),
                    source=self.sources.index(t["source"]))


class EvalTiles(torch.utils.data.Dataset):
    """Deterministic centre crops at the tile's own GSD, for validation during training."""

    def __init__(self, tiles, crop=HN.TILE, limit=None, seed=0):
        tiles = list(tiles)
        if limit and len(tiles) > limit:
            r = random.Random(seed)
            tiles = r.sample(tiles, limit)
        self.tiles, self.crop = tiles, crop

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, i):
        t = self.tiles[i]
        root = t["root"]
        rgb = read_rgb(os.path.join(root, t["rgb"]))
        h = read_height(os.path.join(root, t["height"]))
        cls = read_cls(os.path.join(root, t["cls"]) if t.get("cls") else None, h.shape)
        c = self.crop
        H, W = h.shape
        if H < c or W < c:
            ph, pw = max(0, c - H), max(0, c - W)
            rgb = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="reflect")
            h = np.pad(h, ((0, ph), (0, pw)), constant_values=np.nan)
            cls = np.pad(cls, ((0, ph), (0, pw)), constant_values=HN.IGNORE_INDEX)
            H, W = h.shape
        y, x = (H - c) // 2, (W - c) // 2
        return dict(image=HN.normalise(np.ascontiguousarray(rgb[y:y + c, x:x + c]))[0],
                    height=torch.from_numpy(np.ascontiguousarray(h[y:y + c, x:x + c])),
                    cls=torch.from_numpy(cls[y:y + c, x:x + c].astype(np.int64)),
                    gsd=torch.tensor(float(t["gsd"]), dtype=torch.float32),
                    source=0, idx=i)
