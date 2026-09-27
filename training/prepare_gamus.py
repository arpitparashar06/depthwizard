#!/usr/bin/env python3
"""
prepare_gamus.py - GAMUS (Hugging Face: earthflow/GAMUS) -> DepthWizard tiles.

GAMUS (Xiong et al. 2023, TUM / DLR): 1024 x 1024 aerial tiles at 0.33 m from
US cities, each with a LiDAR nDSM (height above ground, from DSM - DTM) and a
6-class land-cover map. On the Hub it is HDF5, one file per modality:

    images/<split>/<CITY>_<r>_<c>_RGB.h5
    heights/<split>/<CITY>_<r>_<c>_AGL.h5
    classes/<split>/<CITY>_<r>_<c>_CLS.h5

The full set is ~80 GB. This script downloads ONE TILE AT A TIME, converts it
to a ~1 MB JPEG + 16-bit PNG + label PNG, and deletes the HDF5 straight away,
so the peak disk use is a few files and the output is ~10x smaller. It is
restartable: tiles already converted are skipped.

    python prepare_gamus.py --out /kaggle/working/data/gamus
    python prepare_gamus.py --out data/gamus --limit 50          # a quick look
    python prepare_gamus.py --out data/gamus --local-dir ./GAMUS # already downloaded

It prints height statistics from the first tiles so a unit problem (centimetres
instead of metres) shows up in the first minute, not after a day of training.
"""

import argparse
import os
import re
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "mathsandml"))
import data as D  # noqa: E402

REPO = "earthflow/GAMUS"
GSD = 0.333
PAT = re.compile(r"^images/(?P<split>[^/]+)/(?P<id>.+)_RGB\.h5$")


def read_h5(path):
    """The largest array in the file, whatever the dataset key is called."""
    import h5py
    best = []

    def visit(_, obj):
        if isinstance(obj, h5py.Dataset) and obj.ndim >= 2:
            best.append(obj)
    with h5py.File(path, "r") as f:
        f.visititems(visit)
        if not best:
            raise ValueError(f"no 2-D array in {path}")
        ds = max(best, key=lambda d: d.size)
        return np.asarray(ds[()])


def to_rgb(a):
    a = np.squeeze(a)
    if a.ndim == 3 and a.shape[0] in (3, 4) and a.shape[-1] not in (3, 4):
        a = a.transpose(1, 2, 0)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    a = a[..., :3]
    if a.dtype != np.uint8:
        a = a.astype(np.float32)
        if np.nanmax(a) <= 1.5:
            a = a * 255.0
        elif np.nanmax(a) > 255:
            lo, hi = np.nanpercentile(a, [1, 99])
            a = (a - lo) / max(hi - lo, 1e-6) * 255.0
        a = np.clip(np.nan_to_num(a), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(a)


def to_height(a, scale=1.0):
    h = np.squeeze(a).astype(np.float32) * float(scale)
    h[~np.isfinite(h)] = np.nan
    h[h < -5] = np.nan                  # nodata fill values
    h = np.where(np.isfinite(h), np.clip(h, 0, 650), np.nan)
    return h


def to_cls(a):
    c = np.squeeze(a)
    c = np.where(np.isfinite(c), c, 255).astype(np.int64)
    c[(c < 0) | (c > 6)] = 255
    return c.astype(np.uint8)


def list_tiles(local_dir=None, repo=REPO, splits=("train", "val", "test")):
    if local_dir:
        files = []
        for root, _, names in os.walk(local_dir):
            for n in names:
                files.append(os.path.relpath(os.path.join(root, n), local_dir).replace(os.sep, "/"))
    else:
        from huggingface_hub import list_repo_files
        files = list_repo_files(repo, repo_type="dataset")
    fs = set(files)
    out = []
    for f in sorted(files):
        m = PAT.match(f)
        if not m or m["split"] not in splits:
            continue
        tid, split = m["id"], m["split"]
        hf = f"heights/{split}/{tid}_AGL.h5"
        cf = f"classes/{split}/{tid}_CLS.h5"
        if hf not in fs:
            continue
        out.append(dict(id=tid, split=split, rgb_file=f, h_file=hf,
                        cls_file=cf if cf in fs else None))
    return out


def fetch(rel, local_dir, tmp, repo):
    if local_dir:
        return os.path.join(local_dir, rel), False
    from huggingface_hub import hf_hub_download
    for attempt in range(6):
        try:
            p = hf_hub_download(repo, rel, repo_type="dataset", local_dir=tmp)
            return p, True
        except Exception as e:
            if attempt == 5:
                raise
            # 429 = the Hub's rate limit for anonymous downloads: back off
            # hard. An HF_TOKEN (free account) raises the limit a lot.
            wait = 10 * 2 ** attempt if "429" in str(e) else 5 * (attempt + 1)
            time.sleep(wait)


def convert(rec, out, local_dir, tmp, repo, scale, quality):
    tid, split = rec["id"], rec["split"]
    city = tid.split("_")[0]
    entry = dict(id=f"gamus_{tid}", split=split, gsd=GSD, source="gamus", region=city,
                 landscape="urban")
    rp = os.path.join(split, f"gamus_{tid}_rgb.jpg")
    hp = os.path.join(split, f"gamus_{tid}_h.png")
    cp = os.path.join(split, f"gamus_{tid}_cls.png")
    if os.path.exists(os.path.join(out, hp)) and os.path.exists(os.path.join(out, rp)):
        entry.update(rgb=rp, height=hp,
                     cls=cp if os.path.exists(os.path.join(out, cp)) else None, cached=True)
        return entry
    paths = []
    try:
        p, rm = fetch(rec["rgb_file"], local_dir, tmp, repo)
        paths.append((p, rm))
        rgb = to_rgb(read_h5(p))
        p, rm = fetch(rec["h_file"], local_dir, tmp, repo)
        paths.append((p, rm))
        h = to_height(read_h5(p), scale)
        cls = None
        if rec["cls_file"]:
            p, rm = fetch(rec["cls_file"], local_dir, tmp, repo)
            paths.append((p, rm))
            cls = to_cls(read_h5(p))
    finally:
        for p, rm in paths:
            if rm and os.path.exists(p):
                os.remove(p)
    if rgb.shape[:2] != h.shape:
        raise ValueError(f"{tid}: rgb {rgb.shape} vs height {h.shape}")
    rp, hp, cp = D.write_tile(out, split, f"gamus_{tid}", rgb, h, cls, jpeg_q=quality)
    v = h[np.isfinite(h)]
    entry.update(rgb=rp, height=hp, cls=cp,
                 h_p50=float(np.percentile(v, 50)) if v.size else None,
                 h_p99=float(np.percentile(v, 99)) if v.size else None,
                 h_max=float(v.max()) if v.size else None,
                 valid=float(np.isfinite(h).mean()))
    return entry


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--local-dir", default=None,
                    help="read the HDF5 files from here instead of the Hub")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--limit", type=int, default=0, help="per split, 0 = all")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--height-scale", type=float, default=1.0,
                    help="multiply GAMUS heights by this (1.0 = they are metres)")
    ap.add_argument("--jpeg-quality", type=int, default=95)
    a = ap.parse_args()

    splits = tuple(s.strip() for s in a.splits.split(","))
    recs = list_tiles(a.local_dir, a.repo, splits)
    if a.limit:
        per = {}
        keep = []
        for r in recs:
            per[r["split"]] = per.get(r["split"], 0) + 1
            if per[r["split"]] <= a.limit:
                keep.append(r)
        recs = keep
    counts = {s: sum(r["split"] == s for r in recs) for s in splits}
    print(f"[gamus] {len(recs)} tiles to convert  {counts}", flush=True)
    if not recs:
        sys.exit("no GAMUS tiles found - check --repo / --local-dir")

    os.makedirs(a.out, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="gamus_")
    entries, failed = [], []
    t0 = time.time()
    try:
        with ThreadPoolExecutor(a.workers) as ex:
            futs = {ex.submit(convert, r, a.out, a.local_dir, tmp, a.repo,
                              a.height_scale, a.jpeg_quality): r for r in recs}
            for n, f in enumerate(as_completed(futs), 1):
                r = futs[f]
                try:
                    entries.append(f.result())
                except Exception as e:
                    failed.append(r["id"])
                    print(f"[gamus] FAILED {r['id']}: {type(e).__name__}: {e}", flush=True)
                if n == 20 or (n <= 20 and n == len(recs)):
                    p99 = [e["h_p99"] for e in entries if e.get("h_p99") is not None]
                    mx = [e["h_max"] for e in entries if e.get("h_max") is not None]
                    if p99:
                        print(f"[gamus] UNIT CHECK over the first {len(p99)} tiles: "
                              f"p99 height median {np.median(p99):.1f}, max {max(mx):.1f}. "
                              f"Metres expected (p99 roughly 10-60).", flush=True)
                        if np.median(p99) > 300:
                            print("[gamus] WARNING: these look like centimetres or "
                                  "millimetres - rerun with --height-scale 0.01 or 0.001",
                                  flush=True)
                if n % 200 == 0 or n == len(recs):
                    rate = n / max(time.time() - t0, 1e-6)
                    print(f"[gamus] {n}/{len(recs)}  {rate:.1f} tiles/s  "
                          f"eta {(len(recs) - n) / max(rate, 1e-6) / 60:.0f} min", flush=True)
                    D.save_index(a.out, entries, dataset="GAMUS", repo=a.repo)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for e in entries:
        e.pop("cached", None)
    D.save_index(a.out, sorted(entries, key=lambda e: e["id"]), dataset="GAMUS", repo=a.repo)
    by = {s: sum(e["split"] == s for e in entries) for s in splits}
    print(f"[gamus] done: {len(entries)} tiles {by}, {len(failed)} failed, "
          f"{(time.time() - t0) / 60:.1f} min -> {a.out}")


if __name__ == "__main__":
    main()
