#!/usr/bin/env python3
"""The whole training chain on synthetic data, on a CPU, in about a minute."""

import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "mathsandml"))


def fake_scene(rng, n=1024):
    import cv2
    rgb = np.zeros((n, n, 3), np.uint8)
    rgb[:] = (rng.integers(90, 130), rng.integers(110, 150), rng.integers(80, 110))
    h = np.zeros((n, n), np.float32)
    cls = np.ones((n, n), np.uint8)                    # ground
    cv2.rectangle(rgb, (0, n // 2 - 20), (n, n // 2 + 20), (70, 70, 75), -1)
    cls[n // 2 - 20:n // 2 + 20] = 5                   # road
    for _ in range(rng.integers(6, 14)):
        x, y = rng.integers(0, n - 120, 2)
        w, d = rng.integers(30, 120, 2)
        bh = float(rng.choice([4, 8, 15, 30, 60]))
        sh = int(bh * 0.8)
        cv2.rectangle(rgb, (int(x + sh), int(y + sh)), (int(x + w + sh), int(y + d + sh)),
                      (30, 30, 35), -1)
        c = tuple(int(v) for v in rng.integers(120, 220, 3))
        cv2.rectangle(rgb, (int(x), int(y)), (int(x + w), int(y + d)), c, -1)
        h[y:y + d, x:x + w] = bh
        cls[y:y + d, x:x + w] = 3
    for _ in range(rng.integers(5, 15)):
        x, y = rng.integers(20, n - 20, 2)
        r = int(rng.integers(6, 18))
        th = float(rng.uniform(5, 20))
        cv2.circle(rgb, (int(x), int(y)), r, (40, 100, 40), -1)
        yy, xx = np.ogrid[:n, :n]
        m = (yy - y) ** 2 + (xx - x) ** 2 <= r * r
        h[m] = np.maximum(h[m], th)
        cls[m] = 6
    return rgb, h, cls


def write_fake_gamus(root, rng, per_split=dict(train=6, val=2, test=2)):
    import h5py
    for split, k in per_split.items():
        for i in range(k):
            rgb, h, cls = fake_scene(rng)
            tid = f"DC_{i:02d}_{split[:2]}"
            for sub, name, arr in (("images", "RGB", rgb.transpose(2, 0, 1)),
                                   ("heights", "AGL", h),
                                   ("classes", "CLS", cls.astype(np.float32))):
                d = os.path.join(root, sub, split)
                os.makedirs(d, exist_ok=True)
                with h5py.File(os.path.join(d, f"{tid}_{name}.h5"), "w") as f:
                    f.create_dataset("data", data=arr)


def run(cmd):
    print("$ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode:
        sys.exit(f"FAILED: {' '.join(cmd)}")


def main():
    rng = np.random.default_rng(0)
    work = tempfile.mkdtemp(prefix="hn_selftest_")
    try:
        raw = os.path.join(work, "GAMUS")
        write_fake_gamus(raw, rng)
        tiles = os.path.join(work, "gamus")
        run([sys.executable, "prepare_gamus.py", "--local-dir", raw, "--out", tiles,
             "--workers", "2"])
        idx = json.load(open(os.path.join(tiles, "index.json")))
        assert len(idx["tiles"]) == 10, len(idx["tiles"])
        runs = os.path.join(work, "run")
        run([sys.executable, "train.py", "--data", tiles, "--out", runs, "--variant", "tiny",
             "--iters", "12", "--batch", "2", "--accum", "2", "--val-every", "6",
             "--val-n", "2", "--workers", "0", "--warmup", "3", "--log-every", "3"])
        best = os.path.join(runs, "heightnet_best.pt")
        assert os.path.exists(best) and os.path.exists(os.path.join(runs, "state.pt"))
        run([sys.executable, "train.py", "--data", tiles, "--out", runs, "--variant", "tiny",
             "--iters", "14", "--batch", "2", "--val-every", "7", "--val-n", "2",
             "--workers", "0", "--warmup", "3", "--resume", os.path.join(runs, "state.pt")])
        run([sys.executable, "evaluate.py", "--ckpt", best, "--data", tiles,
             "--split", "test", "--out", os.path.join(runs, "eval"), "--rotations", "2"])
        import heightnet as HN
        predictor = HN.HeightPredictor(best, device="cpu")
        print(f"  fusion stored in the checkpoint: {predictor.fusion}")
        img = (rng.random((611, 947, 3)) * 255).astype(np.uint8)
        for gsd in (None, 0.25, 0.7, 2.0):
            r = predictor.predict(img, gsd, rotations=2)
            assert r["height"].shape == img.shape[:2] and r["seg"].shape == img.shape[:2]
            assert r["rel"].shape == img.shape[:2] and np.isfinite(r["rel"]).all()
            assert np.isfinite(r["height"]).all() and (r["sigma"] >= 0).all()
            c = HN.combine(r, gsd or HN.GSD_DEFAULT, fusion=predictor.fusion)
            assert c["fused"].shape == img.shape[:2] and np.isfinite(c["fused"]).all()
            print(f"  predict gsd={gsd}: model gsd {r['gsd_model']:.2f}, {r['tiles']} tiles, "
                  f"B {r['height'].min():.1f}..{r['height'].max():.1f} m, "
                  f"A scale {c['alpha']}, fused mean {c['fused'].mean():.2f} m")
        import data as D
        import evaluate as E

        class FakePredictor:
            fusion = {}

            def predict(self, rgb, gsd, rotations=4):
                t = np.nan_to_num(self._h)
                n = np.random.default_rng(1).normal(0, 0.3, t.shape)
                return dict(height=0.9 * t + n, sigma=np.full(t.shape, 0.5),
                            rel=0.2 * t + 3.0 + 0.05 * n, rel_sigma=np.full(t.shape, 0.05))
        file_path = FakePredictor()
        tl = [t for t in D.load_index(tiles) if t["split"] == "test"]
        for t in tl:
            file_path._h = D.read_height(os.path.join(t["root"], t["height"]))
            c = HN.combine(file_path.predict(None, t["gsd"]), t["gsd"])
            assert c["alpha"] is not None and 3.5 < c["alpha"] < 5.5, c["alpha"]
        file_path._h = D.read_height(os.path.join(tl[0]["root"], tl[0]["height"]))
        fus = E.calibrate(file_path, tl[:1], rotations=1)
        assert fus and np.isfinite(fus["c_a"]) and np.isfinite(fus["c_b"]), fus
        assert fus["mode"] in ("ivw", "stacked") and len(fus["w_a"]) == 4, fus
        assert fus["calib_rmse_stacked"] <= fus["calib_rmse_B"] + 1e-9, fus
        print(f"  hybrid on a stand-in with known errors: A scale {c['alpha']:.2f} "
              f"(expect ~4.5), calibration {fus}")
        # The hybrid maths on a scene with a known answer
        det = np.zeros((200, 200)); det[50:100, 50:100] = 3.0; det[120:140, 20:180] = 1.0
        met = 7.5 * det
        a_ = HN.learned_scale(det, met)
        assert abs(a_ - 7.5) < 1e-6, a_
        fh, fs, wb = HN.fuse(np.full(4, 10.0), np.full(4, 1.0), np.full(4, 20.0), np.full(4, 3.0))
        assert abs(fh[0] - 11.0) < 1e-6 and abs(wb[0] - 0.1) < 1e-6, (fh, wb)
        yy = np.linspace(0, 40, 4000)
        fh, _, wb = HN.fuse(yy + 2, np.ones_like(yy), yy - 2, np.ones_like(yy),
                            dict(mode="stacked", w_a=[0.5] * 4))
        assert np.allclose(fh, yy), "stacked blend"
        print("  learned scale and fusion maths: exact")
        print("\nSELFTEST PASSED - data prep, training, resume, export, evaluation and "
              "inference all run.")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
