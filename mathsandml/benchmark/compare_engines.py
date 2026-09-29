#!/usr/bin/env python3
"""The one table that decides the default engine."""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

CONFIGS = [
    # Key, label, engine, scale source, needs model
    ("zeroshot_prior", "current: zero-shot + scene prior", "zeroshot", "known-height", False),
    ("zeroshot_ghsl", "current: zero-shot + GHSL (automatic)", "zeroshot", "ghsl", False),
    ("a_prior", "A: fine-tuned relative + scene prior", "finetuned", "known-height", True),
    ("a_ghsl", "A: fine-tuned relative + GHSL (automatic)", "finetuned", "ghsl", True),
    ("b", "B: metres directly (automatic)", "metric", "model", True),
    ("hybrid", "A+B: hybrid (automatic)", "hybrid", "model", True),
    ("hybrid_prior", "A+B: hybrid + scene prior", "hybrid", "known-height", True),
]
LANDSCAPES = ("urban", "sparse", "hilly", "forest")


def fmt(v, nd=2):
    return "-" if v is None or v != v else f"{v:.{nd}f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scenes", default=os.path.join(HERE, "scenes"))
    parser.add_argument("--out", default=os.path.join(HERE, "results", "engines"))
    parser.add_argument("--only", default=None, help="comma-separated config keys")
    parser.add_argument("--reuse", action="store_true",
                    help="use results.json already in --out instead of re-running")
    args = parser.parse_args()

    import heightnet as HN
    have_model = HN.available() is not None
    keys = set(args.only.split(",")) if args.only else None
    rows = []
    for key, label, engine, src, needs in CONFIGS:
        if keys and key not in keys:
            continue
        out = os.path.join(args.out, key)
        if needs and not have_model:
            rows.append(dict(key=key, label=label, skipped=f"no trained model at "
                                                           f"{HN.default_checkpoint()}"))
            continue
        result = os.path.join(out, "results.json")
        if not (args.reuse and os.path.exists(result)):
            cmd = [sys.executable, os.path.join(HERE, "run_benchmark.py"), "--scenes", args.scenes,
                   "--out", out, "--scale-source", src, "--engine", engine, "--no-figures"]
            print(f"\n##### {label}\n$ {' '.join(cmd)}", flush=True)
            subprocess.run(cmd, cwd=os.path.dirname(ROOT))
        if not os.path.exists(result):
            rows.append(dict(key=key, label=label, skipped="benchmark run failed"))
            continue
        results = json.load(open(result))
        if not any(not r.get("error") for r in results["scenes"]):
            err = next((str(r.get("error")) for r in results["scenes"]), "no scenes")
            rows.append(dict(key=key, label=label, skipped="every scene failed - "
                                                           + err[:90].replace("|", "/")))
            continue
        rows.append(dict(key=key, label=label, engine=engine, source=src, results=results))

    import run_benchmark as RB
    lines = ["# Engine comparison - 9-scene LiDAR benchmark", "",
             "All metres. RMSE is shift-aligned (a constant vertical datum offset "
             "removed, as in report.md); **RMSE raw** is the untouched output. "
             "Landscape columns are pooled RMSE per landscape.", "",
             "| configuration | scenes | RMSE | **RMSE raw** | MAE | r | "
             + " | ".join(LANDSCAPES) + " |",
             "|---|--:|--:|--:|--:|--:|" + "--:|" * len(LANDSCAPES)]
    summary = []
    for row in rows:
        if row.get("skipped"):
            lines.append(f"| {row['label']} | - | *skipped: {row['skipped']}* | | | | "
                         + " | ".join("" for _ in LANDSCAPES) + " |")
            continue
        records = [r for r in row["results"]["scenes"] if not r.get("error")]
        p = RB.pool([r["headline"] for r in records]) or {}
        praw = RB.pool([r["raw_alignment"] for r in records if r.get("raw_alignment")]) or {}
        land = {}
        for cls in LANDSCAPES:
            q = RB.pool([r["by_landscape"].get(cls) for r in records
                         if r.get("by_landscape", {}).get(cls)])
            land[cls] = q["rmse"] if q else None
        n_all = len(row["results"]["scenes"])
        lines.append(f"| {row['label']} | {len(records)}/{n_all} | {fmt(p.get('rmse'))} | "
                     f"**{fmt(praw.get('rmse'))}** | {fmt(p.get('mae'))} | "
                     f"{fmt(p.get('r'), 3)} | "
                     + " | ".join(fmt(land[c]) for c in LANDSCAPES) + " |")
        summary.append(dict(key=row["key"], label=row["label"], scenes=len(records),
                            rmse=p.get("rmse"), rmse_raw=praw.get("rmse"),
                            mae=p.get("mae"), r=p.get("r"), by_landscape=land))
    lines += ["", "How to read it: the default engine should be the best **RMSE raw** "
                  "among the automatic rows (no human input), without losing any "
                  "landscape badly. 'scene prior' rows use a height a person read off "
                  "each ortho - a fair comparison between engines, but not what an "
                  "unattended run gets."]
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "engines.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(args.out, "engines.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print("\n" + "\n".join(lines))
    print(f"\nWrote {os.path.join(args.out, 'engines.md')}")


if __name__ == "__main__":
    main()
