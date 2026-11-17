"""
PIVOT v2 -- Phase A: is the v1 paper's SWaT headline reproducible?

A single retrain landing near the paper proves little: v1 pinned only the train/val split, so
weight init and batch shuffling varied run to run. The honest test is a seed distribution --
train N independent seeds, and ask whether the published numbers sit inside it.

Target -- "Subsystem AE (Dual-Threshold, Point-wise)" from experiments_comparison.md:
    Acc 96.46%  Prec 97.68%  Recall 72.58%  F1 83.28%  FAR 0.24%

Config (the paper's): per-timestep (SEQ_LEN=1) subsystem AEs trained on normal-only
swat_train_preprocessed.csv; dual-threshold + temporal persistence (W=25, MAX error, MAX
aggregation, ALIGN_BASELINE) evaluated on swat_test_preprocessed.csv.

Usage:  python reproduce_paper_multiseed.py --seeds 5
Writes: paper_reproducibility_swat.csv  (one row per seed + summary stats)
"""

import argparse
import contextlib
import io
import json
import os

import numpy as np
import pandas as pd

import run_generalized_subsystems as gen
import run_subsystem_experiments as v1

TRAIN_DATA = r"Datasets\swat_train_preprocessed.csv"
EVAL_DATA = r"Datasets\swat_test_preprocessed.csv"
SUB_JSON = r"Datasets\swat_subsystems.json"

PAPER = {'accuracy': 0.9646, 'precision': 0.9768, 'recall': 0.7258, 'f1': 0.8328,
         'false_alarm': 0.0024}


def run_seed(seed, subsystems, quiet=True):
    out_dir = f"Oracles/swat_seeds/seed{seed}"
    os.makedirs(out_dir, exist_ok=True)

    # Train only if this seed's bundles are not already on disk (so the script is resumable).
    need_train = any(not os.path.exists(os.path.join(out_dir, f"subsystem_{n}.pth"))
                     for n in subsystems)
    if need_train:
        print(f"  [seed {seed}] training 6 subsystem AEs -> {out_dir}")
        buf = io.StringIO()
        ctx = contextlib.redirect_stdout(buf) if quiet else contextlib.nullcontext()
        with ctx:
            for name, features in subsystems.items():
                gen.train_subsystem(name, features, TRAIN_DATA, out_dir, seed=seed)
    else:
        print(f"  [seed {seed}] reusing existing bundles in {out_dir}")

    # Score with the paper's dual-threshold + persistence logic.
    v1.EVAL_DATA = EVAL_DATA
    v1.MODEL_DIR = out_dir
    buf = io.StringIO()
    ctx = contextlib.redirect_stdout(buf) if quiet else contextlib.nullcontext()
    with ctx:
        res = v1.evaluate_dual_threshold(window_size=v1.WINDOW_SIZE)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-seed reproducibility of the v1 SWaT headline")
    parser.add_argument('--seeds', type=int, default=5, help='Number of seeds (1..N)')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    with open(SUB_JSON) as f:
        subsystems = json.load(f)

    print("=" * 78)
    print(" REPRODUCIBILITY OF THE v1 PAPER HEADLINE (Subsystem AE, dual-threshold, point-wise)")
    print("=" * 78)
    print(f" Paper: F1 {PAPER['f1']:.2%}  Prec {PAPER['precision']:.2%}  "
          f"Rec {PAPER['recall']:.2%}  FAR {PAPER['false_alarm']:.2%}")
    print(f" Seeds: {args.seeds}   Train: {os.path.basename(TRAIN_DATA)}   "
          f"Eval: {os.path.basename(EVAL_DATA)}")
    print("=" * 78)

    rows = []
    for seed in range(1, args.seeds + 1):
        res = run_seed(seed, subsystems, quiet=not args.verbose)
        if res is None:
            print(f"  [seed {seed}] FAILED")
            continue
        row = {'seed': seed, **{k: res[k] for k in PAPER}}
        rows.append(row)
        print(f"  [seed {seed}] F1 {res['f1']:.2%}  Prec {res['precision']:.2%}  "
              f"Rec {res['recall']:.2%}  FAR {res['false_alarm']:.2%}")

    df = pd.DataFrame(rows)

    print("\n" + "=" * 78)
    print(" VERDICT -- does the published number sit inside the seed distribution?")
    print("=" * 78)
    print(f"{'metric':<12} {'paper':>8} {'mean':>8} {'std':>7} {'min':>8} {'max':>8}   verdict")

    summary = {}
    for metric, paper_val in PAPER.items():
        vals = df[metric].values
        mean, std = vals.mean(), vals.std(ddof=1) if len(vals) > 1 else 0.0
        lo, hi = vals.min(), vals.max()
        # "Reproduced" = the paper's value lies within the observed seed range, or within 2 std
        # of the seed mean (a seed we did not happen to draw could plausibly have produced it).
        within_range = lo <= paper_val <= hi
        within_2std = abs(paper_val - mean) <= 2 * std if std > 0 else False
        ok = within_range or within_2std
        verdict = "REPRODUCED" if ok else "OUTSIDE"
        if within_range:
            verdict += " (in range)"
        elif within_2std:
            verdict += " (within 2 sd)"
        summary[metric] = {'paper': paper_val, 'mean': float(mean), 'std': float(std),
                           'min': float(lo), 'max': float(hi), 'reproduced': bool(ok)}
        print(f"{metric:<12} {paper_val:>7.2%} {mean:>7.2%} {std:>6.2%} {lo:>7.2%} {hi:>7.2%}   {verdict}")

    df.to_csv('paper_reproducibility_swat.csv', index=False)
    with open('paper_reproducibility_swat.json', 'w') as f:
        json.dump(summary, f, indent=2)

    headline_ok = summary['f1']['reproduced']
    print("\n" + ("VERDICT: the paper's headline F1 IS reproducible across seeds."
                  if headline_ok else
                  "VERDICT: the paper's headline F1 is NOT inside the seed distribution."))
    print("[SAVED] paper_reproducibility_swat.csv / .json")
