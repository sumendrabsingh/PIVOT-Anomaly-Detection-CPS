"""Label-free fixes for SWaT sensor drift in the v1 dual-threshold array (seed 4).

label_free_dual_threshold.py showed the paper's 83% F1 depends on normalizing with test rows
labelled normal, which silently corrects analyzer drift (AIT201 sits 10.9 training SDs off in
the normal test periods). Two label-free fixes, parameters fixed before looking at results:

  DRIFT  Causal baseline correction of every continuous channel, no retraining:
         x'_t = x_t - (median(x over [t-GAP-H, t-GAP)) - median_train(x)),  H = 6 h, GAP = 1 h.
         Tank levels cycle in minutes, so their offset stays ~0; slow analyzer drift is removed.
         GAP exceeds every SWaT attack except the single 10-hour one.
  GATED  DRIFT, but the baseline skips samples a first DRIFT pass alarmed on, so a long attack
         cannot become the baseline (without it, the 10-hour attack leaves a 4.3-hour false alarm).
         Causal: the baseline at t only excludes earlier samples the first pass flagged.
  ANALYZER  DRIFT restricted to the sensor families that physically drift: chemical analyzers
         (AIT*) and membrane pressures (PIT*, DPIT*). Levels and flows are held at setpoints by
         control loops, so correcting them only lets a long attack leak into the baseline.
  NOAIT  Rule fixed in advance: drop all chemical analyzers (AIT*), which drift with source-water
         chemistry. Stages 2, 4, 5 are retrained without them (seed 4); stages 1, 3, 6 are reused.

Error standardization and thresholds are calibrated on training data only:
tau_H = P99.9, tau_L = P99 of the training aggregate score, causal Tier 2.

Run from the repo root:  python label_free_drift_adaptive.py
"""
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '.')
import label_free_dual_threshold as lf
import run_generalized_subsystems as gen

H, GAP = 6 * 3600, 3600
SEED_DIR, NOAIT_DIR = 'Oracles/swat_seed4', 'Oracles/swat_seed4_noAIT'


def drift_correct(df, cont, ref_median, skip=None):
    out = df.copy()
    for c in cont:
        x = df[c] if skip is None else df[c].mask(skip)
        base = x.shift(GAP).rolling(H, min_periods=H // 4).median()
        out[c] = df[c] - (base.fillna(ref_median[c]) - ref_median[c])
    return out


def build_noait():
    subs = json.load(open('Datasets/swat_subsystems.json'))
    os.makedirs(NOAIT_DIR, exist_ok=True)
    for name, feats in subs.items():
        kept = [f for f in feats if not f.startswith('AIT')]
        dst = f'{NOAIT_DIR}/subsystem_{name}.pth'
        if os.path.exists(dst):
            continue
        if kept == feats:
            shutil.copy(f'{SEED_DIR}/subsystem_{name}.pth', dst)
        else:
            print(f'  retraining {name} without {sorted(set(feats) - set(kept))}')
            gen.train_subsystem(name, kept, lf.TRAIN, NOAIT_DIR, seed=4)


def scores(model_dir, te, tr):
    """Per-subsystem smoothed z-scores, standardized on training errors (no labels)."""
    lf.MODELS = model_dir
    s_te, s_tr = [], []
    for name in lf.SUBS:
        model, ck = lf.load_model(name)
        f = ck['features']
        e_te = lf.errors(model, te[f].fillna(te[f].mean()).values, ck['mean'], ck['std'])
        e_tr = lf.errors(model, tr[f].fillna(tr[f].mean()).values, ck['mean'], ck['std'])
        z = lambda e: pd.Series((e - e_tr.mean()) / (e_tr.std() + 1e-6)).rolling(lf.W, min_periods=1).mean().values
        s_te.append(z(e_te)); s_tr.append(z(e_tr))
    return np.max(s_te, axis=0), np.max(s_tr, axis=0)


if __name__ == '__main__':
    te = pd.read_csv(lf.TEST); te.columns = te.columns.str.strip()
    tr = pd.read_csv(lf.TRAIN); tr.columns = tr.columns.str.strip()
    y = te['Normal/Attack'].astype(str).str.strip().str.lower().str.replace(' ', '').isin(['attack', '1']).astype(int).values
    feats = {f for v in lf.SUBS.values() for f in v}
    cont = [c for c in te.columns if c in feats and te[c].nunique() > 3]
    ref = tr[cont].median()
    print(f'drift-correcting {len(cont)} continuous channels (H={H // 3600} h, gap={GAP // 3600} h)')
    te_d, tr_d = drift_correct(te, cont, ref), drift_correct(tr, cont, ref)
    build_noait()

    # First DRIFT pass, used only to keep alarmed samples out of the GATED baseline
    agg, cal = scores(SEED_DIR, te_d, tr_d)
    th, tl = np.percentile(cal, 99.9), np.percentile(cal, 99.0)
    skip_te = pd.Series(lf.dual(agg, th, tl, 50, causal=True) == 1, index=te.index)
    skip_tr = pd.Series(lf.dual(cal, th, tl, 50, causal=True) == 1, index=tr.index)
    te_g, tr_g = drift_correct(te, cont, ref, skip_te), drift_correct(tr, cont, ref, skip_tr)

    drifting = [c for c in cont if c.startswith(('AIT', 'PIT', 'DPIT'))]
    te_a, tr_a = drift_correct(te, drifting, ref), drift_correct(tr, drifting, ref)

    configs = {'DRIFT (seed-4 models, drift-corrected input)': (SEED_DIR, te_d, tr_d),
               'GATED (DRIFT, baseline skips alarmed samples)': (SEED_DIR, te_g, tr_g),
               'ANALYZER (DRIFT on AIT/PIT/DPIT only)': (SEED_DIR, te_a, tr_a),
               'NOAIT (AIT* excluded, raw input)': (NOAIT_DIR, te, tr),
               'NOAIT + DRIFT': (NOAIT_DIR, te_d, tr_d)}
    rows = []
    for tag, (mdir, a, b) in configs.items():
        agg, cal = scores(mdir, a, b)
        for wd in (50, 200):
            p = (np.percentile(cal, 99.9), np.percentile(cal, 99.0), wd)
            rows.append(lf.summarize(f'{tag}, W={wd}', y, lf.dual(agg, *p, causal=True), p))
        p = lf.oracle_grid(agg, y, causal=True)
        rows.append(lf.summarize(f'{tag}, oracle grid (ceiling)', y, lf.dual(agg, *p, causal=True), p))
        print('  done', tag)

    out = pd.DataFrame(rows)
    fmt = out.copy()
    for c in ('F1', 'Prec', 'Rec', 'FAR'):
        fmt[c] = (fmt[c] * 100).map('{:.2f}'.format)
    pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 70)
    print(fmt.to_string(index=False))
    dest = sys.argv[1] if len(sys.argv) > 1 else 'label_free_drift_adaptive_swat.csv'
    out.to_csv(dest, index=False)
    print(f'[SAVED] {dest}')
