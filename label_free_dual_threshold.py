"""How much of the v1 SWaT headline (dual-threshold, F1 83.35%, seed 4) comes from test-label tuning?

Three label leaks in the paper protocol (run_subsystem_experiments.evaluate_dual_threshold):
  (a) ALIGN_BASELINE: input mean/std recomputed on test rows labelled normal
  (b) per-subsystem error standardization on test rows labelled normal
  (c) (tau_H, tau_L, W_dur) grid-searched to maximize F1 on the test labels
plus (d) Tier-2 backfill: a sustained run is credited from its start, before the alarm could fire.

Run from the repo root:  python <this file>
"""
import json
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, '.')
import run_subsystem_experiments as v1
from evaluate_events import evaluate

MODELS = 'Oracles/swat_seed4'
TEST = 'Datasets/swat_test_preprocessed.csv'
TRAIN = 'Datasets/swat_train_preprocessed.csv'
SUBS = json.load(open('Datasets/swat_subsystems.json'))
W = 25
DEV = v1.DEVICE


def load_model(name):
    ck = torch.load(f'{MODELS}/subsystem_{name}.pth', map_location='cpu', weights_only=False)
    st = ck['model_state']
    d, b = ck['input_dim'], ck['bottleneck_dim']
    h1, h2 = st['encoder.0.weight'].shape[0], st['encoder.4.weight'].shape[0]
    m = v1.SubsystemAutoencoder(d, b)
    m.encoder = nn.Sequential(nn.Linear(d, h1), nn.BatchNorm1d(h1), nn.ReLU(), nn.Dropout(0.2),
                              nn.Linear(h1, h2), nn.BatchNorm1d(h2), nn.ReLU(), nn.Dropout(0.2),
                              nn.Linear(h2, b))
    m.decoder = nn.Sequential(nn.Linear(b, h2), nn.BatchNorm1d(h2), nn.ReLU(), nn.Dropout(0.2),
                              nn.Linear(h2, h1), nn.BatchNorm1d(h1), nn.ReLU(), nn.Linear(h1, d))
    m.load_state_dict(st)
    return m.to(DEV).eval(), ck


def errors(model, X, mean, std):
    Xn = ((torch.tensor(X, dtype=torch.float32) - mean) / std)
    out = [model.get_reconstruction_error(Xn[i:i + 8192].to(DEV), mode='max').cpu()
           for i in range(0, len(Xn), 8192)]
    return torch.cat(out).numpy()


def runs(mask):
    """(start, length) of each maximal True run."""
    m = np.concatenate([[0], mask.astype(np.int8), [0]])
    d = np.diff(m)
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    return s, e - s


def dual(agg, th, tl, wdur, causal=False):
    pred = (agg > th).astype(np.int8)
    s, L = runs(agg > tl)
    for a, n in zip(s, L):
        if n >= wdur:
            pred[(a + wdur - 1 if causal else a):a + n] = 1
    return pred


def oracle_grid(agg, y, causal=False):
    """Exactly the v1 sweep: percentiles of the TEST scores, best F1 on TEST labels."""
    p = np.percentile(agg, [85, 90, 95, 97, 99, 99.5, 99.8])
    best = (-1, None)
    for th in np.linspace(p[3], p[6], 15):
        for tl in np.linspace(p[0], p[3], 15):
            if tl >= th:
                continue
            for wd in [25, 50, 75, 100, 150, 200]:
                pr = dual(agg, th, tl, wd, causal)
                tp = int(((pr == 1) & (y == 1)).sum()); fp = int(((pr == 1) & (y == 0)).sum())
                fn = int(((pr == 0) & (y == 1)).sum())
                f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
                if f1 > best[0]:
                    best = (f1, (th, tl, wd))
    return best[1]


def summarize(tag, y, pred, params=None):
    s, _ = evaluate(y, pred)
    row = {'protocol': tag, 'F1': s['f1'], 'Prec': s['precision'], 'Rec': s['recall'],
           'FAR': s['false_alarm'], 'events': f"{s['n_detected']}/{s['n_events']}",
           'FA_episodes': s['false_alarm_events'],
           'TTD_med_s': s.get('ttd_median_s', np.nan)}
    if params:
        row['params'] = f"tH={params[0]:.2f} tL={params[1]:.2f} W={params[2]}"
    return row


if __name__ == '__main__':
    te = pd.read_csv(TEST); te.columns = te.columns.str.strip()
    tr = pd.read_csv(TRAIN); tr.columns = tr.columns.str.strip()
    lab = 'Label' if 'Label' in te.columns else 'Normal/Attack'
    y = te[lab].astype(str).str.strip().str.lower().str.replace(' ', '').isin(['attack', '1']).astype(int).values
    k0 = int(np.argmax(y == 1))  # attack-free deployment window = everything before the first attack
    print(f'test rows {len(y):,} | attacks {y.sum():,} | first attack at row {k0:,} | train rows {len(tr):,}')

    sc = {k: {} for k in ('paper', 'train', 'deploy')}
    for name in SUBS:
        model, ck = load_model(name)
        f = ck['features']
        Xte = te[f].fillna(te[f].mean()).values
        Xtr = tr[f].fillna(tr[f].mean()).values
        # (a) leaky input normalization vs training statistics
        xt = torch.tensor(Xte, dtype=torch.float32)
        mu_a, sd_a = xt[y == 0].mean(0), xt[y == 0].std(0)
        sd_a = torch.where(sd_a < 1e-4, torch.ones_like(sd_a), sd_a)
        e_paper = errors(model, Xte, mu_a, sd_a)
        e_te = errors(model, Xte, ck['mean'], ck['std'])
        e_tr = errors(model, Xtr, ck['mean'], ck['std'])
        # (b) error standardization: test-normal (leaky) | training set | deployment window
        z = lambda e, ref: (e - ref.mean()) / (ref.std() + 1e-6)
        sm = lambda v: pd.Series(v).rolling(W, min_periods=1).mean().values
        sc['paper'][name] = sm(z(e_paper, e_paper[y == 0]))
        sc['train'][name] = (sm(z(e_te, e_tr)), sm(z(e_tr, e_tr)))
        sc['deploy'][name] = sm(z(e_te, e_te[:k0]))
        print(f'  scored {name}')

    agg_paper = np.max(list(sc['paper'].values()), axis=0)
    agg_train = np.max([v[0] for v in sc['train'].values()], axis=0)
    agg_train_cal = np.max([v[1] for v in sc['train'].values()], axis=0)
    agg_deploy = np.max(list(sc['deploy'].values()), axis=0)

    rows = []
    p = oracle_grid(agg_paper, y)
    rows.append(summarize('A. Paper protocol (a+b+c+d leaks)', y, dual(agg_paper, *p), p))
    p = oracle_grid(agg_paper, y, causal=True)
    rows.append(summarize('B. Paper protocol, causal Tier 2 (no backfill)', y, dual(agg_paper, *p, causal=True), p))

    # Label-free threshold rule, pre-registered: tH = calibration P99.9, tL = calibration P99, W = 200 / 50
    def rule(cal, wd, hi=99.9, lo=99.0):
        return (np.percentile(cal, hi), np.percentile(cal, lo), wd)

    for wd in (200, 50):
        p = rule(agg_paper[y == 0], wd)
        rows.append(summarize(f'C. Leaky norm (a+b), rule thresholds on test-normal, W={wd}', y,
                              dual(agg_paper, *p, causal=True), p))
    for wd in (200, 50):
        p = rule(agg_train_cal, wd)
        rows.append(summarize(f'D. Fully label-free: train-calibrated, W={wd}', y, dual(agg_train, *p, causal=True), p))
    for wd in (200, 50):
        p = rule(agg_deploy[:k0], wd)
        rows.append(summarize(f'E. Fully label-free: deployment window, W={wd}', y, dual(agg_deploy, *p, causal=True), p))
    # Ceiling: honest scores, but thresholds still tuned on test labels (isolates the normalization effect)
    p = oracle_grid(agg_train, y, causal=True)
    rows.append(summarize('F. Train-calibrated scores + oracle grid (ceiling)', y, dual(agg_train, *p, causal=True), p))

    out = pd.DataFrame(rows)
    pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 70)
    fmt = out.copy()
    for c in ('F1', 'Prec', 'Rec', 'FAR'):
        fmt[c] = (fmt[c] * 100).map('{:.2f}'.format)
    print(fmt.to_string(index=False))
    dest = sys.argv[1] if len(sys.argv) > 1 else 'label_free_dual_threshold_swat.csv'
    out.to_csv(dest, index=False)
    print(f'[SAVED] {dest}')
