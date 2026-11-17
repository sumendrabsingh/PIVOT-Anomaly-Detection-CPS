"""PIVOT v2 detectors under the label-free protocol.

v2's headline (SWaT 20->29 of 35, WADI 4->7 of 8, HAI 1->13 of 14) used test labels for calibration:
thresholds and score standardization on test rows labelled normal, a common-FAR operating point
chosen on the test labels, and an HAI calibration window whose length came from the first attack
label. Here every threshold comes from training data or from a commissioning window fixed in
advance. No parameter is tuned on these results.

  preprocessing  FAULT rule: a value > 100 training SDs from the training median is a sensor fault
                 and is replaced by that median (label-free, per sample)
  SWaT   base  = label-free array without analyzers (seed 4) OR analyzer arm   (analyzer_arm.py)
         arm   = physics residual (swat_invariants.json), per-invariant P99.8 of TRAINING residuals
  WADI   base  = label-free subsystem array (Oracles/wadi_none_seed4, lambda = 0), dual threshold P99.9 / P99 of training
         arm   = relationship residual, k = 8, Wa = 3600, k_thr = 5 (as shipped; already label-free),
                 on raw input: its adaptive baseline absorbs the WADI fault, while imputing it breaks
                 partner predictions (FAR 1.8% -> 3.2%; see relationship_detector.py --fault_rule)
  HAI    base  = auto-derived physics residual, per-invariant P99.8 of TRAINING residuals
         arm   = forecaster (Oracles/temporal_hai_W60.pth), P99 calibrated on
                 (a) training data, or (b) the first 15 min of the test stream (fixed in advance)

Each row also reports how many attacks the detector's own alarms hit by chance when shifted to random
offsets (chance_mean, chance_p95), since detectors with many short false alarms hit attacks by luck.

Run from the repo root:  python v2_label_free.py
"""
import json
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, '.')
import analyzer_arm as aa
import label_free as lfp
import label_free_dual_threshold as lf
import label_free_drift_adaptive as da
import physics_residual_detector as phys
import relationship_detector as rel
import temporal_detector as tmp
from evaluate_events import evaluate, to_binary

COMMISSION = 15 * 60


# SWaT array: lambda = 0 headline models since 2026-10-06 (legacy physics term: Oracles/swat_seed4_noAIT)
SWAT_ARRAY = 'Oracles/swat_phys_none_seed4'

def load(ds):
    tr = pd.read_csv(f'Datasets/{ds}_train_preprocessed.csv'); tr.columns = tr.columns.str.strip()
    te = pd.read_csv(f'Datasets/{ds}_test_preprocessed.csv'); te.columns = te.columns.str.strip()
    lab = 'Label' if 'Label' in te.columns else 'Normal/Attack'
    y = to_binary(te[lab])
    num = [c for c in te.columns if c not in ('Label', 'Normal/Attack') and pd.api.types.is_numeric_dtype(te[c])]
    med = tr[num].median()
    fault = lfp.fault_mask(tr, te)[num]   # shared rule: > 100 training SDs for >= 1 h
    raw = te
    te = te.copy()
    te[num] = te[num].mask(fault, med, axis=1)
    fired = {c: f'{v:.1%}' for c, v in fault.mean().items() if v > 0}
    print(f'[{ds}] test {len(te):,} rows, {y.sum():,} attack rows; FAULT rule fires on {fired or "nothing"}')
    return tr, te, y, raw


def physics_label_free(tr, te, inv):
    r_te, r_tr = phys.compute_residuals(tr, te, inv), phys.compute_residuals(tr, tr, inv)
    pred = np.zeros(len(te), dtype=int)
    for k in r_te:
        sm = lambda r: pd.Series(r).rolling(lf.W, min_periods=1).mean().values
        pred = np.maximum(pred, (sm(r_te[k]) > np.percentile(sm(r_tr[k]), 99.8)).astype(int))
    return pred


def array_label_free(model_dir, subs, te, tr):
    lf.MODELS = model_dir
    s_te, s_tr = [], []
    for name in subs:
        model, ck = lf.load_model(name)
        f = ck['features']
        e_te = lf.errors(model, te[f].fillna(te[f].mean()).values, ck['mean'], ck['std'])
        e_tr = lf.errors(model, tr[f].fillna(tr[f].mean()).values, ck['mean'], ck['std'])
        z = lambda e: pd.Series((e - e_tr.mean()) / (e_tr.std() + 1e-6)).rolling(lf.W, min_periods=1).mean().values
        s_te.append(z(e_te)); s_tr.append(z(e_tr))
    agg, cal = np.max(s_te, axis=0), np.max(s_tr, axis=0)
    return lf.dual(agg, np.percentile(cal, 99.9), np.percentile(cal, 99.0), 50, causal=True)


def chance_events(pred, y, n=200):
    """Attacks hit by the detector's own alarm pattern circularly shifted to random offsets:
    same alarm rate and episode structure, no alignment with the attacks."""
    m = np.concatenate([[0], y, [0]]); d = np.diff(m); s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    rng = np.random.default_rng(0)
    hits = [sum(np.roll(pred, rng.integers(5000, len(pred) - 5000))[a:b].any() for a, b in zip(s, e))
            for _ in range(n)]
    return np.mean(hits), np.percentile(hits, 95)


def row(plant, tag, y, pred):
    s, _ = evaluate(y, pred)
    mean, p95 = chance_events(pred, y)
    return {'plant': plant, 'detector': tag, 'F1': s['f1'], 'Prec': s['precision'], 'Rec': s['recall'],
            'FAR': s['false_alarm'], 'events': f"{s['n_detected']}/{s['n_events']}",
            'chance_mean': round(mean, 1), 'chance_p95': p95,
            'FA_episodes': s['false_alarm_events'], 'TTD_med_s': s.get('ttd_median_s', np.nan)}


if __name__ == '__main__':
    rows = []

    # ---- SWaT
    tr, te, y, _ = load('swat')
    ait = sorted(c for c in te.columns if c.startswith('AIT'))
    _, ref = aa.change_scores(tr, ait)
    s_tr, s_te = aa.score(aa.scale_adapted(tr, ait, ref)), aa.score(aa.scale_adapted(te, ait, ref))
    arm_ait = lf.dual(s_te, np.percentile(s_tr, 99.9), np.percentile(s_tr, 99.0), 50, causal=True)
    agg, cal = da.scores(SWAT_ARRAY, te, tr)
    base = np.maximum(lf.dual(agg, np.percentile(cal, 99.9), np.percentile(cal, 99.0), 50, causal=True), arm_ait)
    arm = physics_label_free(tr, te, json.load(open('Datasets/swat_invariants.json')))
    rows += [row('SWaT', 'base: array + analyzer arm', y, base), row('SWaT', 'arm: physics residual', y, arm),
             row('SWaT', 'union', y, np.maximum(base, arm))]

    # ---- WADI
    tr, te, y, te_raw = load('wadi')
    base = array_label_free('Oracles/wadi_none_seed4', json.load(open('Datasets/wadi_subsystems.json')), te, tr)
    cols = [c for c in te.columns if c not in ('Normal/Attack', 'Label', 'Timestamp')]
    Xtr, Xte = tr[cols].values.astype(float), te_raw[cols].values.astype(float)
    mu, sd, rg = Xtr.mean(0), Xtr.std(0), Xtr.max(0) - Xtr.min(0)
    sdf = np.maximum(sd, 0.01 * rg + 1e-6)
    sm = rel.adaptive_score(rel.relationship_residuals((Xtr - mu) / sdf, (Xte - mu) / sdf, 8), Wa=3600)
    arm = (sm > 5).astype(int)
    rows += [row('WADI', 'base: label-free array', y, base), row('WADI', 'arm: relationship residual', y, arm),
             row('WADI', 'union', y, np.maximum(base, arm))]

    # ---- HAI
    tr, te, y, _ = load('hai')
    print(f'[hai] first attack at row {int(np.argmax(y == 1)):,}; commissioning window = first {COMMISSION} rows')
    base = physics_label_free(tr, te, phys.auto_derive_invariants(tr))
    ck = torch.load('Oracles/temporal_hai_W60.pth', map_location=tmp.DEVICE, weights_only=False)
    model = tmp.Forecaster(ck['in_dim'], ck['out_dim']).to(tmp.DEVICE)
    model.load_state_dict(ck['state'])
    Ztr = ((tr[ck['cols']].values - ck['mu']) / ck['sd']).astype(np.float32)
    Zte = ((te[ck['cols']].values - ck['mu']) / ck['sd']).astype(np.float32)
    g_tr, g_te = tmp.aggregate_score(tmp.per_dim_sq_error(model, Ztr, ck['window']),
                                     tmp.per_dim_sq_error(model, Zte, ck['window']), 'max')
    sm_te = pd.Series(g_te).rolling(25, min_periods=1).mean().values
    sm_tr = pd.Series(g_tr).rolling(25, min_periods=1).mean().values
    arm_train = (sm_te > np.percentile(sm_tr, 99.0)).astype(int)
    arm_comm = (sm_te > np.percentile(sm_te[:COMMISSION], 99.0)).astype(int)
    rows += [row('HAI', 'base: physics residual (auto)', y, base),
             row('HAI', 'arm: forecaster, training-calibrated', y, arm_train),
             row('HAI', 'arm: forecaster, 15-min commissioning', y, arm_comm),
             row('HAI', 'union (training-calibrated arm)', y, np.maximum(base, arm_train)),
             row('HAI', 'union (15-min commissioning arm)', y, np.maximum(base, arm_comm))]

    out = pd.DataFrame(rows)
    fmt = out.copy()
    for c in ('F1', 'Prec', 'Rec', 'FAR'):
        fmt[c] = (fmt[c] * 100).map('{:.2f}'.format)
    pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 50)
    print(fmt.to_string(index=False))
    out.to_csv('v2_label_free.csv', index=False)
    print('[SAVED] v2_label_free.csv')
