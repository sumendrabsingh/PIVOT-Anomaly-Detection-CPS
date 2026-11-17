"""Derive and validate WADI actuator -> flow interlocks from normal data (reviewer point R2.12).

The submitted paper used three WADI interlocks (1_P_001 -> 1_FIT_001, 2_MV_003 -> 2_FIT_001,
2_P_003 -> 2_FIT_003) found by a scratchpad run that was not archived, and the archived rule
(physics_residual_detector.auto_derive_invariants) finds none on the preprocessed CSVs: those are
z-scored, so 'flow below 10% of its median' compares z-values with a median of |z|, and it takes an
actuator's minimum as 'off', which for a three-state valve is state 0 (transitioning), not closed.
This script reruns the derivation in RAW units with the rule fixed before looking at the result.

Data: SWaT_Orig_DataSets/wadi/WADI_14days.csv (normal operation only), split in time:
  derivation  the first 10 days      held-out  the last 4 days (never used to choose anything)
Raw encoding (as SWaT): pumps 1 = off, 2 = on; valves 0 = transitioning, 1 = closed, 2 = open.

Rule, for every actuator status column with an off/closed state and every flow column (*FIT*):
  flow 'present' = flow >= 10% of its 95th percentile on the derivation days (its operating value;
                   the median is 0 for flows that are idle most of the time, so the documented
                   'median' scale cannot be used)
  support        off/closed on >= 50 derivation samples and < 98% of them
  off => no flow P(flow present | off)  <= 0.30   (the documented 70% rule)
  on  => flow    P(flow present | on)   >= 0.70   (added: otherwise a flow that is idle most of the
                                                   time passes for every actuator)
Rows are also reported for the documented rule alone (no 'on' condition) and for the legacy
'minimum state = off' choice, to show where the submitted three came from.

Held-out validation for every kept interlock: P(flow present | off) on the last 4 days, and the
false-alarm rate of its residual detector on those days (off-state flow / scale, smoothed W = 25,
threshold = max(P99.8 of the derivation days, 0.1)). On the attack file (labels used only to report):
violation rate inside and outside attacks and attacks caught against chance.

Run from the repo root:  python wadi_interlock_validation.py
                         python wadi_interlock_validation.py --combine preds_wadi_dual_lf.csv
(--combine ORs the interlock arm with another detector's exported predictions and scores the union.)
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '.')
import label_free as lfp
from evaluate_events import evaluate

RAW = 'SWaT_Orig_DataSets/wadi'
DERIVE_DAYS, DAY = 10, 86400
PRESENT, MIN_OFF, MAX_OFF_FRAC, OFF_MAX, ON_MIN = 0.10, 50, 0.98, 0.30, 0.70
W = 25
SUBMITTED = [('1_P_001_STATUS', '1_FIT_001_PV'), ('2_MV_003_STATUS', '2_FIT_001_PV'),
             ('2_P_003_STATUS', '2_FIT_003_PV')]


def load_raw(path, skiprows=0):
    hdr = pd.read_csv(path, skiprows=skiprows, nrows=0).columns
    short = {c: c.strip().split('\\')[-1] for c in hdr}
    use = [c for c in hdr if short[c].endswith('_STATUS') or 'FIT' in short[c]]
    return pd.read_csv(path, skiprows=skiprows, usecols=use).rename(columns=short).ffill().fillna(0)


def stats(act_v, flow_v, off_state, scale):
    off, on = act_v == off_state, act_v == 2
    present = flow_v >= PRESENT * scale
    return {'n_off': int(off.sum()), 'off_frac': float(off.mean()),
            'p_flow_given_off': float(present[off].mean()) if off.any() else np.nan,
            'p_flow_given_on': float(present[on].mean()) if on.any() else np.nan}


def residual(act_v, flow_v, scale):
    return pd.Series((act_v == 1) * flow_v / scale).rolling(W, min_periods=1).mean().values


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--combine', default=None, help='predictions CSV (y_pred column) to OR with the interlocks')
    ap.add_argument('--export', default='preds_wadi_interlock.csv')
    args = ap.parse_args()
    tr = load_raw(f'{RAW}/WADI_14days.csv', skiprows=4)
    te = load_raw(f'{RAW}/WADI_attackdata.csv')
    y = pd.read_csv('Datasets/wadi_test_preprocessed.csv', usecols=['Normal/Attack'])['Normal/Attack'].values
    assert len(y) == len(te), (len(y), len(te))
    cut = DERIVE_DAYS * DAY
    der, held = tr.iloc[:cut], tr.iloc[cut:]
    print(f'normal data {len(tr):,} rows: derivation {len(der):,} (days 1-{DERIVE_DAYS}), held-out {len(held):,}')

    acts = [c for c in tr.columns if c.endswith('_STATUS') and set(der[c].unique()) >= {1, 2}]
    flows = [c for c in tr.columns if 'FIT' in c and der[c].quantile(0.95) > 0]
    scale = {f: float(der[f].quantile(0.95)) for f in flows}
    print(f'{len(acts)} actuators with off and on states, {len(flows)} flows')

    rows = []
    for a in acts:
        av = der[a].values
        for f in flows:
            s = stats(av, der[f].values, 1, scale[f])
            support = s['n_off'] >= MIN_OFF and s['off_frac'] < MAX_OFF_FRAC
            documented = support and s['p_flow_given_off'] <= OFF_MAX
            rows.append({'actuator': a, 'flow': f, **s, 'flow_scale': scale[f],
                         'documented_rule': documented,
                         'kept': documented and s['p_flow_given_on'] >= ON_MIN})
    cand = pd.DataFrame(rows)
    kept = cand[cand.kept].copy()
    print(f'\ndocumented rule (off => no flow) passes {int(cand.documented_rule.sum())} pairs; '
          f'with on => flow, {len(kept)} pairs kept:')
    print(kept[['actuator', 'flow', 'n_off', 'off_frac', 'p_flow_given_off', 'p_flow_given_on']]
          .round(3).to_string(index=False))

    print('\nthe three submitted interlocks:')
    for a, f in SUBMITTED:
        r = cand[(cand.actuator == a) & (cand.flow == f)].iloc[0]
        legacy_off = int(der[a].min())
        lg = stats(der[a].values, der[f].values, legacy_off, scale[f])
        print(f'  {a} -> {f}: closed/off P(flow) {r.p_flow_given_off:.3f}, on P(flow) {r.p_flow_given_on:.3f}, '
              f'kept {bool(r.kept)} | legacy off = state {legacy_off}: n {lg["n_off"]:,}, '
              f'P(flow | state {legacy_off}) {lg["p_flow_given_off"]:.3f}')

    # Held-out normal days and the attack file
    events = lfp.runs(y == 1)
    val, union_held, union_te = [], 0, 0
    for _, r in kept.iterrows():
        a, f, sc = r.actuator, r.flow, r.flow_scale
        h = stats(held[a].values, held[f].values, 1, sc)
        thr = max(np.percentile(residual(der[a].values, der[f].values, sc), 99.8), 0.1)
        far_held = float((residual(held[a].values, held[f].values, sc) > thr).mean())
        pred = (residual(te[a].values, te[f].values, sc) > thr).astype(int)
        union_held = np.maximum(union_held, residual(held[a].values, held[f].values, sc) > thr)
        union_te = np.maximum(union_te, pred)
        s, _ = evaluate(y, pred)
        ch, ch95 = lfp.chance_events(pred, y, n=200)
        present = te[f].values >= PRESENT * sc
        off_te = te[a].values == 1
        val.append({'actuator': a, 'flow': f, 'submitted': (a, f) in SUBMITTED,
                    'derive_p_flow_given_off': r.p_flow_given_off, 'derive_p_flow_given_on': r.p_flow_given_on,
                    'heldout_n_off': h['n_off'], 'heldout_p_flow_given_off': h['p_flow_given_off'],
                    'heldout_p_flow_given_on': h['p_flow_given_on'], 'threshold': thr,
                    'heldout_FAR': far_held,
                    'test_viol_normal': float(present[off_te & (y == 0)].mean()) if (off_te & (y == 0)).any() else np.nan,
                    'test_viol_attack': float(present[off_te & (y == 1)].mean()) if (off_te & (y == 1)).any() else np.nan,
                    'test_FAR': s['false_alarm'], 'attacks_caught': s['n_detected'], 'n_attacks': s['n_events'],
                    'chance': ch, 'chance_p95': ch95})
    s, _ = evaluate(y, union_te.astype(int))
    ch, ch95 = lfp.chance_events(union_te.astype(int), y, n=200)
    val.append({'actuator': 'ALL (union)', 'flow': '', 'heldout_FAR': float(union_held.mean()),
                'test_FAR': s['false_alarm'], 'attacks_caught': s['n_detected'], 'n_attacks': s['n_events'],
                'chance': ch, 'chance_p95': ch95})
    val = pd.DataFrame(val)
    pd.set_option('display.width', 250)
    print('\nvalidation (held-out = last 4 normal days; test = attack file, labels for reporting only):')
    print(val.round(4).to_string(index=False))

    pd.DataFrame({'y_true': y, 'y_pred': union_te.astype(int)}).to_csv(args.export, index=False)
    if args.combine:
        other = pd.read_csv(args.combine)['y_pred'].values
        print(f'\ncombined with {args.combine}:')
        for tag, p in ((args.combine, other), ('interlocks', union_te), ('union', np.maximum(other, union_te))):
            s, _ = evaluate(y, p.astype(int))
            ch, ch95 = lfp.chance_events(p.astype(int), y, n=200)
            print(f"  {tag:28s} attacks {s['n_detected']}/{s['n_events']} (chance {ch:.1f}, 95th pct {ch95:.0f}) | "
                  f"F1 {s['f1']:.2%} | FAR {s['false_alarm']:.2%} | false-alarm episodes {s['false_alarm_events']}")

    cand.to_csv('wadi_interlock_candidates.csv', index=False)
    val.to_csv('wadi_interlock_validation.csv', index=False)
    json.dump({'_comment': 'Derived by wadi_interlock_validation.py from WADI_14days.csv days 1-10, raw units; '
                           'off = state 1 (pump off / valve closed); see the script for the rule.',
               'actuator_flow_interlock': [{'actuator': a, 'flow': f, 'flow_scale': scale[f]}
                                           for a, f in zip(kept.actuator, kept.flow)]},
              open('Datasets/wadi_interlocks_raw.json', 'w'), indent=2)
    print('[SAVED] wadi_interlock_candidates.csv, wadi_interlock_validation.csv, Datasets/wadi_interlocks_raw.json')
