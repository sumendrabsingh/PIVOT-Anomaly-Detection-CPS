"""Pre-registered test H2 (preregistration/H2_physics_or.md, 2026-10-07): does OR-ing physical-invariant detectors
into PIVOT's vote catch more attacks beyond chance at the same total FAR?

Frozen design
  base        the stage array with per-sensor scoring (decomposition_mechanism.sensor_score), seeds 1-5:
              SWaT Oracles/swat_phys_none_seed<k> (six stages, analyzers split out), WADI A2
              Oracles/wadi_a2_none_seed<k>, HAI 23.05 Oracles/hai2305_none_seed<k>
  invariants  SWaT Datasets/swat_invariants.json (3 mass balances, 5 interlocks); WADI A2 interlocks re-derived on
              A2 training data with wadi_interlock_validation.py's rule (first 10 days derive); HAI
              physics_residual_detector.auto_derive_invariants on training data
  residuals   RAW units. Mass balance |dL - (a Fin + b Fout + c)|, fitted and standardized on training data.
              Interlock: off x max(0, flow - training median of flow while off) / training SD of flow, where off =
              state 1 for discrete actuators (SWaT, WADI: valve closed / pump off) and the training minimum for
              continuous ones (HAI, as auto_derive_invariants defines it). Fixes the old off x |flow| residual,
              which on standardized flows scored zero flow high and real violations low.
  rule        each residual smoothed W = 25, dual threshold P99.9 / P99 of its training residual, W_dur = 50,
              causal; the physics arm = OR over invariants (fixed, label-free)
Primary test  at total FAR 0.5%, 1%, 2%: base alone (one threshold at that FAR, set on test normal rows) vs base OR
              physics (base threshold raised until the union reaches the FAR; a level the physics arm alone exceeds
              counts as not supported). Union minus base in attacks beyond chance, seed-averaged, 95% CI
              bootstrapping attacks; H2 supported on a plant if the CI is above zero at >= 2 of the 3 levels.
Secondary     label-free operating point (base dual threshold from training, with and without the physics arm),
              early detection, the physics arm alone.

Run from the repo root:  python physics_or_test.py --dataset {swat,wadi,hai}
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from numpy.linalg import lstsq

sys.path.insert(0, '.')

SEEDS = [1, 2, 3, 4, 5]
TARGETS = [0.005, 0.01, 0.02]
W, W_DUR, N_BOOT = 25, 50, 10000
A2 = 'SWaT_Orig_DataSets/wadi/WADI.A2_19 Nov 2019'


# ---------------------------------------------------------------- raw data and invariants
def raw_data(ds):
    """Raw-unit train/test frames whose rows line up with the preprocessed files."""
    if ds == 'swat':
        tr = pd.read_csv('SWaT_Orig_DataSets/SWaT_Dataset_Normal_v1.csv')
        te = pd.read_csv('SWaT_Orig_DataSets/Original Dataset Backup/SWaT_Dataset_Attack_v0.csv')  # official labels
    elif ds == 'wadi':
        tr = pd.read_csv(f'{A2}/WADI_14days_new.csv')
        te = pd.read_csv(f'{A2}/WADI_attackdataLABLE.csv', header=1)
        te.columns = te.columns.str.strip()
        te = te.dropna(subset=['Row']).reset_index(drop=True)
    else:
        tr = pd.read_csv('Datasets/hai_train_preprocessed.csv')   # HAI files are in raw units
        te = pd.read_csv('Datasets/hai_test_preprocessed.csv')
    tr.columns, te.columns = tr.columns.str.strip(), te.columns.str.strip()
    num = lambda d: d.select_dtypes('number').ffill().fillna(0)
    return num(tr), num(te), (tr['Row'].values if ds == 'wadi' else None)


def invariants(ds, tr, rows):
    if ds == 'swat':
        return json.load(open('Datasets/swat_invariants.json'))
    if ds == 'hai':
        import physics_residual_detector as phys
        return phys.auto_derive_invariants(tr)
    import wadi_interlock_validation as wiv            # A2 re-derivation with the frozen rule
    der = tr[rows <= wiv.DERIVE_DAYS * wiv.DAY]        # Row 1 = 2017-09-25 18:00:00, one row per second
    acts = [c for c in tr.columns if c.endswith('_STATUS') and set(der[c].unique()) >= {1, 2}]
    flows = [c for c in tr.columns if 'FIT' in c and der[c].quantile(0.95) > 0]
    kept = []
    for a in acts:
        for f in flows:
            scale = float(der[f].quantile(0.95))
            s = wiv.stats(der[a].values, der[f].values, 1, scale)
            if (s['n_off'] >= wiv.MIN_OFF and s['off_frac'] < wiv.MAX_OFF_FRAC
                    and s['p_flow_given_off'] <= wiv.OFF_MAX and s['p_flow_given_on'] >= wiv.ON_MIN):
                kept.append({'actuator': a, 'flow': f})
    return {'mass_balance': [], 'actuator_flow_interlock': kept}


def off_mask(tr_col, te_col):
    """Discrete actuators (values within {0, 1, 2}): off/closed = state 1. Continuous: the training minimum."""
    if set(np.unique(tr_col)) <= {0, 1, 2}:
        return tr_col == 1, te_col == 1
    lo = tr_col.min()
    return tr_col <= lo + 1e-6, te_col <= lo + 1e-6


def residuals(tr, te, inv):
    out = {}
    for mb in inv.get('mass_balance', []):
        lit, fin, fout = mb['tank'], mb['inflow'], mb['outflow']
        if not all(c in tr.columns and c in te.columns for c in (lit, fin, fout)):
            continue
        d_tr, d_te = np.diff(tr[lit].values, prepend=tr[lit].values[0]), np.diff(te[lit].values, prepend=te[lit].values[0])
        A_tr = np.c_[tr[fin].values, tr[fout].values, np.ones(len(tr))]
        A_te = np.c_[te[fin].values, te[fout].values, np.ones(len(te))]
        coef, *_ = lstsq(A_tr, d_tr, rcond=None)
        r_tr, r_te = np.abs(d_tr - A_tr @ coef), np.abs(d_te - A_te @ coef)
        mu, sd = r_tr.mean(), r_tr.std() + 1e-9
        out[f'mb_{lit}'] = ((r_tr - mu) / sd, (r_te - mu) / sd)
    for il in inv.get('actuator_flow_interlock', []):
        a, f = il['actuator'], il['flow']
        if not all(c in tr.columns and c in te.columns for c in (a, f)):
            continue
        off_tr, off_te = off_mask(tr[a].values, te[a].values)
        if off_tr.sum() < 50:
            continue
        m_off, sd = np.median(tr[f].values[off_tr]), tr[f].values.std() + 1e-9
        r = lambda d, off: off * np.maximum(0.0, d[f].values - m_off) / sd
        out[f'il_{a}_{f}'] = (r(tr, off_tr), r(te, off_te))
    return out


def physics_arm(res):
    import label_free as lfp
    pred = None
    for r_tr, r_te in res.values():
        s_tr, s_te = lfp.smooth(r_tr, W), lfp.smooth(r_te, W)
        p = lfp.dual_threshold(s_te, np.percentile(s_tr, 99.9), np.percentile(s_tr, 99.0), W_DUR).astype(int)
        pred = p if pred is None else np.maximum(pred, p)
    return pred


# ---------------------------------------------------------------- test
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['swat', 'wadi', 'hai'], required=True)
    ap.add_argument('--gpu', default=None)
    args = ap.parse_args()
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import decomposition_mechanism as dm
    import label_free as lfp
    from baselines_simple_label_free import early_events
    from evaluate_events import evaluate, to_binary
    ds = args.dataset

    # physics arm (deterministic, label-free)
    rtr, rte, rows = raw_data(ds)
    inv = invariants(ds, rtr, rows)
    res = residuals(rtr, rte, inv)
    phys = physics_arm(res)
    print(f'[{ds}] invariants: {len(inv.get("mass_balance", []))} mass balances, '
          f'{len(inv.get("actuator_flow_interlock", []))} interlocks; residuals used: {list(res)}')
    json.dump({'_comment': 'H2 physics arm invariants (physics_or_test.py)', **inv},
              open(f'physics_or_test_{ds}_invariants.json', 'w'), indent=1)
    del rtr, rte

    # base: stage array, per-sensor scoring
    paths = {'swat': ('Datasets/swat_train_preprocessed.csv', 'Datasets/swat_test_preprocessed.csv'),
             'wadi': ('Datasets/wadi_train_preprocessed.csv', 'Datasets/wadi_test_preprocessed.csv'),
             'hai': ('Datasets/hai_train_preprocessed.csv', 'Datasets/hai_test_preprocessed.csv')}[ds]
    tr = pd.read_csv(paths[0]); tr.columns = tr.columns.str.strip()
    te = pd.read_csv(paths[1]); te.columns = te.columns.str.strip()
    y = to_binary(te['Label' if 'Label' in te.columns else 'Normal/Attack'])
    assert len(y) == len(phys), (len(y), len(phys))
    te, _ = lfp.fault_rule(tr, te)
    fill = tr.select_dtypes('number').mean()
    if ds == 'swat':
        import decomposition_ablation_label_free as dab
        stages, stage_dir = list(dab.partition(6, 'split')), 'Oracles/swat_phys_none_seed{}'
    else:
        stages = list(json.load(open({'wadi': 'Datasets/wadi_subsystems.json',
                                      'hai': 'Datasets/hai2305_subsystems.json'}[ds])))
        stage_dir = {'wadi': 'Oracles/wadi_a2_none_seed{}', 'hai': 'Oracles/hai2305_none_seed{}'}[ds]
    events = lfp.runs(y == 1)
    normal = y == 0

    def summary(tag, p, seed=None):
        s, _ = evaluate(y, p)
        ch, _ = lfp.chance_events(p, y, n=100)
        early, early_ch = early_events(p, y)
        return {'detector': tag, 'seed': seed, 'F1': s['f1'], 'FAR': s['false_alarm'], 'caught': s['n_detected'],
                'chance': ch, 'beyond': s['n_detected'] - ch, 'early_beyond': early - early_ch,
                'FA_episodes': s['false_alarm_events']}

    per_event, lf_rows = [], [summary('physics arm alone', phys)]
    phys_far = phys[normal].mean()
    for seed in SEEDS:
        e_tr, e_te = [], []
        for name in stages:
            _, a = dm.sensor_errors(stage_dir.format(seed), name, tr, fill)
            _, b = dm.sensor_errors(stage_dir.format(seed), name, te, fill)
            e_tr.append(a); e_te.append(b)
        e_tr, e_te = np.hstack(e_tr), np.hstack(e_te)
        s_tr, s_te = dm.sensor_score(e_tr, e_tr), dm.sensor_score(e_tr, e_te)
        sn = np.sort(s_te[normal])
        for target in TARGETS:
            preds = {'base': (s_te > np.quantile(sn, 1 - target)).astype(int)}
            if phys_far < target:
                lo, hi = np.quantile(sn, 1 - target), sn[-1] + 1e-9   # union FAR >= target at lo, < target at hi
                for _ in range(50):
                    mid = (lo + hi) / 2
                    far = ((s_te > mid) | (phys == 1))[normal].mean()
                    lo, hi = (lo, mid) if far <= target else (mid, hi)
                preds['union'] = np.maximum((s_te > hi).astype(int), phys)
            for det, p in preds.items():
                caught, pc = dm.per_event(p, events, np.random.default_rng(seed))
                for i in range(len(caught)):
                    per_event.append({'seed': seed, 'target_FAR': target, 'detector': det, 'event': i + 1,
                                      'caught': caught[i], 'p_chance': pc[i], 'FAR': p[normal].mean()})
        base_lf = lfp.dual_threshold(s_te, np.percentile(s_tr, 99.9), np.percentile(s_tr, 99.0), W_DUR).astype(int)
        lf_rows += [summary('PIVOT (stages, per-sensor)', base_lf, seed),
                    summary('PIVOT OR physics', np.maximum(base_lf, phys), seed)]
        print(f'  seed {seed} done', flush=True)

    out = pd.DataFrame(per_event)
    out.to_csv(f'physics_or_test_{ds}.csv', index=False)
    out['excess'] = out.caught - out.p_chance
    rng = np.random.default_rng(0)
    res_rows = []
    for target in TARGETS:
        sub = out[out.target_FAR == target]
        if 'union' not in set(sub.detector):
            res_rows.append({'target_FAR': target, 'note': f'physics arm alone has FAR {phys_far:.2%} >= target'})
            continue
        ev = sub.groupby(['detector', 'event'])['excess'].mean().unstack(0)
        d = (ev['union'] - ev['base']).values
        boot = d[rng.integers(0, len(d), (N_BOOT, len(d)))].sum(1)
        ps = sub.groupby(['detector', 'seed'])['excess'].sum().unstack(0)
        res_rows.append({'target_FAR': target, 'base_beyond': ps['base'].mean(), 'union_beyond': ps['union'].mean(),
                         'diff': d.sum(), 'CI_low': np.percentile(boot, 2.5), 'CI_high': np.percentile(boot, 97.5),
                         'seeds_union_better': int((ps['union'] > ps['base']).sum()), 'note': ''})
    res_df = pd.DataFrame(res_rows)
    res_df.to_csv(f'physics_or_test_{ds}_summary.csv', index=False)
    pd.set_option('display.width', 250)
    print(f'\n[{ds}] {len(events[0])} attacks; physics arm alone FAR {phys_far:.2%}. Matched total FAR, attacks beyond chance:')
    print(res_df.round(2).to_string(index=False))
    n_pass = int((res_df.get('CI_low', pd.Series(dtype=float)).fillna(-1) > 0).sum())
    print(f'\nH2 on {ds}: CI above zero at {n_pass} of {len(TARGETS)} levels -> {"SUPPORTED" if n_pass >= 2 else "NOT SUPPORTED"}')
    lf = pd.DataFrame(lf_rows)
    lf.to_csv(f'physics_or_test_{ds}_labelfree.csv', index=False)
    g = lf.groupby('detector', sort=False).agg(F1=('F1', 'mean'), FAR=('FAR', 'mean'), caught=('caught', 'mean'),
                                               chance=('chance', 'mean'), beyond=('beyond', 'mean'),
                                               early_beyond=('early_beyond', 'mean'), FA_episodes=('FA_episodes', 'mean'))
    g['F1'] = (g.F1 * 100).round(1); g['FAR'] = (g.FAR * 100).round(2)
    print('\nSecondary: label-free operating point (mean of seeds 1-5 where seeded)')
    print(g.round(2).to_string())


if __name__ == '__main__':
    main()
