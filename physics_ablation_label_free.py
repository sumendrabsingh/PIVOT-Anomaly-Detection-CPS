"""Physics-loss ablation on SWaT with the corrected physics (seeds 1-5, label-free).

Arms, identical except for the training physics term (run_generalized_subsystems.train_subsystem):
  none       lambda = 0
  nonneg     non-negativity of flows/levels/pressures in PHYSICAL units
  interlock  nonneg + closed-valve / stopped-pump -> zero-flow rules, gated on observed states
  legacy     the mis-specified pre-2026-10-05 term (penalizes below-mean z-scores; reference only;
             reuses Oracles/swat_seed<k>_noAIT)
All arms are six-subsystem arrays with analyzers split out. Evaluation is label-free (training
calibration): dual threshold alone and with the analyzer arm, plus a matched-FAR comparison (one
threshold on the aggregate score at 0.5% and 1% FAR; choosing the point by FAR uses labels, so it is a
comparison, not an operating point). Attacks caught are reported against the chance level.

Run from the repo root:  python physics_ablation_label_free.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '.')
import decomposition_ablation_label_free as dab
import label_free as lfp
import matched_far_decomposition as mfd
import run_generalized_subsystems as gen
from evaluate_events import evaluate, to_binary

ARMS = ['none', 'nonneg', 'interlock', 'legacy']


def arm_dir(arm, seed):
    return f'Oracles/swat_seed{seed}_noAIT' if arm == 'legacy' else f'Oracles/swat_phys_{arm}_seed{seed}'


if __name__ == '__main__':
    tr = pd.read_csv(dab.TRAIN); tr.columns = tr.columns.str.strip()
    te = pd.read_csv(dab.TEST); te.columns = te.columns.str.strip()
    y = to_binary(te['Normal/Attack'])
    te, _ = lfp.fault_rule(tr, te)
    analyzer = lfp.analyzer_arm_predict(tr, te, [c for c in te.columns if lfp.is_analyzer(c)])
    events = lfp.runs(y == 1)
    subs = dab.partition(6, 'split')

    rows = []
    for seed in dab.SEEDS:
        for arm in ARMS:
            out_dir = arm_dir(arm, seed)
            for name, feats in subs.items():
                if not os.path.exists(f'{out_dir}/subsystem_{name}.pth'):
                    gen.train_subsystem(name, feats, dab.TRAIN, out_dir, seed=seed, physics=arm)
            scores, train_scores = gen._compute_subsystem_scores(subs, te, y, out_dir, 'train', tr)
            dual = dab.decide(scores, train_scores, 'dual')
            for tag, p in (('dual', dual), ('dual + analyzer arm', np.maximum(dual, analyzer))):
                s, _ = evaluate(y, p)
                cm, _ = lfp.chance_events(p, y)
                rows.append({'seed': seed, 'physics': arm, 'protocol': tag, 'F1': s['f1'], 'FAR': s['false_alarm'],
                             'events': s['n_detected'], 'chance': cm, 'FA_episodes': s['false_alarm_events']})
            agg = np.max(list(scores.values()), axis=0)
            for target in (0.005, 0.01):
                r = mfd.curve_point(agg, y, target, events)
                rows.append({'seed': seed, 'physics': arm, 'protocol': f'matched FAR {target:.1%}', 'F1': r['F1'],
                             'FAR': r['FAR'], 'events': r['events'], 'chance': r['chance'],
                             'FA_episodes': r['FA_episodes']})
            print(f'  seed {seed} {arm} done')

    out = pd.DataFrame(rows)
    out['excess'] = out['events'] - out['chance']
    out.to_csv('physics_ablation_label_free_swat.csv', index=False)
    summ = out.groupby(['protocol', 'physics'], sort=False).agg(
        F1=('F1', 'mean'), F1_sd=('F1', 'std'), FAR=('FAR', 'mean'), events=('events', 'mean'),
        chance=('chance', 'mean'), excess=('excess', 'mean'), excess_sd=('excess', 'std'),
        FA_ep=('FA_episodes', 'mean')).reset_index()
    for c in ('F1', 'F1_sd', 'FAR'):
        summ[c] = (summ[c] * 100).round(2)
    pd.set_option('display.width', 250)
    print(summ.round(2).to_string(index=False))
    # paired, per seed: each physics arm minus 'none'
    piv = out.pivot_table(index=['seed', 'protocol'], columns='physics', values=['F1', 'excess']).reset_index()
    print('\npaired difference vs none (mean over seeds; F1 in points):')
    for arm in ('nonneg', 'interlock', 'legacy'):
        d = piv.groupby('protocol').apply(lambda g: pd.Series({
            'dF1': 100 * (g[('F1', arm)] - g[('F1', 'none')]).mean(),
            'dF1_seeds_better': int(((g[('F1', arm)] - g[('F1', 'none')]) > 0).sum()),
            'dExcess': (g[('excess', arm)] - g[('excess', 'none')]).mean()}), include_groups=False)
        print(f'  {arm}:'); print(d.round(2).to_string())
    print('[SAVED] physics_ablation_label_free_swat.csv')
