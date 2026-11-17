"""Six subsystems (K=6) vs one monolithic model (K=1) at MATCHED false-alarm rates (SWaT, seeds 1-5).

decomposition_ablation_label_free.py found K=6 wins on point-wise F1 at a lower FAR while K=1 catches
more attacks at a higher FAR: different operating points. This compares them on one curve. Each
detector's aggregate score (max over subsystems for K=6, the single score for K=1; standardized on
training errors, smoothed W=25, analyzers split out) is thresholded by one knob swept over quantiles
of its own unlabeled test scores. At each target FAR we read attacks caught, the chance level (alarm
pattern shifted to random offsets) and false-alarm episodes. Choosing the point by FAR uses the test
labels, so this is a COMPARISON, not an operating point. Models from decomposition_ablation_label_free.py.

Run from the repo root:  python matched_far_decomposition.py
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '.')
import decomposition_ablation_label_free as dab
import label_free as lfp
import run_generalized_subsystems as gen
from evaluate_events import evaluate, to_binary

TARGETS = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.03]


def curve_point(agg, y, target, events):
    """Lowest-threshold point whose FAR does not exceed target."""
    normal = agg[y == 0]
    t = np.quantile(normal, 1 - target)          # equivalent to scanning thresholds by FAR
    pred = (agg > t).astype(int)
    hit = sum(pred[a:a + n].any() for a, n in zip(*events))
    far = pred[y == 0].mean()
    ep = sum(1 for a, n in zip(*lfp.runs(pred == 1)) if not y[a:a + n].any())
    chance, _ = lfp.chance_events(pred, y, n=100)
    s, _ = evaluate(y, pred)
    return {'FAR': far, 'events': hit, 'chance': chance, 'excess': hit - chance, 'FA_episodes': ep, 'F1': s['f1']}


if __name__ == '__main__':
    tr = pd.read_csv(dab.TRAIN); tr.columns = tr.columns.str.strip()
    te = pd.read_csv(dab.TEST); te.columns = te.columns.str.strip()
    y = to_binary(te['Normal/Attack'])
    te, _ = lfp.fault_rule(tr, te)
    events = lfp.runs(y == 1)

    rows = []
    for seed in dab.SEEDS:
        for k in (6, 1):
            subs, out_dir = dab.partition(k, 'split'), dab.model_dir(k, 'split', seed)
            scores, _ = gen._compute_subsystem_scores(subs, te, y, out_dir, 'train', tr)
            agg = np.max(list(scores.values()), axis=0)
            for target in TARGETS:
                rows.append({'seed': seed, 'K': k, 'target_FAR': target, **curve_point(agg, y, target, events)})
        print(f'  seed {seed} done')

    out = pd.DataFrame(rows)
    out.to_csv('matched_far_decomposition_swat.csv', index=False)
    summ = out.groupby(['target_FAR', 'K']).agg(
        FAR=('FAR', 'mean'), events=('events', 'mean'), events_sd=('events', 'std'), chance=('chance', 'mean'),
        excess=('excess', 'mean'), excess_sd=('excess', 'std'), FA_ep=('FA_episodes', 'mean'), F1=('F1', 'mean')).reset_index()
    summ['FAR'] = (summ['FAR'] * 100).round(2); summ['F1'] = (summ['F1'] * 100).round(2)
    summ['target_FAR'] = (summ['target_FAR'] * 100).round(2)
    pd.set_option('display.width', 250)
    print(summ.round(2).to_string(index=False))
    # paired difference per seed: K=6 minus K=1 excess over chance
    piv = out.pivot_table(index=['seed', 'target_FAR'], columns='K', values='excess').reset_index()
    piv['diff'] = piv[6] - piv[1]
    d = piv.groupby('target_FAR')['diff'].agg(['mean', 'std', lambda v: int((v > 0).sum())])
    d.columns = ['K6_minus_K1_excess', 'sd', 'seeds_K6_better']
    d.index = (d.index * 100).round(2)
    print(d.round(2).to_string())
    print('[SAVED] matched_far_decomposition_swat.csv')
