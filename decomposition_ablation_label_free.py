"""Does subsystem decomposition beat a monolithic model? Controlled, label-free, seeds 1-5 (SWaT).

The v1 claim (subsystem F1 83.35% vs monolithic 78.73%) compared models trained with different code,
data and normalization, all calibrated on test labels. Here the only difference is the partition:

  K=6   the six-stage subsystem array
  K=1   one "Global" autoencoder over the same features, trained with the same function
        (run_generalized_subsystems.train_subsystem), data and seed
Both are trained without a physics term (lambda = 0, PHYSICS below), the paper's headline setting since
2026-10-06: Oracles/swat_phys_none_seed<k> (analyzers split) / swat_none_seed<k> (kept) and
Oracles/swat_mono_noAIT_none_seed<k> / swat_mono_none_seed<k>. The earlier pair trained with the
mis-specified legacy term (Oracles/swat_seed<k>[_noAIT], swat_mono[_noAIT]_seed<k>) is kept for reference.

Evaluation is identical for both: label-free calibration on training data, decision 'orvote'
(per-model P99.8) or 'dual' (P99.9 / P99, W_dur = 50, causal), optionally OR the analyzer arm.
Attacks caught are reported against the chance level. Writes decomposition_ablation_label_free_swat.csv.

Run from the repo root:  python decomposition_ablation_label_free.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '.')
import label_free as lfp
import run_generalized_subsystems as gen
from evaluate_events import evaluate, to_binary

SEEDS = [1, 2, 3, 4, 5]
PHYSICS = 'none'
TRAIN, TEST = 'Datasets/swat_train_preprocessed.csv', 'Datasets/swat_test_preprocessed.csv'
STAGES = json.load(open('Datasets/swat_subsystems.json'))
ALL = [f for feats in STAGES.values() for f in feats]


def partition(k, analyzers):
    subs = STAGES if k == 6 else {'Global': ALL}
    if analyzers == 'split':
        subs = {n: [f for f in feats if not lfp.is_analyzer(f)] for n, feats in subs.items()}
    return {n: feats for n, feats in subs.items() if feats}


def model_dir(k, analyzers, seed):
    if k == 6:
        return f'Oracles/swat_phys_none_seed{seed}' if analyzers == 'split' else f'Oracles/swat_none_seed{seed}'
    return 'Oracles/swat_mono' + ('_noAIT' if analyzers == 'split' else '') + f'_none_seed{seed}'


def ensure_trained(subs, out_dir, seed):
    for name, feats in subs.items():
        if not os.path.exists(f'{out_dir}/subsystem_{name}.pth'):
            print(f'  training {out_dir}/{name} ({len(feats)} features, seed {seed}, physics {PHYSICS})')
            gen.train_subsystem(name, feats, TRAIN, out_dir, seed=seed, physics=PHYSICS)


def decide(scores, train_scores, decision):
    names = list(scores)
    if decision == 'orvote':
        pred = np.zeros(len(next(iter(scores.values()))), dtype=int)
        for n in names:
            pred = np.maximum(pred, (scores[n] > np.percentile(train_scores[n], 99.8)).astype(int))
        return pred
    agg = np.max([scores[n] for n in names], axis=0)
    cal = np.max([train_scores[n] for n in names], axis=0)
    return lfp.dual_threshold(agg, np.percentile(cal, 99.9), np.percentile(cal, 99.0), 50).astype(int)


if __name__ == '__main__':
    tr = pd.read_csv(TRAIN); tr.columns = tr.columns.str.strip()
    te = pd.read_csv(TEST); te.columns = te.columns.str.strip()
    y = to_binary(te['Normal/Attack'])
    te, _ = lfp.fault_rule(tr, te)
    arm = lfp.analyzer_arm_predict(tr, te, [c for c in te.columns if lfp.is_analyzer(c)])

    rows = []
    for seed in SEEDS:
        for k in (6, 1):
            for analyzers in ('keep', 'split'):
                subs, out_dir = partition(k, analyzers), model_dir(k, analyzers, seed)
                ensure_trained(subs, out_dir, seed)
                scores, train_scores = gen._compute_subsystem_scores(subs, te, y, out_dir, 'train', tr)
                for decision in ('orvote', 'dual'):
                    pred = decide(scores, train_scores, decision)
                    variants = [(pred, False)] + ([(np.maximum(pred, arm), True)] if analyzers == 'split' else [])
                    for p, with_arm in variants:
                        s, _ = evaluate(y, p)
                        cm, c95 = lfp.chance_events(p, y)
                        rows.append({'seed': seed, 'K': k, 'analyzers': analyzers, 'arm': with_arm,
                                     'decision': decision, 'F1': s['f1'], 'Prec': s['precision'],
                                     'Rec': s['recall'], 'FAR': s['false_alarm'], 'events': s['n_detected'],
                                     'chance': cm, 'chance_p95': c95, 'FA_episodes': s['false_alarm_events']})
        print(f'  seed {seed} done')

    out = pd.DataFrame(rows)
    out.to_csv('decomposition_ablation_label_free_swat.csv', index=False)
    out['excess'] = out['events'] - out['chance']
    g = out.groupby(['analyzers', 'arm', 'decision', 'K'], sort=False)
    summ = g.agg(F1=('F1', 'mean'), F1_sd=('F1', 'std'), FAR=('FAR', 'mean'), events=('events', 'mean'),
                 events_sd=('events', 'std'), chance=('chance', 'mean'), excess=('excess', 'mean'),
                 FA_ep=('FA_episodes', 'mean')).reset_index()
    for c in ('F1', 'F1_sd', 'FAR'):
        summ[c] = (summ[c] * 100).round(2)
    pd.set_option('display.width', 250)
    print(summ.round(2).to_string(index=False))
    print('[SAVED] decomposition_ablation_label_free_swat.csv')
