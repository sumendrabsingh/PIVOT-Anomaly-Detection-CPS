"""How much does calibrating on test labels inflate PIVOT's numbers? (evaluation paper, claim C5)

The same frozen lambda = 0 stage models, the same scores (run_generalized_subsystems._compute_subsystem_scores, the
paper's code path), four protocols:
  label-free      errors standardized on training errors, per-stage P99.8 of training scores, OR (default protocol)
  oracle          errors standardized on test rows labelled normal, per-stage P99.8 of those rows, OR (v1/v2 paper)
  oracle + F1     as oracle, one per-stage quantile chosen to maximize test F1 (the paper's f1_opt threshold mode)
  oracle + F1 +600s  SWaT only: as oracle + F1, scored on labels extended 600 s past each attack
                  (create_corrected_dataset.py; the submitted paper's Table IV setting)
Plants: SWaT (six stages, analyzers split out: Oracles/swat_phys_none_seed<k>), WADI A2 (Oracles/wadi_a2_none_seed<k>),
HAI 23.05 (Oracles/hai2305_none_seed<k>); seeds 1-5. Attacks caught are reported with the chance level.

Run from the repo root:  python leakage_effect.py   -> leakage_effect.csv
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '.')
import label_free as lfp
import run_generalized_subsystems as gen
from evaluate_events import evaluate, to_binary

SEEDS = [1, 2, 3, 4, 5]
QS = np.r_[np.linspace(0.90, 0.99, 10), np.linspace(0.991, 0.999, 9), 0.9995, 0.9999]
PLANTS = {'SWaT': ('Datasets/swat_train_preprocessed.csv', 'Datasets/swat_test_preprocessed.csv', None,
                   'Oracles/swat_phys_none_seed{}'),
          'WADI': ('Datasets/wadi_train_preprocessed.csv', 'Datasets/wadi_test_preprocessed.csv',
                   'Datasets/wadi_subsystems.json', 'Oracles/wadi_a2_none_seed{}'),
          'HAI': ('Datasets/hai_train_preprocessed.csv', 'Datasets/hai_test_preprocessed.csv',
                  'Datasets/hai2305_subsystems.json', 'Oracles/hai2305_none_seed{}')}


def extend(y, n=600):
    """Each attack extended n samples past its official end (create_corrected_dataset.py, RECOVERY_WINDOW)."""
    out = y.copy()
    for s, length in zip(*lfp.runs(y == 1)):
        out[s + length:s + length + n] = 1
    return out


def orv(scores, refs, q):
    return np.max([s > np.quantile(r, q) for s, r in zip(scores, refs)], axis=0).astype(int)


def row(plant, seed, protocol, y, p, q=None):
    s, _ = evaluate(y, p)
    ch, _ = lfp.chance_events(p, y, n=100)
    return {'plant': plant, 'seed': seed, 'protocol': protocol, 'q': q, 'F1': s['f1'], 'Prec': s['precision'],
            'Rec': s['recall'], 'FAR': s['false_alarm'], 'caught': s['n_detected'], 'n_events': s['n_events'],
            'chance': ch, 'beyond': s['n_detected'] - ch, 'FA_episodes': s['false_alarm_events']}


def main():
    rows = []
    for plant, (tr_p, te_p, js, mdir) in PLANTS.items():
        tr = pd.read_csv(tr_p); te = pd.read_csv(te_p)
        tr.columns, te.columns = tr.columns.str.strip(), te.columns.str.strip()
        y = to_binary(te['Label' if 'Label' in te.columns else 'Normal/Attack']).astype(int)
        te_lf, _ = lfp.fault_rule(tr, te)
        if js is None:
            import decomposition_ablation_label_free as dab
            subs = dab.partition(6, 'split')
        else:
            subs = json.load(open(js))
        for seed in SEEDS:
            sc_lf, tr_sc = gen._compute_subsystem_scores(subs, te_lf, y, mdir.format(seed), 'train', tr)
            sc_or, _ = gen._compute_subsystem_scores(subs, te, y, mdir.format(seed), 'oracle')
            names = list(sc_lf)
            rows.append(row(plant, seed, 'label-free', y, orv([sc_lf[n] for n in names],
                                                              [tr_sc[n] for n in names], 0.998), 0.998))
            ref = [sc_or[n][y == 0] for n in names]
            rows.append(row(plant, seed, 'oracle', y, orv([sc_or[n] for n in names], ref, 0.998), 0.998))
            labels = [('oracle + F1', y)] + ([('oracle + F1 +600s', extend(y))] if plant == 'SWaT' else [])
            for tag, yy in labels:
                best = max(QS, key=lambda q: evaluate(yy, orv([sc_or[n] for n in names], ref, q))[0]['f1'])
                rows.append(row(plant, seed, tag, yy, orv([sc_or[n] for n in names], ref, best), best))
            print(f'  {plant} seed {seed} done', flush=True)
    out = pd.DataFrame(rows)
    out.to_csv('leakage_effect.csv', index=False)
    g = out.groupby(['plant', 'protocol'], sort=False).agg(
        F1=('F1', 'mean'), FAR=('FAR', 'mean'), caught=('caught', 'mean'), n=('n_events', 'first'),
        chance=('chance', 'mean'), beyond=('beyond', 'mean'), FA_episodes=('FA_episodes', 'mean'))
    g[['F1', 'FAR']] *= 100
    pd.set_option('display.width', 200)
    print(g.round(2).to_string())
    print('[SAVED] leakage_effect.csv')


if __name__ == '__main__':
    main()
