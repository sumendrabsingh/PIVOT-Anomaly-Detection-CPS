"""Was PIVOT's own decision rule tested? Per-stage thresholds + OR voting vs the monolithic model, all plants.

The paper's method trains one autoencoder per stage, freezes it, and at inference gives EACH stage its own threshold,
alarming if ANY stage exceeds its threshold (run_subsystem_experiments.evaluate_subsystems_or_voting; there each
threshold maximized that stage's F1 on the test labels). The H1 tests used one threshold on the max of standardized
stage scores instead. This check runs the OR-voting rule itself, fairly:

  OR, stage scoring       per stage: max squared error over its sensors, standardized on training, smoothed W = 25;
                          thresholds = each stage's own (1 - q) quantile on test normal rows, one q for all stages,
                          bisected so the OR reaches the target FAR (an equal FAR share per stage; a comparison, not
                          an operating point)
  OR, per-sensor scoring  the same with each stage's score = max over its sensors of the per-sensor-standardized error
  max-z (as in H1)        one threshold on the max over stages of the standardized stage score
  monolithic              one model on the same sensors, stage-style scoring and per-sensor scoring, one threshold
Metrics at matched FAR 0.1, 0.25, 0.5, 1, 2%: point-wise F1 (the paper's metric) and attacks beyond chance
(per-attack chance from circular shifts), seeds 1-5. Label-free (training-threshold) versions as a secondary table:
OR with each stage at P99.8 of its training score (run_generalized_subsystems --decision orvote) vs the monolithic
model at P99.8. NOT pre-registered: a check that the method actually proposed was the one tested.

Run from the repo root:  python orvote_check.py --dataset {swat,wadi,hai}
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '.')

TARGETS = [0.001, 0.0025, 0.005, 0.01, 0.02]
SEEDS = [1, 2, 3, 4, 5]
CFG = {'swat': ('Datasets/swat_train_preprocessed.csv', 'Datasets/swat_test_preprocessed.csv', 'swat',
                'Oracles/swat_phys_none_seed{}', 'Oracles/swat_mono_noAIT_none_seed{}'),
       'wadi': ('Datasets/wadi_train_preprocessed.csv', 'Datasets/wadi_test_preprocessed.csv',
                'Datasets/wadi_subsystems.json', 'Oracles/wadi_a2_none_seed{}', 'Oracles/wadi_a2_mono_none_seed{}'),
       'hai': ('Datasets/hai_train_preprocessed.csv', 'Datasets/hai_test_preprocessed.csv',
               'Datasets/hai2305_subsystems.json', 'Oracles/hai2305_none_seed{}', 'Oracles/hai2305_mono_none_seed{}')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=list(CFG), required=True)
    ap.add_argument('--gpu', default=None)
    args = ap.parse_args()
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import decomposition_mechanism as dm
    import label_free as lfp
    from evaluate_events import evaluate, to_binary
    ds = args.dataset
    tr_p, te_p, js, stage_dir, mono_dir = CFG[ds]
    if js == 'swat':
        import decomposition_ablation_label_free as dab
        stages = list(dab.partition(6, 'split'))
    else:
        stages = list(json.load(open(js)))
    tr = pd.read_csv(tr_p); te = pd.read_csv(te_p)
    tr.columns, te.columns = tr.columns.str.strip(), te.columns.str.strip()
    y = to_binary(te['Label' if 'Label' in te.columns else 'Normal/Attack']).astype(int)
    te, _ = lfp.fault_rule(tr, te)
    fill = tr.select_dtypes('number').mean()
    events, normal = lfp.runs(y == 1), y == 0
    sm = lambda v: lfp.smooth(v, 25)

    def stage_style(e_tr, e_te):          # max squared error, standardized on training, smoothed
        a, b = e_tr.max(1), e_te.max(1)
        return sm((a - a.mean()) / (a.std() + 1e-6)), sm((b - a.mean()) / (a.std() + 1e-6))

    def sensor_style(e_tr, e_te):         # per-sensor standardized, max, smoothed
        mu, sd = e_tr.mean(0), e_tr.std(0) + 1e-6
        return sm(((e_tr - mu) / sd).max(1)), sm(((e_te - mu) / sd).max(1))

    def or_at(scores, target):
        """Per-stage (1 - q) quantile thresholds on test normal rows, one q, OR reaching the target FAR."""
        qs = [np.sort(s[normal]) for s in scores]
        def pred(q):
            return np.max([s > np.quantile(n, 1 - q) for s, n in zip(scores, qs)], axis=0).astype(int)
        lo, hi = 0.0, target
        for _ in range(30):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if pred(mid)[normal].mean() <= target else (lo, mid)
        return pred(lo)

    rows, lf_rows = [], []
    for seed in SEEDS:
        st_tr = [dm.sensor_errors(stage_dir.format(seed), n, tr, fill)[1] for n in stages]
        st_te = [dm.sensor_errors(stage_dir.format(seed), n, te, fill)[1] for n in stages]
        _, m_tr = dm.sensor_errors(mono_dir.format(seed), 'Global', tr, fill)
        _, m_te = dm.sensor_errors(mono_dir.format(seed), 'Global', te, fill)
        stage_sc = [stage_style(a, b) for a, b in zip(st_tr, st_te)]
        sensor_sc = [sensor_style(a, b) for a, b in zip(st_tr, st_te)]
        single = {'max-z over stages (as in H1)': np.max([b for _, b in stage_sc], axis=0),
                  'monolithic, stage-style scoring': stage_style(m_tr, m_te)[1],
                  'monolithic, per-sensor scoring': sensor_style(m_tr, m_te)[1]}
        for target in TARGETS:
            preds = {'OR vote, stage scoring (PIVOT)': or_at([b for _, b in stage_sc], target),
                     'OR vote, per-sensor scoring': or_at([b for _, b in sensor_sc], target)}
            for name, s in single.items():
                preds[name] = (s > np.quantile(s[normal], 1 - target)).astype(int)
            for name, p in preds.items():
                m, _ = evaluate(y, p)
                caught, pc = dm.per_event(p, events, np.random.default_rng(seed))
                rows.append({'seed': seed, 'target_FAR': target, 'detector': name, 'F1': m['f1'],
                             'FAR': p[normal].mean(), 'beyond': float(caught.sum() - pc.sum())})
        # label-free: each stage at P99.8 of its own training score, OR; monolithic at P99.8
        for name, sc in (('OR vote, stage scoring (PIVOT)', stage_sc), ('OR vote, per-sensor scoring', sensor_sc)):
            p = np.max([b > np.percentile(a, 99.8) for a, b in sc], axis=0).astype(int)
            m, _ = evaluate(y, p); ch, _ = lfp.chance_events(p, y, n=100)
            lf_rows.append({'seed': seed, 'detector': name, 'F1': m['f1'], 'FAR': m['false_alarm'],
                            'beyond': m['n_detected'] - ch})
        for name, (a, b) in (('monolithic, stage-style scoring', stage_style(m_tr, m_te)),
                             ('monolithic, per-sensor scoring', sensor_style(m_tr, m_te))):
            p = (b > np.percentile(a, 99.8)).astype(int)
            m, _ = evaluate(y, p); ch, _ = lfp.chance_events(p, y, n=100)
            lf_rows.append({'seed': seed, 'detector': name, 'F1': m['f1'], 'FAR': m['false_alarm'],
                            'beyond': m['n_detected'] - ch})
        print(f'  seed {seed} done', flush=True)

    out = pd.DataFrame(rows); out.to_csv(f'orvote_check_{ds}.csv', index=False)
    pd.set_option('display.width', 250)
    print(f'\n[{ds}] {len(events[0])} attacks. Matched FAR, mean of seeds 1-5: point-wise F1 (%)')
    f1 = out.pivot_table(index='detector', columns='target_FAR', values='F1', aggfunc='mean') * 100
    f1.columns = [f'{c * 100:g}%' for c in f1.columns]
    print(f1.round(1).to_string())
    print(f'\n[{ds}] attacks beyond chance')
    bc = out.pivot_table(index='detector', columns='target_FAR', values='beyond', aggfunc='mean')
    bc.columns = [f'{c * 100:g}%' for c in bc.columns]
    print(bc.round(1).to_string())
    lf = pd.DataFrame(lf_rows); lf.to_csv(f'orvote_check_{ds}_labelfree.csv', index=False)
    g = lf.groupby('detector').agg(F1=('F1', 'mean'), F1_sd=('F1', 'std'), FAR=('FAR', 'mean'), beyond=('beyond', 'mean'))
    g[['F1', 'F1_sd', 'FAR']] = g[['F1', 'F1_sd', 'FAR']] * 100
    print(f'\n[{ds}] label-free (training P99.8 thresholds), mean of seeds 1-5')
    print(g.round(2).to_string())


if __name__ == '__main__':
    main()
