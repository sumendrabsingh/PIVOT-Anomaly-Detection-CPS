"""Develop an on-site calibration for PIVOT's per-stage thresholds, on the DEVELOPMENT plants only (SWaT, WADI A2,
HAI 23.05). The held-out HAI releases are not touched here; the chosen method is then pre-registered and applied
to them once (orvote_deploy.py --calibration ...).

H4 (orvote_deploy.py) took each stage's thresholds as quantiles of a 30-minute commissioning window; with a quarter
of a 0.1% budget per stage those quantiles are the window maximum, and the OR of four stages multiplied false alarms.

Calibration methods (applied identically to every score: each PIVOT stage, and the monolithic score)
  window   tau = quantile of the score over the commissioning window (the H4 rule)
  shift    tau = med_w + (Q_train(p) - med_train): the window supplies the level only, training the rest
  pooled   PIVOT only: each stage's score robust-standardized by its own window median / MAD, then ONE dual threshold
           on the max over stages, at the quantiles of the same statistic on training data (each stage standardized
           by its training median / MAD). Per-stage normalization keeps drift isolation; one joint threshold avoids
           OR-ing K separately estimated tails
  ls       location-scale recalibration: tau = med_w + MAD_w * c, where med_w / MAD_w are the score's median and
           median absolute deviation (x 1.4826) over the commissioning window, and c is the same quantile of the score
           robust-standardized on NORMAL TRAINING data, (s_train - med_train) / MAD_train. The window supplies the
           level and spread (which drift changes and 1,000-2,000 samples estimate well); the large training set
           supplies the tail in units of spread (which a short window cannot estimate)
Decision rule as in H4: dual threshold (spike / 50-sample persistence). Monolithic at the 0.999 / 0.99 quantiles,
PIVOT's K stages at 1 - 0.001/K and 1 - 0.01/K, OR-ed. Scoring as in H3. Seeds 1-5.
Windows: the longest round window before the first attack: SWaT 25 min, WADI 30 min, HAI 23.05 20 min.

Run from the repo root:  python calibration_dev.py --dataset {swat,wadi,hai}
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '.')

CFG = {'swat': ('Datasets/swat_train_preprocessed.csv', 'Datasets/swat_test_preprocessed.csv', 'swat',
                'Oracles/swat_phys_none_seed{}', 'Oracles/swat_mono_noAIT_none_seed{}', 1500),
       'wadi': ('Datasets/wadi_train_preprocessed.csv', 'Datasets/wadi_test_preprocessed.csv', 'Datasets/wadi_subsystems.json',
                'Oracles/wadi_a2_none_seed{}', 'Oracles/wadi_a2_mono_none_seed{}', 1800),
       'hai': ('Datasets/hai_train_preprocessed.csv', 'Datasets/hai_test_preprocessed.csv', 'Datasets/hai2305_subsystems.json',
               'Oracles/hai2305_none_seed{}', 'Oracles/hai2305_mono_none_seed{}', 1200)}
SEEDS = [1, 2, 3, 4, 5]
W_DUR = 50


def mad(v):
    return 1.4826 * np.median(np.abs(v - np.median(v))) + 1e-6


def thresholds(s_tr, s_te, window, probs, method):
    w = s_te[:window]
    if method == 'window':
        return [np.quantile(w, p) for p in probs]
    if method == 'shift':
        return [np.median(w) + np.quantile(s_tr, p) - np.median(s_tr) for p in probs]
    z_tr = (s_tr - np.median(s_tr)) / mad(s_tr)
    return [np.median(w) + mad(w) * np.quantile(z_tr, p) for p in probs]


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
    tr_p, te_p, js, stage_dir, mono_dir, window = CFG[args.dataset]
    if js == 'swat':
        import decomposition_ablation_label_free as dab
        stages = list(dab.partition(6, 'split'))
    else:
        stages = list(json.load(open(js)))
    K = len(stages)
    tr = pd.read_csv(tr_p); te = pd.read_csv(te_p)
    tr.columns, te.columns = tr.columns.str.strip(), te.columns.str.strip()
    y = to_binary(te['Label' if 'Label' in te.columns else 'Normal/Attack']).astype(int)
    te, _ = lfp.fault_rule(tr, te)
    fill = tr.select_dtypes('number').mean(); events = lfp.runs(y == 1)
    assert not y[:window].any()

    def score(a, b):
        mu, sd = a.mean(0), a.std(0) + 1e-6
        return lfp.smooth(((a - mu) / sd).max(1), 25), lfp.smooth(((b - mu) / sd).max(1), 25)

    rows = []
    for seed in SEEDS:
        st = [score(dm.sensor_errors(stage_dir.format(seed), n, tr, fill)[1],
                    dm.sensor_errors(stage_dir.format(seed), n, te, fill)[1]) for n in stages]
        m_tr, m_te = score(dm.sensor_errors(mono_dir.format(seed), 'Global', tr, fill)[1],
                           dm.sensor_errors(mono_dir.format(seed), 'Global', te, fill)[1])
        for method in ('window', 'shift', 'ls', 'pooled'):
            if method == 'pooled':
                z_tr = np.max([(a - np.median(a)) / mad(a) for a, _ in st], axis=0)
                z_te = np.max([(b - np.median(b[:window])) / mad(b[:window]) for _, b in st], axis=0)
                preds = {'pivot': lfp.dual_threshold(z_te, np.quantile(z_tr, 0.999), np.quantile(z_tr, 0.99), W_DUR).astype(int)}
            else:
                preds = {'pivot': np.max([lfp.dual_threshold(b, *thresholds(a, b, window, [1 - 0.001 / K, 1 - 0.01 / K], method), W_DUR)
                                          for a, b in st], axis=0).astype(int),
                         'mono': lfp.dual_threshold(m_te, *thresholds(m_tr, m_te, window, [0.999, 0.99], method), W_DUR).astype(int)}
            for det, p in preds.items():
                s, _ = evaluate(y, p)
                caught, pc = dm.per_event(p, events, np.random.default_rng(seed))
                rows.append({'seed': seed, 'method': method, 'detector': det, 'F1': s['f1'], 'FAR': s['false_alarm'],
                             'beyond': float(caught.sum() - pc.sum()), 'FA_episodes': s['false_alarm_events']})
        print(f'  seed {seed} done', flush=True)
    r = pd.DataFrame(rows); r.to_csv(f'calibration_dev_{args.dataset}.csv', index=False)
    g = r.groupby(['method', 'detector']).agg(F1=('F1', 'mean'), FAR=('FAR', 'mean'), beyond=('beyond', 'mean'),
                                              beyond_sd=('beyond', 'std'), FA_episodes=('FA_episodes', 'mean'))
    g[['F1', 'FAR']] = g[['F1', 'FAR']] * 100
    pd.set_option('display.width', 250)
    print(f'\n[{args.dataset}] {len(events[0])} attacks, window {window} samples, K = {K}; mean of seeds 1-5')
    print(g.round(2).to_string())


if __name__ == '__main__':
    main()
