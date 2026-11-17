"""Pre-registered test H3 (preregistration/H3_or_vote_rule.md, 2026-10-07): PIVOT's own OR-vote rule on held-out HAI.

H3: one autoencoder per stage, frozen, each stage with its own threshold and an alarm when any stage fires, catches
more attacks beyond chance than a monolithic autoencoder on the same sensors, at the same total FAR.

Frozen design
  models    as trained for H1b/H1c (deployable_test.py), unchanged: Oracles/<rel>_none_seed<k> (the four processes
            by sensor prefix) and Oracles/<rel>_mono_none_seed<k>, seeds 1-5, rel in hai2007 / hai2103 / hai2204
  scoring   each sensor's squared error standardized on its own training errors; a stage's score = the max over its
            sensors; smoothed W = 25; the monolithic model scored the same way over all its sensors
  OR vote   each stage alarms above the (1 - q) quantile of its own score on test normal rows, one q for all stages,
            bisected so the OR reaches the target FAR; monolithic: one threshold at the target FAR (a comparison)
Primary     at total FAR 0.1, 0.5, 1, 2%: OR minus monolithic in attacks beyond chance (per-attack chance from
            circular shifts, seed-averaged), 95% CI bootstrapping attacks; supported on a release if the CI is above
            zero at >= 2 of the 4 levels. Releases judged separately.
Secondary   point-wise F1 at the same levels; label-free (each stage at P99.8 of its training score vs the
            monolithic model at P99.8).

Run from the repo root:  python orvote_confirm.py --release {hai2007,hai2103,hai2204}
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '.')

TARGETS = [0.001, 0.005, 0.01, 0.02]
SEEDS = [1, 2, 3, 4, 5]
N_BOOT = 10000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release', choices=['hai2007', 'hai2103', 'hai2204'], required=True)
    ap.add_argument('--gpu', default=None)
    args = ap.parse_args()
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import decomposition_mechanism as dm
    import label_free as lfp
    from evaluate_events import evaluate
    rel = args.release
    stages = list(json.load(open(f'Datasets/{rel}_subsystems.json')))
    tr = pd.read_csv(f'Datasets/{rel}_train_preprocessed.csv')
    te = pd.read_csv(f'Datasets/{rel}_test_preprocessed.csv')
    y = te['Label'].values.astype(int)
    te, fired = lfp.fault_rule(tr, te)
    fill = tr.select_dtypes('number').mean()
    events, normal = lfp.runs(y == 1), y == 0
    print(f'[{rel}] {len(te):,} test rows, {len(events[0])} attacks, fault rule: {fired or "nothing"}', flush=True)

    def score(e_tr, e_te):
        mu, sd = e_tr.mean(0), e_tr.std(0) + 1e-6
        return lfp.smooth(((e_tr - mu) / sd).max(1), 25), lfp.smooth(((e_te - mu) / sd).max(1), 25)

    def or_at(scores, target):
        qs = [np.sort(s[normal]) for s in scores]
        pred = lambda q: np.max([s > np.quantile(n, 1 - q) for s, n in zip(scores, qs)], axis=0).astype(int)
        lo, hi = 0.0, target
        for _ in range(30):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if pred(mid)[normal].mean() <= target else (lo, mid)
        return pred(lo)

    per_event, f1_rows, lf_rows = [], [], []
    for seed in SEEDS:
        st = [score(dm.sensor_errors(f'Oracles/{rel}_none_seed{seed}', n, tr, fill)[1],
                    dm.sensor_errors(f'Oracles/{rel}_none_seed{seed}', n, te, fill)[1]) for n in stages]
        m_tr, m_te = score(dm.sensor_errors(f'Oracles/{rel}_mono_none_seed{seed}', 'Global', tr, fill)[1],
                           dm.sensor_errors(f'Oracles/{rel}_mono_none_seed{seed}', 'Global', te, fill)[1])
        for target in TARGETS:
            preds = {'or': or_at([b for _, b in st], target),
                     'mono': (m_te > np.quantile(m_te[normal], 1 - target)).astype(int)}
            for det, p in preds.items():
                caught, pc = dm.per_event(p, events, np.random.default_rng(seed))
                for i in range(len(caught)):
                    per_event.append({'seed': seed, 'target_FAR': target, 'detector': det, 'event': i + 1,
                                      'caught': caught[i], 'p_chance': pc[i]})
                m, _ = evaluate(y, p)
                f1_rows.append({'seed': seed, 'target_FAR': target, 'detector': det, 'F1': m['f1'], 'FAR': p[normal].mean()})
        for det, p in (('or', np.max([b > np.percentile(a, 99.8) for a, b in st], axis=0).astype(int)),
                       ('mono', (m_te > np.percentile(m_tr, 99.8)).astype(int))):
            m, _ = evaluate(y, p); ch, _ = lfp.chance_events(p, y, n=100)
            lf_rows.append({'seed': seed, 'detector': det, 'F1': m['f1'], 'FAR': m['false_alarm'],
                            'caught': m['n_detected'], 'beyond': m['n_detected'] - ch})
        print(f'  seed {seed} done', flush=True)

    ev = pd.DataFrame(per_event); ev.to_csv(f'orvote_confirm_{rel}.csv', index=False)
    ev['excess'] = ev.caught - ev.p_chance
    rng = np.random.default_rng(0)
    res = []
    for target in TARGETS:
        sub = ev[ev.target_FAR == target]
        m = sub.groupby(['detector', 'event'])['excess'].mean().unstack(0)
        d = (m['or'] - m['mono']).values
        boot = d[rng.integers(0, len(d), (N_BOOT, len(d)))].sum(1)
        ps = sub.groupby(['detector', 'seed'])['excess'].sum().unstack(0)
        res.append({'target_FAR': target, 'or_beyond': ps['or'].mean(), 'mono_beyond': ps['mono'].mean(),
                    'diff': d.sum(), 'CI_low': np.percentile(boot, 2.5), 'CI_high': np.percentile(boot, 97.5),
                    'seeds_or_better': int((ps['or'] > ps['mono']).sum())})
    res = pd.DataFrame(res); res.to_csv(f'orvote_confirm_{rel}_summary.csv', index=False)
    pd.set_option('display.width', 250)
    res_p = res.copy(); res_p['target_FAR'] = (res_p.target_FAR * 100).map(lambda v: f'{v:g}%')
    print(f'\n[{rel}] {len(events[0])} attacks. Matched FAR, attacks beyond chance (mean of seeds 1-5):')
    print(res_p.round(2).to_string(index=False))
    n_pass = int((res.CI_low > 0).sum())
    print(f'\nH3 on {rel}: CI above zero at {n_pass} of {len(TARGETS)} levels -> {"SUPPORTED" if n_pass >= 2 else "NOT SUPPORTED"}')
    pd.DataFrame(f1_rows).to_csv(f'orvote_confirm_{rel}_f1.csv', index=False)
    f1 = pd.DataFrame(f1_rows).pivot_table(index='detector', columns='target_FAR', values='F1', aggfunc='mean') * 100
    f1.columns = [f'{c * 100:g}%' for c in f1.columns]
    print('\nSecondary: point-wise F1 (%) at matched FAR'); print(f1.round(1).to_string())
    lf = pd.DataFrame(lf_rows); lf.to_csv(f'orvote_confirm_{rel}_labelfree.csv', index=False)
    g = lf.groupby('detector').agg(F1=('F1', 'mean'), FAR=('FAR', 'mean'), caught=('caught', 'mean'), beyond=('beyond', 'mean'))
    g[['F1', 'FAR']] = g[['F1', 'FAR']] * 100
    print('\nSecondary: label-free (P99.8 of training scores)'); print(g.round(2).to_string())


if __name__ == '__main__':
    main()
