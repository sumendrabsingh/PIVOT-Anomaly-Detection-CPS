"""Exploratory control for H3 (not pre-registered): does the OR-vote gain need separate stage models, or only
separate per-stage thresholds? Adds 'monolithic, grouped OR': the monolithic model's own per-sensor errors grouped
by stage (each group's score = max of its per-sensor-standardized errors, smoothed W = 25), with the same equal-share
per-stage thresholds OR-ed as in orvote_confirm.py. Compared with the stage-model OR vote at matched total FAR
0.1, 0.5, 1, 2% on the H3 releases, attacks beyond chance, 95% CI bootstrapping attacks, seeds 1-5.

Run from the repo root:  python orvote_mechanism.py --release {hai2007,hai2103,hai2204}
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, '.')
TARGETS, SEEDS = [0.001, 0.005, 0.01, 0.02], [1, 2, 3, 4, 5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release', choices=['hai2007', 'hai2103', 'hai2204'], required=True)
    ap.add_argument('--gpu', default=None)
    args = ap.parse_args()
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import decomposition_mechanism as dm
    import label_free as lfp
    rel = args.release
    stages = json.load(open(f'Datasets/{rel}_subsystems.json'))
    tr = pd.read_csv(f'Datasets/{rel}_train_preprocessed.csv'); te = pd.read_csv(f'Datasets/{rel}_test_preprocessed.csv')
    y = te['Label'].values.astype(int); te, _ = lfp.fault_rule(tr, te)
    fill = tr.select_dtypes('number').mean(); events, normal = lfp.runs(y == 1), y == 0

    def z(e_tr, e_te):
        mu, sd = e_tr.mean(0), e_tr.std(0) + 1e-6
        return (e_tr - mu) / sd, (e_te - mu) / sd

    def or_at(scores, target):
        qs = [np.sort(s[normal]) for s in scores]
        pred = lambda q: np.max([s > np.quantile(n, 1 - q) for s, n in zip(scores, qs)], axis=0).astype(int)
        lo, hi = 0.0, target
        for _ in range(30):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if pred(mid)[normal].mean() <= target else (lo, mid)
        return pred(lo)

    rows = []
    for seed in SEEDS:
        stage_sc = []
        for name in stages:
            a, b = z(dm.sensor_errors(f'Oracles/{rel}_none_seed{seed}', name, tr, fill)[1],
                     dm.sensor_errors(f'Oracles/{rel}_none_seed{seed}', name, te, fill)[1])
            stage_sc.append(lfp.smooth(b.max(1), 25))
        feats, a = dm.sensor_errors(f'Oracles/{rel}_mono_none_seed{seed}', 'Global', tr, fill)
        _, b = dm.sensor_errors(f'Oracles/{rel}_mono_none_seed{seed}', 'Global', te, fill)
        _, zb = z(a, b)
        grouped = [lfp.smooth(zb[:, [feats.index(c) for c in cols]].max(1), 25) for cols in stages.values()]
        for target in TARGETS:
            for det, p in (('stage models, OR', or_at(stage_sc, target)), ('monolithic, grouped OR', or_at(grouped, target))):
                caught, pc = dm.per_event(p, events, np.random.default_rng(seed))
                for i in range(len(caught)):
                    rows.append({'seed': seed, 'target_FAR': target, 'detector': det, 'event': i + 1, 'excess': caught[i] - pc[i]})
        print(f'  seed {seed} done', flush=True)
    ev = pd.DataFrame(rows); ev.to_csv(f'orvote_mechanism_{rel}.csv', index=False)
    rng = np.random.default_rng(0); res = []
    for target in TARGETS:
        sub = ev[ev.target_FAR == target]
        m = sub.groupby(['detector', 'event'])['excess'].mean().unstack(0)
        d = (m['stage models, OR'] - m['monolithic, grouped OR']).values
        boot = d[rng.integers(0, len(d), (10000, len(d)))].sum(1)
        ps = sub.groupby(['detector', 'seed'])['excess'].sum().unstack(0)
        res.append({'FAR': f'{target * 100:g}%', 'stage_models_OR': ps['stage models, OR'].mean(),
                    'mono_grouped_OR': ps['monolithic, grouped OR'].mean(), 'diff': d.sum(),
                    'CI_low': np.percentile(boot, 2.5), 'CI_high': np.percentile(boot, 97.5)})
    print(f'\n[{rel}] attacks beyond chance (mean of seeds 1-5)')
    print(pd.DataFrame(res).round(2).to_string(index=False))


if __name__ == '__main__':
    main()
