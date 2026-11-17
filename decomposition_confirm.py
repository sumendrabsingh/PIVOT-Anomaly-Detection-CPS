"""Pre-registered confirmation of H1 on WADI (A2) and HAI 23.05 (preregistration/H1_decomposition_matched_far.md, 2026-10-07).

H1: splitting a plant into its physical stages, with per-sensor error scoring, catches more attacks beyond chance
than one autoencoder with the same scoring, at matched FAR. Found on SWaT after looking at results
(decomposition_mechanism.py, 'K6 sensor-scored' vs 'K1 sensor-scored'), so it is tested here, unchanged, on the
two other plants.

Frozen design
  stages      Datasets/wadi_subsystems.json (A2, 3 stages) / Datasets/hai2305_subsystems.json (4 stages)
  control     one autoencoder on the union of the stage features ('Global'), same training code
  models      run_generalized_subsystems.train_subsystem, lambda = 0, seeds 1-5; trained here if missing
  score       each sensor's squared error standardized on its own training errors, max over sensors,
              smoothed W = 25 (decomposition_mechanism.sensor_score), for both detectors
  data        the fault rule as fixed on 2026-10-07; WADI labels from the maintainers' A2 attack table; HAI labels
              aligned by row (52 segments)
Primary test  at matched FAR 0.1%, 0.25%, 0.5% (one threshold on the smoothed score, set on test normal rows: a
              comparison, not an operating point), stages minus monolithic in attacks beyond chance (per-attack
              chance from circular shifts), averaged over seeds, 95% CI bootstrapping attacks. H1 is supported on a
              plant if the CI lies above zero at >= 2 of the 3 levels.
Secondary     1% and 2% FAR; the label-free operating point (dual threshold P99.9 / P99 of training scores,
              W_dur = 50); early detection (an alarm in the first half of the attack window).

Run from the repo root:  python decomposition_confirm.py --dataset wadi   (or hai; --gpu to pick a device)
Writes decomposition_confirm_<dataset>.csv (per attack) and decomposition_confirm_<dataset>_summary.csv.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '.')

DATA = {'wadi': dict(train='Datasets/wadi_train_preprocessed.csv', test='Datasets/wadi_test_preprocessed.csv',
                     subs='Datasets/wadi_subsystems.json', stage_dir='Oracles/wadi_a2_none_seed{}',
                     mono_dir='Oracles/wadi_a2_mono_none_seed{}'),
        'hai': dict(train='Datasets/hai_train_preprocessed.csv', test='Datasets/hai_test_preprocessed.csv',
                    subs='Datasets/hai2305_subsystems.json', stage_dir='Oracles/hai2305_none_seed{}',
                    mono_dir='Oracles/hai2305_mono_none_seed{}')}
SEEDS = [1, 2, 3, 4, 5]
PRIMARY = [0.001, 0.0025, 0.005]
SECONDARY = [0.01, 0.02]
N_SHIFTS, N_BOOT = 100, 10000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=list(DATA), required=True)
    ap.add_argument('--gpu', default=None, help='CUDA device index')
    ap.add_argument('--train_only', action='store_true')
    args = ap.parse_args()
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import decomposition_mechanism as dm          # after CUDA_VISIBLE_DEVICES is set
    import label_free as lfp
    import run_generalized_subsystems as gen
    from baselines_simple_label_free import early_events
    from evaluate_events import evaluate, to_binary

    cfg = DATA[args.dataset]
    stages = json.load(open(cfg['subs']))
    union = [c for feats in stages.values() for c in feats]
    plan = [(cfg['stage_dir'].format(s), stages, s) for s in SEEDS] + \
           [(cfg['mono_dir'].format(s), {'Global': union}, s) for s in SEEDS]
    for out_dir, subs, seed in plan:
        for name, feats in subs.items():
            if not os.path.exists(f'{out_dir}/subsystem_{name}.pth'):
                os.makedirs(out_dir, exist_ok=True)
                gen.train_subsystem(name, feats, cfg['train'], out_dir, seed=seed, physics='none')
    if args.train_only:
        return

    tr = pd.read_csv(cfg['train']); tr.columns = tr.columns.str.strip()
    te = pd.read_csv(cfg['test']); te.columns = te.columns.str.strip()
    y = to_binary(te['Label' if 'Label' in te.columns else 'Normal/Attack'])
    te, fired = lfp.fault_rule(tr, te)
    print(f'[{args.dataset}] {len(te):,} test rows, {int(y.sum()):,} attack rows, fault rule: {fired or "nothing"}')
    events = lfp.runs(y == 1)
    fill = tr.select_dtypes('number').mean()

    def errors(out_dir, subs):
        e_tr, e_te = [], []
        for name in subs:
            _, a = dm.sensor_errors(out_dir, name, tr, fill)
            _, b = dm.sensor_errors(out_dir, name, te, fill)
            e_tr.append(a); e_te.append(b)
        return np.hstack(e_tr), np.hstack(e_te)

    rows, lf_rows = [], []
    for seed in SEEDS:
        for det, out_dir, subs in (('stages', cfg['stage_dir'].format(seed), stages),
                                   ('monolithic', cfg['mono_dir'].format(seed), {'Global': union})):
            e_tr, e_te = errors(out_dir, subs)
            s_tr, s_te = dm.sensor_score(e_tr, e_tr), dm.sensor_score(e_tr, e_te)
            for target in PRIMARY + SECONDARY:
                pred = (s_te > np.quantile(s_te[y == 0], 1 - target)).astype(int)
                caught, p = dm.per_event(pred, events, np.random.default_rng(seed))
                for i in range(len(caught)):
                    rows.append({'seed': seed, 'detector': det, 'target_FAR': target, 'event': i + 1,
                                 'caught': caught[i], 'p_chance': p[i], 'FAR': pred[y == 0].mean()})
            pred = lfp.dual_threshold(s_te, np.percentile(s_tr, 99.9), np.percentile(s_tr, 99.0), 50).astype(int)
            s, _ = evaluate(y, pred)
            ch, _ = lfp.chance_events(pred, y, n=N_SHIFTS)
            early, early_ch = early_events(pred, y)
            lf_rows.append({'seed': seed, 'detector': det, 'F1': s['f1'], 'FAR': s['false_alarm'],
                            'caught': s['n_detected'], 'chance': ch, 'beyond': s['n_detected'] - ch,
                            'early_beyond': early - early_ch, 'FA_episodes': s['false_alarm_events']})
        print(f'  seed {seed} done', flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(f'decomposition_confirm_{args.dataset}.csv', index=False)
    out['excess'] = out.caught - out.p_chance
    rng = np.random.default_rng(0)
    res = []
    for target in PRIMARY + SECONDARY:
        sub = out[out.target_FAR == target]
        ev = sub.groupby(['detector', 'event'])['excess'].mean().unstack(0)
        d = (ev['stages'] - ev['monolithic']).values
        boot = d[rng.integers(0, len(d), (N_BOOT, len(d)))].sum(1)
        per_seed = sub.groupby(['detector', 'seed'])['excess'].sum().unstack(0)
        res.append({'target_FAR': target, 'primary': target in PRIMARY,
                    'stages_beyond': per_seed['stages'].mean(), 'mono_beyond': per_seed['monolithic'].mean(),
                    'diff': d.sum(), 'CI_low': np.percentile(boot, 2.5), 'CI_high': np.percentile(boot, 97.5),
                    'seeds_stages_better': int((per_seed['stages'] > per_seed['monolithic']).sum())})
    res = pd.DataFrame(res)
    res.to_csv(f'decomposition_confirm_{args.dataset}_summary.csv', index=False)
    pd.set_option('display.width', 250)
    print(f'\n[{args.dataset}] {len(events[0])} attacks. Matched FAR, attacks beyond chance (mean of seeds 1-5):')
    print(res.round(2).to_string(index=False))
    n_pass = int(((res.primary) & (res.CI_low > 0)).sum())
    print(f'\nH1 on {args.dataset}: CI above zero at {n_pass} of {len(PRIMARY)} primary FAR levels -> '
          f'{"SUPPORTED" if n_pass >= 2 else "NOT SUPPORTED"}')
    lf = pd.DataFrame(lf_rows)
    lf.to_csv(f'decomposition_confirm_{args.dataset}_labelfree.csv', index=False)
    g = lf.groupby('detector').agg(F1=('F1', 'mean'), FAR=('FAR', 'mean'), caught=('caught', 'mean'),
                                   chance=('chance', 'mean'), beyond=('beyond', 'mean'), beyond_sd=('beyond', 'std'),
                                   early_beyond=('early_beyond', 'mean'), FA_episodes=('FA_episodes', 'mean'))
    g['F1'] = (g.F1 * 100).round(1); g['FAR'] = (g.FAR * 100).round(2)
    print('\nSecondary: label-free operating point (dual threshold from training scores), mean of seeds 1-5')
    print(g.round(2).to_string())


if __name__ == '__main__':
    main()
