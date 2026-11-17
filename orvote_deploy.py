"""Pre-registered test H4 (preregistration/H4_or_vote_on_site.md, 2026-10-07): PIVOT's OR vote calibrated on site.

H4: with every threshold taken from a 30-minute commissioning window, PIVOT's OR vote raises no more false alarms
than the monolithic detector and catches more attacks beyond chance.

Frozen design
  releases, models, scoring   as in H3 (orvote_confirm.py): HAI 20.07 / 21.03 / 22.04, four stages, seeds 1-5,
                              per-sensor standardized error, max per stage, smoothed W = 25
  commissioning window        the first WINDOW = 1800 samples (30 min) of the test stream; longest round window free
                              of attacks on all three releases (21.03's first attack at 35 min)
  decision rule               dual threshold, W_dur = 50. Monolithic: tau_H / tau_L = window P99.9 / P99 of its score.
                              PIVOT: each of K stages at the 1 - 0.001/K and 1 - 0.01/K quantiles of its own window
                              score (the same budget split over stages), OR-ed
Success, per release          supported if (a) PIVOT's FAR <= monolithic FAR on >= 4 of 5 seeds and (b) the 95% CI of
                              PIVOT minus monolithic in attacks beyond chance lies above zero; (a) with a lower bound
                              above -1 only = non-inferior, not supported
Secondary                     F1, early detection, false-alarm episodes; the same rule with a 15-minute window

Run from the repo root:  python orvote_deploy.py --release {hai2007,hai2103,hai2204}

H5 (pre-registered 2026-10-07, after calibration_dev.py chose it on the development plants): --calibration shift.
Every threshold = window median + (training quantile - training median) of the same score, for both detectors;
everything else as H4. Secondary detector 'mono_window': the monolithic detector with H4's window rule.
  python orvote_deploy.py --release hai2007 --calibration shift

H6 (pre-registered 2026-10-09, preregistration/H6_swat_new_releases.md): H5's rule on newer SWaT releases (Jul 2019, Dec 2019,
Nov 2022; preprocess_swat_new.py) with the 2015 models (six stages without analyzers, monolithic without analyzers).
  python orvote_deploy.py --release swat2019jul --calibration shift
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '.')

SEEDS = [1, 2, 3, 4, 5]
SWAT_NEW = ['swat2019jul', 'swat2019dec', 'swat2022nov']
WINDOW, WINDOW_SHORT = 1800, 900
W_DUR, N_BOOT = 50, 10000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release', choices=['hai2007', 'hai2103', 'hai2204'] + SWAT_NEW, required=True)
    ap.add_argument('--gpu', default=None)
    ap.add_argument('--calibration', choices=['window', 'shift'], default='window')
    args = ap.parse_args()
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import decomposition_mechanism as dm
    import label_free as lfp
    from baselines_simple_label_free import early_events
    from evaluate_events import evaluate
    rel, cal = args.release, args.calibration
    sfx = '' if cal == 'window' else f'_{cal}'
    if rel in SWAT_NEW:   # 2015 models and training data, never retrained
        import decomposition_ablation_label_free as dab
        from evaluate_events import to_binary
        stages = list(dab.partition(6, 'split'))
        stage_dir, mono_dir = 'Oracles/swat_phys_none_seed{}', 'Oracles/swat_mono_noAIT_none_seed{}'
        tr = pd.read_csv('Datasets/swat_train_preprocessed.csv'); te = pd.read_csv(f'Datasets/{rel}_test_preprocessed.csv')
        tr.columns, te.columns = tr.columns.str.strip(), te.columns.str.strip()
        y = to_binary(te['Normal/Attack']).astype(int)
    else:
        stages = list(json.load(open(f'Datasets/{rel}_subsystems.json')))
        stage_dir, mono_dir = f'Oracles/{rel}_none_seed{{}}', f'Oracles/{rel}_mono_none_seed{{}}'
        tr = pd.read_csv(f'Datasets/{rel}_train_preprocessed.csv'); te = pd.read_csv(f'Datasets/{rel}_test_preprocessed.csv')
        y = te['Label'].values.astype(int)
    K = len(stages)
    te, fired = lfp.fault_rule(tr, te)
    if fired:
        print(f'[{rel}] fault rule replaced: {fired}')
    fill = tr.select_dtypes('number').mean(); events = lfp.runs(y == 1)
    assert not y[:WINDOW].any(), 'commissioning window contains an attack'
    print(f'[{rel}] {len(y):,} test rows, {len(events[0])} attacks, window {WINDOW} samples, K = {K}', flush=True)

    def score(a, b):
        mu, sd = a.mean(0), a.std(0) + 1e-6
        return lfp.smooth(((a - mu) / sd).max(1), 25), lfp.smooth(((b - mu) / sd).max(1), 25)

    def dual(sc, w, h, l, rule=None):
        s_tr, s = sc
        if (rule or cal) == 'window':
            th, tl = np.quantile(s[:w], h), np.quantile(s[:w], l)
        else:   # shift: the window sets the level, training the rest
            base = np.median(s[:w]) - np.median(s_tr)
            th, tl = base + np.quantile(s_tr, h), base + np.quantile(s_tr, l)
        return lfp.dual_threshold(s, th, tl, W_DUR).astype(int)

    per_event, rows = [], []
    for seed in SEEDS:
        st = [score(dm.sensor_errors(stage_dir.format(seed), n, tr, fill)[1],
                    dm.sensor_errors(stage_dir.format(seed), n, te, fill)[1]) for n in stages]
        m = score(dm.sensor_errors(mono_dir.format(seed), 'Global', tr, fill)[1],
                  dm.sensor_errors(mono_dir.format(seed), 'Global', te, fill)[1])
        for w, tag in ((WINDOW, 'primary'), (WINDOW_SHORT, 'short')):
            preds = {'pivot': np.max([dual(s, w, 1 - 0.001 / K, 1 - 0.01 / K) for s in st], axis=0),
                     'mono': dual(m, w, 0.999, 0.99)}
            if cal != 'window':
                preds['mono_window'] = dual(m, w, 0.999, 0.99, rule='window')
            for det, p in preds.items():
                caught, pc = dm.per_event(p, events, np.random.default_rng(seed))
                if tag == 'primary':
                    for i in range(len(caught)):
                        per_event.append({'seed': seed, 'detector': det, 'event': i + 1, 'excess': caught[i] - pc[i]})
                s, _ = evaluate(y, p)
                early, early_ch = early_events(p, y)
                rows.append({'window': tag, 'seed': seed, 'detector': det, 'F1': s['f1'], 'FAR': s['false_alarm'],
                             'caught': s['n_detected'], 'beyond': float(caught.sum() - pc.sum()),
                             'early_beyond': early - early_ch, 'FA_episodes': s['false_alarm_events']})
        print(f'  seed {seed} done', flush=True)

    ev = pd.DataFrame(per_event); ev.to_csv(f'orvote_deploy_{rel}{sfx}.csv', index=False)
    r = pd.DataFrame(rows); r.to_csv(f'orvote_deploy_{rel}{sfx}_seeds.csv', index=False)
    mm = ev.groupby(['detector', 'event'])['excess'].mean().unstack(0)
    d = (mm['pivot'] - mm['mono']).values
    boot = d[np.random.default_rng(0).integers(0, len(d), (N_BOOT, len(d)))].sum(1)
    lo, hi = np.percentile(boot, 2.5), np.percentile(boot, 97.5)
    prim = r[r.window == 'primary']
    far = prim.pivot(index='seed', columns='detector', values='FAR')
    far_ok = int((far['pivot'] <= far['mono']).sum())
    verdict = 'SUPPORTED' if far_ok >= 4 and lo > 0 else ('NON-INFERIOR (not supported)' if far_ok >= 4 and lo > -1 else 'NOT SUPPORTED')
    pd.DataFrame([{'release': rel, 'far_ok_seeds': far_ok, 'diff': d.sum(), 'CI_low': lo, 'CI_high': hi,
                   'verdict': verdict}]).to_csv(f'orvote_deploy_{rel}{sfx}_summary.csv', index=False)
    pd.set_option('display.width', 250)
    for tag in ('primary', 'short'):
        g = r[r.window == tag].groupby('detector').agg(F1=('F1', 'mean'), FAR=('FAR', 'mean'), caught=('caught', 'mean'),
                                                       beyond=('beyond', 'mean'), early_beyond=('early_beyond', 'mean'),
                                                       FA_episodes=('FA_episodes', 'mean'))
        g[['F1', 'FAR']] = g[['F1', 'FAR']] * 100
        print(f'\n[{rel}] {"30" if tag == "primary" else "15"}-minute commissioning window, mean of seeds 1-5 ({len(events[0])} attacks)')
        print(g.round(2).to_string())
    print(f'\n(a) PIVOT FAR <= monolithic FAR on {far_ok} of 5 seeds (need >= 4)')
    print(f'(b) PIVOT minus monolithic, attacks beyond chance: {d.sum():+.2f} (95% CI {lo:+.2f} .. {hi:+.2f}); need lower bound > 0')
    print(f'{"H4" if cal == "window" else ("H6" if rel in SWAT_NEW else "H5")} on {rel}: {verdict}')
    if cal != 'window':   # secondary: against the monolithic detector with the window rule
        d2 = (mm['pivot'] - mm['mono_window']).values
        b2 = d2[np.random.default_rng(1).integers(0, len(d2), (N_BOOT, len(d2)))].sum(1)
        ok2 = int((far['pivot'] <= far['mono_window']).sum())
        print(f'Secondary vs monolithic with window rule: PIVOT FAR lower on {ok2} of 5 seeds; '
              f'beyond chance {d2.sum():+.2f} (95% CI {np.percentile(b2, 2.5):+.2f} .. {np.percentile(b2, 97.5):+.2f})')


if __name__ == '__main__':
    main()
