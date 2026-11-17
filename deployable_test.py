"""Pre-registered test H1b on held-out HAI releases (preregistration/H1b_deployable_training_thresholds.md, 2026-10-07).

H1b: at the label-free operating point, the stage array with per-sensor scoring raises no more false alarms than the
monolithic control with the same scoring, and catches at least as many attacks beyond chance.

Held-out data   HAI 22.04 (86 sensors, 58 attack segments) and HAI 21.03 (79 sensors, 50 attack segments); training
                and test files concatenated in file order; labels = the release's own attack column.
Frozen design   stages = the four processes (every sensor named P1_..P4_ by its prefix); the monolithic control
                uses the same sensors. Models (run_generalized_subsystems.train_subsystem, lambda = 0, seeds 1-5),
                scoring (decomposition_mechanism.sensor_score) and decision rule (dual threshold P99.9 / P99 of
                training scores, W_dur = 50) exactly as in H1; the fault rule fixed on 2026-10-07.
Success         per release, both: (a) stage FAR <= monolithic FAR on >= 4 of 5 seeds; (b) the 95% CI of stages
                minus monolithic in attacks beyond chance (per-attack chance from circular shifts, seed-averaged,
                bootstrapping attacks) has its lower bound above -1 attack (above 0 = the stronger result).

Run from the repo root:  python deployable_test.py --release {hai2204,hai2103} [--gpu N]

H1c (pre-registered 2026-10-07): --calibration commission takes the dual-threshold rule's thresholds from the first
COMMISSION samples (15 min) of the test stream instead of from training scores; primary held-out release HAI 20.07
(trained on its normal-labelled training rows only), secondary --release swat / wadi / hai2305 / hai2204 / hai2103
(already seen under training calibration; a plant whose commissioning window contains an attack is skipped).
  python deployable_test.py --release hai2007 --calibration commission
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '.')

RELEASES = {'hai2204': ('SWaT_Orig_DataSets/hai/hai-22.04', 'Attack'),
            'hai2103': ('SWaT_Orig_DataSets/hai/hai-21.03', 'attack'),
            'hai2007': ('SWaT_Orig_DataSets/hai/hai-20.07', 'attack')}
# plants prepared and trained elsewhere: (train csv, test csv, stages, stage dir, monolithic dir)
EXISTING = {'swat': ('Datasets/swat_train_preprocessed.csv', 'Datasets/swat_test_preprocessed.csv', 'swat',
                     'Oracles/swat_phys_none_seed{}', 'Oracles/swat_mono_noAIT_none_seed{}'),
            'wadi': ('Datasets/wadi_train_preprocessed.csv', 'Datasets/wadi_test_preprocessed.csv',
                     'Datasets/wadi_subsystems.json', 'Oracles/wadi_a2_none_seed{}', 'Oracles/wadi_a2_mono_none_seed{}'),
            'hai2305': ('Datasets/hai_train_preprocessed.csv', 'Datasets/hai_test_preprocessed.csv',
                        'Datasets/hai2305_subsystems.json', 'Oracles/hai2305_none_seed{}', 'Oracles/hai2305_mono_none_seed{}')}
COMMISSION = 900
SEEDS = [1, 2, 3, 4, 5]
N_BOOT = 10000


def prepare(rel):
    """Datasets/<rel>_{train,test}_preprocessed.csv (raw units, 'Label') and Datasets/<rel>_subsystems.json."""
    src, lab = RELEASES[rel]
    tr_p, te_p, js = f'Datasets/{rel}_train_preprocessed.csv', f'Datasets/{rel}_test_preprocessed.csv', f'Datasets/{rel}_subsystems.json'
    if all(os.path.exists(p) for p in (tr_p, te_p, js)):
        return tr_p, te_p, js
    sep = ';' if ';' in open(sorted(glob.glob(f'{src}/train*.csv'))[0]).readline() else ','   # HAI 20.07 uses ';'
    trs = [pd.read_csv(f, sep=sep) for f in sorted(glob.glob(f'{src}/train*.csv'))]
    tes = [pd.read_csv(f, sep=sep) for f in sorted(glob.glob(f'{src}/test*.csv'))]
    tags = [c for c in trs[0].columns if re.match(r'P[1-4]_', c)]
    n_att = int(sum((t[lab] > 0).sum() for t in trs))
    if n_att:   # HAI 20.07: its training files label 776 rows as attacks; train on the normal rows only
        print(f'[{rel}] dropping {n_att} attack-labelled training rows')
    tr = pd.concat([t.loc[t[lab] == 0, tags] for t in trs], ignore_index=True)
    te = pd.concat([t[tags].assign(Label=(t[lab].values > 0).astype(int)) for t in tes], ignore_index=True)
    tr.to_csv(tr_p, index=False); te.to_csv(te_p, index=False)
    json.dump({f'P{k}': [c for c in tags if c.startswith(f'P{k}_')] for k in range(1, 5)}, open(js, 'w'), indent=2)
    print(f'[{rel}] prepared: train {len(tr):,} rows, test {len(te):,} rows, {int(te.Label.sum()):,} attack rows, {len(tags)} sensors')
    return tr_p, te_p, js


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release', choices=list(RELEASES) + list(EXISTING), required=True)
    ap.add_argument('--calibration', choices=['train', 'commission'], default='train')
    ap.add_argument('--gpu', default=None)
    args = ap.parse_args()
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import decomposition_mechanism as dm
    import label_free as lfp
    import run_generalized_subsystems as gen
    from baselines_simple_label_free import early_events
    from evaluate_events import evaluate, to_binary
    rel, cal = args.release, args.calibration
    tag = rel if cal == 'train' else f'{rel}_commission'

    if rel in EXISTING:
        tr_p, te_p, js, stage_dir, mono_dir = EXISTING[rel]
        if js == 'swat':
            import decomposition_ablation_label_free as dab
            stages = dab.partition(6, 'split')
        else:
            stages = json.load(open(js))
    else:
        tr_p, te_p, js = prepare(rel)
        stages = json.load(open(js))
        stage_dir, mono_dir = f'Oracles/{rel}_none_seed{{}}', f'Oracles/{rel}_mono_none_seed{{}}'
    union = [c for f in stages.values() for c in f]
    for out_dir, subs, seed in [(stage_dir.format(s), stages, s) for s in SEEDS] + \
                               [(mono_dir.format(s), {'Global': union}, s) for s in SEEDS]:
        for name, feats in subs.items():
            if not os.path.exists(f'{out_dir}/subsystem_{name}.pth'):
                os.makedirs(out_dir, exist_ok=True)
                gen.train_subsystem(name, feats, tr_p, out_dir, seed=seed, physics='none')

    tr = pd.read_csv(tr_p); te = pd.read_csv(te_p)
    tr.columns, te.columns = tr.columns.str.strip(), te.columns.str.strip()
    y = to_binary(te['Label' if 'Label' in te.columns else 'Normal/Attack']).astype(int)
    te, fired = lfp.fault_rule(tr, te)
    events = lfp.runs(y == 1)
    fill = tr.select_dtypes('number').mean()
    print(f'[{rel}] {len(te):,} test rows, {len(events[0])} attacks, fault rule: {fired or "nothing"}, '
          f'calibration: {cal}', flush=True)
    if cal == 'commission' and y[:COMMISSION].any():
        print(f'[{rel}] the first {COMMISSION} test samples contain an attack: commissioning not applicable, skipped')
        return

    per_event, rows = [], []
    for seed in SEEDS:
        for det, out_dir, subs in (('stages', stage_dir.format(seed), stages),
                                   ('monolithic', mono_dir.format(seed), {'Global': union})):
            e_tr, e_te = [], []
            for name in subs:
                _, a = dm.sensor_errors(out_dir, name, tr, fill)
                _, b = dm.sensor_errors(out_dir, name, te, fill)
                e_tr.append(a); e_te.append(b)
            e_tr, e_te = np.hstack(e_tr), np.hstack(e_te)
            s_tr, s_te = dm.sensor_score(e_tr, e_tr), dm.sensor_score(e_tr, e_te)
            ref = s_tr if cal == 'train' else s_te[:COMMISSION]
            pred = lfp.dual_threshold(s_te, np.percentile(ref, 99.9), np.percentile(ref, 99.0), 50).astype(int)
            caught, pc = dm.per_event(pred, events, np.random.default_rng(seed))
            for i in range(len(caught)):
                per_event.append({'seed': seed, 'detector': det, 'event': i + 1, 'caught': caught[i], 'p_chance': pc[i]})
            s, _ = evaluate(y, pred)
            early, early_ch = early_events(pred, y)
            rows.append({'seed': seed, 'detector': det, 'F1': s['f1'], 'FAR': s['false_alarm'], 'caught': s['n_detected'],
                         'chance': float(pc.sum()), 'beyond': float(caught.sum() - pc.sum()),
                         'early_beyond': early - early_ch, 'FA_episodes': s['false_alarm_events']})
        print(f'  seed {seed} done', flush=True)

    ev = pd.DataFrame(per_event); ev.to_csv(f'deployable_test_{tag}.csv', index=False)
    ev['excess'] = ev.caught - ev.p_chance
    m = ev.groupby(['detector', 'event'])['excess'].mean().unstack(0)
    d = (m['stages'] - m['monolithic']).values
    boot = d[np.random.default_rng(0).integers(0, len(d), (N_BOOT, len(d)))].sum(1)
    lo, hi = np.percentile(boot, 2.5), np.percentile(boot, 97.5)
    r = pd.DataFrame(rows); r.to_csv(f'deployable_test_{tag}_seeds.csv', index=False)
    far = r.pivot(index='seed', columns='detector', values='FAR')
    far_ok = int((far['stages'] <= far['monolithic']).sum())
    g = r.groupby('detector').agg(F1=('F1', 'mean'), FAR=('FAR', 'mean'), caught=('caught', 'mean'), chance=('chance', 'mean'),
                                  beyond=('beyond', 'mean'), beyond_sd=('beyond', 'std'), early_beyond=('early_beyond', 'mean'),
                                  FA_episodes=('FA_episodes', 'mean'))
    g['F1'] = (g.F1 * 100).round(1); g['FAR'] = (g.FAR * 100).round(2)
    pd.set_option('display.width', 250)
    print(f'\n[{rel}] {cal}-calibrated operating point, mean of seeds 1-5 ({len(events[0])} attacks):')
    print(g.round(2).to_string())
    print(f'\n(a) stage FAR <= monolithic FAR on {far_ok} of 5 seeds (need >= 4)')
    print(f'(b) stages minus monolithic, attacks beyond chance: {d.sum():+.2f} (95% CI {lo:+.2f} .. {hi:+.2f}); need lower bound > -1')
    ok = far_ok >= 4 and lo > -1
    print(f'{"H1b" if cal == "train" else "H1c"} on {rel}: {"SUPPORTED" if ok else "NOT SUPPORTED"}' + (' (stronger: lower bound > 0)' if ok and lo > 0 else ''))
    pd.DataFrame([{'release': rel, 'calibration': cal, 'far_ok_seeds': far_ok, 'diff': d.sum(), 'CI_low': lo, 'CI_high': hi, 'supported': ok}]) \
        .to_csv(f'deployable_test_{tag}_summary.csv', index=False)


if __name__ == '__main__':
    main()
