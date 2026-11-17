"""Does decomposition make the detector harder to evade? Six stages vs one monolithic model under the same
physically valid adaptive attack, at MATCHED false-alarm rates (SWaT, seeds 1-5, lambda = 0, analyzers split out).

feasible_attack_label_free.py attacks the six-stage array at its own operating point (about 0.2% FAR). The
monolithic control at its own label-free thresholds runs at about 4% FAR, where weak alarms are easier to
suppress, so comparing the two at their own operating points confounds decomposition with strictness. Here
both detectors keep the deployed dual-threshold rule, with tau_L / tau_H set so the CLEAN detector reaches the
same FAR (feasible_attack_label_free.py --far_target). Choosing thresholds by FAR reads the test labels, so this
is a COMPARISON, not an operating point (as matched_far_decomposition.py).

Per run the attacker is feasible_attack_label_free.py unchanged: sensor-only and sensor-plus-actuator threat
models, reconstruction and decision objectives, per-event best iterate. The score at budget eps is the worst case
over budgets <= eps (an attack at a smaller budget is also feasible at eps), reported as attacks caught minus the
chance level of the attacked alarm pattern.

Run from the repo root:  python matched_far_attack.py            (runs, then summarizes)
                         python matched_far_attack.py --summary  (summarize existing runs only)
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

SEEDS = [1, 2, 3, 4, 5]
TARGETS = [0.002, 0.01]
EPSILONS = '0.5,1,2,4,8'
DETECTORS = {'array': ('Oracles/swat_phys_none_seed{}', None),
             'mono': ('Oracles/swat_mono_noAIT_none_seed{}', 'Global')}
OUT_DIR = 'results_matched_far_attack'
SUMMARY = 'matched_far_attack_swat.csv'


def run_name(det, seed, target):
    return f'{OUT_DIR}/{det}_seed{seed}_far{target}.csv'


def run(job, gpu):
    det, seed, target = job
    out = run_name(det, seed, target)
    if os.path.exists(out):
        return f'skip {out}'
    models, subs = DETECTORS[det]
    cmd = [sys.executable, 'feasible_attack_label_free.py', '--models', models.format(seed),
           '--threats', 'sensors,actuators', '--epsilons', EPSILONS, '--far_target', str(target), '--out', out]
    if subs:
        cmd += ['--subs', subs]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    with open(out.replace('.csv', '.log'), 'w') as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
    return f'{"done" if rc == 0 else f"FAILED ({rc})"} {out}'


def summarize():
    rows = []
    for det in DETECTORS:
        for seed in SEEDS:
            for target in TARGETS:
                f = run_name(det, seed, target)
                if not os.path.exists(f):
                    continue
                d = pd.read_csv(f)
                d = d[d.detector == 'array']  # the attacked model; the analyzer arm is identical for both
                clean = d[d.threat == 'clean'].iloc[0]
                base = {'detector': det, 'seed': seed, 'target_FAR': target, 'clean_FAR': clean.FAR}
                rows.append({**base, 'threat': 'clean', 'epsilon': 0.0, 'events': clean.events,
                             'chance': clean.chance, 'excess': clean.events - clean.chance})
                for threat in ('sensors', 'actuators'):
                    b = d[(d.threat == threat) & (d.objective == 'best')].sort_values('epsilon')
                    best = (clean.events, clean.chance)
                    for _, r in b.iterrows():  # worst case over budgets <= eps
                        if r.events - r.chance < best[0] - best[1]:
                            best = (r.events, r.chance)
                        rows.append({**base, 'threat': threat, 'epsilon': r.epsilon, 'events': best[0],
                                     'chance': best[1], 'excess': best[0] - best[1]})
    out = pd.DataFrame(rows)
    out.to_csv(SUMMARY, index=False)
    pd.set_option('display.width', 250)
    s = out.groupby(['target_FAR', 'threat', 'epsilon', 'detector']).agg(
        n=('seed', 'size'), FAR=('clean_FAR', 'mean'), events=('events', 'mean'), chance=('chance', 'mean'),
        excess=('excess', 'mean'), excess_sd=('excess', 'std')).reset_index()
    s['FAR'] = (s['FAR'] * 100).round(2)
    print(s.round(2).to_string(index=False))
    piv = out.pivot_table(index=['target_FAR', 'threat', 'epsilon', 'seed'], columns='detector',
                          values='excess').dropna().reset_index()
    piv['diff'] = piv['array'] - piv['mono']
    d = piv.groupby(['target_FAR', 'threat', 'epsilon'])['diff'].agg(
        ['mean', 'std', 'size', lambda v: int((v > 0).sum())])
    d.columns = ['array_minus_mono_excess', 'sd', 'seeds', 'seeds_array_better']
    print(d.round(2).to_string())
    print(f'[SAVED] {SUMMARY}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary', action='store_true')
    ap.add_argument('--gpus', default='0,1')
    ap.add_argument('--per_gpu', type=int, default=2)
    args = ap.parse_args()
    if not args.summary:
        os.makedirs(OUT_DIR, exist_ok=True)
        gpus = args.gpus.split(',')
        jobs = [(det, seed, t) for seed in SEEDS for t in TARGETS for det in DETECTORS]
        slots = [gpus[i % len(gpus)] for i in range(len(gpus) * args.per_gpu)]
        with ThreadPoolExecutor(len(slots)) as ex:
            futs = [ex.submit(run, j, slots[i % len(slots)]) for i, j in enumerate(jobs)]
            for f in futs:
                print(f.result(), flush=True)
    summarize()
