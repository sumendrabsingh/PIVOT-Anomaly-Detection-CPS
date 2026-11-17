"""Where does the decomposition gain come from, and is it reliable across attacks? (SWaT, seeds 1-5, lambda = 0,
analyzers split out.)

matched_far_decomposition.py compares six stage autoencoders (K=6) with one monolithic autoencoder (K=1) at matched
FAR. Decomposition changes two things at once: (1) each model reconstructs its stage from that stage's sensors only,
and (2) each stage's error is standardized on its own training errors before the max over stages. This script
separates them by re-scoring the SAME monolithic model's per-sensor errors in four ways:

  K6                six stage models, per-stage max error, per-stage standardization, max over stages (PIVOT)
  K1                one model, max error over all sensors, one standardization (the matched-FAR control)
  K1 stage-scored   one model, errors grouped by stage: per-stage max, per-stage standardization, max over stages
  K1 sensor-scored  one model, each sensor's squared error standardized on its training errors, max over sensors
  K6 sensor-scored  six stage models scored like K1 sensor-scored (does decomposition add anything on top?)

If "K1 stage-scored" matches K6, the gain comes from how errors are scored, not from training separate models.

Reliability: seeds measure training noise only. To include the sampling of the attacks, each attack gets a
per-seed excess (caught minus its own chance of being hit by the shifted alarm pattern); the K6 - K1 difference is
averaged over seeds and bootstrapped over attacks. Matched FAR uses a single threshold on the smoothed score, set on
test normal rows (as matched_far_decomposition.py): a COMPARISON, not an operating point.

Run from the repo root:  python decomposition_mechanism.py
"""
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, '.')
import decomposition_ablation_label_free as dab
import label_free as lfp
import run_generalized_subsystems as gen
from evaluate_events import to_binary

TARGETS = [0.001, 0.0025, 0.005, 0.01, 0.02]
N_SHIFTS = 100
N_BOOT = 10000
OUT = 'decomposition_mechanism_swat.csv'


def sensor_errors(out_dir, name, df, fill):
    """Per-sensor squared reconstruction error (standardized units) of one bundle, rows x features."""
    model, ck = gen._load_subsystem_model(f'{out_dir}/subsystem_{name}.pth')
    f = ck['features']
    mean = np.asarray(ck['mean'], dtype=np.float32)
    std = np.asarray(ck['std'], dtype=np.float32)
    std = np.where(std == 0, 1, std)
    X = torch.tensor((df[f].fillna(fill[f]).values - mean) / std, dtype=torch.float32)
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 65536):
            b = X[i:i + 65536].to(gen.DEVICE)
            out.append(((b - model(b)) ** 2).cpu().numpy())
    return f, np.concatenate(out)


def stage_score(err_tr, err_te, groups):
    """Max error within each group, standardized on that group's training errors, smoothed; max over groups."""
    out = []
    for g in groups:
        tr_, te_ = err_tr[:, g].max(1), err_te[:, g].max(1)
        out.append(lfp.smooth((te_ - tr_.mean()) / (tr_.std() + 1e-6), gen.WINDOW_SIZE))
    return np.max(out, axis=0)


def sensor_score(err_tr, err_te):
    mu, sd = err_tr.mean(0), err_tr.std(0) + 1e-6
    return lfp.smooth(((err_te - mu) / sd).max(1), gen.WINDOW_SIZE)


def per_event(pred, events, rng):
    """Caught (0/1) and chance of being hit under circular shifts, per attack event."""
    s, length = events
    caught = np.array([pred[a:a + n].any() for a, n in zip(s, length)], dtype=float)
    margin = max(1, len(pred) // 20)
    hits = np.zeros(len(s))
    for _ in range(N_SHIFTS):
        p = np.roll(pred, rng.integers(margin, len(pred) - margin))
        hits += [p[a:a + n].any() for a, n in zip(s, length)]
    return caught, hits / N_SHIFTS


if __name__ == '__main__':
    tr = pd.read_csv(dab.TRAIN); tr.columns = tr.columns.str.strip()
    te = pd.read_csv(dab.TEST); te.columns = te.columns.str.strip()
    y = to_binary(te['Normal/Attack'])
    te, _ = lfp.fault_rule(tr, te)
    events = lfp.runs(y == 1)
    fill = tr.select_dtypes('number').mean()
    stages6 = dab.partition(6, 'split')

    rows = []
    for seed in dab.SEEDS:
        scores = {}
        # K6: six stage models
        err6_tr, err6_te, groups6, feats6, start = [], [], [], [], 0
        for name in stages6:
            f, e_tr = sensor_errors(dab.model_dir(6, 'split', seed), name, tr, fill)
            _, e_te = sensor_errors(dab.model_dir(6, 'split', seed), name, te, fill)
            err6_tr.append(e_tr); err6_te.append(e_te); feats6.append(f)
            groups6.append(list(range(start, start + len(f)))); start += len(f)
        scores['K6'] = stage_score(np.hstack(err6_tr), np.hstack(err6_te), groups6)
        scores['K6 sensor-scored'] = sensor_score(np.hstack(err6_tr), np.hstack(err6_te))  # six models, per-sensor scoring
        # K1 and its re-scorings: one monolithic model
        f1, e1_tr = sensor_errors(dab.model_dir(1, 'split', seed), 'Global', tr, fill)
        _, e1_te = sensor_errors(dab.model_dir(1, 'split', seed), 'Global', te, fill)
        scores['K1'] = stage_score(e1_tr, e1_te, [list(range(len(f1)))])
        groups1 = [[f1.index(c) for c in feats] for feats in feats6]  # stage bundles' own (one-hot) columns
        scores['K1 stage-scored'] = stage_score(e1_tr, e1_te, groups1)
        scores['K1 sensor-scored'] = sensor_score(e1_tr, e1_te)

        for target in TARGETS:
            for det, agg in scores.items():
                t = np.quantile(agg[y == 0], 1 - target)
                pred = (agg > t).astype(int)
                caught, p = per_event(pred, events, np.random.default_rng(seed))
                for i in range(len(caught)):
                    rows.append({'seed': seed, 'target_FAR': target, 'detector': det, 'event': i + 1,
                                 'caught': caught[i], 'p_chance': p[i], 'FAR': pred[y == 0].mean()})
        print(f'  seed {seed} done', flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    out['excess'] = out.caught - out.p_chance

    tot = out.groupby(['target_FAR', 'detector', 'seed']).agg(
        caught=('caught', 'sum'), chance=('p_chance', 'sum'), excess=('excess', 'sum')).reset_index()
    summ = tot.groupby(['target_FAR', 'detector']).agg(
        caught=('caught', 'mean'), chance=('chance', 'mean'), excess=('excess', 'mean'),
        excess_sd=('excess', 'std')).reset_index()
    summ['target_FAR'] = (summ['target_FAR'] * 100).round(2)
    pd.set_option('display.width', 250)
    print(f'\nAttacks caught beyond chance (of {len(events[0])}), mean over seeds 1-5')
    print(summ.pivot_table(index='target_FAR', columns='detector', values='excess').round(2).to_string())

    # Paired over attacks: each detector vs K1, per-event excess averaged over seeds, bootstrap over the attacks
    rng = np.random.default_rng(0)
    print(f'\nDifference vs K1 in attacks beyond chance; 95% CI bootstraps the {len(events[0])} attacks ({N_BOOT} resamples)')
    res = []
    for target in TARGETS:
        ev = out[out.target_FAR == target].groupby(['detector', 'event'])['excess'].mean().unstack(0)
        for det in ['K6', 'K6 sensor-scored', 'K1 stage-scored', 'K1 sensor-scored']:
            d = (ev[det] - ev['K1']).values
            boot = d[rng.integers(0, len(d), (N_BOOT, len(d)))].sum(1)
            res.append({'target_FAR %': round(target * 100, 2), 'detector': det, 'diff': d.sum(),
                        'CI_low': np.percentile(boot, 2.5), 'CI_high': np.percentile(boot, 97.5),
                        'P(diff<=0)': (boot <= 0).mean(),
                        'attacks_better': int((d > 0.1).sum()), 'attacks_worse': int((d < -0.1).sum())})
    print(pd.DataFrame(res).round(3).to_string(index=False))
    print(f'[SAVED] {OUT}')
