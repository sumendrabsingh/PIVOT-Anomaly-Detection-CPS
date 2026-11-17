"""Simple label-free baselines on SWaT, on the same features and drift handling as PIVOT (reviewer-style Issue 4).

Table II compares PIVOT with monolithic deep models that keep the drifting analyzers (or were trained on a different
40-sensor file), so they alarm on most of the test week. Every detector here gets PIVOT's treatment instead:
the preprocessed features, the fault rule, analyzers removed from the model and, in the '+ analyzer arm' rows,
OR-ed with the same analyzer change detector. All thresholds come from the training file.

  MinMax          a continuous sensor outside its training [min, max] widened by tol x range
                  (Wolsing et al., ESORICS 2022, "SIMPLE")
  Gradient        |x_t - x_(t-1)| of a continuous sensor above its training maximum x (1 + tol) (same paper)
  PCA-SPE         squared prediction error of a PCA model keeping 95% of training variance (Kravchik & Shabtai style)
  AE per-sensor   the monolithic lambda = 0 autoencoder (Oracles/swat_mono_noAIT_none_seed<k>), each sensor's
                  squared error standardized on its training errors, max over sensors (decomposition_mechanism.py)
  PIVOT array     the six stage autoencoders (Oracles/swat_phys_none_seed<k>)
Score-based detectors use PIVOT's rule: smoothed W = 25, dual threshold P99.9 / P99 of the training score,
W_dur = 50, causal. Autoencoders: seeds 1-5 (mean). Attacks caught are reported against the chance level, both
anywhere in the attack window and early (an alarm in its first half; a late alarm is often the attack ending).

Run from the repo root:  python baselines_simple_label_free.py
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '.')
import decomposition_ablation_label_free as dab
import decomposition_mechanism as dm
import label_free as lfp
import run_generalized_subsystems as gen
from evaluate_events import evaluate, to_binary

OUT = 'baselines_simple_label_free_swat.csv'
EARLY = 0.5  # 'caught early': an alarm in the first half of the attack window


def dual_from_train(s_tr, s_te):
    return lfp.dual_threshold(s_te, np.percentile(s_tr, 99.9), np.percentile(s_tr, 99.0), 50).astype(int)


def early_events(pred, y, frac=EARLY, n_shifts=100, seed=0):
    """Attacks with an alarm in the first `frac` of their window, and the same count for the alarm pattern
    circularly shifted to random offsets (its chance level). An alarm late in the window is often the attacker
    restoring the true value, i.e. the attack ending."""
    s, length = lfp.runs(y == 1)
    win = [(a, max(1, int(np.ceil(n * frac)))) for a, n in zip(s, length)]
    hit = lambda p: sum(p[a:a + n].any() for a, n in win)
    rng = np.random.default_rng(seed)
    margin = max(1, len(pred) // 20)
    return hit(pred), float(np.mean([hit(np.roll(pred, rng.integers(margin, len(pred) - margin)))
                                     for _ in range(n_shifts)]))


def score_row(name, pred, y, arm, seed=None):
    out = []
    for tag, p in ((name, pred), (name + ' + analyzer arm', np.maximum(pred, arm))):
        s, _ = evaluate(y, p)
        ch, _ = lfp.chance_events(p, y, n=100)
        ep = sum(1 for a, n in zip(*lfp.runs(p == 1)) if not y[a:a + n].any())
        early, early_ch = early_events(p, y)
        out.append({'detector': tag, 'seed': seed, 'F1': s['f1'], 'FAR': s['false_alarm'],
                    'events': s['n_detected'], 'chance': ch, 'excess': s['n_detected'] - ch, 'FA_episodes': ep,
                    'early': early, 'early_chance': early_ch, 'early_excess': early - early_ch})
    return out


if __name__ == '__main__':
    tr = pd.read_csv(dab.TRAIN); tr.columns = tr.columns.str.strip()
    te = pd.read_csv(dab.TEST); te.columns = te.columns.str.strip()
    y = to_binary(te['Normal/Attack'])
    te, _ = lfp.fault_rule(tr, te)
    fill = tr.select_dtypes('number').mean()
    ait = sorted(c for c in te.columns if lfp.is_analyzer(c))
    arm = lfp.analyzer_arm_predict(tr, te, ait, w_dur=50).astype(int)

    _, ck = gen._load_subsystem_model(f'{dab.model_dir(1, "split", 1)}/subsystem_Global.pth')
    feats = ck['features']                                   # non-analyzer columns, valves one-hot
    cont = [c for c in feats if tr[c].nunique() > 3]        # continuous sensors
    Xtr, Xte = tr[cont].fillna(fill[cont]), te[cont].fillna(fill[cont])
    print(f'{len(feats)} model columns, {len(cont)} continuous sensors, {len(ait)} analyzers')

    rows = []
    lo, hi = Xtr.min(), Xtr.max()
    for tol in (0.0, 0.05, 0.10):
        pad = tol * (hi - lo)
        pred = ((Xte < lo - pad) | (Xte > hi + pad)).any(axis=1).astype(int).values
        rows += score_row(f'MinMax (tol {tol:.0%} of range)', pred, y, arm)
    gmax = Xtr.diff().abs().max()
    for tol in (0.0, 0.25, 0.50):
        pred = (Xte.diff().abs() > gmax * (1 + tol)).any(axis=1).astype(int).values
        rows += score_row(f'Gradient (tol {tol:.0%})', pred, y, arm)

    # PCA squared prediction error on the standardized model columns
    mu, sd = tr[feats].fillna(fill[feats]).mean(), tr[feats].fillna(fill[feats]).std().replace(0, 1)
    Ztr = ((tr[feats].fillna(fill[feats]) - mu) / sd).values
    Zte = ((te[feats].fillna(fill[feats]) - mu) / sd).values
    _, sv, vt = np.linalg.svd(Ztr[::5] - Ztr[::5].mean(0), full_matrices=False)
    k = int(np.searchsorted(np.cumsum(sv ** 2) / np.sum(sv ** 2), 0.95) + 1)
    P = vt[:k].T
    spe = lambda Z: ((Z - (Z @ P) @ P.T) ** 2).sum(1)  # Z is standardized on training: already centred
    s_tr, s_te = lfp.smooth(spe(Ztr), gen.WINDOW_SIZE), lfp.smooth(spe(Zte), gen.WINDOW_SIZE)
    rows += score_row(f'PCA-SPE ({k} components)', dual_from_train(s_tr, s_te), y, arm)

    stages6 = dab.partition(6, 'split')
    for seed in dab.SEEDS:
        f1, e_tr = dm.sensor_errors(dab.model_dir(1, 'split', seed), 'Global', tr, fill)
        _, e_te = dm.sensor_errors(dab.model_dir(1, 'split', seed), 'Global', te, fill)
        rows += score_row('AE per-sensor (monolithic)', dual_from_train(dm.sensor_score(e_tr, e_tr),
                                                                        dm.sensor_score(e_tr, e_te)), y, arm, seed)
        rows += score_row('AE (monolithic, as in paper)', dual_from_train(
            dm.stage_score(e_tr, e_tr, [list(range(len(f1)))]), dm.stage_score(e_tr, e_te, [list(range(len(f1)))])),
            y, arm, seed)
        etr6, ete6, groups, start = [], [], [], 0
        for name in stages6:
            f, a = dm.sensor_errors(dab.model_dir(6, 'split', seed), name, tr, fill)
            _, b = dm.sensor_errors(dab.model_dir(6, 'split', seed), name, te, fill)
            etr6.append(a); ete6.append(b); groups.append(list(range(start, start + len(f)))); start += len(f)
        etr6, ete6 = np.hstack(etr6), np.hstack(ete6)
        rows += score_row('PIVOT array (6 stages)', dual_from_train(dm.stage_score(etr6, etr6, groups),
                                                                   dm.stage_score(etr6, ete6, groups)), y, arm, seed)
        print(f'  seed {seed} done', flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    summ = out.groupby('detector', sort=False).agg(
        n=('F1', 'size'), F1=('F1', 'mean'), FAR=('FAR', 'mean'), caught=('events', 'mean'), chance=('chance', 'mean'),
        beyond=('excess', 'mean'), beyond_sd=('excess', 'std'), FA_episodes=('FA_episodes', 'mean'),
        early=('early', 'mean'), early_chance=('early_chance', 'mean'), early_beyond=('early_excess', 'mean'))
    summ['F1'] = (summ['F1'] * 100).round(1); summ['FAR'] = (summ['FAR'] * 100).round(2)
    pd.set_option('display.width', 250)
    print(summ.round(2).to_string())
    print(f'[SAVED] {OUT}')
