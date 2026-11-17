"""Simple label-free baselines on WADI and HAI 23.05 (reviewer-style Issue 6; SWaT: baselines_simple_label_free.py).

Same preprocessed files and fault rule as the PIVOT runners. Detectors:
  MinMax      a continuous sensor outside its training [min, max] widened by tol x range (Wolsing et al. 2022)
  Gradient    |x_t - x_(t-1)| of a continuous sensor above its training maximum
  PCA-SPE     PCA squared prediction error (95% of training variance), smoothed W = 25, dual threshold P99.9 / P99,
              W_dur = 50, causal, thresholds from the training score ('train') or, for HAI, from the first 900
              samples of the test stream ('commission'), the window the paper's HAI forecaster uses
Attacks caught are reported against the chance level, anywhere in the window and early (first half of the window).

Run from the repo root:  python baselines_simple_generalization.py
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '.')
import label_free as lfp
from baselines_simple_label_free import early_events
from evaluate_events import evaluate, to_binary

DATA = {'WADI': ('Datasets/wadi_train_preprocessed.csv', 'Datasets/wadi_test_preprocessed.csv'),
        'HAI': ('Datasets/hai_train_preprocessed.csv', 'Datasets/hai_test_preprocessed.csv')}
COMMISSION = 900
OUT = 'baselines_simple_generalization.csv'


def row(plant, name, p, y):
    s, _ = evaluate(y, p)
    ch, _ = lfp.chance_events(p, y, n=100)
    early, early_ch = early_events(p, y)
    ep = sum(1 for a, n in zip(*lfp.runs(p == 1)) if not y[a:a + n].any())
    return {'plant': plant, 'detector': name, 'F1': s['f1'], 'FAR': s['false_alarm'], 'events': s['n_detected'],
            'n_events': s['n_events'], 'chance': ch, 'beyond': s['n_detected'] - ch,
            'early': early, 'early_chance': early_ch, 'early_beyond': early - early_ch, 'FA_episodes': ep}


if __name__ == '__main__':
    rows = []
    for plant, (ftr, fte) in DATA.items():
        tr = pd.read_csv(ftr); tr.columns = tr.columns.str.strip()
        te = pd.read_csv(fte); te.columns = te.columns.str.strip()
        lab = 'Label' if 'Label' in te.columns else 'Normal/Attack'
        y = to_binary(te[lab])
        te, _ = lfp.fault_rule(tr, te)
        num = [c for c in tr.select_dtypes('number').columns if c in te.columns and c not in ('Normal/Attack', 'Label')]
        cont = [c for c in num if tr[c].nunique() > 3]
        fill = tr[num].mean()
        Xtr, Xte = tr[cont].fillna(fill[cont]), te[cont].fillna(fill[cont])
        print(f'{plant}: {len(num)} numeric columns, {len(cont)} continuous, {int(y.sum())} attack rows, '
              f'{len(lfp.runs(y == 1)[0])} attacks', flush=True)

        lo, hi = Xtr.min(), Xtr.max()
        for tol in (0.0, 0.05, 0.10):
            pad = tol * (hi - lo)
            p = ((Xte < lo - pad) | (Xte > hi + pad)).any(axis=1).astype(int).values
            rows.append(row(plant, f'MinMax (tol {tol:.0%})', p, y))
        p = (Xte.diff().abs() > Xtr.diff().abs().max()).any(axis=1).astype(int).values
        rows.append(row(plant, 'Gradient', p, y))

        mu, sd = tr[num].fillna(fill).mean(), tr[num].fillna(fill).std().replace(0, 1)
        Ztr, Zte = ((tr[num].fillna(fill) - mu) / sd).values, ((te[num].fillna(fill) - mu) / sd).values
        _, sv, vt = np.linalg.svd(Ztr[::5] - Ztr[::5].mean(0), full_matrices=False)
        k = int(np.searchsorted(np.cumsum(sv ** 2) / np.sum(sv ** 2), 0.95) + 1)
        P = vt[:k].T
        spe = lambda Z: ((Z - (Z @ P) @ P.T) ** 2).sum(1)
        s_tr, s_te = lfp.smooth(spe(Ztr), 25), lfp.smooth(spe(Zte), 25)
        for cal_name, cal in (('train', s_tr), ('commission', s_te[:COMMISSION])):
            if cal_name == 'commission' and plant != 'HAI':
                continue
            p = lfp.dual_threshold(s_te, np.percentile(cal, 99.9), np.percentile(cal, 99.0), 50).astype(int)
            rows.append(row(plant, f'PCA-SPE ({k} comp., {cal_name})', p, y))

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    out['F1'] = (out['F1'] * 100).round(1); out['FAR'] = (out['FAR'] * 100).round(2)
    pd.set_option('display.width', 250)
    print(out.round(2).to_string(index=False))
    print(f'[SAVED] {OUT}')
