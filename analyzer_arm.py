"""Analyzer arm for SWaT: a label-free abrupt-change detector on the chemical analyzers, OR-combined
with the subsystem array retrained without them (Oracles/swat_seed4_noAIT).

Why: analyzers drift for days (AIT201 sits 10.9 training SDs off in the test week), which breaks any
reconstruction model that includes them; dropping them loses attacks 5, 15, 23, 24, 32. Drift is slow
and attacks are fast, so the arm looks only at short-horizon change. Parameters fixed in advance:

  r_t   = x_t - median(x over [t - GAP - H, t - GAP)),   H = 30 min, GAP = 5 min   (causal)
  z_t   = r_t standardized by the training residual mean / SD, per channel
  score = max over channels of |z|, smoothed W = 25
  alarm = dual threshold, tau_H = P99.9, tau_L = P99 of the training score, W_dur = 50, causal

Iteration 2 (chosen after iteration 1 showed AIT201 and AIT503 are 3-4x noisier in the test week):
  z_t   = r_t / max(1.4826 * MAD of r over [t - GAP - HS, t - GAP), training SD of r),  HS = 6 h

Run from the repo root:  python analyzer_arm.py
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '.')
import label_free_dual_threshold as lf
import label_free_drift_adaptive as da
from evaluate_events import evaluate

H, GAP, HS = 30 * 60, 5 * 60, 6 * 3600
LOST_BY_NOAIT = [5, 15, 23, 24, 32]


def change_scores(df, cols, ref=None):
    r = pd.DataFrame({c: df[c] - df[c].shift(GAP).rolling(H, min_periods=H // 4).median() for c in cols}).fillna(0.0)
    if ref is None:
        ref = (r.mean(), r.std() + 1e-6)
    z = ((r - ref[0]) / ref[1]).abs()
    return z, ref


def scale_adapted(df, cols, ref):
    """Iteration 2: divide by a causal robust scale, floored at the training SD."""
    r = pd.DataFrame({c: df[c] - df[c].shift(GAP).rolling(H, min_periods=H // 4).median() for c in cols}).fillna(0.0)
    past = r.shift(GAP)
    mad = (past - past.rolling(HS, min_periods=HS // 4).median()).abs().rolling(HS, min_periods=HS // 4).median() * 1.4826
    scale = np.maximum(mad.fillna(0.0).values, ref[1].values)
    return pd.DataFrame(np.abs(r.values - ref[0].values) / scale, columns=cols)


def score(z):
    return pd.Series(z.max(axis=1).values).rolling(lf.W, min_periods=1).mean().values


def detected_events(lab, pred):
    m = np.concatenate([[0], lab, [0]]); d = np.diff(m)
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    return {i + 1 for i, (a, b) in enumerate(zip(s, e)) if pred[a:b].any()}


if __name__ == '__main__':
    te = pd.read_csv(lf.TEST); te.columns = te.columns.str.strip()
    tr = pd.read_csv(lf.TRAIN); tr.columns = tr.columns.str.strip()
    labels = pd.read_csv('swat_test_labels_recovery.csv')
    y, y_rec = labels['y_original'].values, labels['y_recovery'].values
    ait = sorted(c for c in te.columns if c.startswith('AIT'))
    print('analyzer channels:', ait)

    z_tr, ref = change_scores(tr, ait)
    z_te, _ = change_scores(te, ait, ref)
    a_tr, a_te = score(z_tr), score(z_te)
    th, tl = np.percentile(a_tr, 99.9), np.percentile(a_tr, 99.0)
    arm = lf.dual(a_te, th, tl, 50, causal=True)

    agg, cal = da.scores(da.NOAIT_DIR, te, tr)
    array = lf.dual(agg, np.percentile(cal, 99.9), np.percentile(cal, 99.0), 50, causal=True)
    union = np.maximum(array, arm)
    zs_tr, zs_te = scale_adapted(tr, ait, ref), scale_adapted(te, ait, ref)
    s_tr, s_te = score(zs_tr), score(zs_te)
    arm2 = lf.dual(s_te, np.percentile(s_tr, 99.9), np.percentile(s_tr, 99.0), 50, causal=True)
    union2 = np.maximum(array, arm2)

    rows = []
    for tag, pr in (('Array without analyzers', array), ('Analyzer arm alone', arm), ('Union (OR)', union),
                    ('Analyzer arm, scale-adapted', arm2), ('Union, scale-adapted arm', union2)):
        for lab_name, lab in (('original', y), ('recovery', y_rec)):
            s, _ = evaluate(lab, pr)
            r, L = lf.runs((pr == 1) & (lab == 0))
            rows.append({'detector': tag, 'labels': lab_name, 'F1': s['f1'], 'Prec': s['precision'],
                         'Rec': s['recall'], 'FAR': s['false_alarm'], 'events': f"{s['n_detected']}/{s['n_events']}",
                         'FA_episodes': s['false_alarm_events'], 'longest_FA_s': int(L.max()) if len(L) else 0,
                         'TTD_med_s': s.get('ttd_median_s', np.nan)})
    out = pd.DataFrame(rows)
    fmt = out.copy()
    for c in ('F1', 'Prec', 'Rec', 'FAR'):
        fmt[c] = (fmt[c] * 100).map('{:.2f}'.format)
    pd.set_option('display.width', 250)
    print(fmt.to_string(index=False))

    m = np.concatenate([[0], y, [0]]); d = np.diff(m); st, en = np.where(d == 1)[0], np.where(d == -1)[0]
    rng = np.random.default_rng(0)
    for tag, a in (('iteration 1', arm), ('iteration 2', arm2)):
        hits = [sum(np.roll(a, rng.integers(20000, len(a) - 20000))[p:q].any() for p, q in zip(st, en)) for _ in range(200)]
        print(f'{tag}: attacks hit {sum(a[p:q].any() for p, q in zip(st, en))} | by chance (shifted alarms): '
              f'mean {np.mean(hits):.1f}, 95th pct {np.percentile(hits, 95):.0f}')
    arm = arm2
    ev_arr, ev_arm = detected_events(y, array), detected_events(y, arm)
    print(f'\nattacks the arm catches: {sorted(ev_arm)}')
    print(f'recovers the analyzer attacks lost by dropping AITs {LOST_BY_NOAIT}: '
          f'{sorted(set(LOST_BY_NOAIT) & ev_arm)} | new beyond array: {sorted(ev_arm - ev_arr)}')
    fa = (arm == 1) & (y == 0)
    blame = pd.Series(np.array(ait)[zs_te.values[fa].argmax(axis=1)]).value_counts() if fa.any() else pd.Series(dtype=int)
    print('arm false-alarm samples by driving channel:', blame.to_dict())
    out.to_csv('analyzer_arm_swat.csv', index=False)
    print('[SAVED] analyzer_arm_swat.csv')
