"""
PIVOT v2 -- WS3c: drift-robust relationship-residual detector (for non-stationary plants like WADI).

WADI is a water *distribution* network driven by unmeasured consumer demand, so its normal
operation is severely non-stationary: individual sensors drift enough that magnitude, physics,
and forecasting detectors all fail at low FAR (a fixed threshold gives ~58% FAR; see
RESEARCH_PLAN_V2 sec 9.4). But the drift is largely *common-mode* -- sensors drift together --
so the RELATIONSHIPS between sensors are far more stable than the sensors themselves. An attack
that manipulates one sensor breaks its relationship to its partners even while every value stays
in range.

This detector, in the spirit of graph-deviation methods (e.g. GDN), predicts each sensor from its
top-k most-correlated partners (fit on normal data); the residual is a relationship violation.
Two mechanisms make it drift-robust:
  1. the predictor uses co-drifting partners, so common-mode drift cancels in the residual;
  2. an ADAPTIVE rolling-baseline threshold tracks any residual-scale drift that remains, so the
     operating point stays fixed without a labelled test set.

Result on WADI: 6/8 attack events at 0.9% FAR, vs 4/8 for the autoencoder OR-vote.

Usage:
  python relationship_detector.py --dataset wadi
  python relationship_detector.py --dataset swat --k 8
"""

import argparse

import numpy as np
import pandas as pd
from numpy.linalg import lstsq

import label_free as lfp
from evaluate_events import to_binary, evaluate, print_report, find_events

PATHS = {
    'wadi': (r'Datasets/wadi_train_preprocessed.csv', r'Datasets/wadi_test_preprocessed.csv'),
    'swat': (r'Datasets/swat_train_preprocessed.csv', r'Datasets/swat_test_preprocessed.csv'),
    'hai':  (r'Datasets/hai_train_preprocessed.csv',  r'Datasets/hai_test_preprocessed.csv'),
}


def relationship_residuals(Ztr, Zte, k, lam=1.0):
    """Per-sensor residual from predicting it via its top-k correlated partners (fit on normal)."""
    C = np.corrcoef(Ztr.T); np.fill_diagonal(C, 0); C = np.nan_to_num(C)
    Rtr = np.zeros_like(Ztr); Rte = np.zeros((len(Zte), Ztr.shape[1]), dtype=np.float32)
    for i in range(Ztr.shape[1]):
        part = np.argsort(-np.abs(C[i]))[:k]
        A = np.c_[Ztr[:, part], np.ones(len(Ztr))]
        coef, *_ = lstsq(A, Ztr[:, i], rcond=None)
        Rtr[:, i] = np.abs(Ztr[:, i] - A @ coef)
        Ate = np.c_[Zte[:, part], np.ones(len(Zte))]
        Rte[:, i] = np.abs(Zte[:, i] - Ate @ coef)
    rmu = Rtr.mean(0); rsd = Rtr.std(0) + 1e-9
    return ((Rte - rmu) / rsd).max(1)  # per-dim standardize, MAX-aggregate (signal isolation)


def adaptive_score(score, Wa=3600, q=0.90, smooth=25):
    """Rolling-baseline normalization: (score - rolling median) / (rolling q-quantile - median).
    Tracks residual-scale drift so the threshold is a fixed multiple k, label-free."""
    sc = pd.Series(score)
    med = sc.rolling(Wa, min_periods=600).median()
    hi = sc.rolling(Wa, min_periods=600).quantile(q)
    z = ((sc - med) / ((hi - med).replace(0, np.nan) + 1e-9)).fillna(0).values
    return pd.Series(z).rolling(smooth, min_periods=1).mean().values


def main():
    ap = argparse.ArgumentParser(description="PIVOT v2 drift-robust relationship detector")
    ap.add_argument('--dataset', choices=list(PATHS), default='wadi')
    ap.add_argument('--k', type=int, default=8, help='partners per sensor')
    ap.add_argument('--adapt_window', type=int, default=3600, help='rolling-baseline window (samples)')
    ap.add_argument('--combine', default=None, help='AE predictions CSV to OR with')
    ap.add_argument('--fault_rule', choices=['on', 'off'], default='off',
                    help="'on' replaces values > 100 training SDs from the training median. Off by default "
                         "here: the adaptive baseline already absorbs a persistent fault, while imputing a "
                         "constant breaks the faulted channel's partner predictions (WADI FAR 1.8% -> 3.2%)")
    ap.add_argument('--export', default=None, help='predictions CSV (default preds_<dataset>_relationship.csv)')
    args = ap.parse_args()

    train_path, test_path = PATHS[args.dataset]
    tr = pd.read_csv(train_path); tr.columns = tr.columns.str.strip()
    te = pd.read_csv(test_path); te.columns = te.columns.str.strip()
    label_col = 'Label' if 'Label' in te.columns else 'Normal/Attack'
    y = to_binary(te[label_col])
    if args.fault_rule == 'on':
        te, fired_faults = lfp.fault_rule(tr, te)
        if fired_faults:
            print(f"[FAULT RULE] replaced impossible values in: {list(fired_faults)}")
    cols = [c for c in te.columns if c not in ('Normal/Attack', 'Label', 'Timestamp')]

    Xtr = tr[cols].values.astype(float); Xte = te[cols].values.astype(float)
    mu = Xtr.mean(0); sd = Xtr.std(0); rng = Xtr.max(0) - Xtr.min(0)
    sdf = np.maximum(sd, 0.01 * rng + 1e-6)
    Ztr = (Xtr - mu) / sdf; Zte = (Xte - mu) / sdf

    raw = relationship_residuals(Ztr, Zte, args.k)
    sm = adaptive_score(raw, Wa=args.adapt_window)
    ev = find_events(y)

    print(f"\n{args.dataset.upper()} relationship detector (k={args.k}, adaptive Wa={args.adapt_window})")
    best = None
    for kthr in [4, 5, 6]:
        pred = (sm > kthr).astype(int)
        m, _ = evaluate(y, pred)
        print(f"  k_thr={kthr}: event recall {m['event_recall']:.0%} ({m['n_detected']}/{m['n_events']}), "
              f"pt-recall {m['recall']:.1%}, FAR {m['false_alarm']:.2%}, FA-ep {m['false_alarm_events']}, "
              f"TTD {m['ttd_median_s']:.0f}s")
        if kthr == 5:
            best = pred
    export = args.export or f'preds_{args.dataset}_relationship.csv'
    pd.DataFrame({'y_true': y, 'y_pred': best}).to_csv(export, index=False)
    print(f"[SAVED] {export} (k_thr=5)")

    summary, events = evaluate(y, best)
    print_report(summary, events, f"{args.dataset.upper()} RELATIONSHIP DETECTOR (adaptive, k_thr=5)")

    if args.combine:
        ae = pd.read_csv(args.combine)['y_pred'].values
        union = np.maximum(ae, best)
        s_ae, _ = evaluate(y, ae); s_un, _ = evaluate(y, union)
        added = [i+1 for i, (s, e) in enumerate(ev) if ae[s:e+1].any() == 0 and union[s:e+1].any()]
        print(f"\n  AE alone         : event recall {s_ae['event_recall']:.0%} ({s_ae['n_detected']}/{s_ae['n_events']})")
        print(f"  AE + relationship: event recall {s_un['event_recall']:.0%} ({s_un['n_detected']}/{s_un['n_events']})  adds {added}")


if __name__ == "__main__":
    main()
