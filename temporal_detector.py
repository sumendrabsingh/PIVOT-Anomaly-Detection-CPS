"""
PIVOT v2 -- WS3b: windowed (sequence) reconstruction detector for stealthy attacks.

Instantaneous detectors fail on HAI because its attacks are stealthy setpoint manipulations that
stay inside the normal operating envelope of every sensor (best single sensor separates only 5.5%
of attack timesteps; see RESEARCH_PLAN_V2 sec 9.2). The attack signal is not in any instantaneous
value but in the *temporal dynamics* -- how the multivariate state evolves. Linear VAR models
recover only ~6/14 events at 1% FAR, so the dynamics are nonlinear.

This detector reconstructs a sliding WINDOW of W consecutive standardized timesteps with an
autoencoder trained on normal data. An attack that forces the plant onto a trajectory it never
takes in normal operation -- even one whose individual samples look normal -- produces a window
the AE cannot reconstruct. Same detection wrapper as the other arms (standardize, W=25 smooth,
per-normal P99.x threshold), so it composes with the physics and instantaneous-AE arms.

Usage:
  python temporal_detector.py --dataset hai --window 60 --train
  python temporal_detector.py --dataset hai --window 60           # eval (loads saved model)
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import label_free as lfp
from evaluate_events import to_binary, evaluate, print_report, find_events

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PATHS = {
    'hai':  (r'Datasets/hai_train_preprocessed.csv',  r'Datasets/hai_test_preprocessed.csv'),
    'swat': (r'Datasets/swat_train_preprocessed.csv', r'Datasets/swat_test_preprocessed.csv'),
    'wadi': (r'Datasets/wadi_train_preprocessed.csv', r'Datasets/wadi_test_preprocessed.csv'),
}


class Forecaster(nn.Module):
    """Nonlinear one-step forecaster: predict the next state vector (d) from a flattened window
    of the previous W states (W*d). A reconstruction AE copies its input, so it reconstructs a
    self-consistent stealthy-attack window faithfully; a forecaster must predict the *future*
    from the past, so an attack that bends the trajectory raises the residual even when every
    individual value stays in range."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        h1, h2 = 512, 256
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(h2, h1), nn.ReLU(),
            nn.Linear(h1, out_dim))

    def forward(self, x):
        return self.net(x)


def make_pairs(Z, W, stride=1):
    """(N,d) series -> (past window Z[i:i+W] flattened, next state Z[i+W]) forecasting pairs."""
    n, d = Z.shape
    idx = np.arange(0, n - W, stride)
    Xp = np.empty((len(idx), W * d), dtype=np.float32)
    Yn = np.empty((len(idx), d), dtype=np.float32)
    for j, i in enumerate(idx):
        Xp[j] = Z[i:i + W].ravel()
        Yn[j] = Z[i + W]
    return Xp, Yn


def per_dim_sq_error(model, Z, W, batch=2048):
    """Per-dimension squared one-step forecast error, shape (n, d).

    Localized attacks touch a few of the d sensors; averaging the error over all d dilutes them
    (mean-error forecasting caught only 3/14 HAI events). We return the full per-dimension error
    so the caller can standardize each dimension and MAX-aggregate -- the signal-isolation trick
    from the original PIVOT paper, here on temporal residuals."""
    n, d = Z.shape
    starts = np.arange(0, n - W)  # window Z[s:s+W] predicts Z[s+W]
    E = np.zeros((n, d), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(starts), batch):
            bs = starts[i:i+batch]
            wb = np.empty((len(bs), W * d), dtype=np.float32)
            for j, s in enumerate(bs):
                wb[j] = Z[s:s+W].ravel()
            xb = torch.from_numpy(wb).to(DEVICE)
            tgt = torch.from_numpy(Z[bs + W]).to(DEVICE)
            e = ((tgt - model(xb)) ** 2).cpu().numpy()
            E[bs + W] = e
    E[:W] = E[W]
    return E


def aggregate_score(E_tr, E_te, mode='max'):
    """Standardize each dimension's error on the normal (train) period, then aggregate over dims.
    max = signal isolation (localized anomalies survive); mean = the diluted baseline."""
    mu = E_tr.mean(0); sd = E_tr.std(0) + 1e-9
    Ztr = (E_tr - mu) / sd
    Zte = (E_te - mu) / sd
    if mode == 'max':
        return Ztr.max(1), Zte.max(1)
    return Ztr.mean(1), Zte.mean(1)


def main():
    ap = argparse.ArgumentParser(description="PIVOT v2 windowed reconstruction detector")
    ap.add_argument('--dataset', choices=list(PATHS), default='hai')
    ap.add_argument('--window', type=int, default=60)
    ap.add_argument('--bottleneck', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--stride', type=int, default=5, help='training-window stride (speed)')
    ap.add_argument('--seed', type=int, default=4)
    ap.add_argument('--train', action='store_true')
    ap.add_argument('--combine', default=None, help='AE-arm predictions CSV to OR with')
    ap.add_argument('--model', default=None)
    ap.add_argument('--calibration', choices=['commission', 'train', 'oracle'], default='commission',
                    help="'commission' (default): thresholds from the first --commission_s samples of the "
                         "test stream, a window fixed in advance (label-free). 'train': from training "
                         "scores. 'oracle': from the samples before the first attack LABEL (leaks labels)")
    ap.add_argument('--export', default=None, help='predictions CSV (default preds_<dataset>_temporal.csv)')
    ap.add_argument('--commission_s', type=int, default=900,
                    help='commissioning window length in samples (default 900 = 15 min at 1 Hz)')
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    train_path, test_path = PATHS[args.dataset]
    model_path = args.model or f'Oracles/temporal_{args.dataset}_W{args.window}.pth'

    tr = pd.read_csv(train_path); tr.columns = tr.columns.str.strip()
    te = pd.read_csv(test_path); te.columns = te.columns.str.strip()
    label_col = 'Label' if 'Label' in te.columns else 'Normal/Attack'
    y = to_binary(te[label_col])
    if args.calibration != 'oracle':
        te, fired_faults = lfp.fault_rule(tr, te)
        if fired_faults:
            print(f"[FAULT RULE] replaced impossible values in: {list(fired_faults)}")
    cols = [c for c in te.columns if c not in ('Normal/Attack', 'Label', 'Timestamp')]

    Xtr = tr[cols].values.astype(np.float32)
    Xte = te[cols].values.astype(np.float32)
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd < 1e-9] = 1.0
    Ztr = (Xtr - mu) / sd
    Zte = (Xte - mu) / sd
    d = len(cols); W = args.window; in_dim = W * d

    if args.train or not os.path.exists(model_path):
        print(f"[TRAIN] {args.dataset} forecaster  W={W} d={d} in_dim={in_dim}")
        Xp, Yn = make_pairs(Ztr, W, stride=args.stride)
        Xp = torch.tensor(Xp); Yn = torch.tensor(Yn)
        model = Forecaster(in_dim, d).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        lossf = nn.MSELoss()
        bs = 1024
        n = len(Xp)
        for ep in range(args.epochs):
            model.train(); perm = torch.randperm(n)
            tot = 0.0
            for i in range(0, n, bs):
                idx = perm[i:i+bs]
                xb = Xp[idx].to(DEVICE); yb = Yn[idx].to(DEVICE)
                opt.zero_grad(); loss = lossf(model(xb), yb)
                loss.backward(); opt.step(); tot += loss.item() * len(xb)
            if ep % 5 == 0 or ep == args.epochs - 1:
                print(f"  epoch {ep:>3}  train MSE {tot/n:.5f}")
        os.makedirs('Oracles', exist_ok=True)
        torch.save({'state': model.state_dict(), 'in_dim': in_dim, 'out_dim': d,
                    'window': W, 'cols': cols, 'mu': mu, 'sd': sd}, model_path)
        print(f"  [SAVED] {model_path}")
    else:
        ck = torch.load(model_path, map_location=DEVICE, weights_only=False)
        model = Forecaster(ck['in_dim'], ck['out_dim']).to(DEVICE)
        model.load_state_dict(ck['state'])
        print(f"[LOAD] {model_path}")

    # per-dimension forecast error -> per-dim-standardized MAX aggregation (signal isolation for
    # localized attacks; mean-aggregation dilutes them, see the header).
    E_tr = per_dim_sq_error(model, Ztr, W)
    E_te = per_dim_sq_error(model, Zte, W)
    s_tr, s_te = aggregate_score(E_tr, E_te, 'max')
    sm = pd.Series(s_te).rolling(25, min_periods=1).mean().values
    ev = find_events(y)

    # DEPLOYMENT calibration: HAI's normal dynamics drift across the test period, so a
    # train-calibrated threshold does not transfer (40% FAR). Standard ICS practice is to calibrate
    # on a commissioning window of the deployment stream, whose length is fixed in advance -- the
    # previous version ended it at the first attack LABEL, which leaks labels (now 'oracle').
    if args.calibration == 'commission':
        cal = sm[:args.commission_s]
    elif args.calibration == 'train':
        cal = pd.Series(s_tr).rolling(25, min_periods=1).mean().values
    else:
        print("[WARNING] --calibration oracle ends the calibration window at the first attack label.")
        first_attack = int(np.argmax(y == 1)) if y.any() else len(y)
        cal = sm[:first_attack] if first_attack > 100 else sm[y == 0]
    print(f"\n{args.dataset.upper()} forecaster (W={W}, max-agg) -- calibration={args.calibration} "
          f"on {len(cal)} samples")
    best = None
    for pct in [99.5, 99.0, 98.0]:
        thr = np.percentile(cal, pct)
        pred = (sm > thr).astype(int)
        m, _ = evaluate(y, pred)
        print(f"  P{pct}: event recall {m['event_recall']:.0%} "
              f"({m['n_detected']}/{m['n_events']}), pt-recall {m['recall']:.1%}, "
              f"FAR {m['false_alarm']:.2%}, FA-episodes {m['false_alarm_events']}, "
              f"median TTD {m['ttd_median_s']:.0f}s")
        if pct == 99.0:
            best = pred
    export = args.export or f'preds_{args.dataset}_temporal.csv'
    pd.DataFrame({'y_true': y, 'y_pred': best}).to_csv(export, index=False)
    print(f"\n[SAVED] {export} (calibrated P99.0)")

    summary, events = evaluate(y, best)
    print_report(summary, events, f"{args.dataset.upper()} FORECASTER (calibrated P99.0)")

    if args.combine and os.path.exists(args.combine):
        ae = pd.read_csv(args.combine)['y_pred'].values
        union = np.maximum(ae, best)
        s_ae, _ = evaluate(y, ae); s_un, _ = evaluate(y, union)
        added = [i+1 for i, (s, e) in enumerate(ev) if ae[s:e+1].any() == 0 and union[s:e+1].any()]
        print(f"\n  AE alone     : event recall {s_ae['event_recall']:.1%} ({s_ae['n_detected']}/{s_ae['n_events']})")
        print(f"  AE + temporal: event recall {s_un['event_recall']:.1%} ({s_un['n_detected']}/{s_un['n_events']})")
        print(f"  temporal adds events: {added}")


if __name__ == "__main__":
    main()
