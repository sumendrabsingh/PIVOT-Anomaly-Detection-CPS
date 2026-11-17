"""
PIVOT v2 -- WS3: model-free physics-residual detector.

The autoencoder OR-vote misses ~17 SWaT attacks in which sensors move 9-16 sigma but the AE
reconstructs the anomaly faithfully and stays quiet (see the Phase-A diagnosis). A physics
residual does not reconstruct -- it checks whether the sensor values are physically consistent,
so it fires on exactly those events. Two invariant families, both fit / calibrated on normal
data only:

  mass balance:  d(level)/dt should equal (a*inflow + b*outflow + c); the fitted residual flags
                 tanks that fill or drain without a matching flow.
  actuator-flow interlock: a closed valve / stopped pump implies its flow is ~0; a large flow
                 while the actuator is off is a violation.

Each residual is standardized by its normal-period statistics, smoothed over W=25, OR-combined,
and thresholded at P99.8 of the TRAINING residuals (label-free; --calibration oracle uses test rows
labelled normal, the leaky v2 protocol, for reference only) -- the same detection wrapper the AE arm
uses, so the two are directly composable.

Usage:
  python physics_residual_detector.py                      # physics arm alone
  python physics_residual_detector.py --combine preds_swat_v1_orvoting.csv   # union with an AE arm
"""

import argparse
import json

import numpy as np
import pandas as pd
from numpy.linalg import lstsq

import label_free as lfp
from evaluate_events import to_binary, evaluate, print_report

WINDOW = 25


def _valve_closed_indicator(df, actuator):
    """Return a per-row 'actuator is off/closed' mask in [0,1].

    Valves are one-hot (MV101_0/1/2 -> state 0/1/2, state 1 == closed in SWaT's encoding);
    pumps are a single 0/1 column (0 == off)."""
    onehot = [c for c in df.columns if c.startswith(actuator + '_')]
    if onehot:  # valve: closed == the '_1' class
        col = f"{actuator}_1"
        return df[col].values if col in df.columns else np.zeros(len(df))
    if actuator in df.columns:  # pump: off == value near 0
        v = df[actuator].values
        return (v <= (v.min() + 0.5 * (np.median(v) - v.min() + 1e-9))).astype(float)
    return np.zeros(len(df))


def auto_derive_invariants(tr, min_mb_corr=0.15, min_interlock=0.7):
    """Discover invariants from normal data alone (portable, no hand-tuning).

    mass_balance: for each level sensor, take the flow with the most positive correlation to
      d(level)/dt as inflow and the most negative as outflow, if either clears min_mb_corr.
    actuator_flow_interlock: for each (binary actuator, flow) pair, keep it if, on normal data,
      'actuator off' implies 'flow near zero' for at least min_interlock of off-samples -- i.e.
      the actuator genuinely gates the flow.
    """
    cols = [c for c in tr.columns if c not in ('Normal/Attack', 'Label', 'Timestamp')]
    up = {c: c.upper() for c in cols}
    levels = [c for c in cols if any(k in up[c] for k in ('LIT', 'LT_', '_LT', 'LEVEL'))]
    flows = [c for c in cols if any(k in up[c] for k in ('FIT', 'FT0', 'FT_', '_FT', 'FLOW'))]
    actuators = [c for c in cols if any(k in up[c] for k in ('MV', 'P_', '_P_', 'PUMP', 'STATUS', 'PCV', 'FCV', 'LCV'))]

    mass_balance = []
    for lit in levels:
        d = np.diff(tr[lit].values, prepend=tr[lit].values[0])[1:]
        cors = []
        for f in flows:
            c = np.corrcoef(d, tr[f].values[1:])[0, 1]
            if not np.isnan(c):
                cors.append((f, c))
        if not cors:
            continue
        inflow = max(cors, key=lambda x: x[1])
        outflow = min(cors, key=lambda x: x[1])
        if max(abs(inflow[1]), abs(outflow[1])) < min_mb_corr:
            continue
        mass_balance.append({'tank': lit, 'inflow': inflow[0], 'outflow': outflow[0]})

    interlocks = []
    for act in actuators:
        v = tr[act].values
        lo = v.min()
        off = v <= lo + 1e-6  # binary/status actuators sit at their min when off
        if off.sum() < 50 or off.mean() > 0.98:  # need a real 'off' population
            continue
        for f in flows:
            fv = np.abs(tr[f].values)
            fscale = np.median(fv) + 1e-9
            # fraction of off-samples where the flow is genuinely near zero
            frac_zero = (fv[off] < 0.1 * fscale).mean()
            if frac_zero >= min_interlock:
                interlocks.append({'actuator': act, 'flow': f})
    return {'mass_balance': mass_balance, 'actuator_flow_interlock': interlocks}


def compute_residuals(tr, te, inv):
    """Per-invariant standardized residual on the test set, calibrated on normal (train) data."""
    residuals = {}

    for mb in inv.get('mass_balance', []):
        lit, fin, fout = mb['tank'], mb['inflow'], mb['outflow']
        if not all(c in tr.columns and c in te.columns for c in (lit, fin, fout)):
            continue
        d_tr = np.diff(tr[lit].values, prepend=tr[lit].values[0])
        A_tr = np.c_[tr[fin].values, tr[fout].values, np.ones(len(tr))]
        coef, *_ = lstsq(A_tr, d_tr, rcond=None)
        r_tr = np.abs(d_tr - A_tr @ coef)
        d_te = np.diff(te[lit].values, prepend=te[lit].values[0])
        A_te = np.c_[te[fin].values, te[fout].values, np.ones(len(te))]
        r_te = np.abs(d_te - A_te @ coef)
        residuals[f"mb_{lit}"] = (r_te - r_tr.mean()) / (r_tr.std() + 1e-9)

    for il in inv.get('actuator_flow_interlock', []):
        act, flow = il['actuator'], il['flow']
        if flow not in te.columns:
            continue
        # violation = flow present while actuator is off. Calibrate the (off & flow) product.
        off_tr = _valve_closed_indicator(tr, act)
        off_te = _valve_closed_indicator(te, act)
        v_tr = off_tr * np.abs(tr[flow].values)
        v_te = off_te * np.abs(te[flow].values)
        residuals[f"il_{act}_{flow}"] = (v_te - v_tr.mean()) / (v_tr.std() + 1e-9)

    return residuals


def detect(residuals, reference=None, y_true=None):
    """OR-combine standardized residuals: smooth W=25, per-residual P99.8, union.
    Label-free: thresholds from `reference` (training residuals, same keys). Oracle (reference=None):
    thresholds from the test rows labelled normal in `y_true` -- the leaky v2 protocol."""
    n = len(next(iter(residuals.values())))
    preds = np.zeros(n, dtype=int)
    fired = {}
    for name, r in residuals.items():
        sm = pd.Series(r).rolling(WINDOW, min_periods=1).mean().values
        if reference is None:
            thr = np.percentile(sm[y_true == 0], 99.8)
        else:
            thr = np.percentile(pd.Series(reference[name]).rolling(WINDOW, min_periods=1).mean().values, 99.8)
        p = (sm > thr).astype(int)
        fired[name] = p
        preds = np.maximum(preds, p)
    return preds, fired


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PIVOT v2 WS3 physics-residual detector")
    ap.add_argument('--train', default=r'Datasets/swat_train_preprocessed.csv')
    ap.add_argument('--test', default=r'Datasets/swat_test_preprocessed.csv')
    ap.add_argument('--invariants', default=r'Datasets/swat_invariants.json')
    ap.add_argument('--auto', action='store_true',
                    help='derive invariants from normal data instead of reading the JSON')
    ap.add_argument('--combine', default=None, help='predictions CSV (AE arm) to OR with physics')
    ap.add_argument('--export', default='preds_swat_physics.csv')
    ap.add_argument('--calibration', choices=['train', 'oracle'], default='train',
                    help="'train' (default): thresholds from training residuals, label-free. 'oracle': "
                         "thresholds from test rows labelled normal (leaks labels; reference only)")
    args = ap.parse_args()

    tr = pd.read_csv(args.train); tr.columns = tr.columns.str.strip()
    te = pd.read_csv(args.test); te.columns = te.columns.str.strip()
    label_col = 'Label' if 'Label' in te.columns else 'Normal/Attack'
    y = to_binary(te[label_col])
    if args.calibration == 'train':
        te, fired_faults = lfp.fault_rule(tr, te)
        if fired_faults:
            print(f"[FAULT RULE] replaced impossible values in: {list(fired_faults)}")
    else:
        print("[WARNING] --calibration oracle thresholds on test rows labelled normal (leaks labels).")
    if args.auto:
        inv = auto_derive_invariants(tr)
        derived_path = args.invariants.replace('.json', '_auto.json')
        json.dump(inv, open(derived_path, 'w'), indent=2)
        print(f"[AUTO] derived {len(inv['mass_balance'])} mass-balance + "
              f"{len(inv['actuator_flow_interlock'])} interlock invariants -> {derived_path}")
    else:
        inv = json.load(open(args.invariants))

    residuals = compute_residuals(tr, te, inv)
    print(f"invariants active: {list(residuals.keys())}")
    if args.calibration == 'train':
        phys_pred, fired = detect(residuals, reference=compute_residuals(tr, tr, inv))
    else:
        phys_pred, fired = detect(residuals, y_true=y)

    summary, events = evaluate(y, phys_pred)
    print_report(summary, events, "WS3 PHYSICS-RESIDUAL DETECTOR (alone)")

    # per-invariant event contribution
    from evaluate_events import find_events
    ev = find_events(y)
    print("\n-- per-invariant event coverage --")
    for name, p in fired.items():
        d = sum(1 for s, e in ev if p[s:e + 1].any())
        print(f"  {name:22s} {d:>2}/{len(ev)} events")

    pd.DataFrame({'y_true': y, 'y_pred': phys_pred}).to_csv(args.export, index=False)
    print(f"\n[SAVED] {args.export}")

    if args.combine:
        ae = pd.read_csv(args.combine)
        ae_pred = ae['y_pred'].values
        union = np.maximum(ae_pred, phys_pred)
        s_ae, _ = evaluate(y, ae_pred)
        s_un, _ = evaluate(y, union)
        print("\n" + "=" * 60)
        print(" AE ARM vs AE + PHYSICS (union)")
        print("=" * 60)
        print(f"  AE alone      : event recall {s_ae['event_recall']:.1%} "
              f"({s_ae['n_detected']}/{s_ae['n_events']})  point-F1 {s_ae['f1']:.1%}  FAR {s_ae['false_alarm']:.3%}")
        print(f"  AE + physics  : event recall {s_un['event_recall']:.1%} "
              f"({s_un['n_detected']}/{s_un['n_events']})  point-F1 {s_un['f1']:.1%}  FAR {s_un['false_alarm']:.3%}")
        gained = [i + 1 for i, (s, e) in enumerate(ev)
                  if ae_pred[s:e + 1].any() == 0 and union[s:e + 1].any()]
        print(f"  events physics ADDS over AE: {gained}")
        pd.DataFrame({'y_true': y, 'y_pred': union}).to_csv('preds_swat_ae_plus_physics.csv', index=False)
        print("  [SAVED] preds_swat_ae_plus_physics.csv")
