"""Detector-independent recovery horizons for SWaT attacks (reviewer point R2.8).

The v1 label correction (fix_labels.py) extended labels only where the detector fired, so it was
circular. Here recovery is read from the plant state alone, using normal training data and no
detector output:

  in_band  every level (LIT*) and flow (FIT*) channel inside its training [P0.5, P99.5]
  in_mode  the joint pump/valve configuration occurs >= 0.1% of the time in training
  normal   in_band and in_mode;  an attack's recovery ends at the first run of 60 normal seconds
           after the attack, capped at the next attack's start

Analyzers (AIT*) and pressures (PIT*, DPIT*) are excluded because they drift (see issue #1).
The check is validated on normal test periods far (> 2 h) from any attack: how often does it call
those abnormal?

Run from the repo root:  python recovery_check.py
Writes recovery_horizons_swat.csv and swat_test_labels_recovery.csv (corrected labels).
"""
import numpy as np
import pandas as pd

TEST, TRAIN = 'Datasets/swat_test_preprocessed.csv', 'Datasets/swat_train_preprocessed.csv'
HOLD, MODE_MIN, FAR_FROM = 60, 0.001, 2 * 3600


def run_starts(mask, hold):
    """First index of each run of >= hold consecutive True values."""
    m = np.concatenate([[0], mask.astype(np.int8), [0]])
    d = np.diff(m)
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    return s[(e - s) >= hold]


def plant_state(df, tr):
    levels_flows = [c for c in tr.columns if c.startswith(('LIT', 'FIT'))]
    actuators = [c for c in tr.columns if c.startswith(('P', 'MV', 'UV')) and not c.startswith('PIT')]
    lo, hi = tr[levels_flows].quantile(0.005), tr[levels_flows].quantile(0.995)
    in_band = ((df[levels_flows] >= lo) & (df[levels_flows] <= hi)).all(axis=1).values
    key = lambda d: d[actuators].round(3).astype(str).agg('|'.join, axis=1)
    freq = key(tr).value_counts(normalize=True)
    in_mode = key(df).map(freq).fillna(0.0).values >= MODE_MIN
    return in_band, in_mode, levels_flows, actuators


if __name__ == '__main__':
    te = pd.read_csv(TEST); te.columns = te.columns.str.strip()
    tr = pd.read_csv(TRAIN); tr.columns = tr.columns.str.strip()
    y = te['Normal/Attack'].astype(str).str.strip().str.lower().str.replace(' ', '').isin(['attack', '1']).astype(int).values
    in_band, in_mode, lf_cols, act_cols = plant_state(te, tr)
    normal = in_band & in_mode
    print(f'{len(lf_cols)} level/flow channels, {len(act_cols)} actuator columns')

    m = np.concatenate([[0], y, [0]]); d = np.diff(m)
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]

    # Validation: normal test rows more than 2 h from any attack
    dist = np.full(len(y), np.inf)
    for a, b in zip(s, e):
        idx = np.arange(len(y))
        dist = np.minimum(dist, np.where(idx < a, a - idx, np.where(idx >= b, idx - b + 1, 0)))
    far = (y == 0) & (dist > FAR_FROM)
    print(f'validation on {far.sum():,} normal rows > 2 h from any attack: '
          f'out of band {(~in_band[far]).mean():.2%}, rare mode {(~in_mode[far]).mean():.2%}, '
          f'not normal {(~normal[far]).mean():.2%}')

    starts = run_starts(normal, HOLD)
    rows, y_rec = [], y.copy()
    for k, (a, b) in enumerate(zip(s, e)):
        nxt = s[k + 1] if k + 1 < len(s) else len(y)
        cand = starts[starts >= b]
        rec = int(cand[0]) if len(cand) and cand[0] < nxt else nxt
        horizon = rec - b
        post = slice(b, rec)
        rows.append({'event': k + 1, 'attack_end': int(b), 'recovered_at': rec, 'horizon_s': horizon,
                     'capped_by_next_attack': rec == nxt and not (len(cand) and cand[0] < nxt),
                     'out_of_band_s': int((~in_band[post]).sum()), 'rare_mode_s': int((~in_mode[post]).sum())})
        y_rec[b:rec] = 1

    out = pd.DataFrame(rows)
    out.to_csv('recovery_horizons_swat.csv', index=False)
    pd.DataFrame({'y_original': y, 'y_recovery': y_rec}).to_csv('swat_test_labels_recovery.csv', index=False)
    h = out['horizon_s']
    print(f'\nrecovery horizon (s): median {h.median():.0f} | p90 {h.quantile(0.9):.0f} | max {h.max()} | '
          f'0 s for {(h == 0).sum()} of {len(h)} | capped by next attack: {out.capped_by_next_attack.sum()}')
    print(f'horizons > 600 s (beyond the paper\'s fixed window): {(h > 600).sum()}')
    print(f'labels added: {int(y_rec.sum() - y.sum()):,} rows')
    print(out.sort_values('horizon_s', ascending=False).head(8).to_string(index=False))

    b22 = e[21]
    post = slice(b22, s[22])
    print(f'\nevent 22 aftermath (rows {b22:,}-{s[22]:,}): out of band {(~in_band[post]).mean():.0%}, '
          f'rare mode {(~in_mode[post]).mean():.0%}')
    oob = {c: float(((te[c].values[post] < tr[c].quantile(0.005)) | (te[c].values[post] > tr[c].quantile(0.995))).mean())
           for c in lf_cols}
    print('  channels out of band:', {k: f'{v:.0%}' for k, v in sorted(oob.items(), key=lambda x: -x[1]) if v > 0.01})
    changed = {c: (te[c].values[post].mean(), tr[c].mean()) for c in act_cols}
    diff = {c: round(a - b, 2) for c, (a, b) in changed.items() if abs(a - b) > 0.5}
    print('  actuators whose mean state differs from training (z units):', diff)
    print('[SAVED] recovery_horizons_swat.csv, swat_test_labels_recovery.csv')
