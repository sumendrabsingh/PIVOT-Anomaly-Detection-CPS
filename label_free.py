"""Label-free evaluation protocol shared by the PIVOT entry points.

Nothing here reads test labels. Calibration (score standardization, thresholds) comes from normal
training data, or from a commissioning window of the deployment stream whose length is fixed in
advance. The leaky protocol used for the v1 paper numbers (standardizing and thresholding on test
rows labelled normal, sweeping thresholds for test F1) is kept only behind explicit
`--calibration oracle` flags, for reproducing those numbers as a reference row.

Drift handling, all label-free:
  fault_rule      a value > 100 training SDs from its training median is a sensor fault (WADI's
                  2B_AIT_002_PV reads ~8,100 vs ~9 in training) and is replaced by that median
  analyzer split  chemical analyzers (*AIT*) drift for days (SWaT AIT201: 10.9 training SDs), which
                  breaks reconstruction models; they are removed from the subsystem models and
                  watched by analyzer_arm_predict(), an abrupt-change detector that ignores drift
"""
import numpy as np
import pandas as pd

FAULT_SDS = 100
FAULT_MIN_RUN = 3600                  # samples (1 h at 1 Hz) a reading must stay impossible to count as a fault
SMOOTH = 25                           # samples, same persistence window as the subsystem array
ARM_H, ARM_GAP, ARM_HS = 30 * 60, 5 * 60, 6 * 3600


def is_analyzer(col):
    return 'AIT' in col


def fault_mask(train_df, test_df, k=FAULT_SDS, min_run=FAULT_MIN_RUN):
    """Boolean frame over the numeric columns: True where a value is > k training SDs from its
    training median AND stays so for at least min_run consecutive samples, i.e. a broken sensor
    rather than a process state. The run condition (added 2026-10-07) keeps attacks: on HAI the old
    per-sample rule replaced 642 values, 614 of them inside attacks (longest run 102 s), while WADI's
    2B_AIT_002_PV fault is one 26-hour run. Deciding that a run is long needs the whole run, so this
    is an offline cleaning step, not a causal one."""
    num = [c for c in test_df.columns if c in train_df.columns and c not in ('Label', 'Normal/Attack')
           and pd.api.types.is_numeric_dtype(test_df[c])]
    med, sd = train_df[num].median(), train_df[num].std().replace(0, 1.0)
    out = (test_df[num] - med).abs() > k * sd
    for c in out.columns[out.any()]:
        m = out[c].values.copy()
        for a, n in zip(*runs(m)):
            if n < min_run:
                m[a:a + n] = False
        out[c] = m
    return out


def fault_rule(train_df, test_df, k=FAULT_SDS):
    """Replace physically impossible values with the training median. Returns (fixed_df, fired).
    Right for models that need every input (autoencoders, forecasters). A detector that scores each
    channel separately should also suspend the faulted channel's own score (see fault_mask)."""
    fault = fault_mask(train_df, test_df, k)
    fixed = test_df.copy()
    fixed[fault.columns] = test_df[fault.columns].mask(fault, train_df[fault.columns].median(), axis=1)
    return fixed, {c: float(v) for c, v in fault.mean().items() if v > 0}


def smooth(v, w=SMOOTH):
    return pd.Series(v).rolling(w, min_periods=1).mean().values


def runs(mask):
    """(start, length) of each maximal True run."""
    m = np.concatenate([[0], np.asarray(mask).astype(np.int8), [0]])
    d = np.diff(m)
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    return s, e - s


def dual_threshold(score, tau_h, tau_l, w_dur):
    """Tier 1: score > tau_h. Tier 2: score > tau_l for w_dur samples, alarmed from the sample at
    which the run reaches w_dur (causal: no credit before the alarm could fire)."""
    pred = (score > tau_h).astype(np.int8)
    s, n = runs(score > tau_l)
    for a, length in zip(s, n):
        if length >= w_dur:
            pred[a + w_dur - 1:a + length] = 1
    return pred


def analyzer_arm_predict(train_df, test_df, cols, w_dur=50):
    """Abrupt-change detector on analyzer channels. Residual = value minus its median over
    [t - GAP - H, t - GAP); divided by a causal robust spread over the past HS, floored at the
    training SD; max over channels, smoothed; dual threshold P99.9 / P99 of the training score."""
    def residual(df):
        return pd.DataFrame({c: df[c] - df[c].shift(ARM_GAP).rolling(ARM_H, min_periods=ARM_H // 4).median()
                             for c in cols}).fillna(0.0)

    r_tr = residual(train_df)
    mu, sd = r_tr.mean(), r_tr.std() + 1e-6

    def score(r):
        past = r.shift(ARM_GAP)
        mad = (past - past.rolling(ARM_HS, min_periods=ARM_HS // 4).median()).abs() \
            .rolling(ARM_HS, min_periods=ARM_HS // 4).median() * 1.4826
        z = np.abs(r.values - mu.values) / np.maximum(mad.fillna(0.0).values, sd.values)
        return smooth(z.max(axis=1))

    cal = score(r_tr)
    return dual_threshold(score(residual(test_df)), np.percentile(cal, 99.9), np.percentile(cal, 99.0), w_dur)


def chance_events(pred, y, n=200, seed=0):
    """Attacks hit by the detector's own alarm pattern circularly shifted to random offsets: same
    alarm rate and episode structure, no alignment with the attacks. Detectors with many short false
    alarms hit attacks by luck; compare attacks caught against this. Returns (mean, 95th percentile)."""
    pred, y = np.asarray(pred), np.asarray(y)
    s, length = runs(y == 1)
    rng = np.random.default_rng(seed)
    margin = max(1, len(pred) // 20)
    hits = [sum(np.roll(pred, rng.integers(margin, len(pred) - margin))[a:a + n_].any() for a, n_ in zip(s, length))
            for _ in range(n)]
    return float(np.mean(hits)), float(np.percentile(hits, 95))


def swat_test_labels(raw=None):
    """SWaT test labels from Datasets/swat_test_preprocessed.csv, the single source of SWaT ground truth (since
    2026-10-07: SWaT_Dataset_Attack_v0_Corrected.csv, official labels extended 600 s past each attack). Scripts that
    score the raw 'Original Dataset Backup' files take their labels from here; raw must have the same rows."""
    te = pd.read_csv('Datasets/swat_test_preprocessed.csv', usecols=lambda c: c.strip() in ('Timestamp', 'Normal/Attack'))
    te.columns = te.columns.str.strip()
    if raw is not None:
        ts = raw['Timestamp'].astype(str).str.strip().values
        if len(ts) != len(te) or not (ts == te['Timestamp'].astype(str).str.strip().values).all():
            raise SystemExit('raw SWaT test rows do not line up with Datasets/swat_test_preprocessed.csv')
    return te['Normal/Attack'].astype(str).str.strip().str.lower().str.replace(' ', '').eq('attack').astype(int).values
