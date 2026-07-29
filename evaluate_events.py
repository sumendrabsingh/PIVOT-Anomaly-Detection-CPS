"""
PIVOT v2 -- Event-level evaluation protocol (WS5).

v1 reported point-wise metrics only: every timestep is an independent sample. That hides
the two numbers an ICS operator actually asks for:

  1. "Of the N attacks launched at the plant, how many did you catch at all?"   -> event recall
  2. "How long after the attack started did the alarm fire?"                    -> time-to-detect

This module scores a prediction vector at the *event* level, where a ground-truth event is
a maximal contiguous run of label==1 (one attack scenario), and reports:

  - point-wise metrics          (must reproduce the v1 numbers exactly -- the Phase A gate)
  - event recall / TTD          (limitations 8 and 11)
  - point-adjusted metrics      (the convention in the HAI/SWaT literature: an event that is
                                 detected anywhere counts as fully detected)

Input is a predictions CSV with a y_true and a y_pred column, as written by the
--export_predictions flag of run_generalized_subsystems.py. Scores are optional and only
used for threshold sweeps.

Usage:
    python evaluate_events.py --predictions preds_swat_orvote.csv
    python evaluate_events.py --predictions preds_swat_orvote.csv --sample_period 1.0 --output events_swat.csv
"""

import argparse
import json

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------

def to_binary(series):
    """Coerce a label column to 0/1.

    Handles both the numeric labels in the preprocessed CSVs and the raw 'Normal'/'Attack'
    strings. Note a plain `dtype == object` test is not enough: recent pandas reads string
    columns as an Arrow-backed dtype, so we branch on whether the column is numeric instead.
    """
    if not pd.api.types.is_numeric_dtype(series):
        return (series.astype(str).str.strip().str.lower() != 'normal').astype(int).values
    return series.astype(int).values


# ---------------------------------------------------------------------------
# Event extraction
# ---------------------------------------------------------------------------

def find_events(labels):
    """Maximal contiguous runs of 1s. Returns [(start, end_inclusive), ...]."""
    labels = np.asarray(labels).astype(int)
    if labels.sum() == 0:
        return []

    # Pad with zeros so a run touching either boundary still produces an edge.
    padded = np.concatenate([[0], labels, [0]])
    diff = np.diff(padded)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def point_wise_metrics(y_true, y_pred):
    """The v1 metric set. Kept here so v2 can prove it reproduces v1 before changing it."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        'accuracy': (tp + tn) / len(y_true) if len(y_true) else 0.0,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'false_alarm': fp / (fp + tn) if (fp + tn) else 0.0,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
    }


def event_metrics(y_true, y_pred, sample_period=1.0):
    """Per-attack-scenario detection rate and time-to-detect.

    An event is detected if the detector fires on at least one sample inside it. TTD is the
    offset of the first such sample from event onset, in seconds (sample_period scales the
    index; SWaT/WADI/HAI are all 1 Hz, so the default is a no-op).
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    events = find_events(y_true)
    rows = []
    for i, (start, end) in enumerate(events, start=1):
        window = y_pred[start:end + 1]
        hit_offsets = np.flatnonzero(window == 1)
        detected = hit_offsets.size > 0

        rows.append({
            'event_id': i,
            'start_index': start,
            'end_index': end,
            'duration_samples': end - start + 1,
            'duration_seconds': (end - start + 1) * sample_period,
            'detected': bool(detected),
            # TTD is undefined for a miss -- NaN, not 0, so it cannot silently drag a mean down.
            'ttd_samples': int(hit_offsets[0]) if detected else np.nan,
            'ttd_seconds': float(hit_offsets[0] * sample_period) if detected else np.nan,
            'coverage': float(window.mean()),  # fraction of the event flagged
        })

    events_df = pd.DataFrame(rows)
    n_events = len(events_df)
    n_detected = int(events_df['detected'].sum()) if n_events else 0
    ttd = events_df.loc[events_df['detected'], 'ttd_seconds'] if n_events else pd.Series(dtype=float)

    summary = {
        'n_events': n_events,
        'n_detected': n_detected,
        'event_recall': n_detected / n_events if n_events else 0.0,
        'ttd_mean_s': float(ttd.mean()) if len(ttd) else np.nan,
        'ttd_median_s': float(ttd.median()) if len(ttd) else np.nan,
        'ttd_p90_s': float(ttd.quantile(0.90)) if len(ttd) else np.nan,
        'ttd_max_s': float(ttd.max()) if len(ttd) else np.nan,
        'mean_coverage': float(events_df['coverage'].mean()) if n_events else 0.0,
    }
    return summary, events_df


def point_adjusted_metrics(y_true, y_pred):
    """Point-adjusted (PA) scoring, as used in most SWaT/WADI/HAI papers.

    If any sample inside a ground-truth event is flagged, the whole event is credited as
    detected. PA inflates recall by construction -- we report it only for comparability with
    the literature, never as PIVOT's headline number.
    """
    y_true = np.asarray(y_true).astype(int)
    adjusted = np.asarray(y_pred).astype(int).copy()

    for start, end in find_events(y_true):
        if adjusted[start:end + 1].any():
            adjusted[start:end + 1] = 1

    metrics = point_wise_metrics(y_true, adjusted)
    return {f'pa_{k}': v for k, v in metrics.items()}


def false_alarm_events(y_true, y_pred):
    """Count alarm runs that lie entirely outside any ground-truth event.

    Operators experience false alarms as *episodes*, not as timesteps: one 500-sample false
    alarm is a single nuisance, not 500. This is the operational complement to point-wise FAR.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    fa = 0
    for start, end in find_events(y_pred):
        if not y_true[start:end + 1].any():
            fa += 1
    return {'false_alarm_events': fa}


def evaluate(y_true, y_pred, sample_period=1.0):
    """The full v2 metric set for one prediction vector."""
    summary = {}
    summary.update(point_wise_metrics(y_true, y_pred))
    ev_summary, events_df = event_metrics(y_true, y_pred, sample_period)
    summary.update(ev_summary)
    summary.update(point_adjusted_metrics(y_true, y_pred))
    summary.update(false_alarm_events(y_true, y_pred))
    return summary, events_df


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(summary, events_df, title="EVENT-LEVEL EVALUATION"):
    print("\n" + "=" * 78)
    print(f" {title}")
    print("=" * 78)

    print("\n-- Point-wise (v1-comparable) --")
    print(f"  Accuracy:          {summary['accuracy']:.4%}")
    print(f"  Precision:         {summary['precision']:.4%}")
    print(f"  Recall:            {summary['recall']:.4%}")
    print(f"  F1:                {summary['f1']:.4%}")
    print(f"  False Alarm Rate:  {summary['false_alarm']:.4%}")
    print(f"  TP {summary['tp']:,} | FN {summary['fn']:,} | FP {summary['fp']:,} | TN {summary['tn']:,}")

    print("\n-- Event-level (NEW: limitations 8 and 11) --")
    print(f"  Attack events:     {summary['n_events']}")
    print(f"  Detected:          {summary['n_detected']}  ->  event recall {summary['event_recall']:.2%}")
    print(f"  Mean coverage:     {summary['mean_coverage']:.2%} of each event flagged")
    print(f"  False-alarm episodes (outside any attack): {summary['false_alarm_events']}")

    print("\n-- Time-to-detect (detected events only) --")
    if summary['n_detected']:
        print(f"  Median:            {summary['ttd_median_s']:.1f} s")
        print(f"  Mean:              {summary['ttd_mean_s']:.1f} s")
        print(f"  P90:               {summary['ttd_p90_s']:.1f} s")
        print(f"  Worst:             {summary['ttd_max_s']:.1f} s")
    else:
        print("  no events detected")

    print("\n-- Point-adjusted (literature convention -- NOT our headline) --")
    print(f"  PA Precision:      {summary['pa_precision']:.4%}")
    print(f"  PA Recall:         {summary['pa_recall']:.4%}")
    print(f"  PA F1:             {summary['pa_f1']:.4%}")

    if len(events_df):
        missed = events_df[~events_df['detected']]
        if len(missed):
            print(f"\n-- Missed events ({len(missed)}) --")
            for _, r in missed.iterrows():
                print(f"  event {int(r['event_id']):>3}  idx {int(r['start_index']):>7}-{int(r['end_index']):<7}"
                      f"  {int(r['duration_samples']):>6} samples")
        slowest = events_df[events_df['detected']].nlargest(3, 'ttd_seconds')
        if len(slowest):
            print("\n-- Slowest detections --")
            for _, r in slowest.iterrows():
                print(f"  event {int(r['event_id']):>3}  TTD {r['ttd_seconds']:>8.1f} s"
                      f"  ({int(r['duration_samples'])} samples long)")
    print("=" * 78)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_predictions(path):
    df = pd.read_csv(path)

    def pick(*candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    true_col = pick('y_true', 'label', 'Label', 'Normal/Attack')
    pred_col = pick('y_pred', 'prediction', 'pred')
    if true_col is None or pred_col is None:
        raise SystemExit(
            f"[ERROR] {path} needs a ground-truth column (y_true/label) and a prediction "
            f"column (y_pred/prediction); found: {list(df.columns)}")

    return to_binary(df[true_col]), df[pred_col].astype(int).values, df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PIVOT v2 event-level evaluation (event recall, time-to-detect, point-adjusted)")
    parser.add_argument('--predictions', type=str, required=True,
                        help='CSV with y_true and y_pred columns (see --export_predictions '
                             'on run_generalized_subsystems.py)')
    parser.add_argument('--sample_period', type=float, default=1.0,
                        help='Seconds per sample (SWaT/WADI/HAI are 1 Hz)')
    parser.add_argument('--output', type=str, default=None,
                        help='Write the per-event table to this CSV')
    parser.add_argument('--summary_json', type=str, default=None,
                        help='Write the summary metrics to this JSON')
    parser.add_argument('--title', type=str, default='EVENT-LEVEL EVALUATION')
    args = parser.parse_args()

    y_true, y_pred, _ = load_predictions(args.predictions)
    summary, events_df = evaluate(y_true, y_pred, args.sample_period)
    print_report(summary, events_df, args.title)

    if args.output:
        events_df.to_csv(args.output, index=False)
        print(f"\n[SAVED] per-event table -> {args.output}")
    if args.summary_json:
        with open(args.summary_json, 'w') as f:
            json.dump(summary, f, indent=2, default=float)
        print(f"[SAVED] summary -> {args.summary_json}")
