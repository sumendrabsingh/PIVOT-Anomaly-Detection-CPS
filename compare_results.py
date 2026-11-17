"""Compare every result CSV of the evaluation paper between the repo and an independent re-run (PaperV4 validation).

The re-run (a separate directory, ../PIVOT_repro) rebuilt every dataset from the raw files, retrained every model from scratch
and re-ran every evaluation script. For each file: rows are aligned on the non-numeric/key columns, then per numeric
column the max absolute difference is reported, and verdict-like columns must match exactly.

Run from the repo root:  python compare_results.py [--other ../PIVOT_repro]  -> PaperV4/validation_files.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

FILES = """decomposition_ablation_label_free_swat.csv physics_ablation_label_free_swat.csv decomposition_mechanism_swat.csv
baselines_simple_label_free_swat.csv baselines_simple_generalization.csv leakage_effect.csv
orvote_check_swat.csv orvote_check_swat_labelfree.csv orvote_check_wadi.csv orvote_check_wadi_labelfree.csv
orvote_check_hai.csv orvote_check_hai_labelfree.csv
physics_or_test_swat_summary.csv physics_or_test_swat_labelfree.csv physics_or_test_wadi_summary.csv
physics_or_test_wadi_labelfree.csv physics_or_test_hai_summary.csv physics_or_test_hai_labelfree.csv
decomposition_confirm_wadi_summary.csv decomposition_confirm_wadi_labelfree.csv decomposition_confirm_hai_summary.csv
decomposition_confirm_hai_labelfree.csv
deployable_test_hai2204_summary.csv deployable_test_hai2204_seeds.csv deployable_test_hai2103_summary.csv
deployable_test_hai2103_seeds.csv deployable_test_hai2007_commission_summary.csv deployable_test_hai2007_commission_seeds.csv
deployable_test_swat_commission_summary.csv deployable_test_wadi_commission_summary.csv
deployable_test_hai2305_commission_summary.csv deployable_test_hai2204_commission_summary.csv
deployable_test_hai2103_commission_summary.csv
orvote_confirm_hai2007_summary.csv orvote_confirm_hai2204_summary.csv orvote_confirm_hai2103_summary.csv
orvote_confirm_hai2007_f1.csv orvote_confirm_hai2204_f1.csv orvote_confirm_hai2103_f1.csv
orvote_confirm_hai2007_labelfree.csv orvote_confirm_hai2204_labelfree.csv orvote_confirm_hai2103_labelfree.csv
orvote_mechanism_hai2007.csv orvote_mechanism_hai2204.csv orvote_mechanism_hai2103.csv
calibration_dev_swat.csv calibration_dev_wadi.csv calibration_dev_hai.csv
orvote_deploy_hai2007_summary.csv orvote_deploy_hai2204_summary.csv orvote_deploy_hai2103_summary.csv
orvote_deploy_hai2007_seeds.csv orvote_deploy_hai2204_seeds.csv orvote_deploy_hai2103_seeds.csv
orvote_deploy_hai2007_shift_summary.csv orvote_deploy_hai2204_shift_summary.csv orvote_deploy_hai2103_shift_summary.csv
orvote_deploy_hai2007_shift_seeds.csv orvote_deploy_hai2204_shift_seeds.csv orvote_deploy_hai2103_shift_seeds.csv
orvote_deploy_swat2019jul_shift_summary.csv orvote_deploy_swat2019dec_shift_summary.csv
orvote_deploy_swat2022nov_shift_summary.csv orvote_deploy_swat2019jul_shift_seeds.csv
orvote_deploy_swat2019dec_shift_seeds.csv orvote_deploy_swat2022nov_shift_seeds.csv
matched_far_attack_swat.csv feasible_attack_label_free_swat.csv drift_isolation_hai2204.csv""".split()
VERDICT = {'verdict', 'supported', 'primary', 'note', 'calibration', 'release', 'plant', 'detector', 'method',
           'protocol', 'physics', 'threat', 'objective', 'analyzers', 'decision', 'window', 'stage'}


def compare(a, b):
    keys = [c for c in a.columns if c in VERDICT or c in ('seed', 'K', 'arm', 'target_FAR', 'event', 'epsilon')]
    keys = [k for k in keys if k not in ('verdict', 'supported', 'primary', 'note')]
    if len(a) != len(b):
        return {'status': 'ROWS DIFFER', 'rows': f'{len(a)} vs {len(b)}'}
    if keys:
        a, b = a.sort_values(keys).reset_index(drop=True), b.sort_values(keys).reset_index(drop=True)
        if not a[keys].astype(str).equals(b[keys].astype(str)):
            return {'status': 'KEYS DIFFER', 'rows': len(a)}
    out = {'status': 'ok', 'rows': len(a)}
    worst, worst_col = 0.0, ''
    for c in a.columns:
        if c in ('verdict', 'supported', 'primary', 'note'):
            if not a[c].astype(str).equals(b[c].astype(str)):
                out['status'] = 'VERDICT DIFFERS'
                out['verdicts'] = f'{c}: {list(a[c])} vs {list(b[c])}'
        elif c not in keys and pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
            d = np.nanmax(np.abs(a[c].values.astype(float) - b[c].values.astype(float))) if len(a) else 0.0
            if d > worst:
                worst, worst_col = d, c
    out['max_abs_diff'], out['worst_column'] = worst, worst_col
    if out['status'] == 'ok' and worst > 1e-9:
        out['status'] = 'numbers differ'
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--other', default='../PIVOT_repro')
    args = ap.parse_args()
    rows = []
    for f in FILES:
        pa, pb = f, os.path.join(args.other, f)
        if not os.path.exists(pa) or not os.path.exists(pb):
            rows.append({'file': f, 'status': 'MISSING ' + ('repo' if not os.path.exists(pa) else 'rerun')})
            continue
        try:
            rows.append({'file': f, **compare(pd.read_csv(pa), pd.read_csv(pb))})
        except pd.errors.EmptyDataError:
            rows.append({'file': f, 'status': 'EMPTY'})
    out = pd.DataFrame(rows)
    os.makedirs('PaperV4', exist_ok=True)
    out.to_csv('PaperV4/validation_files.csv', index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 80)
    print(out.to_string(index=False))
    print(out.status.value_counts().to_string())


if __name__ == '__main__':
    main()
