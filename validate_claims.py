"""Claim ledger of the evaluation paper (PaperV4): key numbers and verdict of every claim, from the repo's result
files and from the independent re-run (../PIVOT_repro: data rebuilt from raw, models retrained, every
script re-run). Writes PaperV4/CLAIMS.md and PaperV4/claims.csv.

Agreement: verdict text identical, and every key number within tolerance (attack counts 1.5 attacks; percentages
2 points; F1 3 points) -> 'reproduced'; identical numbers -> 'identical'; otherwise 'DIFFERS'.

Run from the repo root:  python validate_claims.py [--other ../PIVOT_repro]
"""
import argparse
import os

import numpy as np
import pandas as pd

TOL = {'n': 1.5, 'pct': 2.0, 'f1': 3.0}


def rd(root, f):
    return pd.read_csv(os.path.join(root, f))


def pct(x):
    return 100 * float(x)


# each extractor: root -> (dict name -> (value, kind), verdict string)
def c4(r):
    d = rd(r, 'leakage_effect.csv').query('plant == "SWaT"').groupby('protocol')
    return ({'F1 oracle+F1': (pct(d.F1.mean()['oracle + F1']), 'f1'), 'F1 +600s': (pct(d.F1.mean()['oracle + F1 +600s']), 'f1'),
             'attacks official': (d.n_events.first()['oracle + F1'], 'n'), 'attacks +600s': (d.n_events.first()['oracle + F1 +600s'], 'n')},
            '+600 s labels lower F1 and merge attacks' if d.F1.mean()['oracle + F1 +600s'] < d.F1.mean()['oracle + F1'] else 'no drop')


def c5(r):
    d = rd(r, 'leakage_effect.csv').groupby(['plant', 'protocol'])[['F1', 'beyond']].mean()
    out, v = {}, []
    for p in ('SWaT', 'WADI', 'HAI'):
        lf, orf = d.loc[(p, 'label-free')], d.loc[(p, 'oracle + F1')]
        out[f'{p} F1 label-free'] = (pct(lf.F1), 'f1'); out[f'{p} F1 oracle+F1'] = (pct(orf.F1), 'f1')
        out[f'{p} beyond label-free'] = (lf.beyond, 'n'); out[f'{p} beyond oracle+F1'] = (orf.beyond, 'n')
        v.append(f'{p}: oracle F1 {"higher" if orf.F1 > lf.F1 else "lower"}')
    return out, '; '.join(v)


def c6(r):
    g = rd(r, 'baselines_simple_generalization.csv').set_index(['plant', 'detector'])
    x = g.loc[('HAI', 'MinMax (tol 0%)')]
    lf = rd(r, 'leakage_effect.csv').query('plant == "SWaT" and protocol == "label-free"')
    return ({'HAI MinMax0 caught': (x.events, 'n'), 'HAI MinMax0 chance': (x.chance, 'n'), 'HAI MinMax0 FAR%': (pct(x.FAR), 'pct'),
             'SWaT PIVOT caught': (lf.caught.mean(), 'n'), 'SWaT PIVOT chance': (lf.chance.mean(), 'n')},
            'chance explains most catches of high-FAR detectors' if x.chance > 0.5 * x.events else 'chance small')


def c7(r):
    b = rd(r, 'baselines_simple_label_free_swat.csv').groupby('detector').mean(numeric_only=True)
    x = b.loc['Gradient (tol 0%)']
    return ({'Gradient caught': (x.events, 'n'), 'Gradient caught early': (x.early, 'n')},
            'most gradient catches are late' if x.early < 0.6 * x.events else 'mostly early')


def baselines_swat(r):
    b = rd(r, 'baselines_simple_label_free_swat.csv').groupby('detector').mean(numeric_only=True)
    keys = {'PIVOT': 'PIVOT array (6 stages)', 'PIVOT+arm': 'PIVOT array (6 stages) + analyzer arm',
            'MinMax5': 'MinMax (tol 5% of range)', 'MinMax5+arm': 'MinMax (tol 5% of range) + analyzer arm',
            'mono AE': 'AE (monolithic, as in paper)', 'PCA': 'PCA-SPE (14 components)'}
    out = {}
    for k, name in keys.items():
        out[f'{k} beyond'] = (b.loc[name].excess, 'n'); out[f'{k} FAR%'] = (pct(b.loc[name].FAR), 'pct')
    best = max(keys, key=lambda k: b.loc[keys[k]].excess)
    return out, f'most attacks beyond chance: {best}; MinMax5+arm {"beats" if b.loc[keys["MinMax5+arm"]].excess > b.loc[keys["PIVOT+arm"]].excess else "does not beat"} PIVOT+arm'


def baselines_other(plant, lf_file):
    def f(r):
        g = rd(r, 'baselines_simple_generalization.csv')
        g = g[g.plant == plant].set_index('detector')
        s = rd(r, lf_file).groupby('detector').mean(numeric_only=True)
        out = {'PIVOT stages (label-free) beyond': (s.loc['stages'].beyond, 'n'), 'PIVOT FAR%': (pct(s.loc['stages'].FAR), 'pct'),
               'mono (label-free) beyond': (s.loc['monolithic'].beyond, 'n')}
        for d in g.index:
            out[f'{d} beyond'] = (g.loc[d].beyond, 'n')
        best_simple = g.beyond.max()
        return out, f'PIVOT {"above" if s.loc["stages"].beyond > best_simple else "not above"} best simple baseline'
    return f


def c11(r):
    p = rd(r, 'physics_ablation_label_free_swat.csv').query('protocol == "dual"').groupby('physics').mean(numeric_only=True)
    out = {f'{a} F1': (pct(p.loc[a].F1), 'f1') for a in p.index}
    out.update({f'{a} beyond': (p.loc[a].excess, 'n') for a in p.index})
    best = p.excess.idxmax()
    return out, f'no physics arm beats lambda=0 by >1 attack' if p.excess.max() - p.loc['none'].excess <= 1 else f'{best} better'


def h_summary(files, diff='diff', lo='CI_low'):
    def f(r):
        out, v = {}, []
        for tag, fn, q in files:
            d = rd(r, fn)
            d = d.query(q) if q else d
            for _, x in d.iterrows():
                key = f'{tag} {x.get("target_FAR", "")}'.strip()
                if pd.isna(x[diff]):
                    continue
                out[f'{key} diff'] = (x[diff], 'n'); out[f'{key} CI_low'] = (x[lo], 'n')
                v.append(f'{key}: CI>0' if x[lo] > 0 else f'{key}: no')
        return out, '; '.join(v)
    return f


def verdict_files(files):
    def f(r):
        out, v = {}, []
        for tag, fn in files:
            x = rd(r, fn).iloc[0]
            out[f'{tag} diff'] = (x['diff'], 'n'); out[f'{tag} CI_low'] = (x.CI_low, 'n')
            out[f'{tag} FAR-ok seeds'] = (x.far_ok_seeds, 'n')
            verdict = x['verdict'] if 'verdict' in x else ('SUPPORTED' if x['supported'] else 'NOT SUPPORTED')
            v.append(f'{tag}: {verdict}')
        return out, '; '.join(v)
    return f


def c13(r):
    m = rd(r, 'matched_far_attack_swat.csv').groupby(['target_FAR', 'threat', 'epsilon', 'detector']).excess.mean()
    out = {}
    for t in (0.002, 0.01):
        for th in ('sensors', 'actuators'):
            for d in ('array', 'mono'):
                out[f'{t:.1%} {th} eps8 {d}'] = (m.loc[(t, th, 8.0, d)], 'n')
    v = 'array loses as much or more than mono' if out['1.0% sensors eps8 array'][0] <= out['1.0% sensors eps8 mono'][0] else 'array more robust at 1%'
    return out, v


def c14(r):
    f = rd(r, 'feasible_attack_label_free_swat.csv')
    a = f.query('detector == "union" and objective == "best"').groupby(['threat', 'epsilon']).events.mean()
    clean = f.query('threat == "clean" and detector == "union"').events.mean()
    out = {'union clean': (clean, 'n')}
    for th in ('sensors', 'actuators', 'unconstrained'):
        out[f'union {th} eps32 (best objective)'] = (a.loc[(th, 32.0)], 'n')
    return out, 'no rise to 35 at large eps' if max(v for v, _ in out.values()) < 30 else 'rises'


def c15(r):
    d = rd(r, 'decomposition_ablation_label_free_swat.csv')
    d['ex'] = d.events - d.chance
    g = d.query('analyzers == "split" and decision == "dual"').groupby(['arm', 'K']).agg(F1=('F1', 'mean'), FAR=('FAR', 'mean'), ex=('ex', 'mean'))
    out = {}
    for arm in (False, True):
        for k in (6, 1):
            out[f'arm={arm} K={k} F1'] = (pct(g.loc[(arm, k)].F1), 'f1'); out[f'arm={arm} K={k} FAR%'] = (pct(g.loc[(arm, k)].FAR), 'pct')
            out[f'arm={arm} K={k} beyond'] = (g.loc[(arm, k)].ex, 'n')
    return out, f'K1 beyond {"above" if g.loc[(True, 1)].ex > g.loc[(True, 6)].ex else "below"} K6 (with arm), at higher FAR'


def c21(r):
    out, v = {}, []
    for ds in ('swat', 'wadi', 'hai'):
        o = rd(r, f'orvote_check_{ds}.csv').groupby(['target_FAR', 'detector']).beyond.mean().unstack()
        for t in (0.001, 0.005, 0.01):
            a, b = o.loc[t, 'OR vote, per-sensor scoring'], o.loc[t, 'monolithic, per-sensor scoring']
            out[f'{ds} {t:.1%} OR'] = (a, 'n'); out[f'{ds} {t:.1%} mono'] = (b, 'n')
            v.append(f'{ds} {t:.1%}: {"OR" if a > b else "mono"}')
    return out, '; '.join(v)


def c22(r):
    d = rd(r, 'drift_isolation_hai2204.csv').set_index('stage')
    out = {f'{s} test-normal median': (d.loc[s].test_normal_median, 'n') for s in d.index}
    out.update({f'{s} train P99': (d.loc[s].train_p99, 'n') for s in d.index})
    drifting = [s for s in d.index if d.loc[s].test_normal_median > d.loc[s].train_p99]
    return out, f'drifting stages: {drifting}'


def c23(r):
    out, v = {}, []
    for rel in ('hai2007', 'hai2204', 'hai2103'):
        e = rd(r, f'orvote_mechanism_{rel}.csv').groupby(['target_FAR', 'detector', 'event']).excess.mean().groupby(['target_FAR', 'detector']).sum().unstack()
        for t in e.index:
            for d in e.columns:
                out[f'{rel} {t:.1%} {d}'] = (e.loc[t, d], 'n')
    return out, f'{len(out)} values'


def c25(r):
    out, v = {}, []
    for ds in ('swat', 'wadi', 'hai'):
        c = rd(r, f'calibration_dev_{ds}.csv').groupby(['method', 'detector']).agg(FAR=('FAR', 'mean'), beyond=('beyond', 'mean'))
        for (m, d), x in c.iterrows():
            out[f'{ds} {m} {d} FAR%'] = (pct(x.FAR), 'pct'); out[f'{ds} {m} {d} beyond'] = (x.beyond, 'n')
    return out, 'calibration development values'


def c27(r):
    out = {}
    for rel in ('swat2019jul', 'swat2019dec', 'swat2022nov'):
        s = rd(r, f'orvote_deploy_{rel}_shift_seeds.csv').query('window == "primary"').groupby('detector').FAR.mean()
        out[f'{rel} PIVOT FAR%'] = (pct(s['pivot']), 'pct'); out[f'{rel} mono FAR%'] = (pct(s['mono']), 'pct')
    return out, 'FAR >= 10% everywhere' if min(v for v, _ in out.values()) >= 9.5 else 'some FAR < 10%'


def c28(r):
    """HAI 23.05 forecaster, seeds 1-5 (preds_hai_temporal_seed<k>.csv, commissioning calibration)."""
    import sys
    sys.path.insert(0, '.')
    import label_free as lfp
    from evaluate_events import evaluate
    rows = []
    for k in range(1, 6):
        d = rd(r, f'preds_hai_temporal_seed{k}.csv')
        y, p = d.y_true.values.astype(int), d.y_pred.values.astype(int)
        s, _ = evaluate(y, p)
        ch, _ = lfp.chance_events(p, y)
        rows.append((s['n_detected'], ch, pct(s['false_alarm']), pct(s['f1'])))
    a = np.array(rows, dtype=float)
    m = a.mean(0)
    return ({'caught (mean)': (m[0], 'n'), 'chance (mean)': (m[1], 'n'), 'beyond (mean)': (m[0] - m[1], 'n'),
             'FAR% (mean)': (m[2], 'pct'), 'F1 (mean)': (m[3], 'f1'), 'caught min': (a[:, 0].min(), 'n'),
             'caught max': (a[:, 0].max(), 'n')},
            f"forecaster {a[:, 0].min():.0f}-{a[:, 0].max():.0f} of 52 attacks over seeds 1-5")


CLAIMS = [
    ('C4', 'SWaT +600 s label convention lowers F1 and merges 35 attacks into 25', c4),
    ('C5', 'Test-label calibration moves F1 by a few points and does not raise attacks beyond chance', c5),
    ('C6', 'Chance level explains most catches of high-FAR detectors', c6),
    ('C7', '"Any sample caught" credits late alarms (gradient detector)', c7),
    ('C8', 'SWaT: simple baselines vs PIVOT (label-free, beyond chance)', baselines_swat),
    ('C9', 'WADI A2: simple baselines vs PIVOT (label-free)', baselines_other('WADI', 'decomposition_confirm_wadi_labelfree.csv')),
    ('C10', 'HAI 23.05: simple baselines vs PIVOT (label-free)', baselines_other('HAI', 'decomposition_confirm_hai_labelfree.csv')),
    ('C11', 'Physics loss term adds nothing (SWaT, seeds 1-5)', c11),
    ('C12', 'H2: OR-ing a physics arm, matched FAR', h_summary([('SWaT', 'physics_or_test_swat_summary.csv', None),
                                                               ('WADI', 'physics_or_test_wadi_summary.csv', None),
                                                               ('HAI', 'physics_or_test_hai_summary.csv', None)])),
    ('C13', 'Matched-FAR white-box attack: stage array not more robust than monolithic', c13),
    ('C14', 'Feasible attacks: detection does not rise to 35 at large epsilon', c14),
    ('C15', 'SWaT K6 vs K1, label-free dual rule', c15),
    ('C16', 'H1 WADI A2: stages vs monolithic at matched FAR', h_summary([('WADI', 'decomposition_confirm_wadi_summary.csv', 'primary')])),
    ('C17', 'H1 HAI 23.05: stages vs monolithic at matched FAR', h_summary([('HAI', 'decomposition_confirm_hai_summary.csv', 'primary')])),
    ('C18', 'H1b training thresholds, HAI 22.04 / 21.03', verdict_files([('hai2204', 'deployable_test_hai2204_summary.csv'),
                                                                        ('hai2103', 'deployable_test_hai2103_summary.csv')])),
    ('C19', 'H1c commissioning thresholds, HAI 20.07', verdict_files([('hai2007', 'deployable_test_hai2007_commission_summary.csv')])),
    ('C20', 'H3 OR vote vs monolithic at matched FAR, held-out HAI', h_summary([('20.07', 'orvote_confirm_hai2007_summary.csv', None),
                                                                              ('22.04', 'orvote_confirm_hai2204_summary.csv', None),
                                                                              ('21.03', 'orvote_confirm_hai2103_summary.csv', None)])),
    ('C21', 'OR vote vs monolithic on development plants (exploratory)', c21),
    ('C22', 'Drift isolation: HAI 22.04 drifting stage', c22),
    ('C23', 'Separate stage models vs grouped monolithic errors (mechanism)', c23),
    ('C24', 'H4 window calibration', verdict_files([('hai2007', 'orvote_deploy_hai2007_summary.csv'), ('hai2204', 'orvote_deploy_hai2204_summary.csv'),
                                                    ('hai2103', 'orvote_deploy_hai2103_summary.csv')])),
    ('C25', 'Calibration development (dev plants) -> shift rule', c25),
    ('C25b', 'H5 shift calibration', verdict_files([('hai2007', 'orvote_deploy_hai2007_shift_summary.csv'), ('hai2204', 'orvote_deploy_hai2204_shift_summary.csv'),
                                                    ('hai2103', 'orvote_deploy_hai2103_shift_summary.csv')])),
    ('C26', 'H6 cross-year SWaT, shift calibration', verdict_files([('Jul2019', 'orvote_deploy_swat2019jul_shift_summary.csv'),
                                                                   ('Dec2019', 'orvote_deploy_swat2019dec_shift_summary.csv'),
                                                                   ('Nov2022', 'orvote_deploy_swat2022nov_shift_summary.csv')])),
    ('C27', 'H6: 2015 models raise false alarms on 10-62% of normal rows in 2019/2022', c27),
    ('C28', 'HAI 23.05 forecaster, seeds 1-5 (commissioning calibration): attacks caught, FAR', c28),
]


def run(root):
    res = {}
    for cid, text, fn in CLAIMS:
        try:
            res[cid] = fn(root)
        except Exception as e:  # missing file etc.
            res[cid] = ({}, f'ERROR: {type(e).__name__}: {e}')
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--other', default='../PIVOT_repro')
    args = ap.parse_args()
    a, b = run('.'), run(args.other)
    rows, md = [], ['# PaperV4 claim ledger', '',
                    f'Repo results vs independent re-run in `{args.other}` (datasets rebuilt from raw files, every model '
                    'retrained, every script re-run). Generated by `validate_claims.py`. C1-C3 (data errors) are checked '
                    'from the raw files by `validate_data_claims.py` -> `PaperV4/validation_data.txt`.', '',
                    '| Claim | Statement | Verdict (repo) | Verdict (re-run) | Status | Largest difference |', '|---|---|---|---|---|---|']
    for cid, text, _ in CLAIMS:
        (na, va), (nb, vb) = a[cid], b[cid]
        worst, wkey, status = 0.0, '', 'identical'
        for k, (x, kind) in na.items():
            y = nb.get(k, (np.nan, kind))[0]
            if pd.isna(x) and pd.isna(y):
                d = 0.0
            else:
                d = abs(float(x) - float(y)) if not (pd.isna(x) or pd.isna(y)) else np.inf
            rows.append({'claim': cid, 'key': k, 'repo': x, 'rerun': y, 'abs_diff': d, 'kind': kind})
            if d > worst:
                worst, wkey = d, k
            if d > TOL[kind]:
                status = 'DIFFERS'
            elif d > 1e-9 and status == 'identical':
                status = 'reproduced'
        if va != vb or va.startswith('ERROR'):
            status = 'DIFFERS' if not va.startswith('ERROR') else 'ERROR'
        md.append(f'| {cid} | {text} | {va} | {vb} | {status} | {wkey}: {worst:.3g} |')
    pd.DataFrame(rows).to_csv('PaperV4/claims.csv', index=False)
    open('PaperV4/CLAIMS.md', 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
