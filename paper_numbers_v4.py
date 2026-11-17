"""Write PaperV4/numbers.tex: every number in the evaluation paper, read from the validated result files.

Validation (PaperV4/VALIDATION.md): every file read here was reproduced by an independent re-run from the raw data;
re-run values quoted in the paper (instability of H2 on SWaT, forecaster seeds) come from PaperV4/claims.csv.

Sources (run their scripts first; see PaperV4/OUTLINE.md for the claim map)
  PaperV4/validation_data.txt, PaperV4/claims.csv      validate_data_claims.py, validate_claims.py
  leakage_effect.csv                                    leakage_effect.py
  baselines_simple_label_free_swat.csv                  baselines_simple_label_free.py
  baselines_simple_generalization.csv                   baselines_simple_generalization.py
  decomposition_confirm_{wadi,hai}_{summary,labelfree}.csv, physics_or_test_*_{summary,labelfree}.csv
  physics_ablation_label_free_swat.csv, decomposition_ablation_label_free_swat.csv
  matched_far_attack_swat.csv, feasible_attack_label_free_swat.csv
  deployable_test_*.csv, orvote_confirm_*.csv, orvote_mechanism_*.csv, drift_isolation_hai2204.csv
  calibration_dev_{swat,wadi,hai}.csv, orvote_deploy_*.csv, preds_hai_temporal_seed<k>.csv
  Datasets/*_preprocessed.csv, Datasets/*_subsystems.json (dataset table)

Run from the repo root:  python paper_numbers_v4.py
"""
import json
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '.')
import label_free as lfp
from evaluate_events import evaluate, to_binary

OUT = 'PaperV4/numbers.tex'
macros = {}


def put(name, value):
    assert re.fullmatch(r'[A-Za-z]+', name), name
    assert name not in macros, name
    macros[name] = value


n0 = lambda x: f'{x:.0f}'
n1 = lambda x: f'{x:.1f}'
mi = lambda s: s.replace('-', '\\textminus{}')
sg = lambda x: mi(f'{x:+.1f}')
s1 = lambda x: mi(f'{x:.1f}')
pc = lambda x: f'{100 * x:.1f}'
p2 = lambda x: f'{100 * x:.2f}'
th = lambda x: f'{x:,.0f}'.replace(',', '{,}')
FARS = {0.001: 'a', 0.0025: 'q', 0.005: 'b', 0.01: 'c', 0.02: 'd'}
REL = {'hai2007': ('A', 'HAI 20.07'), 'hai2204': ('B', 'HAI 22.04'), 'hai2103': ('C', 'HAI 21.03')}
SNEW = {'swat2019jul': ('J', 'SWaT Jul 2019'), 'swat2019dec': ('D', 'SWaT Dec 2019'), 'swat2022nov': ('N', 'SWaT Nov 2022')}


def segments(y):
    return int((np.diff(np.r_[0, np.asarray(y).astype(int)]) == 1).sum())


# ------------------------------------------------------------------ Table 1: plants and releases
def label_of(df):
    c = 'Label' if 'Label' in df.columns else 'Normal/Attack'
    return to_binary(df[c]).astype(int)


def nrows(path):
    with open(path) as f:
        return sum(1 for _ in f) - 1


rows = []
swat_tags = None
import decomposition_ablation_label_free as dab
swat_tags = sum(len(v) for v in dab.partition(6, 'split').values())
spec = [('SWaT', '2015 (A1)', 'swat', 'Datasets/swat_train_preprocessed.csv', 'Datasets/swat_test_preprocessed.csv',
         swat_tags, 'development'),
        ('SWaT', 'Jul 2019 (A4/A5)', None, None, 'Datasets/swat2019jul_test_preprocessed.csv', swat_tags, 'cross-year'),
        ('SWaT', 'Dec 2019 (A6)', None, None, 'Datasets/swat2019dec_test_preprocessed.csv', swat_tags, 'cross-year'),
        ('SWaT', 'Nov 2022 (A9)', None, None, 'Datasets/swat2022nov_test_preprocessed.csv', swat_tags, 'cross-year'),
        ('WADI', 'A2 (2019)', 'wadi', 'Datasets/wadi_train_preprocessed.csv', 'Datasets/wadi_test_preprocessed.csv',
         sum(len(v) for v in json.load(open('Datasets/wadi_subsystems.json')).values()), 'development'),
        ('HAI', '23.05', 'hai', 'Datasets/hai_train_preprocessed.csv', 'Datasets/hai_test_preprocessed.csv',
         sum(len(v) for v in json.load(open('Datasets/hai2305_subsystems.json')).values()), 'development')]
for rel, (k, name) in REL.items():
    spec.append(('HAI', name.split()[1], rel, f'Datasets/{rel}_train_preprocessed.csv', f'Datasets/{rel}_test_preprocessed.csv',
                 sum(len(v) for v in json.load(open(f'Datasets/{rel}_subsystems.json')).values()), 'held out'))
for plant, release, key, trp, tep, tags, role in spec:
    te = pd.read_csv(tep, usecols=lambda c: c.strip() in ('Label', 'Normal/Attack'))
    te.columns = te.columns.str.strip()
    y = label_of(te)
    tr_rows = th(nrows(trp)) if trp else '(2015)'
    rows.append(f'{plant} & {release} & {tags} & {tr_rows} & {th(len(y))} & {segments(y)} & {role} \\\\')
    if key == 'swat':
        put('swatN', str(segments(y)))
    if key == 'wadi':
        put('wadiN', str(segments(y)))
    if key == 'hai':
        put('haiN', str(segments(y)))
put('dataRows', '\n'.join(rows))
for rel, (k, name) in SNEW.items():
    y = label_of(pd.read_csv(f'Datasets/{rel}_test_preprocessed.csv', usecols=['Normal/Attack']))
    put(f'new{k}N', str(segments(y)))

# ------------------------------------------------------------------ C1-C3 data errors (validate_data_claims.py)
txt = open('PaperV4/validation_data.txt').read()
m = re.search(r'test2: ([\d,]+) data rows.*?attack segments by join: (\d+), by row: (\d+)', txt)
put('haiTestTwoRows', th(int(m.group(1).replace(',', '')))); put('haiTestTwoSeg', m.group(3))
m = re.search(r'total attack segments: timestamp join (\d+), row alignment (\d+)', txt)
put('haiJoinSeg', m.group(1)); put('haiRowSeg', m.group(2))
m = re.search(r'\[y_submitted\]: (\d+) events', txt); put('wadiOldN', m.group(1))
m = re.search(r'median offset ([-+\d.]+) s over (\d+)', txt)
put('julOffset', n1(abs(float(m.group(1))))); put('julAnchors', m.group(2))

# ------------------------------------------------------------------ C4-C5 label convention and leakage
lk = pd.read_csv('leakage_effect.csv').groupby(['plant', 'protocol'], sort=False).agg(
    F1=('F1', 'mean'), FAR=('FAR', 'mean'), caught=('caught', 'mean'), n=('n_events', 'first'),
    chance=('chance', 'mean'), beyond=('beyond', 'mean'))
PROT = [('label-free', 'Label-free'), ('oracle', 'Test-normal calibration'), ('oracle + F1', '\\quad + F1-tuned threshold'),
        ('oracle + F1 +600s', '\\quad + labels extended 600\\,s')]
lrows = []
for plant in ('SWaT', 'WADI', 'HAI'):
    lrows.append(f'\\multicolumn{{6}}{{@{{}}l}}{{\\emph{{{plant if plant != "HAI" else "HAI 23.05"}}}}} \\\\')
    for p, name in PROT:
        if (plant, p) not in lk.index:
            continue
        x = lk.loc[(plant, p)]
        lrows.append(f'{name} & {pc(x.F1)} & {p2(x.FAR)} & {n1(x.caught)}/{int(x.n)} & {n1(x.chance)} & {n1(x.beyond)} \\\\')
        key = {'label-free': 'Lf', 'oracle': 'Or', 'oracle + F1': 'Of', 'oracle + F1 +600s': 'Os'}[p] + plant.capitalize()
        put(f'fOne{key}', pc(x.F1)); put(f'beyond{key}', n1(x.beyond))
put('leakRows', '\n'.join(lrows))
put('swatNsix', str(int(lk.loc[('SWaT', 'oracle + F1 +600s')].n)))

# ------------------------------------------------------------------ C6-C10 chance, late alarms, simple baselines
g = pd.read_csv('baselines_simple_generalization.csv').set_index(['plant', 'detector'])
x = g.loc[('HAI', 'MinMax (tol 0%)')]
put('hmmCaught', n0(x.events)); put('hmmChance', n1(x.chance)); put('hmmFar', pc(x.FAR))
b = pd.read_csv('baselines_simple_label_free_swat.csv').groupby('detector').mean(numeric_only=True)
x = b.loc['Gradient (tol 0%)']
put('gradCaught', n0(x.events)); put('gradEarly', n0(x.early))
SW = [('PIVOT array (6 stages)', 'PIVOT'), ('AE (monolithic, as in paper)', 'Monolithic AE'),
      ('MinMax (tol 5% of range)', 'Range check (5\\%)'), ('PCA-SPE (14 components)', 'PCA-SPE')]
for det, name in SW:
    key = {'PIVOT': 'Pv', 'Monolithic AE': 'Mo', 'Range check (5\\%)': 'Mm', 'PCA-SPE': 'Pca'}[name]
    for arm, sfx in (('', ''), (' + analyzer arm', 'Arm')):
        r = b.loc[det + arm]
        put(f'sw{key}{sfx}Beyond', n1(r.excess)); put(f'sw{key}{sfx}Far', p2(r.FAR)); put(f'sw{key}{sfx}Ep', n0(r.FA_episodes))
for plant, f in (('WADI', 'decomposition_confirm_wadi_labelfree.csv'), ('HAI', 'decomposition_confirm_hai_labelfree.csv')):
    s = pd.read_csv(f).groupby('detector').mean(numeric_only=True)
    k = plant.capitalize()
    put(f'lf{k}StBeyond', n1(s.loc['stages'].beyond)); put(f'lf{k}StFar', p2(s.loc['stages'].FAR))
    put(f'lf{k}MoBeyond', n1(s.loc['monolithic'].beyond)); put(f'lf{k}MoFar', p2(s.loc['monolithic'].FAR))
    put(f'lf{k}StEp', n0(s.loc['stages'].FA_episodes)); put(f'lf{k}MoEp', n0(s.loc['monolithic'].FA_episodes))
for plant, det, key in (('WADI', 'MinMax (tol 10%)', 'WadiMmTen'), ('WADI', 'MinMax (tol 0%)', 'WadiMmZero'),
                        ('HAI', 'MinMax (tol 5%)', 'HaiMmFive'), ('HAI', 'PCA-SPE (25 comp., train)', 'HaiPca')):
    r = g.loc[(plant, det)]
    put(f'b{key}Beyond', n1(r.beyond)); put(f'b{key}Far', p2(r.FAR))
wpca = [d for d in g.loc['WADI'].index if d.startswith('PCA')][0]
put('bWadiPcaBeyond', n1(g.loc[('WADI', wpca)].beyond)); put('bWadiPcaFar', p2(g.loc[('WADI', wpca)].FAR))
# PIVOT's own rule label-free (per-stage P99.8 of training scores, OR; per-sensor scoring) vs monolithic, 3 plants
for plant in ('swat', 'wadi', 'hai'):
    o = pd.read_csv(f'orvote_check_{plant}_labelfree.csv').groupby('detector').mean(numeric_only=True)
    k = plant.capitalize()
    put(f'or{k}Beyond', n1(o.loc['OR vote, per-sensor scoring'].beyond)); put(f'or{k}Far', p2(o.loc['OR vote, per-sensor scoring'].FAR))
    put(f'om{k}Beyond', n1(o.loc['monolithic, per-sensor scoring'].beyond)); put(f'om{k}Far', p2(o.loc['monolithic, per-sensor scoring'].FAR))
# forecaster, seeds 1-5
fc = []
for k in range(1, 6):
    d = pd.read_csv(f'preds_hai_temporal_seed{k}.csv')
    yy, pp = d.y_true.values.astype(int), d.y_pred.values.astype(int)
    s, _ = evaluate(yy, pp); ch, _ = lfp.chance_events(pp, yy)
    fc.append((s['n_detected'], ch, s['false_alarm']))
fc = np.array(fc, dtype=float)
put('fcCaught', n1(fc[:, 0].mean())); put('fcBeyond', n1((fc[:, 0] - fc[:, 1]).mean())); put('fcFar', p2(fc[:, 2].mean()))
put('fcMin', n0(fc[:, 0].min())); put('fcMax', n0(fc[:, 0].max()))

# ------------------------------------------------------------------ C11-C15 physics, robustness, decomposition on SWaT
p = pd.read_csv('physics_ablation_label_free_swat.csv').query('protocol == "dual"').groupby('physics').mean(numeric_only=True)
for arm in ('none', 'nonneg', 'interlock', 'legacy'):
    put(f'phys{arm.capitalize()}FOne', pc(p.loc[arm].F1)); put(f'phys{arm.capitalize()}Beyond', n1(p.loc[arm].excess))
cl = pd.read_csv('PaperV4/claims.csv')


def claim(cid, key, col):
    return float(cl[(cl.claim == cid) & (cl.key == key)][col].iloc[0])


for plant, f in (('Swat', 'physics_or_test_swat_summary.csv'), ('Wadi', 'physics_or_test_wadi_summary.csv'),
                 ('Hai', 'physics_or_test_hai_summary.csv')):
    s = pd.read_csv(f).set_index('target_FAR')
    for t in (0.01, 0.02):
        put(f'hTwo{plant}Diff{FARS[t]}', sg(s.loc[t, 'diff'])); put(f'hTwo{plant}Lo{FARS[t]}', s1(s.loc[t, 'CI_low']))
        put(f'hTwo{plant}Hi{FARS[t]}', s1(s.loc[t, 'CI_high']))
    put(f'hTwo{plant}ArmFar', p2(float(re.search(r'FAR ([\d.]+)%', s.loc[0.005, 'note']).group(1)) / 100))
put('hTwoSwatRerunDiffc', sg(claim('C12', 'SWaT 0.01 diff', 'rerun'))); put('hTwoSwatRerunLoc', s1(claim('C12', 'SWaT 0.01 CI_low', 'rerun')))
lf = pd.read_csv('physics_or_test_swat_labelfree.csv').groupby('detector', sort=False).agg(
    ep=('FA_episodes', 'mean'), beyond=('beyond', 'mean'), FAR=('FAR', 'mean'))
put('hTwoSwatEpBase', n0(lf.loc['PIVOT (stages, per-sensor)'].ep)); put('hTwoSwatEpUnion', n0(lf.loc['PIVOT OR physics'].ep))
put('hTwoSwatLfBase', n1(lf.loc['PIVOT (stages, per-sensor)'].beyond)); put('hTwoSwatLfUnion', n1(lf.loc['PIVOT OR physics'].beyond))
put('hTwoSwatLfBaseFar', p2(lf.loc['PIVOT (stages, per-sensor)'].FAR)); put('hTwoSwatLfUnionFar', p2(lf.loc['PIVOT OR physics'].FAR))
mf = pd.read_csv('matched_far_attack_swat.csv').groupby(['target_FAR', 'threat', 'epsilon', 'detector']).excess.mean()
for t, tk in ((0.002, 'Lo'), (0.01, 'Hi')):
    for det, dk in (('array', 'Ar'), ('mono', 'Mo')):
        put(f'mf{tk}{dk}Clean', n1(mf.loc[(t, 'clean', 0.0, det)]))
        put(f'mf{tk}{dk}Sens', s1(mf.loc[(t, 'sensors', 8.0, det)])); put(f'mf{tk}{dk}Act', s1(mf.loc[(t, 'actuators', 8.0, det)]))
fa = pd.read_csv('feasible_attack_label_free_swat.csv')
put('feasClean', n0(fa.query('threat == "clean" and detector == "union"').events.mean()))
big = fa.query('detector == "union" and objective == "best" and epsilon == 32.0')
put('feasBigMax', n0(big.events.max()))
d = pd.read_csv('decomposition_ablation_label_free_swat.csv')
d['ex'] = d.events - d.chance
gg = d.query('analyzers == "split" and decision == "dual"').groupby(['arm', 'K']).agg(F1=('F1', 'mean'), FAR=('FAR', 'mean'), ex=('ex', 'mean'), ep=('FA_episodes', 'mean'))
for arm, ak in ((False, ''), (True, 'Arm')):
    for k, kk in ((6, 'Six'), (1, 'One')):
        r = gg.loc[(arm, k)]
        put(f'k{kk}{ak}FOne', pc(r.F1)); put(f'k{kk}{ak}Far', p2(r.FAR)); put(f'k{kk}{ak}Beyond', n1(r.ex)); put(f'k{kk}{ak}Ep', n0(r.ep))

# ------------------------------------------------------------------ pre-registered tests H1, H1b, H1c, H3-H6
for plant in ('wadi', 'hai'):
    s = pd.read_csv(f'decomposition_confirm_{plant}_summary.csv').query('primary').set_index('target_FAR')
    for t in s.index:
        put(f'hOne{plant.capitalize()}Diff{FARS[t]}', sg(s.loc[t, 'diff']))
        put(f'hOne{plant.capitalize()}Lo{FARS[t]}', s1(s.loc[t, 'CI_low'])); put(f'hOne{plant.capitalize()}Hi{FARS[t]}', s1(s.loc[t, 'CI_high']))
    put(f'hOne{plant.capitalize()}Pass', str(int((s.CI_low > 0).sum())))
for rel, cal, key in (('hai2204', '', 'hOneBB'), ('hai2103', '', 'hOneBC'), ('hai2007', '_commission', 'hOneCA')):
    s = pd.read_csv(f'deployable_test_{rel}{cal}_summary.csv').iloc[0]
    sd = pd.read_csv(f'deployable_test_{rel}{cal}_seeds.csv').groupby('detector').FAR.mean()
    put(f'{key}Diff', sg(s['diff'])); put(f'{key}Lo', s1(s.CI_low)); put(f'{key}Hi', s1(s.CI_high))
    put(f'{key}StFar', pc(sd['stages'])); put(f'{key}MoFar', pc(sd['monolithic']))
for rel, (k, name) in REL.items():
    s = pd.read_csv(f'orvote_confirm_{rel}_summary.csv').set_index('target_FAR')
    for t in s.index:
        put(f'hThree{k}Diff{FARS[t]}', sg(s.loc[t, 'diff'])); put(f'hThree{k}Lo{FARS[t]}', s1(s.loc[t, 'CI_low']))
        put(f'hThree{k}Hi{FARS[t]}', s1(s.loc[t, 'CI_high']))
    put(f'hThree{k}Pass', str(int((s.CI_low > 0).sum())))
    put(f'hThree{k}Min', sg(s['diff'].min())); put(f'hThree{k}Max', sg(s['diff'].max()))
    put(f'hThree{k}MinU', n1(s['diff'].min())); put(f'hThree{k}MaxU', n1(s['diff'].max()))  # unsigned, for prose
    e = pd.read_csv(f'orvote_mechanism_{rel}.csv').groupby(['target_FAR', 'detector', 'event']).excess.mean() \
        .groupby(['target_FAR', 'detector']).sum().unstack()
    sep = e['stage models, OR'] - e['monolithic, grouped OR']
    put(f'mech{k}Min', sg(sep.min())); put(f'mech{k}Max', sg(sep.max()))
dr = pd.read_csv('drift_isolation_hai2204.csv').set_index('stage')
put('driftPTwoTrain', n1(dr.loc['P2'].train_p99)); put('driftPTwoTest', n1(dr.loc['P2'].test_normal_median))
for cal, h in (('', 'Four'), ('_shift', 'Five')):
    for rel, (k, name) in REL.items():
        s = pd.read_csv(f'orvote_deploy_{rel}{cal}_summary.csv').iloc[0]
        sd = pd.read_csv(f'orvote_deploy_{rel}{cal}_seeds.csv').query('window == "primary"').groupby('detector').agg(
            FAR=('FAR', 'mean'), beyond=('beyond', 'mean'), ep=('FA_episodes', 'mean'))
        put(f'h{h}{k}Diff', sg(s['diff'])); put(f'h{h}{k}Lo', s1(s.CI_low)); put(f'h{h}{k}Hi', s1(s.CI_high))
        put(f'h{h}{k}Ok', str(int(s.far_ok_seeds)))
        put(f'h{h}{k}PvFar', pc(sd.loc['pivot'].FAR)); put(f'h{h}{k}MoFar', pc(sd.loc['mono'].FAR))
for rel, (k, name) in SNEW.items():
    s = pd.read_csv(f'orvote_deploy_{rel}_shift_summary.csv').iloc[0]
    sd = pd.read_csv(f'orvote_deploy_{rel}_shift_seeds.csv').query('window == "primary"').groupby('detector').agg(
        FAR=('FAR', 'mean'), beyond=('beyond', 'mean'), caught=('caught', 'mean'))
    put(f'hSix{k}Diff', sg(s['diff'])); put(f'hSix{k}Lo', s1(s.CI_low)); put(f'hSix{k}Hi', s1(s.CI_high))
    put(f'hSix{k}Ok', str(int(s.far_ok_seeds)))
    put(f'hSix{k}PvFar', pc(sd.loc['pivot'].FAR)); put(f'hSix{k}MoFar', pc(sd.loc['mono'].FAR))
    put(f'hSix{k}PvBeyond', n1(sd.loc['pivot'].beyond)); put(f'hSix{k}MoBeyond', n1(sd.loc['mono'].beyond))
    put(f'hSix{k}PvCaught', n1(sd.loc['pivot'].caught))
put('hSixJMoFarRerun', n1(claim('C27', 'swat2019jul mono FAR%', 'rerun')))
cd = {ds: pd.read_csv(f'calibration_dev_{ds}.csv').groupby(['method', 'detector']).agg(FAR=('FAR', 'mean'), beyond=('beyond', 'mean'))
      for ds in ('swat', 'wadi', 'hai')}
for ds, c in cd.items():
    for m_ in ('window', 'shift'):
        put(f'cal{ds.capitalize()}{m_.capitalize()}PvFar', pc(c.loc[(m_, 'pivot')].FAR))
        put(f'cal{ds.capitalize()}{m_.capitalize()}PvBeyond', s1(c.loc[(m_, 'pivot')].beyond))

# ------------------------------------------------------------------ Table: the pre-registered tests
def verdict(ok):
    return 'yes' if ok else 'no'


h1 = {p: int(macros[f'hOne{p}Pass']) >= 2 for p in ('Wadi', 'Hai')}
h2 = {}
for plant in ('Swat', 'Wadi', 'Hai'):
    s = pd.read_csv(f'physics_or_test_{plant.lower()}_summary.csv')
    h2[plant] = int((s.CI_low > 0).sum()) >= 2
sup = lambda f: bool(pd.read_csv(f).iloc[0]['supported'])
vrd = lambda f: pd.read_csv(f).iloc[0]['verdict'] == 'SUPPORTED'
prereg = [
    ('H1', 'Stage models beat one model at matched FAR (one threshold)', f'WADI {verdict(h1["Wadi"])}, HAI 23.05 {verdict(h1["Hai"])}'),
    ('H2', 'OR-ing physics invariants adds attacks at matched FAR', f'SWaT {verdict(h2["Swat"])}$^\\dagger$, WADI {verdict(h2["Wadi"])}, HAI 23.05 {verdict(h2["Hai"])}'),
    ('H1b', 'Stages, training thresholds: fewer false alarms, not fewer attacks',
     f'22.04 {verdict(sup("deployable_test_hai2204_summary.csv"))}, 21.03 {verdict(sup("deployable_test_hai2103_summary.csv"))}'),
    ('H1c', 'As H1b, commissioning-window thresholds', f'20.07 {verdict(sup("deployable_test_hai2007_commission_summary.csv"))}'),
    ('H3', 'Per-stage thresholds + OR beat one model at matched FAR',
     ', '.join(f'{n.split()[1]} {verdict(int(macros[f"hThree{k}Pass"]) >= 2)}' for r, (k, n) in REL.items())),
    ('H4', 'H3 calibrated on a 30-min window (window quantiles)',
     ', '.join(f'{n.split()[1]} {verdict(vrd(f"orvote_deploy_{r}_summary.csv"))}' for r, (k, n) in REL.items())),
    ('H5', 'H3 calibrated on a 30-min window (shift rule)',
     ', '.join(f'{n.split()[1]} {verdict(vrd(f"orvote_deploy_{r}_shift_summary.csv"))}' for r, (k, n) in REL.items())),
    ('H6', 'H5 on SWaT 3.5--7 years after training',
     ', '.join(f'{n.split(" ", 1)[1]} {verdict(vrd(f"orvote_deploy_{r}_shift_summary.csv"))}' for r, (k, n) in SNEW.items())),
]
put('preregRows', '\n'.join(f'{h} & {t} & {v} \\\\' for h, t, v in prereg))
n_tests = sum(v.count('yes') + v.count('no') for _, _, v in prereg)
n_yes = sum(v.count('yes') for _, _, v in prereg)
put('nReleaseTests', str(n_tests)); put('nReleaseYes', str(n_yes)); put('nTests', str(len(prereg)))

with open(OUT, 'w') as f:
    f.write('%% generated by paper_numbers_v4.py -- do not edit by hand\n')
    for k, v in macros.items():
        f.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
print(f'[SAVED] {OUT}: {len(macros)} macros; release-level tests {n_yes} of {n_tests} supported')
