"""Independent checks of the evaluation paper's data claims (C1-C3), from the raw files only (no pipeline code).

C1  HAI 23.05: label-test2.csv has minute-resolution timestamps, so a join on the timestamp text labels the 40 test2
    attacks normal; aligning by row recovers them. Also: segment counts per test file vs summary_label*.txt.
C2  WADI: labelled attack events in the A1 test file as preprocessed before the fix (8) vs the A2 attack table (14).
C3  SWaT Jul 2019: offset between the attack PDF times (GMT+8) and actuator/spoof transitions in the data.

Run from the repo root:  python validate_data_claims.py  -> PaperV4/validation_data.txt
"""
import os

import numpy as np
import pandas as pd

OUT = []


def say(s=''):
    print(s); OUT.append(s)


def segments(y):
    y = np.asarray(y).astype(int)
    return int(((np.diff(np.r_[0, y]) == 1)).sum())


# ------------------------------------------------------------------ C1 HAI 23.05 labels
H = 'SWaT_Orig_DataSets/hai/hai-23.05'
say('C1  HAI 23.05 test labels')
tot_join, tot_pos = 0, 0
for k in (1, 2):
    lab = pd.read_csv(f'{H}/label-test{k}.csv')
    dat = pd.read_csv(f'{H}/hai-test{k}.csv', usecols=['timestamp'])
    has_sec = lab['timestamp'].str.count(':').eq(2).mean()
    merged = dat.merge(lab.drop_duplicates('timestamp'), on='timestamp', how='left')
    join_y = merged['label'].fillna(0).astype(int)
    pos_y = lab['label'].astype(int)
    tdat, tlab = pd.to_datetime(dat['timestamp']), pd.to_datetime(lab['timestamp'])
    aligned = (tdat.dt.floor('s' if has_sec > 0.5 else 'min') == tlab).mean()
    tot_join += segments(join_y); tot_pos += segments(pos_y)
    say(f'  test{k}: {len(dat):,} data rows, {len(lab):,} label rows, label timestamps with seconds: {has_sec:.0%}; '
        f'rows matched by timestamp join: {merged["label"].notna().mean():.1%}; attack segments by join: '
        f'{segments(join_y)}, by row: {segments(pos_y)} ({int(pos_y.sum()):,} attack rows); '
        f'row-wise time agreement at label resolution: {aligned:.2%}')
    summ = f'{H}/summary_label{k}.txt'
    if os.path.exists(summ):
        say(f'    summary_label{k}.txt: ' + ' | '.join(open(summ).read().split('\n')[:3]))
say(f'  total attack segments: timestamp join {tot_join}, row alignment {tot_pos}')

# ------------------------------------------------------------------ C2 WADI A1 vs A2
say('\nC2  WADI attack events')
for name, f in (('A1 file as preprocessed before the 2026-10-06 fix', 'label_corrections/wadi_a1_test_labels_corrected.csv'),
                ('A2 (Datasets/wadi_test_preprocessed.csv, maintainers\' table)', 'Datasets/wadi_test_preprocessed.csv')):
    if not os.path.exists(f):
        say(f'  {name}: file {f} missing'); continue
    d = pd.read_csv(f)
    d.columns = d.columns.str.strip()
    cols = [c for c in d.columns if 'label' in c.lower() or c == 'Normal/Attack' or c.startswith('y_')]
    for c in cols:
        v = d[c]
        y = v.astype(str).str.lower().str.contains('attack').astype(int) if v.dtype == object else (v > 0).astype(int)
        say(f'  {name} [{c}]: {segments(y)} events, {int(y.sum()):,} attack rows')

# ------------------------------------------------------------------ C3 SWaT Jul 2019 offset
say('\nC3  SWaT Jul 2019 clock offset (data time minus attack-PDF time, GMT+8)')
x = pd.read_excel('SWaT_Orig_DataSets/SWaT.A4 & A5_Jul 2019/SWaT_dataset_Jul 19 v2.xlsx', header=1, skiprows=[2])
ts = pd.to_datetime(x.iloc[:, 0], utc=True, format='ISO8601').dt.tz_convert('Asia/Singapore').dt.tz_localize(None)
x = x.set_axis([str(c).replace(' Status', '').replace(' ', '') for c in x.columns], axis=1)
x.index = ts.dt.round('s')
anchors = []


def first_change(col, after, before, to=None):
    s = pd.to_numeric(x[col], errors='coerce').loc[after:before]
    ch = s[s.diff().fillna(0) != 0]
    if to is not None:
        ch = ch[ch == to]
    return ch.index[0] if len(ch) else None


for col, pdf, to, desc in (('P601', '2019-07-20 15:26:57', 2, 'P601 OFF->ON'),
                           ('MV201', '2019-07-20 15:38:50', 2, 'MV201 CLOSE->OPEN'),
                           ('P301', '2019-07-20 16:02:56', 1, 'P301 ON->OFF')):
    t = first_change(col, pd.Timestamp(pdf) - pd.Timedelta('5min'), pd.Timestamp(pdf) + pd.Timedelta('5min'), to)
    off = (t - pd.Timestamp(pdf)).total_seconds()
    anchors.append(off); say(f'  {desc}: PDF {pdf[11:]}, data {t.time()}, offset {off:+.0f} s')
f = pd.to_numeric(x['FIT401'], errors='coerce').loc['2019-07-20 15:04':'2019-07-20 15:12']
spoof = f[(f - 0.5).abs() < 0.02]
end = spoof.index[-1]
off = (end - pd.Timestamp('2019-07-20 15:10:31')).total_seconds()
anchors.append(off); say(f'  FIT401 spoof (0.5) end: PDF 15:10:31, data {end.time()}, offset {off:+.0f} s')
say(f'  median offset {np.median(anchors):+.1f} s over {len(anchors)} second-resolution anchors')

os.makedirs('PaperV4', exist_ok=True)
open('PaperV4/validation_data.txt', 'w').write('\n'.join(OUT) + '\n')
