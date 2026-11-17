"""Preprocess the newer iTrust SWaT releases with the 2015 scaler (pre-registered H6, preregistration/H6_swat_new_releases.md).

Releases (raw files in SWaT_Orig_DataSets/):
  swat2019jul  SWaT.A4 & A5_Jul 2019/SWaT_dataset_Jul 19 v2.xlsx  (GMT+0; stream from plant start 12:35 GMT+8; 6 attacks)
  swat2019dec  SWaT.A6_Dec 2019/csv/Dec2019.xlsx                  (GMT+8; 5 sensor/actuator disruption cycles)
  swat2022nov  SWaT.A9_Nov 2022/11-Nov-2022_0900_1143.csv         (17 attacks of Umer et al. 2026, Tables 2-3)

Output: Datasets/<release>_test_preprocessed.csv (2015 z-score scaler from Datasets/preprocessing_meta.json,
valves one-hot, 'Normal/Attack' label, 'Timestamp' local GMT+8).

Run from the repo root:  python preprocess_swat_new.py
"""
import json
import re

import pandas as pd

from preprocess_swat import transform_data

RAW = 'SWaT_Orig_DataSets/'
# attack windows, local time (GMT+8), inclusive
ATTACKS = {
    'swat2019jul': [('2019-07-20 15:08:46', '2019-07-20 15:10:31'), ('2019-07-20 15:15:00', '2019-07-20 15:19:32'),
                    ('2019-07-20 15:26:57', '2019-07-20 15:30:48'), ('2019-07-20 15:38:50', '2019-07-20 15:46:20'),
                    ('2019-07-20 15:54:00', '2019-07-20 15:56:00'), ('2019-07-20 16:02:56', '2019-07-20 16:16:18')],
    'swat2019dec': [(f'2019-12-06 {s}:00', f'2019-12-06 {e}:00') for s, e in
                    (('12:30', '12:33'), ('12:43', '12:46'), ('12:56', '12:59'), ('13:09', '13:12'), ('13:22', '13:25'))],
    # Umer et al. rules 7-9, 11-24 (start hh:mm:00 .. end hh:mm:59)
    'swat2022nov': [(f'2022-11-11 {s}:00', f'2022-11-11 {e}:59') for s, e in
                    (('09:51', '09:55'), ('09:59', '10:01'), ('10:03', '10:05'), ('10:09', '10:12'), ('10:14', '10:17'),
                     ('10:20', '10:22'), ('10:24', '10:26'), ('10:29', '10:32'), ('10:34', '10:37'), ('10:41', '10:44'),
                     ('10:47', '10:50'), ('10:53', '10:56'), ('11:05', '11:08'), ('11:10', '11:13'), ('11:16', '11:19'),
                     ('11:22', '11:25'), ('11:28', '11:31'))],
}
# Label-alignment check (allowed by the pre-registration): in the Jul 2019 data every documented action appears
# ~105 s before the PDF time (second-resolution starts: FIT401 -106, P601 -104, MV201 -98, P301 -110 s; median -105),
# so all Jul times (attacks and plant start) are shifted by -105 s. Dec 2019 (12:43 actuator cycle) and Nov 2022
# (FIT101 10:02:50, MV201 11:15:31) agree with the documentation to within a minute: no shift.
CLOCK_OFFSET_S = {'swat2019jul': -105, 'swat2019dec': 0, 'swat2022nov': 0}
STREAM_START = {'swat2019jul': '2019-07-20 12:35:00', 'swat2019dec': '2019-12-06 10:05:00',
                'swat2022nov': '2022-11-11 09:00:00'}


def tag(c):
    """'FIT 101' / 'FIT101.Pv' / 'P101 Status' / 'P101.Status' -> 'FIT101' / 'P101'."""
    c = re.sub(r'(\.Pv|\.Status|\.Alarm| Status)$', '', str(c).strip())
    return c.replace(' ', '')


def load(rel):
    if rel == 'swat2019jul':
        df = pd.read_excel(RAW + 'SWaT.A4 & A5_Jul 2019/SWaT_dataset_Jul 19 v2.xlsx', header=1, skiprows=[2])
        ts = pd.to_datetime(df.iloc[:, 0], utc=True, format='ISO8601').dt.tz_convert('Asia/Singapore').dt.tz_localize(None)
    elif rel == 'swat2019dec':
        df = pd.read_excel(RAW + 'SWaT.A6_Dec 2019/csv/Dec2019.xlsx', header=9)
        ts = pd.to_datetime(df.iloc[:, 0])
    else:
        df = pd.read_csv(RAW + 'SWaT.A9_Nov 2022/11-Nov-2022_0900_1143.csv')
        ts = pd.to_datetime(df.iloc[:, 0], format='%d-%b-%Y %H:%M:%S')
    df = df.iloc[:, 1:]
    df.columns = [tag(c) for c in df.columns]
    df = df.apply(pd.to_numeric, errors='coerce')
    df.insert(0, 'Timestamp', ts.dt.round('s').values)
    return df


def main():
    meta = json.load(open('Datasets/preprocessing_meta.json'))
    for rel in ATTACKS:
        df = load(rel)
        off = pd.Timedelta(seconds=CLOCK_OFFSET_S[rel])
        df = df[df.Timestamp >= pd.Timestamp(STREAM_START[rel]) + off].reset_index(drop=True)
        y = pd.Series(0, index=df.index)
        for s, e in ATTACKS[rel]:
            y[(df.Timestamp >= pd.Timestamp(s) + off) & (df.Timestamp <= pd.Timestamp(e) + off)] = 1
        df['Normal/Attack'] = y.map({0: 'Normal', 1: 'Attack'})
        need = meta['continuous_features'] + meta['valves_encoded']
        missing = [c for c in need if c not in df.columns]
        nan_cols = {c: int(df[c].isna().sum()) for c in need if c in df.columns and df[c].isna().any()}
        gaps = df.Timestamp.diff().dt.total_seconds().value_counts().head(3).to_dict()
        out = transform_data(df, meta)
        out.to_csv(f'Datasets/{rel}_test_preprocessed.csv', index=False)
        n_ev = (y.diff().fillna(y.iloc[0]) == 1).sum()
        print(f'{rel}: {len(df):,} rows {df.Timestamp.iloc[0]} .. {df.Timestamp.iloc[-1]}, {n_ev} attacks, '
              f'{int(y.sum()):,} attack rows; step counts {gaps}; missing {missing}; NaN {nan_cols}')


if __name__ == '__main__':
    main()
