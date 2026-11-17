"""WADI -> Datasets/wadi_{train,test}_preprocessed.csv and Datasets/wadi_subsystems.json.

--release a2 (default since 2026-10-07): WaDi.A2 (iTrust, Dec 2019). WADI_14days_new.csv has the plant's unstable
  periods removed (rows 336,000-599,999, 73 h) and ends at row 1,048,571; WADI_attackdataLABLE.csv is the A1 attack
  file (identical values) plus a label column. Its Time column is Excel-mangled (mm:ss.0), so time comes from Row.
--release a1: the 2017 files (WADI_14days.csv, WADI_attackdata.csv), as used before 2026-10-07.
--labels table (default): the maintainers' corrected attack table (table_WADI.pdf in the A2 folder), 14 windows.
  The A2 label file agrees except that it places attack 13 (2PIC003, 12:07:30-12:10:52 per the table) at
  00:07:33-00:10:55, an AM/PM error, and starts most windows 3 s later.
--labels file: the A2 label column as distributed (-1 = attack), for comparison with work that used it.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser()
ap.add_argument('--release', choices=['a1', 'a2'], default='a2')
ap.add_argument('--labels', choices=['table', 'file'], default='table')
ap.add_argument('--out_dir', default='Datasets')
args = ap.parse_args()

wadi_dir = r"SWaT_Orig_DataSets/wadi"
a2_dir = os.path.join(wadi_dir, "WADI.A2_19 Nov 2019")
output_dir = args.out_dir
os.makedirs(output_dir, exist_ok=True)
TEST_START = pd.Timestamp('2017-10-09 18:00:00')   # Row 1 of the attack file (both releases)

print(f"Loading WADI Training Data ({args.release})...")
if args.release == 'a2':
    train_df = pd.read_csv(os.path.join(a2_dir, "WADI_14days_new.csv"))
    test_df = pd.read_csv(os.path.join(a2_dir, "WADI_attackdataLABLE.csv"), header=1)  # line 1 is column numbers
    test_df.columns = test_df.columns.str.strip()
    test_df = test_df.dropna(subset=['Row']).reset_index(drop=True)                   # trailing empty line
else:
    # WADI files have 4 rows of metadata at the top, so we skip them
    train_df = pd.read_csv(os.path.join(wadi_dir, "WADI_14days.csv"), skiprows=4)
    test_df = pd.read_csv(os.path.join(wadi_dir, "WADI_attackdata.csv"))
print("Loading WADI Testing Data...")
if args.labels == 'file' and args.release != 'a2':
    raise SystemExit('--labels file needs --release a2 (A1 has no label column)')

def clean_wadi_columns(df):
    # Strip whitespace
    df.columns = df.columns.str.strip()
    # WADI column names look like: \\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\1_AIT_001_PV
    # We want just '1_AIT_001_PV'
    new_cols = []
    for col in df.columns:
        if '\\' in col:
            new_cols.append(col.split('\\')[-1])
        else:
            new_cols.append(col)
    df.columns = new_cols
    return df

train_df = clean_wadi_columns(train_df)
test_df = clean_wadi_columns(test_df)

# Wait to drop non-sensor columns until after label extraction
cols_to_drop = ['Row', 'Date', 'Time']

# Ground-truth attack windows (local time), from attack_description.xlsx. They are written out here
# because the spreadsheet's date cells are corrupted: attack 5 has date 17450 and start '11.30:40',
# and attacks 11-15 carry serial 42927 (2017-07-11) instead of 2017-10-11. Parsing the xlsx dropped
# those six attacks (labels had 8 events instead of 14). Times are the xlsx time fractions; attack 5's
# start and the 11-15 date come from the official table (table_WADI.pdf). Two xlsx rows are secondary
# points of attacks 2 and 7 (inside or overlapping their windows) and are kept as in the spreadsheet.
ATTACK_WINDOWS = [
    (1, '2017-10-09 19:25:00', '2017-10-09 19:50:16'),   # 1MV001
    (2, '2017-10-10 10:24:10', '2017-10-10 10:34:00'),   # 1FIT001
    (3, '2017-10-10 10:55:00', '2017-10-10 11:24:00'),   # 2LIT002 (attacks 3-4)
    (2, '2017-10-10 11:07:46', '2017-10-10 11:12:15'),   # 1AIT001, second point of attack 2
    (5, '2017-10-10 11:30:40', '2017-10-10 11:44:50'),   # 2MCV101..601 (corrupted date in xlsx)
    (6, '2017-10-10 13:39:30', '2017-10-10 13:50:40'),   # 2MCV101, 2MCV201
    (7, '2017-10-10 14:48:17', '2017-10-10 14:59:55'),   # 1_AIT_002
    (7, '2017-10-10 14:53:44', '2017-10-10 15:00:32'),   # 2MV003, second point of attack 7
    (8, '2017-10-10 17:40:00', '2017-10-10 17:49:40'),   # 2MCV007
    (9, '2017-10-11 10:55:00', '2017-10-11 10:56:27'),   # 1P005 on, 1P006 off
    (10, '2017-10-11 11:17:54', '2017-10-11 11:31:20'),  # 1MV001
    (11, '2017-10-11 11:36:31', '2017-10-11 11:47:00'),  # 2MCV007 (xlsx date 2017-07-11)
    (12, '2017-10-11 11:59:00', '2017-10-11 12:05:00'),  # 2MCV007 (xlsx date 2017-07-11)
    (13, '2017-10-11 12:07:30', '2017-10-11 12:10:52'),  # 2PIC003 (xlsx date 2017-07-11)
    (14, '2017-10-11 12:16:00', '2017-10-11 12:25:36'),  # 1P001 and 1P003 (xlsx date 2017-07-11)
    (15, '2017-10-11 15:26:30', '2017-10-11 15:37:00'),  # 2LIT002 (xlsx date 2017-07-11)
]

# Secondary points of attacks 2 and 7 come from the A1 spreadsheet, not the official table; A2 uses the table only.
SECONDARY = {('2017-10-10 11:07:46', '2017-10-10 11:12:15'), ('2017-10-10 14:53:44', '2017-10-10 15:00:32')}
label_col_file = [c for c in test_df.columns if 'LABLE' in c.upper() or 'LABEL' in c.upper()]
if args.release == 'a2':
    test_dt = TEST_START + pd.to_timedelta(test_df['Row'].astype(int) - 1, unit='s')
else:
    test_dt = pd.to_datetime(test_df['Date'].str.strip() + ' ' + test_df['Time'].str.strip(),
                             format='%m/%d/%Y %I:%M:%S.%f %p')
if args.labels == 'file':
    y_test = (test_df[label_col_file[0]].values == -1).astype(float)
else:
    y_test = np.zeros(len(test_df))
    for _, start, end in ATTACK_WINDOWS:
        if args.release == 'a2' and (start, end) in SECONDARY:
            continue
        mask = (test_dt >= start) & (test_dt <= end)
        assert mask.any(), f"attack window {start} - {end} matches no test row"
        y_test[mask.values] = 1
test_df = test_df.drop(columns=label_col_file)
print(f"Found {int(sum(y_test))} attack samples out of {len(y_test)} ({args.release}, labels: {args.labels})")

train_df = train_df.drop(columns=[c for c in cols_to_drop if c in train_df.columns], errors='ignore')
test_df = test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns], errors='ignore')

# Ensure train and test have exactly the same sensor columns
common_cols = [c for c in train_df.columns if c in test_df.columns and not c.startswith('LEAK') and not c.startswith('PLANT') and not c.startswith('TOTAL')]
empty = [c for c in common_cols if train_df[c].isna().all()] if args.release == 'a2' else []  # A2: 2_LS_001_AL, 2_LS_002_AL, 2_P_001/002_STATUS (A1 keeps them as before)
if empty:
    print(f"Dropping {len(empty)} columns that are empty in training: {empty}")
    common_cols = [c for c in common_cols if c not in empty]
train_df = train_df[common_cols]
test_df = test_df[common_cols]

# Fill NaNs
train_df = train_df.ffill().fillna(0)
test_df = test_df.ffill().fillna(0)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_df)
X_test_scaled = scaler.transform(test_df)

train_processed = pd.DataFrame(X_train_scaled, columns=common_cols)
test_processed = pd.DataFrame(X_test_scaled, columns=common_cols)
train_processed['Normal/Attack'] = 0
test_processed['Normal/Attack'] = y_test

# Save
train_processed.to_csv(os.path.join(output_dir, "wadi_train_preprocessed.csv"), index=False)
test_processed.to_csv(os.path.join(output_dir, "wadi_test_preprocessed.csv"), index=False)

# Define Subsystems based on prefixes
subsystems = {
    "P1": [c for c in common_cols if c.startswith('1_')],
    "P2": [c for c in common_cols if c.startswith('2_') or c.startswith('2A_') or c.startswith('2B_')],
    "P3": [c for c in common_cols if c.startswith('3_')]
}

with open(os.path.join(output_dir, "wadi_subsystems.json"), "w") as f:
    json.dump(subsystems, f, indent=4)

print("WADI Preprocessing Complete!")
print(f"Total Sensors: {len(common_cols)}")
for k, v in subsystems.items():
    print(f"Subsystem {k}: {len(v)} sensors")
