"""HAI 23.05 -> Datasets/hai_{train,test}_preprocessed.csv.

Labels are aligned to the test rows by POSITION, not by joining on the timestamp text. label-test2.csv stores
minute-resolution timestamps ('2022-08-17 1:27') while hai-test2.csv has seconds ('2022-08-17 00:00:01'), so the
earlier left join matched no row and a fillna(0) labelled all 40 test2 attacks normal (fixed 2026-10-07; 14 of the
54 attacks were labelled before). Alignment is checked: equal row counts and, row by row, the same timestamp at
the label file's resolution.
"""
import argparse
import os

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--out_dir', default='Datasets')
ap.add_argument('--test_only', action='store_true', help='skip rewriting the (unchanged) training file')
args = ap.parse_args()

hai_dir = r"SWaT_Orig_DataSets/hai/hai-23.05"
out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)

if not args.test_only:
    print("Processing HAI Training Data...")
    train_files = ["hai-train1.csv", "hai-train2.csv", "hai-train3.csv", "hai-train4.csv"]
    train_dfs = [pd.read_csv(os.path.join(hai_dir, f)) for f in train_files]
    df_train = pd.concat(train_dfs, ignore_index=True)

    # Remove timestamp
    if 'timestamp' in df_train.columns:
        df_train = df_train.drop(columns=['timestamp'])

    df_train.to_csv(os.path.join(out_dir, "hai_train_preprocessed.csv"), index=False)
    print(f"Saved {len(df_train)} training rows.")

print("Processing HAI Testing Data...")
test_files = ["hai-test1.csv", "hai-test2.csv"]
label_files = ["label-test1.csv", "label-test2.csv"]

test_dfs = []
for tf, lf in zip(test_files, label_files):
    df_t = pd.read_csv(os.path.join(hai_dir, tf))
    df_l = pd.read_csv(os.path.join(hai_dir, lf))
    if len(df_t) != len(df_l):
        raise SystemExit(f"{tf}: {len(df_t)} rows but {lf} has {len(df_l)} labels")
    t_data = pd.to_datetime(df_t['timestamp'])
    t_label = pd.to_datetime(df_l['timestamp'])
    has_seconds = (t_label.dt.second != 0).any()
    if not (t_data.dt.floor('s' if has_seconds else 'min') == t_label).all():
        raise SystemExit(f"{tf} and {lf}: timestamps do not line up row by row")

    df_merged = df_t.drop(columns=['timestamp'])
    df_merged['Label'] = df_l['label'].astype(int).values
    n_att = int((df_merged['Label'].diff().fillna(df_merged['Label']) == 1).sum())
    print(f"  {tf}: {len(df_merged)} rows, {int(df_merged['Label'].sum())} attack rows, {n_att} attack segments")
    test_dfs.append(df_merged)

df_test = pd.concat(test_dfs, ignore_index=True)
df_test.to_csv(os.path.join(out_dir, "hai_test_preprocessed.csv"), index=False)
print(f"Saved {len(df_test)} test rows.")
print("Preprocessing complete!")
