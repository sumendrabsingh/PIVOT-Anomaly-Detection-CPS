import pandas as pd
import os

hai_dir = r"SWaT_Orig_DataSets/hai/hai-23.05"
out_dir = r"Datasets"
os.makedirs(out_dir, exist_ok=True)

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
    
    # Merge on timestamp
    df_merged = pd.merge(df_t, df_l, on='timestamp', how='left')
    
    # Fill missing labels with 0 just in case
    df_merged['label'] = df_merged['label'].fillna(0).astype(int)
    
    # Rename label to Label for consistency with other scripts
    df_merged = df_merged.rename(columns={'label': 'Label'})
    
    if 'timestamp' in df_merged.columns:
        df_merged = df_merged.drop(columns=['timestamp'])
        
    test_dfs.append(df_merged)

df_test = pd.concat(test_dfs, ignore_index=True)
df_test.to_csv(os.path.join(out_dir, "hai_test_preprocessed.csv"), index=False)
print(f"Saved {len(df_test)} test rows.")
print("Preprocessing complete!")
