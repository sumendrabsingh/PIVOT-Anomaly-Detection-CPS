import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import json

wadi_dir = r"SWaT_Orig_DataSets/wadi"
output_dir = r"Datasets"
os.makedirs(output_dir, exist_ok=True)

print("Loading WADI Training Data...")
# WADI files have 4 rows of metadata at the top, so we skip them
train_df = pd.read_csv(os.path.join(wadi_dir, "WADI_14days.csv"), skiprows=4)
print("Loading WADI Testing Data...")
test_df = pd.read_csv(os.path.join(wadi_dir, "WADI_attackdata.csv"))

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

# Identify the labels. WADI attack data labels are in attack_description.xlsx
print("Parsing attack_description.xlsx for ground-truth labels...")
try:
    attack_info = pd.read_excel(os.path.join(wadi_dir, "attack_description.xlsx"), skiprows=4)
    attack_info.columns = ['S.No', 'Date', 'Start Time', 'End Time', 'Attack Point', 'Initial State', 'State', 'Impact', 'Remarks']
    attack_info = attack_info.dropna(subset=['Date', 'Start Time', 'End Time'])
    
    # Create combined datetime for test dataset
    test_df['Datetime'] = pd.to_datetime(test_df['Date'] + ' ' + test_df['Time'], format='%m/%d/%Y %I:%M:%S.%f %p', errors='coerce')
    # Some rows might not have .000 or might be AM/PM, so let's try a broader format if there's NaT
    if test_df['Datetime'].isnull().any():
        test_df['Datetime'] = pd.to_datetime(test_df['Date'] + ' ' + test_df['Time'], errors='coerce')

    # Initialize all as normal
    y_test = np.zeros(len(test_df))
    
    # Iterate through the attack descriptions
    for _, row in attack_info.iterrows():
        try:
            # Handle possible float/string dates and times
            date_str = str(row['Date']).split(' ')[0]
            start_time_str = str(row['Start Time'])
            end_time_str = str(row['End Time'])
            
            start_dt = pd.to_datetime(f"{date_str} {start_time_str}")
            end_dt = pd.to_datetime(f"{date_str} {end_time_str}")
            
            # Mask
            mask = (test_df['Datetime'] >= start_dt) & (test_df['Datetime'] <= end_dt)
            y_test[mask] = 1
        except Exception as e:
            print(f"Error parsing attack row: {e}")
            continue

    print(f"Found {int(sum(y_test))} attack samples out of {len(y_test)}")
    test_df = test_df.drop(columns=['Datetime'])
    
except Exception as e:
    print(f"Failed to load attack_description.xlsx: {e}")
    y_test = np.zeros(len(test_df))

train_df = train_df.drop(columns=[c for c in cols_to_drop if c in train_df.columns], errors='ignore')
test_df = test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns], errors='ignore')

# Ensure train and test have exactly the same sensor columns
common_cols = [c for c in train_df.columns if c in test_df.columns and not c.startswith('LEAK') and not c.startswith('PLANT') and not c.startswith('TOTAL')]
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
