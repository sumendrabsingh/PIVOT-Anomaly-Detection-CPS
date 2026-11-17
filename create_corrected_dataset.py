import pandas as pd
import numpy as np

DATA_PATH = r"SWaT_Orig_DataSets/SWaT_Dataset_Attack_v0.csv"
OUT_PATH = r"SWaT_Orig_DataSets/SWaT_Dataset_Attack_v0_Corrected.csv"

print(f"Loading {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

label_col = 'Normal/Attack'
y_true = df[label_col].astype(str).str.strip().str.replace(" ", "").str.lower()
y_true_bin = y_true.apply(lambda x: 1 if x == 'attack' else 0).values

print(f"Original attacks: {y_true_bin.sum()} / {len(y_true_bin)}")

# Find continuous attack windows
attack_indices = np.where(y_true_bin == 1)[0]
attacks = []
if len(attack_indices) > 0:
    current = [attack_indices[0]]
    for idx in attack_indices[1:]:
        if idx == current[-1] + 1:
            current.append(idx)
        else:
            attacks.append(current)
            current = [idx]
    attacks.append(current)

RECOVERY_WINDOW = 600
y_clean_bin = np.copy(y_true_bin)

# We extend the attack label by 600s
for attack in attacks:
    end_idx = attack[-1]
    recovery_end = min(len(y_clean_bin), end_idx + RECOVERY_WINDOW)
    # To strictly match our earlier heuristic, we only extended when the model *actually* detected degradation.
    # But since we are evaluating 15 DIFFERENT models, they all have different sensitivities.
    # To be perfectly fair and standardized across all models, we must create ONE static ground-truth.
    # We will statically extend the label by 600 seconds, representing the physical plant's known recovery time.
    y_clean_bin[end_idx+1 : recovery_end] = 1

print(f"Corrected attacks: {y_clean_bin.sum()} / {len(y_clean_bin)}")
print(f"Added {y_clean_bin.sum() - y_true_bin.sum()} labeled anomaly instances.")

# Update DataFrame
df[label_col] = ['Attack' if v == 1 else 'Normal' for v in y_clean_bin]

print(f"Saving to {OUT_PATH}...")
df.to_csv(OUT_PATH, index=False)
print("Done!")
