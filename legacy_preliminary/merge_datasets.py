import pandas as pd
import os

NETWORK_CSV = r"Datasets/swat_network_aggregated.csv"
NORMAL_CSV = r"SWaT_Orig_DataSets/SWaT_Dataset_Normal_v1.csv"
ATTACK_CSV = r"SWaT_Orig_DataSets/SWaT_Dataset_Attack_v0.csv"

OUT_DIR = r"Datasets"

def merge_network(physical_path, out_name):
    print(f"Loading {physical_path}...", flush=True)
    df_phys = pd.read_csv(physical_path)
    df_phys.columns = df_phys.columns.str.strip()
    
    # Clean timestamp string for matching
    df_phys['Timestamp_clean'] = df_phys['Timestamp'].astype(str).str.strip()
    
    print(f"Loading {NETWORK_CSV}...", flush=True)
    df_net = pd.read_csv(NETWORK_CSV)
    df_net['Timestamp_clean'] = df_net['Timestamp'].astype(str).str.strip()
    
    # Drop Timestamp from network to avoid collision
    df_net = df_net.drop(columns=['Timestamp'])
    
    # Fill NA for missing network values (e.g. seconds where no packets occurred, though unlikely)
    net_cols = ['total_packets', 'read_tags', 'write_tags', 'error_count', 'unique_src', 'unique_dst']
    
    print(f"Merging DataFrames...", flush=True)
    merged_df = pd.merge(df_phys, df_net, on='Timestamp_clean', how='left')
    
    # Fill NAs with 0
    merged_df[net_cols] = merged_df[net_cols].fillna(0)
    
    merged_df = merged_df.drop(columns=['Timestamp_clean'])
    
    out_path = os.path.join(OUT_DIR, out_name)
    print(f"Saving merged dataset to {out_path}...", flush=True)
    merged_df.to_csv(out_path, index=False)
    print(f"[DONE] Merged dataset shape: {merged_df.shape}", flush=True)

if __name__ == "__main__":
    if not os.path.exists(NETWORK_CSV):
        print(f"ERROR: Network aggregated CSV not found at {NETWORK_CSV}")
    else:
        merge_network(NORMAL_CSV, "swat_normal_with_network.csv")
        merge_network(ATTACK_CSV, "swat_attack_with_network.csv")
