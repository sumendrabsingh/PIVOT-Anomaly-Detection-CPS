import pandas as pd
import numpy as np
import os
import json
import argparse

# Config
VALVE_COLUMNS = ['MV101', 'MV201', 'MV301', 'MV302', 'MV303', 'MV304']
VALVE_CATEGORIES = [0, 1, 2]

def clean_labels(df, label_col):
    """
    Standardizes label strings, removing whitespaces and typos.
    Maps everything to 'Normal' or 'Attack'.
    """
    if label_col in df.columns:
        df[label_col] = df[label_col].astype(str).str.strip().str.replace(" ", "").str.lower()
        df[label_col] = df[label_col].apply(lambda x: 'Attack' if x == 'attack' else 'Normal')
    return df

def fit_preprocessing(train_df, scaler_type):
    """
    Learns preprocessing parameters from the training set.
    """
    print(f"[FIT] Fitting preprocessing pipeline on training set using {scaler_type.upper()} scaling...")
    
    # Strip column names
    train_df.columns = train_df.columns.str.strip()
    
    # 1. Label column identification
    label_col = 'Normal/Attack' if 'Normal/Attack' in train_df.columns else ('Label' if 'Label' in train_df.columns else None)
    
    # 2. Find constant columns (numeric columns with std < 1e-4)
    numeric_df = train_df.select_dtypes(include=[np.number])
    stds = numeric_df.std(axis=0)
    constant_cols = stds[stds < 1e-4].index.tolist()
    
    # EXCLUDE NETWORK FEATURES FROM DROPPING
    network_features = ['total_packets', 'read_tags', 'write_tags', 'error_count', 'unique_src', 'unique_dst']
    constant_cols = [c for c in constant_cols if c not in network_features]
    
    print(f"[FIT] Identified {len(constant_cols)} constant columns to drop: {constant_cols}")
    
    # Keep track of features to scale (numerical features excluding Timestamp and label)
    cols_to_keep = [col for col in train_df.columns if col not in ['Timestamp', label_col]]
    cols_after_drop = [col for col in cols_to_keep if col not in constant_cols]
    
    # Identify which of the kept columns are valves vs continuous sensors
    valves_to_encode = [col for col in VALVE_COLUMNS if col in cols_after_drop]
    continuous_features = [col for col in cols_after_drop if col not in valves_to_encode]
    
    # 3. Fit Scaler parameters for continuous features
    min_vals = {}
    max_vals = {}
    mean_vals = {}
    std_vals = {}
    
    for col in continuous_features:
        min_vals[col] = float(train_df[col].min())
        max_vals[col] = float(train_df[col].max())
        mean_vals[col] = float(train_df[col].mean())
        std_v = float(train_df[col].std())
        std_vals[col] = std_v if std_v >= 1e-4 else 1.0
        
    # Build final column sequence including one-hot valve outputs
    final_columns = []
    for col in cols_after_drop:
        if col in valves_to_encode:
            for cat in VALVE_CATEGORIES:
                final_columns.append(f"{col}_{cat}")
        else:
            final_columns.append(col)
            
    meta = {
        "scaler_type": scaler_type,
        "label_col": label_col,
        "constant_columns": constant_cols,
        "valves_encoded": valves_to_encode,
        "continuous_features": continuous_features,
        "min_vals": min_vals,
        "max_vals": max_vals,
        "mean_vals": mean_vals,
        "std_vals": std_vals,
        "final_columns": final_columns
    }
    
    return meta

def transform_data(df, meta):
    """
    Applies the preprocessing transformation to a dataset using fitted metadata.
    """
    df_clean = df.copy()
    df_clean.columns = df_clean.columns.str.strip()
    
    # 1. Clean labels if present
    label_col = meta["label_col"]
    if label_col and label_col in df_clean.columns:
        df_clean = clean_labels(df_clean, label_col)
        
    # Preserve Timestamp and Label columns (unmodified)
    reserved_cols = {}
    for col in ['Timestamp', label_col]:
        if col in df_clean.columns:
            reserved_cols[col] = df_clean[col].values
            
    # 2. Extract and transform features
    transformed_features = pd.DataFrame(index=df_clean.index)
    scaler_type = meta.get("scaler_type", "minmax")
    
    # Scale continuous features
    for col in meta["continuous_features"]:
        if col in df_clean.columns:
            val = df_clean[col].astype(float)
            if scaler_type == "minmax":
                col_min = meta["min_vals"][col]
                col_max = meta["max_vals"][col]
                denom = col_max - col_min
                if denom < 1e-4:
                    denom = 1.0
                transformed_features[col] = (val - col_min) / denom
            else:  # zscore
                col_mean = meta["mean_vals"][col]
                col_std = meta["std_vals"][col]
                transformed_features[col] = (val - col_mean) / col_std
        else:
            print(f"[WARNING] Column {col} missing in dataset during transform! Imputing with 0.0")
            transformed_features[col] = 0.0
            
    # One-hot encode valve features deterministically
    for col in meta["valves_encoded"]:
        if col in df_clean.columns:
            val = df_clean[col].astype(int)
            for cat in VALVE_CATEGORIES:
                transformed_features[f"{col}_{cat}"] = (val == cat).astype(float)
        else:
            print(f"[WARNING] Valve column {col} missing in dataset during transform! Imputing with 0.0")
            for cat in VALVE_CATEGORIES:
                transformed_features[f"{col}_{cat}"] = 0.0
                
    # 3. Align to final columns order
    transformed_features = transformed_features[meta["final_columns"]]
    
    # 4. Re-attach reserved columns
    for col_name, col_vals in reserved_cols.items():
        transformed_features[col_name] = col_vals
        
    # Reorder columns so that Timestamp and Label are at the beginning/end respectively
    col_order = []
    if 'Timestamp' in transformed_features.columns:
        col_order.append('Timestamp')
    for col in meta["final_columns"]:
        col_order.append(col)
    if label_col in transformed_features.columns:
        col_order.append(label_col)
        
    return transformed_features[col_order]

def main():
    parser = argparse.ArgumentParser(description="SWaT Dataset Preprocessing Utility")
    parser.add_argument('--train', type=str, default=r'SWaT_Orig_DataSets/SWaT_Dataset_Normal_v1.csv',
                        help='Path to the raw training CSV dataset')
    parser.add_argument('--test', type=str, default=r'SWaT_Orig_DataSets/SWaT_Dataset_Attack_v0.csv',
                        help='Path to the raw testing/evaluation CSV dataset')
    parser.add_argument('--out_dir', type=str, default=r'Datasets',
                        help='Output directory to save the preprocessed datasets')
    parser.add_argument('--scaler', type=str, default='zscore', choices=['minmax', 'zscore'],
                        help='Feature scaling method: "minmax" or "zscore" (default: "zscore")')
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 1. Load Training Data
    print(f"\nLoading Training Data from {args.train}...")
    train_df = pd.read_csv(args.train)
    
    # 2. Fit Preprocessing
    meta = fit_preprocessing(train_df, args.scaler)
    
    # 3. Save Metadata Config
    meta_path = os.path.join(args.out_dir, 'preprocessing_meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=4)
    print(f"[SAVE] Saved preprocessing metadata to {meta_path}")
    
    # 4. Transform & Save Training Set
    print("\nTransforming Training Set...")
    train_processed = transform_data(train_df, meta)
    train_out_path = os.path.join(args.out_dir, 'swat_train_preprocessed.csv')
    train_processed.to_csv(train_out_path, index=False)
    print(f"[SAVE] Saved preprocessed training set to {train_out_path} (Shape: {train_processed.shape})")
    
    # 5. Load, Transform & Save Testing Data if provided
    if args.test and os.path.exists(args.test):
        print(f"\nLoading Testing Data from {args.test}...")
        test_df = pd.read_csv(args.test)
        print("Transforming Testing Set...")
        test_processed = transform_data(test_df, meta)
        test_out_path = os.path.join(args.out_dir, 'swat_test_preprocessed.csv')
        test_processed.to_csv(test_out_path, index=False)
        print(f"[SAVE] Saved preprocessed testing set to {test_out_path} (Shape: {test_processed.shape})")
        
        # Print class distribution verification
        label_col = meta['label_col']
        if label_col in test_processed.columns:
            print("\nVerification - Test Label Distribution:")
            print(test_processed[label_col].value_counts())
            
    print("\n[COMPLETE] Preprocessing finished successfully!")

if __name__ == "__main__":
    main()
