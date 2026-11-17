import pandas as pd
import numpy as np
import os

TRAIN_IN = r"Datasets/hai_train_preprocessed.csv"
TEST_IN = r"Datasets/hai_test_preprocessed.csv"
TRAIN_OUT = r"Datasets/hai_train_engineered.csv"
TEST_OUT = r"Datasets/hai_test_engineered.csv"

def engineer_features(df):
    print(f"Engineering features for {len(df)} rows...")
    df_new = df.copy()
    
    # 1. Cross-Subsystem Physics Invariants
    if 'P1_PIT01' in df_new.columns and 'P4_ST_PT01' in df_new.columns:
        df_new['Physics_P1_P4_Pressure_Delta'] = df_new['P1_PIT01'] - df_new['P4_ST_PT01']
    if 'P1_TIT01' in df_new.columns and 'P4_ST_TT01' in df_new.columns:
        df_new['Physics_P1_P4_Temp_Delta'] = df_new['P1_TIT01'] - df_new['P4_ST_TT01']
    if 'P3_FIT01' in df_new.columns and 'P4_HT_PO' in df_new.columns:
        df_new['Physics_P3_P4_Hydro_Delta'] = df_new['P3_FIT01'] - df_new['P4_HT_PO']
        
    # 2. Control Errors (Setpoint vs Process Variable)
    if 'P1_B3004' in df_new.columns and 'P1_LIT01' in df_new.columns:
        df_new['Error_WaterLevel'] = df_new['P1_B3004'] - df_new['P1_LIT01']
    if 'P1_B4022' in df_new.columns and 'P1_TIT01' in df_new.columns:
        df_new['Error_Temperature'] = df_new['P1_B4022'] - df_new['P1_TIT01']
    if 'P2_AutoSD' in df_new.columns and 'P2_SIT01' in df_new.columns:
        df_new['Error_TurbineSpeed'] = df_new['P2_AutoSD'] - df_new['P2_SIT01']
        
    # 3. Rate of Change (First Derivative) for continuous variables
    # For speed, skip Label columns
    num_cols = [c for c in df.columns if c != 'Label' and not c.endswith('Lamp') and not c.endswith('GO')]
    diff_df = df[num_cols].diff().fillna(0)
    diff_df.columns = [f"{c}_diff" for c in num_cols]
    df_new = pd.concat([df_new, diff_df], axis=1)
    
    # 4. Noise Smoothing (Rolling Average) for sensitive sensors
    smooth_cols = [c for c in df.columns if 'VIBTR' in c or c == 'P4_LD']
    if smooth_cols:
        roll_df = df[smooth_cols].rolling(window=5, min_periods=1).mean()
        roll_df.columns = [f"{c}_ma5" for c in smooth_cols]
        df_new = pd.concat([df_new, roll_df], axis=1)
        
    return df_new

print("Processing Train Data...")
df_train = pd.read_csv(TRAIN_IN)
df_train_eng = engineer_features(df_train)
df_train_eng.to_csv(TRAIN_OUT, index=False)

print("Processing Test Data...")
df_test = pd.read_csv(TEST_IN)
df_test_eng = engineer_features(df_test)
df_test_eng.to_csv(TEST_OUT, index=False)

print("Done! Feature engineering complete.")
