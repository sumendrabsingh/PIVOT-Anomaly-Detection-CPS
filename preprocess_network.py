import os
import glob
import pandas as pd
from multiprocessing import Pool
import warnings

# Suppress DtypeWarnings
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)

NETWORK_DIR = r"F:\SWaT.A1 & A2_Dec 2015\SWaT.A1 & A2_Dec 2015\Network"
OUTPUT_FILE = r"Datasets/swat_network_aggregated.csv"

def process_file(file_path):
    print(f"Reading {os.path.basename(file_path)}...", flush=True)
    try:
        # Some rows might be malformed, skip them
        df = pd.read_csv(file_path, usecols=['date', 'time', 'src', 'dst', 'Modbus_Function_Description'], dtype=str, on_bad_lines='skip')
        
        df['datetime'] = df['date'] + ' ' + df['time']
        df['Modbus_Function_Description'] = df['Modbus_Function_Description'].fillna('')
        
        df['is_read'] = df['Modbus_Function_Description'].str.contains('Read', case=False, na=False).astype(int)
        df['is_write'] = df['Modbus_Function_Description'].str.contains('Write', case=False, na=False).astype(int)
        df['is_error'] = df['Modbus_Function_Description'].str.contains('Error|Exception', case=False, na=False).astype(int)
        
        agg = df.groupby('datetime').agg(
            total_packets=('datetime', 'size'),
            read_tags=('is_read', 'sum'),
            write_tags=('is_write', 'sum'),
            error_count=('is_error', 'sum'),
            unique_src=('src', 'nunique'),
            unique_dst=('dst', 'nunique')
        ).reset_index()
        
        return agg
    except Exception as e:
        print(f"Error processing {file_path}: {e}", flush=True)
        return pd.DataFrame()

def process_network_logs():
    csv_files = sorted(glob.glob(os.path.join(NETWORK_DIR, "*_sorted.csv")))
    print(f"[INFO] Found {len(csv_files)} files.", flush=True)
    
    with Pool(processes=8) as pool:
        results = pool.map(process_file, csv_files)
        
    print("[INFO] Merging results...", flush=True)
    final_df = pd.concat([r for r in results if not r.empty], ignore_index=True)
    
    print("[INFO] Final group by across files...", flush=True)
    final_agg = final_df.groupby('datetime').agg({
        'total_packets': 'sum',
        'read_tags': 'sum',
        'write_tags': 'sum',
        'error_count': 'sum',
        'unique_src': 'max',
        'unique_dst': 'max'
    }).reset_index()
    
    print("[INFO] Formatting timestamps...", flush=True)
    final_agg['Timestamp'] = pd.to_datetime(final_agg['datetime'], format='%d%b%Y %H:%M:%S', errors='coerce')
    
    # Drop rows where date parsing failed
    final_agg = final_agg.dropna(subset=['Timestamp'])
    
    # Sort
    final_agg = final_agg.sort_values('Timestamp')
    
    # Format to match physical dataset
    final_agg['Timestamp'] = final_agg['Timestamp'].dt.strftime('%d/%m/%Y %I:%M:%S %p')
    final_agg = final_agg.drop(columns=['datetime'])
    
    final_agg.to_csv(OUTPUT_FILE, index=False)
    print(f"[COMPLETE] Saved aggregated data to {OUTPUT_FILE}", flush=True)

if __name__ == "__main__":
    process_network_logs()
