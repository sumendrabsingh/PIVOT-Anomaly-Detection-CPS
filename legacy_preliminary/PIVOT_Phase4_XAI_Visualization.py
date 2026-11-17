import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_preprocessed_subsystem_experiments import (
    subsystems, 
    SubsystemAutoencoder, 
    pgd_attack_subsystem, 
    DEVICE, 
    ERROR_MODE, 
    ALIGN_BASELINE,
    SEQ_LEN
)

META_DATA = r"Datasets/preprocessing_meta.json"
DATASET = r"SWaT_Orig_DataSets/SWaT_Dataset_Attack_v0.csv"
OUT_DIR = "Oracles/XAI_Visualizations"

with open(META_DATA, 'r') as f:
    meta = json.load(f)

def run_xai_visualization():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[INIT] Loading dataset: {DATASET}")
    
    # We load a small chunk of the dataset that contains an attack.
    # The first attack in SWaT v0 happens around index 41700 to 42500 (approx)
    # We will slice index 41500 to 42500 to see normal -> attack.
    
    df_eval = pd.read_csv(DATASET)
    df_eval.columns = df_eval.columns.str.strip()
    
    label_col = meta['label_col']
    df_eval[label_col] = df_eval[label_col].astype(str).str.strip().str.replace(" ", "").str.lower()
    df_eval[label_col] = df_eval[label_col].apply(lambda x: 'Attack' if x == 'attack' else 'Normal')
    y_true = df_eval[label_col].apply(lambda val: 1 if val == "Attack" else 0).values

    # Find the first major attack window
    attack_indices = np.where(y_true == 1)[0]
    if len(attack_indices) == 0:
        print("[ERROR] No attacks found!")
        return
        
    start_idx = attack_indices[0] - 100
    end_idx = attack_indices[500] + 100
    
    print(f"[DATA] Slicing dataset from index {start_idx} to {end_idx}")
    
    df_slice = df_eval.iloc[start_idx:end_idx].reset_index(drop=True)
    y_slice = y_true[start_idx:end_idx]
    y_true_tensor = torch.tensor(y_slice, dtype=torch.long, device=DEVICE)
    
    time_axis = np.arange(len(y_slice))

    clean_errors = {}
    adv_errors = {}
    
    # Run Inference
    for name, features in subsystems.items():
        model_path = f"Oracles/subsystem_{name}.pth"
        if not os.path.exists(model_path):
            print(f"  [ERROR] Model missing: {model_path}")
            continue

        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        input_dim = checkpoint['input_dim']
        bottleneck_dim = checkpoint['bottleneck_dim']

        model = SubsystemAutoencoder(input_dim, bottleneck_dim).to(DEVICE)
        model.load_state_dict(checkpoint['model_state'])
        model.eval()

        stage_valves = [f for f in features if any(f.startswith(v) for v in meta["valves_encoded"])]
        stage_continuous = [f for f in features if f not in stage_valves]
        
        scaled_data = pd.DataFrame(index=df_slice.index)
        
        for col in stage_continuous:
            val = df_slice[col].astype(float)
            col_mean = meta["mean_vals"].get(col, val.mean())
            col_std = meta["std_vals"].get(col, val.std() + 1e-6)
            scaled_data[col] = (val - col_mean) / col_std
            
        for f in stage_valves:
            valve_name = f.split('_')[0]
            cat = int(f.split('_')[1])
            val = df_slice[valve_name].astype(int)
            scaled_data[f] = (val == cat).astype(float)
            
        scaled_data = scaled_data[features]
        data_tensor = torch.tensor(scaled_data.values, dtype=torch.float32).to(DEVICE)

        if SEQ_LEN > 1:
            pad = data_tensor[0].unsqueeze(0).repeat(SEQ_LEN - 1, 1)
            padded_data = torch.cat([pad, data_tensor], dim=0)
            norm_seqs = padded_data.unfold(0, SEQ_LEN, 1).transpose(1, 2).reshape(-1, SEQ_LEN * data_tensor.shape[1])
        else:
            norm_seqs = data_tensor

        # Clean Inference
        with torch.no_grad():
            clean_recon_errors = model.get_reconstruction_error(norm_seqs, mode=ERROR_MODE)
            clean_errors[name] = clean_recon_errors.cpu().numpy()

        # Adversarial Inference
        for param in model.parameters():
            param.requires_grad = True
        
        adv_seqs = pgd_attack_subsystem(model, norm_seqs, y_true_tensor, epsilon=0.1, alpha=0.01, num_iter=10)
        
        for param in model.parameters():
            param.requires_grad = False
            
        with torch.no_grad():
            adv_recon_errors = model.get_reconstruction_error(adv_seqs, mode=ERROR_MODE)
            adv_errors[name] = adv_recon_errors.cpu().numpy()

    # Standardization
    for name in clean_errors.keys():
        sc = clean_errors[name]
        mean_sc = np.mean(sc[y_slice == 0])
        std_sc = np.std(sc[y_slice == 0]) + 1e-6
        clean_errors[name] = (sc - mean_sc) / std_sc
        adv_errors[name] = (adv_errors[name] - mean_sc) / std_sc

    # -------------------------------------------------------------
    # Plot 1: Global XAI Subsystem Heatmap
    # -------------------------------------------------------------
    print("[XAI] Rendering Global Heatmap...")
    heatmap_data = pd.DataFrame(clean_errors)
    heatmap_data = heatmap_data.rolling(window=10, min_periods=1).mean() # Smooth for visualization
    
    plt.figure(figsize=(12, 6))
    im = plt.imshow(heatmap_data.T.values, aspect='auto', cmap='hot', interpolation='nearest')
    plt.colorbar(im, label='Reconstruction Error (Z-Score)')
    
    # Highlight attack window
    att_indices = np.where(y_slice == 1)[0]
    if len(att_indices) > 0:
        plt.axvspan(att_indices[0], att_indices[-1], color='cyan', alpha=0.3, label='True Attack Window')
        
    plt.title('Global XAI Subsystem Heatmap (Physical Isolation)')
    plt.xlabel('Time Step')
    plt.ylabel('Physical Stage')
    plt.yticks(ticks=np.arange(len(heatmap_data.columns)), labels=[s.replace("Stage", "S") for s in heatmap_data.columns])
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "xai_subsystem_heatmap.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # Plot 2: The Adversarial Structural Trap (Bar Chart)
    # -------------------------------------------------------------
    print("[XAI] Rendering Adversarial Structural Trap Plot...")
    stages = list(clean_errors.keys())
    clean_means = [np.mean(clean_errors[s][y_slice == 1]) for s in stages]
    adv_means = [np.mean(adv_errors[s][y_slice == 1]) for s in stages]
    
    x = np.arange(len(stages))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 7))
    rects1 = ax.bar(x - width/2, clean_means, width, label='Clean Attack', color='#1f77b4')
    rects2 = ax.bar(x + width/2, adv_means, width, label='Adversarial Spoofing (PGD)', color='#d62728')

    ax.set_ylabel('Average Error Signal (Z-Score)')
    ax.set_title('The Structural Trap: PGD Minimizes Error Locally, But Violates Adjacent Physical Invariants')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("Stage", "S") for s in stages], rotation=15)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "xai_adversarial_trap.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # Plot 3: Sensor-Level Root Cause Trace
    # -------------------------------------------------------------
    print("[XAI] Rendering Sensor-Level Trace...")
    # Find a sensor trace from Ultrafiltration or Dechlorination (Stage 3 or 4)
    # E.g. FIT301 (Flow) and MV301 (Valve)
    
    plt.figure(figsize=(14, 5))
    
    # We plot the physical values from df_slice
    fit301 = df_slice['FIT301'].values
    mv301 = df_slice['MV301'].values
    
    plt.plot(time_axis, fit301, label='FIT301 Flow Meter (L/h)', color='blue')
    plt.plot(time_axis, mv301 * max(fit301), label='MV301 Valve State (1=Open, 0=Closed)', color='green', alpha=0.5, linestyle='--')
    
    if len(att_indices) > 0:
        plt.axvspan(att_indices[0], att_indices[-1], color='red', alpha=0.2, label='Attack Event')
        
    plt.title('Root Cause XAI Trace: Stage 3 Ultrafiltration Sensor Drift')
    plt.xlabel('Time Step')
    plt.ylabel('Physical Reading')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "xai_sensor_drift.png"), dpi=300)
    plt.close()

    print("[SUCCESS] XAI Visualizations generated in Oracles/XAI_Visualizations/")

if __name__ == "__main__":
    run_xai_visualization()
