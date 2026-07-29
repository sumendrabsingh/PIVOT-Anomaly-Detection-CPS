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
    SEQ_LEN
)

META_DATA = r"Datasets/preprocessing_meta.json"
DATASET = r"SWaT_Orig_DataSets/SWaT_Dataset_Attack_v0.csv"
OUT_DIR = "Oracles/XAI_Visualizations"

# Custom C&W L_inf formulation for Anomaly Evasion
def cw_attack_subsystem(model, x_clean, y_true, epsilon=0.1, num_iter=50, lr=0.01):
    attack_mask = (y_true == 1)
    if not attack_mask.any(): return x_clean.clone()
    
    x_clean_attacks = x_clean[attack_mask].clone().detach()
    delta = torch.zeros_like(x_clean_attacks, requires_grad=True)
    
    optimizer = torch.optim.Adam([delta], lr=lr)
    
    for _ in range(num_iter):
        optimizer.zero_grad()
        x_adv = x_clean_attacks + delta
        reconstructed = model(x_adv)
        squared_diff = (x_adv - reconstructed) ** 2
        loss = torch.mean(torch.max(squared_diff, dim=1)[0]) if ERROR_MODE == 'max' else torch.mean(squared_diff)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            delta.clamp_(-epsilon, epsilon)
            
    x_adv_final = x_clean.clone().detach()
    x_adv_final[attack_mask] = (x_clean_attacks + delta).detach()
    return x_adv_final

# Custom Auto-PGD (APGD) formulation for Anomaly Evasion
def apgd_attack_subsystem(model, x_clean, y_true, epsilon=0.1, num_iter=20):
    attack_mask = (y_true == 1)
    if not attack_mask.any(): return x_clean.clone()
    
    x_clean_attacks = x_clean[attack_mask].clone().detach()
    x_adv = x_clean_attacks.clone().detach()
    x_adv.requires_grad = True
    
    alpha = 1.5 * epsilon / (num_iter * 0.25)
    momentum = torch.zeros_like(x_adv)
    
    for i in range(num_iter):
        reconstructed = model(x_adv)
        squared_diff = (x_adv - reconstructed) ** 2
        loss = torch.mean(torch.max(squared_diff, dim=1)[0]) if ERROR_MODE == 'max' else torch.mean(squared_diff)
        
        if i % 5 == 0 and i > 0:
            alpha = alpha / 1.5
            
        model.zero_grad()
        if x_adv.grad is not None: x_adv.grad.zero_()
        loss.backward()
        
        with torch.no_grad():
            grad = x_adv.grad
            momentum = 0.75 * momentum + grad / (torch.norm(grad, p=1) + 1e-12)
            x_adv = x_adv - alpha * momentum.sign()
            delta = torch.clamp(x_adv - x_clean_attacks, min=-epsilon, max=epsilon)
            x_adv = x_clean_attacks + delta
            x_adv.requires_grad = True
            
    x_adv_final = x_clean.clone().detach()
    x_adv_final[attack_mask] = x_adv.detach()
    return x_adv_final

def run_advanced_adversarial_comparison():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(META_DATA, 'r') as f:
        meta = json.load(f)
        
    print(f"[INIT] Loading dataset: {DATASET}")
    df_eval = pd.read_csv(DATASET)
    df_eval.columns = df_eval.columns.str.strip()
    
    label_col = meta['label_col']
    df_eval[label_col] = df_eval[label_col].astype(str).str.strip().str.replace(" ", "").str.lower()
    df_eval[label_col] = df_eval[label_col].apply(lambda x: 'Attack' if x == 'attack' else 'Normal')
    y_true = df_eval[label_col].apply(lambda val: 1 if val == "Attack" else 0).values

    # Using the same attack window as Phase 4 for consistent comparison
    attack_indices = np.where(y_true == 1)[0]
    start_idx = attack_indices[0] - 100
    end_idx = attack_indices[500] + 100
    
    print(f"[DATA] Slicing dataset from index {start_idx} to {end_idx}")
    
    df_slice = df_eval.iloc[start_idx:end_idx].reset_index(drop=True)
    y_slice = y_true[start_idx:end_idx]
    y_true_tensor = torch.tensor(y_slice, dtype=torch.long, device=DEVICE)
    time_axis = np.arange(len(y_slice))

    results = {
        'clean': {},
        'pgd': {},
        'cw': {},
        'apgd': {}
    }
    
    # Extract the physical sensor tracks from the dataframe directly for the bottom plot
    # FIT301 (Flow) and MV301 (Valve) from Stage 3 Ultrafiltration
    fit301_base = df_slice['FIT301'].values
    mv301_base = df_slice['MV301'].values
    
    for name, features in subsystems.items():
        model_path = f"Oracles/subsystem_{name}.pth"
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        model = SubsystemAutoencoder(checkpoint['input_dim'], checkpoint['bottleneck_dim']).to(DEVICE)
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

        def infer(seqs):
            with torch.no_grad():
                return model.get_reconstruction_error(seqs, mode=ERROR_MODE).cpu().numpy()

        # Clean
        results['clean'][name] = infer(norm_seqs)
        
        # PGD
        for p in model.parameters(): p.requires_grad = True
        adv_pgd = pgd_attack_subsystem(model, norm_seqs, y_true_tensor, epsilon=0.1, alpha=0.01, num_iter=10)
        results['pgd'][name] = infer(adv_pgd)
        
        # C&W
        adv_cw = cw_attack_subsystem(model, norm_seqs, y_true_tensor, epsilon=0.1, num_iter=50, lr=0.01)
        results['cw'][name] = infer(adv_cw)
        
        # APGD (AutoAttack Core)
        adv_apgd = apgd_attack_subsystem(model, norm_seqs, y_true_tensor, epsilon=0.1, num_iter=20)
        results['apgd'][name] = infer(adv_apgd)
        
        for p in model.parameters(): p.requires_grad = False

    # Z-Score Standardization based on clean normal data
    for attack_type in results.keys():
        for name in results[attack_type].keys():
            clean_sc = results['clean'][name]
            mean_sc = np.mean(clean_sc[y_slice == 0])
            std_sc = np.std(clean_sc[y_slice == 0]) + 1e-6
            results[attack_type][name] = (results[attack_type][name] - mean_sc) / std_sc

    print("[XAI] Rendering Advanced Multi-Attack Comparison Graph...")
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 1])
    
    # -------------------------------------------------------------
    # Top Panel: Structural Trap Comparison
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0])
    
    stages = list(subsystems.keys())
    x = np.arange(len(stages))
    width = 0.2
    
    clean_means = [np.mean(results['clean'][s][y_slice == 1]) for s in stages]
    pgd_means = [np.mean(results['pgd'][s][y_slice == 1]) for s in stages]
    cw_means = [np.mean(results['cw'][s][y_slice == 1]) for s in stages]
    apgd_means = [np.mean(results['apgd'][s][y_slice == 1]) for s in stages]
    
    ax1.bar(x - 1.5*width, clean_means, width, label='Clean Attack', color='#1f77b4')
    ax1.bar(x - 0.5*width, pgd_means, width, label='PGD Evasion', color='#ff7f0e')
    ax1.bar(x + 0.5*width, cw_means, width, label='C&W Evasion (Optimization)', color='#2ca02c')
    ax1.bar(x + 1.5*width, apgd_means, width, label='AutoAttack APGD (Adaptive)', color='#d62728')
    
    ax1.set_ylabel('Average Error Signal (Z-Score)', fontsize=12)
    ax1.set_title('Subsystem PINN Structural Trap vs. Advanced Adversarial Attacks', fontsize=16, pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.replace("Stage", "S") for s in stages], rotation=15, fontsize=11)
    ax1.legend(fontsize=12, loc='upper left')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # -------------------------------------------------------------
    # Bottom Panel: Sensor-Level Trace
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1])
    
    att_indices = np.where(y_slice == 1)[0]
    
    ax2.plot(time_axis, fit301_base, label='FIT301 Flow Meter (L/h)', color='blue', linewidth=2)
    ax2.plot(time_axis, mv301_base * max(fit301_base), label='MV301 Valve Constraint (Expected 0 flow when closed)', color='black', linestyle='--', linewidth=2)
    
    if len(att_indices) > 0:
        ax2.axvspan(att_indices[0], att_indices[-1], color='red', alpha=0.15, label='Attack Event')
        
    ax2.set_title('Root Cause Trace: Physics Violations During Advanced Attacks (Ultrafiltration Stage)', fontsize=14, pad=10)
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('Physical Reading', fontsize=12)
    ax2.legend(fontsize=11, loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(OUT_DIR, "xai_advanced_attacks_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print("[SUCCESS] Advanced comparison visualization generated in Oracles/XAI_Visualizations/")

if __name__ == "__main__":
    run_advanced_adversarial_comparison()
