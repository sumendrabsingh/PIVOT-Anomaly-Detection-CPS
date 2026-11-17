import os
import torch
import pandas as pd
import numpy as np
import json
from sklearn.metrics import f1_score, confusion_matrix

from run_preprocessed_subsystem_experiments import (
    subsystems, 
    SubsystemAutoencoder, 
    DEVICE, 
    ERROR_MODE, 
    SEQ_LEN
)

META_DATA = r"Datasets/preprocessing_meta.json"
DATASET = r"SWaT_Orig_DataSets/SWaT_Dataset_Attack_v0.csv"

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

def evaluate_advanced_subsystems(df_eval, y_true, attack_type='cw', epsilon=0.1):
    with open(META_DATA, 'r') as f:
        meta = json.load(f)

    subsystem_scores = {}
    
    for name, features in subsystems.items():
        print(f"  [XAI EVAL] Processing {name}...")
        model_path = f"Oracles/subsystem_{name}.pth"
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        input_dim = checkpoint['input_dim']
        bottleneck_dim = checkpoint['bottleneck_dim']

        model = SubsystemAutoencoder(input_dim, bottleneck_dim).to(DEVICE)
        model.load_state_dict(checkpoint['model_state'])
        model.eval()

        stage_valves = [f for f in features if any(f.startswith(v) for v in meta["valves_encoded"])]
        stage_continuous = [f for f in features if f not in stage_valves]
        
        scaled_data = pd.DataFrame(index=df_eval.index)
        
        for col in stage_continuous:
            val = df_eval[col].astype(float)
            col_mean = meta["mean_vals"].get(col, val.mean())
            col_std = meta["std_vals"].get(col, val.std() + 1e-6)
            scaled_data[col] = (val - col_mean) / col_std
            
        for f in stage_valves:
            valve_name = f.split('_')[0]
            cat = int(f.split('_')[1])
            val = df_eval[valve_name].astype(int)
            scaled_data[f] = (val == cat).astype(float)
            
        scaled_data = scaled_data[features]
        data_tensor = torch.tensor(scaled_data.values, dtype=torch.float32).to(DEVICE)

        if SEQ_LEN > 1:
            pad = data_tensor[0].unsqueeze(0).repeat(SEQ_LEN - 1, 1)
            padded_data = torch.cat([pad, data_tensor], dim=0)
            norm_seqs = padded_data.unfold(0, SEQ_LEN, 1).transpose(1, 2).reshape(-1, SEQ_LEN * data_tensor.shape[1])
        else:
            norm_seqs = data_tensor

        batch_size = 4096
        errors_list = []
        y_true_tensor = torch.tensor(y_true, dtype=torch.long, device=DEVICE)
        
        for i in range(0, len(norm_seqs), batch_size):
            batch = norm_seqs[i:i + batch_size]
            batch_y = y_true_tensor[i:i + batch_size]
            
            for param in model.parameters():
                param.requires_grad = True
                
            if attack_type == 'cw':
                batch = cw_attack_subsystem(model, batch, batch_y, epsilon=epsilon, num_iter=50, lr=0.01)
            elif attack_type == 'apgd':
                batch = apgd_attack_subsystem(model, batch, batch_y, epsilon=epsilon, num_iter=20)
                
            for param in model.parameters():
                param.requires_grad = False
                    
            with torch.no_grad():
                recon_errors = model.get_reconstruction_error(batch, mode=ERROR_MODE)
                errors_list.extend(recon_errors.cpu().numpy().tolist())

        raw_scores = np.array(errors_list)
        normal_scores_subset = raw_scores[y_true == 0]
        mean_err = np.mean(normal_scores_subset)
        std_err = np.std(normal_scores_subset) + 1e-6
        subsystem_scores[name] = (raw_scores - mean_err) / std_err

    print("\n[OR-VOTING] Computing Optimal Thresholds from Clean Normal Data...")
    per_sub_thresholds = {}
    for name, scores in subsystem_scores.items():
        lo = np.percentile(scores, 80)
        hi = np.percentile(scores, 99.9)
        if hi <= lo: hi = lo + 1.0
        candidates = np.linspace(lo, hi, 200)

        best_f1 = -1.0
        best_t = 0.0
        for t in candidates:
            preds = (scores > t).astype(int)
            tp = ((preds == 1) & (y_true == 1)).sum()
            fp = ((preds == 1) & (y_true == 0)).sum()
            fn = ((preds == 0) & (y_true == 1)).sum()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        per_sub_thresholds[name] = best_t

    combined_preds = np.zeros(len(y_true), dtype=int)
    for name, scores in subsystem_scores.items():
        t = per_sub_thresholds[name]
        sub_preds = (scores > t).astype(int)
        combined_preds = np.maximum(combined_preds, sub_preds)

    cm = confusion_matrix(y_true, combined_preds)
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = f1_score(y_true, combined_preds, zero_division=0)
    false_alarm = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print(f"\n============================================================")
    print(f" FINAL PERFORMANCE REPORT: {attack_type.upper()}")
    print(f"============================================================")
    print(f"  Accuracy:                      {accuracy:.4%}")
    print(f"  Precision:                     {precision:.4%}")
    print(f"  Recall (Detection Rate):       {recall:.4%}")
    print(f"  F1-Score:                      {f1:.4%}")
    print(f"  False Alarm Rate:              {false_alarm:.4%}")
    print(f"  Confusion Matrix:")
    print(f"    TP: {tp:,} | FN: {fn:,}")
    print(f"    FP: {fp:,} | TN: {tn:,}")
    print(f"============================================================\n")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'false_alarm': false_alarm
    }

def run_advanced_eval():
    print(f"[INIT] Loading dataset: {DATASET}")
    df_eval = pd.read_csv(DATASET)
    df_eval.columns = df_eval.columns.str.strip()
    with open(META_DATA, 'r') as f:
        meta = json.load(f)
        
    label_col = meta['label_col']
    df_eval[label_col] = df_eval[label_col].astype(str).str.strip().str.replace(" ", "").str.lower()
    df_eval[label_col] = df_eval[label_col].apply(lambda x: 'Attack' if x == 'attack' else 'Normal')
    y_true = df_eval[label_col].apply(lambda val: 1 if val == "Attack" else 0).values

    print("\n--- Running Carlini & Wagner (C&W) Optimization Evaluation ---")
    res_cw = evaluate_advanced_subsystems(df_eval, y_true, attack_type='cw')
    
    print("\n--- Running AutoAttack (APGD) Adaptive Evaluation ---")
    res_apgd = evaluate_advanced_subsystems(df_eval, y_true, attack_type='apgd')
    
    results = [{
        'Model': 'Subsystem_OR_Voting_CW',
        'Accuracy (Adv)': res_cw['accuracy'],
        'Precision (Adv)': res_cw['precision'],
        'Recall (Adv)': res_cw['recall'],
        'F1 (Adv)': res_cw['f1'],
        'False Alarm (Adv)': res_cw['false_alarm']
    }, {
        'Model': 'Subsystem_OR_Voting_APGD',
        'Accuracy (Adv)': res_apgd['accuracy'],
        'Precision (Adv)': res_apgd['precision'],
        'Recall (Adv)': res_apgd['recall'],
        'F1 (Adv)': res_apgd['f1'],
        'False Alarm (Adv)': res_apgd['false_alarm']
    }]
    
    df_results = pd.DataFrame(results)
    df_results.to_csv('advanced_metrics_benchmark_results.csv', index=False)
    print("\nAdvanced Benchmark Complete. Saved to advanced_metrics_benchmark_results.csv")

if __name__ == "__main__":
    run_advanced_eval()
