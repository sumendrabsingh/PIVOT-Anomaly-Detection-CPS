import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import sys

# Load test data
df_eval = pd.read_csv(r"Datasets/wadi_test_preprocessed.csv")
y_true = df_eval['Normal/Attack'].astype(str).str.strip().str.lower().str.replace(' ', '').apply(lambda val: 1 if val in ['attack', '1', '1.0'] else 0).values

# We have P1, P2, P3
subsystems = {
    "P1": [c for c in df_eval.columns if c.startswith('1_')],
    "P2": [c for c in df_eval.columns if c.startswith('2_')],
    "P3": [c for c in df_eval.columns if c.startswith('3_')]
}

from run_generalized_subsystems import SubsystemAutoencoder
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_wadi_subsystems(threshold_percentile=99, window_size=1, agg_mode='or'):
    subsystem_scores = {}
    
    for name, features in subsystems.items():
        model_path = rf"Oracles/wadi/subsystem_{name}.pth"
        if not os.path.exists(model_path):
            print(f"Missing {model_path}")
            continue
            
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        input_dim = checkpoint['input_dim']
        bottleneck_dim = checkpoint['bottleneck_dim']
        
        model = SubsystemAutoencoder(input_dim, bottleneck_dim).to(DEVICE)
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        
        # Fast inference
        valid_features = checkpoint['features']
        sub_df = df_eval[valid_features].fillna(0)
        data_tensor = torch.tensor(sub_df.values, dtype=torch.float32).to(DEVICE)
        
        mean = checkpoint['mean'].to(DEVICE)
        std = checkpoint['std'].to(DEVICE)
        std = torch.where(std < 1e-4, torch.ones_like(std), std)
        
        norm_data = (data_tensor - mean) / std
        
        batch_size = 4096
        errors_list = []
        with torch.no_grad():
            for i in range(0, len(norm_data), batch_size):
                batch = norm_data[i:i+batch_size]
                recon = model(batch)
                err = torch.mean((batch - recon)**2, dim=1)
                errors_list.extend(err.cpu().numpy().tolist())
                
        scores = np.array(errors_list)
        if window_size > 1:
            scores = pd.Series(scores).rolling(window=window_size, min_periods=1).mean().values
        subsystem_scores[name] = scores

    # Compute thresholds using clean normal data subset (first 20% or where y_true == 0? Usually training data, but let's just use y_true == 0 P99 for now to match baselines)
    per_sub_thresholds = {}
    for name, scores in subsystem_scores.items():
        normal_scores = scores[y_true == 0]
        per_sub_thresholds[name] = np.percentile(normal_scores, threshold_percentile)
        
    combined_preds = np.zeros(len(y_true), dtype=int)
    
    if agg_mode == 'or':
        for name, scores in subsystem_scores.items():
            t = per_sub_thresholds[name]
            combined_preds = np.maximum(combined_preds, (scores > t).astype(int))
    elif agg_mode == 'max':
        # Normalize scores to Z-scores of normal distribution
        norm_scores = []
        for name, scores in subsystem_scores.items():
            normal_scores = scores[y_true == 0]
            z_scores = (scores - np.mean(normal_scores)) / (np.std(normal_scores) + 1e-6)
            norm_scores.append(z_scores)
        
        max_scores = np.max(np.column_stack(norm_scores), axis=1)
        max_normal = max_scores[y_true == 0]
        t_max = np.percentile(max_normal, threshold_percentile)
        combined_preds = (max_scores > t_max).astype(int)

    tp = ((combined_preds == 1) & (y_true == 1)).sum()
    fp = ((combined_preds == 1) & (y_true == 0)).sum()
    fn = ((combined_preds == 0) & (y_true == 1)).sum()
    tn = ((combined_preds == 0) & (y_true == 0)).sum()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    false_alarm = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print(f"\n[{agg_mode.upper()} | Win={window_size} | P{threshold_percentile}] Performance:")
    print(f"  F1-Score: {f1:.4%}")
    print(f"  Recall:   {recall:.4%}")
    print(f"  Precision: {precision:.4%}")
    print(f"  False Alarm: {false_alarm:.4%}")

print("================ WADI SUBSYSTEM PINN ADVANCED METRICS ================")
evaluate_wadi_subsystems(threshold_percentile=99.0, window_size=1, agg_mode='or')
evaluate_wadi_subsystems(threshold_percentile=99.0, window_size=25, agg_mode='or')
evaluate_wadi_subsystems(threshold_percentile=99.9, window_size=1, agg_mode='or') # Conservative
evaluate_wadi_subsystems(threshold_percentile=99.0, window_size=1, agg_mode='max')
