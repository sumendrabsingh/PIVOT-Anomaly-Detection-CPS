import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAIN_DATA = r'Datasets/hai_train_engineered.csv'
EVAL_DATA = r'Datasets/hai_test_engineered.csv'

print('Loading Engineered HAI data for Subsystem PINN...')
df_train = pd.read_csv(TRAIN_DATA)
df_eval = pd.read_csv(EVAL_DATA)

numeric_cols = [c for c in df_train.columns if 'Label' not in c]
subsystems = {
    'P1': [c for c in numeric_cols if 'P1_' in c or 'Error_WaterLevel' in c or 'Error_Temperature' in c],
    'P2': [c for c in numeric_cols if 'P2_' in c or 'Error_TurbineSpeed' in c],
    'P3': [c for c in numeric_cols if 'P3_' in c],
    'P4': [c for c in numeric_cols if 'P4_' in c]
}

class SubsystemAE(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        hd = max(2, input_dim // 2)
        self.enc = nn.Sequential(nn.Linear(input_dim, hd), nn.ReLU(), nn.Linear(hd, hd))
        self.dec = nn.Sequential(nn.Linear(hd, hd), nn.ReLU(), nn.Linear(hd, input_dim))
    def forward(self, x):
        return self.dec(self.enc(x))

subsystem_scores = {}

for name, features in subsystems.items():
    print(f"\nTraining Subsystem PINN for {name} ({len(features)} features)...")
    train_vals = df_train[features].fillna(0).values
    test_vals = df_eval[features].fillna(0).values
    
    mean = np.mean(train_vals, axis=0)
    std = np.std(train_vals, axis=0) + 1e-6
    
    x_train = torch.tensor((train_vals - mean) / std, dtype=torch.float32).to(DEVICE)
    x_test = torch.tensor((test_vals - mean) / std, dtype=torch.float32).to(DEVICE)
    
    model = SubsystemAE(len(features)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    batch_size = 4096
    model.train()
    for epoch in range(10):
        total_loss = 0
        for i in range(0, len(x_train), batch_size):
            batch = x_train[i:i+batch_size]
            optimizer.zero_grad()
            recon = model(batch)
            loss = torch.mean((batch - recon)**2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        if epoch == 9:
            print(f"  Epoch {epoch+1} Loss: {total_loss / len(x_train):.4f}")
            
    model.eval()
    errors_list = []
    with torch.no_grad():
        for i in range(0, len(x_test), batch_size):
            batch = x_test[i:i+batch_size]
            recon = model(batch)
            err = torch.mean((batch - recon)**2, dim=1)
            errors_list.extend(err.cpu().numpy().tolist())
    subsystem_scores[name] = np.array(errors_list)

y_true = df_eval['Label'].astype(int).values

def evaluate_subsystems(threshold_percentile, window_size, agg_mode):
    scores_dict = subsystem_scores.copy()
    for name in scores_dict:
        if window_size > 1:
            scores_dict[name] = pd.Series(scores_dict[name]).rolling(window=window_size, min_periods=1).mean().values

    per_sub_thresholds = {}
    for name, scores in scores_dict.items():
        normal_scores = scores[y_true == 0]
        per_sub_thresholds[name] = np.percentile(normal_scores, threshold_percentile)
        
    combined_preds = np.zeros(len(y_true), dtype=int)
    
    if agg_mode == 'or':
        for name, scores in scores_dict.items():
            t = per_sub_thresholds[name]
            combined_preds = np.maximum(combined_preds, (scores > t).astype(int))
    elif agg_mode == 'max':
        norm_scores = []
        for name, scores in scores_dict.items():
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

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    false_alarm = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print(f"\n[ENGINEERED {agg_mode.upper()} | Win={window_size} | P{threshold_percentile}] Performance:")
    print(f"  F1-Score: {f1:.4%}")
    print(f"  Recall:   {recall:.4%}")
    print(f"  Precision:{precision:.4%}")
    print(f"  False Alarm: {false_alarm:.4%}")

print("\n================ HAI ENGINEERED SUBSYSTEM PINN ADVANCED METRICS ================")
evaluate_subsystems(threshold_percentile=99.0, window_size=1, agg_mode='or')
evaluate_subsystems(threshold_percentile=99.0, window_size=25, agg_mode='or')
evaluate_subsystems(threshold_percentile=99.9, window_size=1, agg_mode='or')
evaluate_subsystems(threshold_percentile=99.0, window_size=1, agg_mode='max')
