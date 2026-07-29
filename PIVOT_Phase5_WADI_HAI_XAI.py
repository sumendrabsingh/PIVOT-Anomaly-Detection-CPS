import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SubsystemAE(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        hd = max(2, input_dim // 2)
        self.enc = nn.Sequential(nn.Linear(input_dim, hd), nn.ReLU(), nn.Linear(hd, hd))
        self.dec = nn.Sequential(nn.Linear(hd, hd), nn.ReLU(), nn.Linear(hd, input_dim))
    def forward(self, x):
        return self.dec(self.enc(x))

def generate_wadi_xai():
    print("Loading WADI...")
    df_train = pd.read_csv(r'Datasets/wadi_train_preprocessed.csv', nrows=50000) # Fast train
    df_eval = pd.read_csv(r'Datasets/wadi_test_preprocessed.csv')
    
    numeric_cols = [c for c in df_train.columns if 'Normal' not in c and 'Label' not in c]
    subsystems = {
        'P1 (Raw Water)': [c for c in numeric_cols if c.startswith('1_')],
        'P2 (Elevated Tanks)': [c for c in numeric_cols if c.startswith('2_')],
        'P3 (Return Water)': [c for c in numeric_cols if c.startswith('3_')]
    }
    
    y_true = df_eval['Normal/Attack'].astype(str).str.strip().str.lower().str.replace(' ', '').apply(lambda val: 1 if val in ['attack', '1', '1.0'] else 0).values
    attack_indices = np.where(y_true == 1)[0]
    if len(attack_indices) == 0: return
    
    start_idx = attack_indices[0] - 200
    end_idx = attack_indices[0] + 800
    if start_idx < 0: start_idx = 0
    
    errors_dict = {}
    for name, features in subsystems.items():
        if len(features) == 0: continue
        train_vals = df_train[features].fillna(0).values
        eval_vals = df_eval[features].iloc[start_idx:end_idx].fillna(0).values
        mean = np.mean(train_vals, axis=0)
        std = np.std(train_vals, axis=0) + 1e-6
        x_train = torch.tensor((train_vals - mean) / std, dtype=torch.float32).to(DEVICE)
        x_eval = torch.tensor((eval_vals - mean) / std, dtype=torch.float32).to(DEVICE)
        
        model = SubsystemAE(len(features)).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        model.train()
        for _ in range(5):
            optimizer.zero_grad()
            loss = torch.mean((x_train - model(x_train))**2)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            err = torch.mean((x_eval - model(x_eval))**2, dim=1).cpu().numpy()
            # Normalize error for heatmap visualization
            err = (err - np.min(err)) / (np.max(err) - np.min(err) + 1e-6)
            errors_dict[name] = err
            
    # Plot WADI Heatmap
    plot_df = pd.DataFrame(errors_dict).T
    plt.figure(figsize=(12, 4))
    im = plt.imshow(plot_df.values, cmap='Reds', aspect='auto')
    plt.colorbar(im, label='Normalized Reconstruction Error')
    plt.yticks(ticks=np.arange(len(plot_df.index)), labels=plot_df.index)
    plt.axvline(x=200, color='blue', linestyle='--', linewidth=2, label='Attack Start')
    plt.title('WADI XAI: Subsystem PINN Root-Cause Isolation')
    plt.xlabel('Time Steps')
    plt.ylabel('Sub-Plant')
    plt.legend()
    plt.tight_layout()
    plt.savefig(r'C:\Users\sumen\.gemini\antigravity-ide\brain\561c5052-8640-4032-829d-cbcfa9a21b49\wadi_xai_heatmap.png', dpi=300)
    plt.close()

def generate_hai_xai():
    print("Loading HAI Engineered...")
    df_train = pd.read_csv(r'Datasets/hai_train_engineered.csv', nrows=50000)
    df_eval = pd.read_csv(r'Datasets/hai_test_engineered.csv')
    
    numeric_cols = [c for c in df_train.columns if 'Label' not in c]
    subsystems = {
        'P1 (Boiler Thermodynamics)': [c for c in numeric_cols if 'P1_' in c or 'Error_WaterLevel' in c or 'Error_Temperature' in c],
        'P2 (Turbine Speed/Vibration)': [c for c in numeric_cols if 'P2_' in c or 'Error_TurbineSpeed' in c],
        'P3 (Water Treatment Hydrodynamics)': [c for c in numeric_cols if 'P3_' in c],
        'P4 (HIL Simulator Grid)': [c for c in numeric_cols if 'P4_' in c]
    }
    
    y_true = df_eval['Label'].astype(int).values
    attack_indices = np.where(y_true == 1)[0]
    if len(attack_indices) == 0: return
    
    start_idx = attack_indices[0] - 200
    end_idx = attack_indices[0] + 800
    if start_idx < 0: start_idx = 0
    
    errors_dict = {}
    for name, features in subsystems.items():
        if len(features) == 0: continue
        train_vals = df_train[features].fillna(0).values
        eval_vals = df_eval[features].iloc[start_idx:end_idx].fillna(0).values
        mean = np.mean(train_vals, axis=0)
        std = np.std(train_vals, axis=0) + 1e-6
        x_train = torch.tensor((train_vals - mean) / std, dtype=torch.float32).to(DEVICE)
        x_eval = torch.tensor((eval_vals - mean) / std, dtype=torch.float32).to(DEVICE)
        
        model = SubsystemAE(len(features)).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        model.train()
        for _ in range(5):
            optimizer.zero_grad()
            loss = torch.mean((x_train - model(x_train))**2)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            err = torch.mean((x_eval - model(x_eval))**2, dim=1).cpu().numpy()
            err = (err - np.min(err)) / (np.max(err) - np.min(err) + 1e-6)
            errors_dict[name] = err
            
    # Plot HAI Heatmap
    plot_df = pd.DataFrame(errors_dict).T
    plt.figure(figsize=(12, 5))
    im = plt.imshow(plot_df.values, cmap='Reds', aspect='auto')
    plt.colorbar(im, label='Normalized Reconstruction Error')
    plt.yticks(ticks=np.arange(len(plot_df.index)), labels=plot_df.index)
    plt.axvline(x=200, color='blue', linestyle='--', linewidth=2, label='Attack Start')
    plt.title('HAI XAI: Extreme Noise Isolation via Thermodynamic Partitions')
    plt.xlabel('Time Steps')
    plt.ylabel('Sub-Plant')
    plt.legend()
    plt.tight_layout()
    plt.savefig(r'C:\Users\sumen\.gemini\antigravity-ide\brain\561c5052-8640-4032-829d-cbcfa9a21b49\hai_xai_heatmap.png', dpi=300)
    plt.close()

if __name__ == '__main__':
    generate_wadi_xai()
    generate_hai_xai()
    print("Done generating XAI heatmaps!")
