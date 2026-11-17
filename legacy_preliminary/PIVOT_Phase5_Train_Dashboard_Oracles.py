import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(r"Oracles", exist_ok=True)

class SubsystemAutoencoder(torch.nn.Module):
    def __init__(self, input_dim, bottleneck_dim=8):
        super().__init__()
        if input_dim <= 8:
            hidden1 = 32
            hidden2 = 16
        elif input_dim <= 16:
            hidden1 = 64
            hidden2 = 32
        elif input_dim <= 32:
            hidden1 = 128
            hidden2 = 64
        else:
            hidden1 = 256
            hidden2 = 128
            
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden1),
            torch.nn.BatchNorm1d(hidden1),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            
            torch.nn.Linear(hidden1, hidden2),
            torch.nn.BatchNorm1d(hidden2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            
            torch.nn.Linear(hidden2, bottleneck_dim)
        )
        
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(bottleneck_dim, hidden2),
            torch.nn.BatchNorm1d(hidden2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            
            torch.nn.Linear(hidden2, hidden1),
            torch.nn.BatchNorm1d(hidden1),
            torch.nn.ReLU(),
            
            torch.nn.Linear(hidden1, input_dim)
        )
        
    def forward(self, x):
        return self.decoder(self.encoder(x))

def train_and_save_wadi():
    print("Pre-training WADI Dashboard Oracles...")
    df_train = pd.read_csv(r'Datasets/wadi_train_preprocessed.csv', nrows=50000)
    
    numeric_cols = [c for c in df_train.columns if 'Normal' not in c and 'Label' not in c]
    subsystems = {
        'P1_RawWater': [c for c in numeric_cols if c.startswith('1_')],
        'P2_ElevatedTanks': [c for c in numeric_cols if c.startswith('2_')],
        'P3_ReturnWater': [c for c in numeric_cols if c.startswith('3_')]
    }
    
    for name, features in subsystems.items():
        if len(features) == 0: continue
        
        # NOTE: wadi_train_preprocessed.csv is NOT normalized. We must normalize it and SAVE the mean/std
        # wait! wadi_test_preprocessed.csv is NOT scaled! I must scale it inside the Dashboard.
        # Oh, in wadi_subsystem_fast.py, I used StandardScaler on the fly. 
        # I need to save the scaler parameters so the dashboard can use them.
        
        train_vals = df_train[features].fillna(0).values
        mean = np.mean(train_vals, axis=0)
        std = np.std(train_vals, axis=0) + 1e-6
        x_train = torch.tensor((train_vals - mean) / std, dtype=torch.float32).to(DEVICE)
        
        model = SubsystemAutoencoder(len(features), bottleneck_dim=max(2, len(features)//2)).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        
        model.train()
        for epoch in range(10):
            optimizer.zero_grad()
            loss = torch.mean((x_train - model(x_train))**2)
            loss.backward()
            optimizer.step()
            
        # Save model and its specific scaler parameters
        checkpoint = {
            'input_dim': len(features),
            'bottleneck_dim': max(2, len(features)//2),
            'model_state': model.state_dict(),
            'mean': mean.tolist(),
            'std': std.tolist(),
            'features': features
        }
        torch.save(checkpoint, f"Oracles/wadi_subsystem_{name}.pth")
        print(f"  -> Saved wadi_subsystem_{name}.pth")

def train_and_save_hai():
    print("Pre-training HAI Dashboard Oracles...")
    df_train = pd.read_csv(r'Datasets/hai_train_engineered.csv', nrows=50000)
    
    numeric_cols = [c for c in df_train.columns if 'Label' not in c]
    subsystems = {
        'P1_Boiler': [c for c in numeric_cols if 'P1_' in c or 'Error_WaterLevel' in c or 'Error_Temperature' in c],
        'P2_Turbine': [c for c in numeric_cols if 'P2_' in c or 'Error_TurbineSpeed' in c],
        'P3_WaterTreatment': [c for c in numeric_cols if 'P3_' in c],
        'P4_HILSimulator': [c for c in numeric_cols if 'P4_' in c]
    }
    
    for name, features in subsystems.items():
        if len(features) == 0: continue
        
        train_vals = df_train[features].fillna(0).values
        mean = np.mean(train_vals, axis=0)
        std = np.std(train_vals, axis=0) + 1e-6
        x_train = torch.tensor((train_vals - mean) / std, dtype=torch.float32).to(DEVICE)
        
        model = SubsystemAutoencoder(len(features), bottleneck_dim=max(2, len(features)//2)).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        
        model.train()
        for epoch in range(10):
            optimizer.zero_grad()
            loss = torch.mean((x_train - model(x_train))**2)
            loss.backward()
            optimizer.step()
            
        checkpoint = {
            'input_dim': len(features),
            'bottleneck_dim': max(2, len(features)//2),
            'model_state': model.state_dict(),
            'mean': mean.tolist(),
            'std': std.tolist(),
            'features': features
        }
        torch.save(checkpoint, f"Oracles/hai_subsystem_{name}.pth")
        print(f"  -> Saved hai_subsystem_{name}.pth")

def train_and_save_swat():
    print("Pre-training SWaT Dashboard Oracles...")
    df_train = pd.read_csv(r'Datasets/swat_train_preprocessed.csv', nrows=50000)
    
    numeric_cols = [c for c in df_train.columns if 'Normal' not in c and 'Label' not in c]
    subsystems = {
        'Stage1_RawWater': [c for c in numeric_cols if c.startswith('FIT1') or c.startswith('LIT1') or c.startswith('MV1') or c.startswith('P1')],
        'Stage2_ChemicalDosing': [c for c in numeric_cols if c.startswith('FIT2') or c.startswith('AIT2') or c.startswith('MV2') or c.startswith('P2')],
        'Stage3_Ultrafiltration': [c for c in numeric_cols if c.startswith('FIT3') or c.startswith('DPIT3') or c.startswith('MV3') or c.startswith('P3')],
        'Stage4_Dechlorination': [c for c in numeric_cols if c.startswith('AIT4') or c.startswith('LIT4') or c.startswith('MV4') or c.startswith('P4')],
        'Stage5_ReverseOsmosis': [c for c in numeric_cols if c.startswith('AIT5') or c.startswith('FIT5') or c.startswith('PIT5') or c.startswith('MV5') or c.startswith('P5')],
        'Stage6_ROCleaning': [c for c in numeric_cols if c.startswith('FIT6') or c.startswith('P6')]
    }
    
    for name, features in subsystems.items():
        if len(features) == 0: continue
        
        train_vals = df_train[features].fillna(0).values
        mean = np.mean(train_vals, axis=0)
        std = np.std(train_vals, axis=0) + 1e-6
        x_train = torch.tensor((train_vals - mean) / std, dtype=torch.float32).to(DEVICE)
        
        model = SubsystemAutoencoder(len(features), bottleneck_dim=max(2, len(features)//2)).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        
        model.train()
        for epoch in range(10):
            optimizer.zero_grad()
            loss = torch.mean((x_train - model(x_train))**2)
            loss.backward()
            optimizer.step()
            
        checkpoint = {
            'input_dim': len(features),
            'bottleneck_dim': max(2, len(features)//2),
            'model_state': model.state_dict(),
            'mean': mean.tolist(),
            'std': std.tolist(),
            'features': features
        }
        torch.save(checkpoint, f"Oracles/subsystem_{name}.pth")
        print(f"  -> Saved subsystem_{name}.pth")

if __name__ == '__main__':
    train_and_save_swat()
    train_and_save_wadi()
    train_and_save_hai()
    print("Done!")
