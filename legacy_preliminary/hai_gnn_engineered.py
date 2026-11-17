import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAIN_DATA = r'Datasets/hai_train_engineered.csv'
EVAL_DATA = r'Datasets/hai_test_engineered.csv'

print('Loading Engineered HAI data for ST-GAE (GNN)...')
df_train = pd.read_csv(TRAIN_DATA)
df_eval = pd.read_csv(EVAL_DATA)

numeric_cols = [c for c in df_train.columns if 'Label' not in c]

subsystems = {
    'P1': [c for c in numeric_cols if 'P1_' in c or 'Error_WaterLevel' in c or 'Error_Temperature' in c],
    'P2': [c for c in numeric_cols if 'P2_' in c or 'Error_TurbineSpeed' in c],
    'P3': [c for c in numeric_cols if 'P3_' in c],
    'P4': [c for c in numeric_cols if 'P4_' in c]
}

num_nodes = 4
A = torch.zeros((num_nodes, num_nodes))
# P1(Boiler) is connected to P4(Simulator)
A[0, 3] = 1.0; A[3, 0] = 1.0
# P3(Water) is connected to P4(Simulator)
A[2, 3] = 1.0; A[3, 2] = 1.0
# P2(Turbine) is connected to P4(Simulator)
A[1, 3] = 1.0; A[3, 1] = 1.0
for i in range(num_nodes): A[i, i] = 1.0

D = torch.sum(A, dim=1)
D_inv_sqrt = torch.pow(D, -0.5)
D_mat = torch.diag(D_inv_sqrt)
A_norm = (D_mat @ A @ D_mat).to(DEVICE)

class GNNAutoencoder(nn.Module):
    def __init__(self, feature_dims, hidden_dim):
        super().__init__()
        self.A_norm = A_norm
        self.encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)) for dim in feature_dims
        ])
        self.gcn1_weights = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        self.gcn2_weights = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        self.decoders = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, dim)) for dim in feature_dims
        ])
        
    def forward(self, x_list):
        h_list = [enc(x) for enc, x in zip(self.encoders, x_list)]
        H = torch.stack(h_list, dim=1)
        H = torch.matmul(self.A_norm, H)
        H = torch.matmul(H, self.gcn1_weights)
        H = torch.relu(H)
        H = torch.matmul(self.A_norm, H)
        H = torch.matmul(H, self.gcn2_weights)
        recon_list = [dec(H[:, i, :]) for i, dec in enumerate(self.decoders)]
        return recon_list

print('Normalizing...')
data_tensors = []
test_tensors = []
for k in ['P1', 'P2', 'P3', 'P4']:
    train_vals = df_train[subsystems[k]].fillna(0).values
    test_vals = df_eval[subsystems[k]].fillna(0).values
    mean = np.mean(train_vals, axis=0)
    std = np.std(train_vals, axis=0) + 1e-6
    data_tensors.append(torch.tensor((train_vals - mean) / std, dtype=torch.float32).to(DEVICE))
    test_tensors.append(torch.tensor((test_vals - mean) / std, dtype=torch.float32).to(DEVICE))

model = GNNAutoencoder([len(subsystems[k]) for k in ['P1', 'P2', 'P3', 'P4']], hidden_dim=32).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.001)

print('Training Engineered GNN for 10 epochs...')
batch_size = 4096
model.train()
for epoch in range(10):
    total_loss = 0
    for i in range(0, len(data_tensors[0]), batch_size):
        batch_list = [t[i:i+batch_size] for t in data_tensors]
        optimizer.zero_grad()
        recon_list = model(batch_list)
        loss = sum(torch.mean((recon_list[j] - batch_list[j])**2) for j in range(4))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch_list[0])
    if epoch == 9:
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(data_tensors[0]):.4f}')

model.eval()
errors_list = []
with torch.no_grad():
    for i in range(0, len(test_tensors[0]), batch_size):
        batch_list = [t[i:i+batch_size] for t in test_tensors]
        recon_list = model(batch_list)
        err = torch.stack([torch.mean((recon_list[j] - batch_list[j])**2, dim=1) for j in range(4)], dim=1)
        errors_list.append(err.cpu().numpy())

errors = np.concatenate(errors_list, axis=0)
max_errors = np.max(errors, axis=1)
y_true = df_eval['Label'].astype(int).values

def eval_gnn(t_p, w_size):
    if w_size > 1:
        smoothed = pd.Series(max_errors).rolling(window=w_size, min_periods=1).mean().values
    else:
        smoothed = max_errors
    
    max_normal = smoothed[y_true == 0]
    t_max = np.percentile(max_normal, t_p)
    
    preds = (smoothed > t_max).astype(int)
    tp = ((preds == 1) & (y_true == 1)).sum()
    fp = ((preds == 1) & (y_true == 0)).sum()
    fn = ((preds == 0) & (y_true == 1)).sum()
    tn = ((preds == 0) & (y_true == 0)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print(f"\n[ENGINEERED GNN MAX | Win={w_size} | P{t_p}]")
    print(f"  F1-Score: {f1:.4%}")
    print(f"  Recall:   {recall:.4%}")
    print(f"  Precision:{precision:.4%}")
    print(f"  False Alarm: {fp / (fp + tn):.4%}")

eval_gnn(99.0, 1)
eval_gnn(99.0, 25)
