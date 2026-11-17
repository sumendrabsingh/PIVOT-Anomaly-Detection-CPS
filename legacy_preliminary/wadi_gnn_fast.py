import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import confusion_matrix, f1_score

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAIN_DATA = r'Datasets/wadi_train_preprocessed.csv'
EVAL_DATA = r'Datasets/wadi_test_preprocessed.csv'

print('Loading WADI data...')
df_train = pd.read_csv(TRAIN_DATA)
df_eval = pd.read_csv(EVAL_DATA)

numeric_cols = [c for c in df_train.columns if 'Normal' not in c and 'Label' not in c]

subsystems = {
    'P1': [c for c in numeric_cols if c.startswith('1_')],
    'P2': [c for c in numeric_cols if c.startswith('2_')],
    'P3': [c for c in numeric_cols if c.startswith('3_')]
}

num_nodes = 3
A = torch.zeros((num_nodes, num_nodes))
for i in range(num_nodes): A[i, i] = 1.0
for i in range(num_nodes - 1):
    A[i, i+1] = 1.0
    A[i+1, i] = 1.0
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
for k in ['P1', 'P2', 'P3']:
    train_vals = df_train[subsystems[k]].fillna(0).values
    test_vals = df_eval[subsystems[k]].fillna(0).values
    mean = np.mean(train_vals, axis=0)
    std = np.std(train_vals, axis=0) + 1e-6
    data_tensors.append(torch.tensor((train_vals - mean) / std, dtype=torch.float32).to(DEVICE))
    test_tensors.append(torch.tensor((test_vals - mean) / std, dtype=torch.float32).to(DEVICE))

model = GNNAutoencoder([len(subsystems[k]) for k in ['P1', 'P2', 'P3']], hidden_dim=32).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.001)

print('Training GNN for 5 epochs (fast proof of concept)...')
model.train()
for epoch in range(5):
    optimizer.zero_grad()
    recon_list = model(data_tensors)
    loss = sum(torch.mean((recon_list[i] - data_tensors[i])**2) for i in range(3))
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    recon_list = model(test_tensors)
    errors = torch.stack([torch.mean((recon_list[i] - test_tensors[i])**2, dim=1) for i in range(3)], dim=1)
    max_errors = torch.max(errors, dim=1)[0].cpu().numpy()

y_true = df_eval['Normal/Attack'].astype(str).str.strip().str.lower().str.replace(' ', '').apply(lambda val: 1 if val in ['attack', '1', '1.0'] else 0).values

print('Computing Optimal Threshold...')
max_normal = max_errors[y_true == 0]
t_max = np.percentile(max_normal, 99.0)

preds = (max_errors > t_max).astype(int)
tp = ((preds == 1) & (y_true == 1)).sum()
fp = ((preds == 1) & (y_true == 0)).sum()
fn = ((preds == 0) & (y_true == 1)).sum()
tn = ((preds == 0) & (y_true == 0)).sum()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

print(f'\nWADI GNN Performance (Aggregated Max | P99):')
print(f'  F1-Score: {f1:.4%}')
print(f'  Recall:   {recall:.4%}')
print(f'  Precision:{precision:.4%}')
print(f'  False Alarm: {fp / (fp + tn):.4%}')
