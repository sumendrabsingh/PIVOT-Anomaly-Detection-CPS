import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
import json
import argparse
from sklearn.metrics import confusion_matrix, f1_score

# --- Config ---
AGGREGATION_MODE = 'max'
ERROR_MODE = 'max'
WINDOW_SIZE = 25
SEQ_LEN = 1
ALIGN_BASELINE = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SubsystemAutoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim):
        super().__init__()
        
        if input_dim > 64:
            hidden1 = 256
            hidden2 = 128
        elif input_dim > 16:
            hidden1 = 128
            hidden2 = 64
        elif input_dim > 8:
            hidden1 = 64
            hidden2 = 32
        else:
            hidden1 = 32
            hidden2 = 16
            
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden2, bottleneck_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden2, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            
            nn.Linear(hidden1, input_dim)
        )
        
    def forward(self, x):
        return self.decoder(self.encoder(x))
        
    def get_reconstruction_error(self, x, mode='mean'):
        with torch.no_grad():
            reconstructed = self.forward(x)
            squared_diff = (x - reconstructed) ** 2
            if mode == 'max':
                return torch.max(squared_diff, dim=1)[0]
            else:
                return torch.mean(squared_diff, dim=1)

class GenericPINNLoss(nn.Module):
    def __init__(self, features, mean, std, lambda_phys=0.1):
        super().__init__()
        self.mse = nn.MSELoss(reduction='mean')
        self.features = features
        self.mean = mean
        self.std = std
        self.lambda_phys = lambda_phys

    def forward(self, x_hat, x):
        mse_loss = self.mse(x_hat, x)
        
        if self.lambda_phys <= 0:
            return mse_loss
            
        x_hat_phys = x_hat * self.std + self.mean
        phys_loss = torch.tensor(0.0, device=x.device)
        
        # Generic non-negativity constraint for flows, levels, pressures, temperatures
        # Common in SWaT, WADI, HAI: FIT, LIT, PIT, AIT, P_, L_, F_, T_
        for i, feat in enumerate(self.features):
            feat_upper = feat.upper()
            if any(key in feat_upper for key in ['FIT', 'LIT', 'PIT', 'AIT', 'P_', 'L_', 'F_', 'T_', 'PRESSURE', 'FLOW', 'LEVEL']):
                phys_loss += torch.mean(torch.relu(-x_hat_phys[:, i]))
                
        return mse_loss + self.lambda_phys * phys_loss

def pgd_attack_subsystem(model, x_clean, y_true, epsilon=0.1, alpha=0.01, num_iter=10):
    x_adv = x_clean.clone().detach().to(x_clean.device)
    attack_mask = (y_true == 1)
    if not attack_mask.any():
        return x_adv

    x_adv_attacks = x_adv[attack_mask].clone().detach()
    x_clean_attacks = x_clean[attack_mask].clone().detach()
    x_adv_attacks.requires_grad = True
    
    for _ in range(num_iter):
        reconstructed = model(x_adv_attacks)
        squared_diff = (x_adv_attacks - reconstructed) ** 2
        loss = torch.mean(torch.max(squared_diff, dim=1)[0]) if ERROR_MODE == 'max' else torch.mean(squared_diff)
            
        model.zero_grad()
        if x_adv_attacks.grad is not None:
            x_adv_attacks.grad.zero_()
        loss.backward()
        
        with torch.no_grad():
            grad_sign = x_adv_attacks.grad.sign()
            x_adv_attacks = x_adv_attacks - alpha * grad_sign
            delta = torch.clamp(x_adv_attacks - x_clean_attacks, min=-epsilon, max=epsilon)
            x_adv_attacks = x_clean_attacks + delta
            x_adv_attacks.requires_grad = True

    with torch.no_grad():
        x_adv[attack_mask] = x_adv_attacks.detach()
    return x_adv

def expand_features(features, columns):
    """Map a subsystem's logical feature names onto the actual dataframe columns.

    Preprocessing one-hot-encodes the valve actuators, so the subsystem JSON's 'MV101' becomes
    'MV101_0'/'MV101_1'/'MV101_2' in the CSV. A plain `f in df.columns` test silently drops
    every valve -- i.e. exactly the actuator states the physics invariants are built on -- so
    match a logical name to itself OR to its one-hot children. Column order is preserved so the
    feature vector is stable across train and eval.
    """
    return [c for c in columns
            if any(c == f or c.startswith(f + '_') for f in features)]


def train_subsystem(name, features, train_data_path, out_dir, seed=None):
    print(f"\nTraining Subsystem Autoencoder for {name} with {len(features)} features")
    if seed is not None:
        # The train/val split below was already seeded, but weight init and batch shuffling
        # were not, so repeated runs drifted by ~1 F1 point. Pin all three.
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
    df = pd.read_csv(train_data_path)
    df.columns = df.columns.str.strip()
    
    valid_features = expand_features(features, df.columns)
    if len(valid_features) == 0:
        print(f"Skipping {name}: No matching features found in dataset.")
        return
        
    sub_df = df[valid_features].fillna(df[valid_features].mean())
    
    # Downsample huge datasets to speed up
    if len(sub_df) > 100000:
        sub_df = sub_df.iloc[::5].reset_index(drop=True)
        
    data_val = sub_df.values
    data_tensor = torch.tensor(data_val, dtype=torch.float32)
    
    mean = data_tensor.mean(dim=0)
    std = data_tensor.std(dim=0)
    std = torch.where(std < 1e-4, torch.ones_like(std), std)
    
    norm_data = (data_tensor - mean) / std
    norm_data = norm_data.to(DEVICE)
    
    dataset = TensorDataset(norm_data)
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4096, shuffle=False)
    
    input_dim = data_tensor.shape[1]
    bottleneck_dim = max(1, input_dim // 2)
    
    model = SubsystemAutoencoder(input_dim, bottleneck_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = GenericPINNLoss(valid_features, mean.to(DEVICE), std.to(DEVICE), lambda_phys=0.1)
    
    best_val_loss = float('inf')
    patience = 20
    epochs_no_improve = 0
    best_model_state = None
    
    for epoch in range(100):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x_batch = batch[0]
            optimizer.zero_grad()
            reconstructed = model(x_batch)
            loss = criterion(reconstructed, x_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x_batch = batch[0]
                reconstructed = model(x_batch)
                loss = criterion(reconstructed, x_batch)
                val_loss += loss.item() * x_batch.size(0)
        val_loss /= len(val_loader.dataset)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            break
            
    checkpoint = {
        'model_state': best_model_state,
        'mean': mean,
        'std': std,
        'features': valid_features,
        'input_dim': input_dim,
        'bottleneck_dim': bottleneck_dim
    }
    
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"subsystem_{name}.pth")
    torch.save(checkpoint, out_path)
    print(f"  Saved subsystem model to {out_path} (Best Val: {best_val_loss:.6f})")

def _compute_subsystem_scores(subsystems, test_data_path, out_dir, run_adversarial=False, epsilon=0.1, alpha=0.01, num_iter=10):
    df_eval = pd.read_csv(test_data_path)
    df_eval.columns = df_eval.columns.str.strip()

    label_col = 'Label' if 'Label' in df_eval.columns else 'Normal/Attack'
    if label_col in df_eval.columns:
        if df_eval[label_col].dtype == object:
            y_true = df_eval[label_col].astype(str).str.strip().str.lower().str.replace(" ", "").apply(
                lambda val: 1 if val in ["attack", "1", "1.0"] else 0).values
        else:
            y_true = df_eval[label_col].astype(str).str.strip().str.lower().str.replace(" ", "").apply(
                lambda val: 1 if val in ["attack", "1", "1.0"] else 0).values
    else:
        y_true = np.zeros(len(df_eval))

    subsystem_scores = {}

    for name, features in subsystems.items():
        model_path = os.path.join(out_dir, f"subsystem_{name}.pth")
        if not os.path.exists(model_path):
            continue

        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        input_dim = checkpoint['input_dim']
        bottleneck_dim = checkpoint['bottleneck_dim']
        ckpt_features = checkpoint.get('features', features)
        model_state = checkpoint['model_state']
        
        hidden1 = model_state['encoder.0.weight'].shape[0]
        hidden2 = model_state['encoder.4.weight'].shape[0]
        
        model = SubsystemAutoencoder(input_dim, bottleneck_dim).to(DEVICE)
        model.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden1), nn.BatchNorm1d(hidden1), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden1, hidden2), nn.BatchNorm1d(hidden2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden2, bottleneck_dim)
        ).to(DEVICE)
        model.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden2), nn.BatchNorm1d(hidden2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden2, hidden1), nn.BatchNorm1d(hidden1), nn.ReLU(),
            nn.Linear(hidden1, input_dim)
        ).to(DEVICE)
        
        model.load_state_dict(model_state)
        model.eval()

        sub_df = df_eval[ckpt_features].fillna(df_eval[ckpt_features].mean())
        data_tensor = torch.tensor(sub_df.values, dtype=torch.float32)

        # Bundles store per-feature mean/std as lists; coerce to a broadcastable tensor.
        MEAN = torch.as_tensor(checkpoint.get('mean', 0.0), dtype=torch.float32)
        STD = torch.as_tensor(checkpoint.get('std', 1.0), dtype=torch.float32)
        STD = torch.where(STD == 0, torch.ones_like(STD), STD)

        norm_data = (data_tensor - MEAN) / STD

        batch_size = 4096
        errors_list = []
        y_true_tensor = torch.tensor(y_true, dtype=torch.long, device=DEVICE)
        
        for i in range(0, len(norm_data), batch_size):
            batch = norm_data[i:i + batch_size].to(DEVICE)
            batch_y = y_true_tensor[i:i + batch_size]
            
            if run_adversarial:
                for param in model.parameters():
                    param.requires_grad = True
                batch = pgd_attack_subsystem(model, batch, batch_y, epsilon, alpha, num_iter)
                for param in model.parameters():
                    param.requires_grad = False
                    
            recon_errors = model.get_reconstruction_error(batch, mode=ERROR_MODE)
            errors_list.extend(recon_errors.cpu().numpy().tolist())

        raw_scores = np.array(errors_list)
        normal_scores_subset = raw_scores[y_true == 0] if len(raw_scores[y_true==0]) > 0 else raw_scores
        mean_err = np.mean(normal_scores_subset)
        std_err = np.std(normal_scores_subset)
        
        standardized_scores = (raw_scores - mean_err) / (std_err + 1e-6)
        subsystem_scores[name] = standardized_scores

    return y_true, subsystem_scores

def evaluate_subsystems_or_voting(subsystems, test_data_path, out_dir, window_size=25, run_adversarial=False):
    print("\n[OR-VOTING] Evaluating Subsystems...")
    y_true, subsystem_scores = _compute_subsystem_scores(subsystems, test_data_path, out_dir, run_adversarial)
    if y_true is None or len(subsystem_scores) == 0:
        return None

    if y_true.sum() == 0:
        print("[WARNING] Test dataset has NO ATTACK labels! Evaluation metrics will be zero.")

    n_samples = len(y_true)
    smoothed = {}
    for name, scores in subsystem_scores.items():
        if window_size > 1:
            smoothed[name] = pd.Series(scores).rolling(window=window_size, min_periods=1).mean().values
        else:
            smoothed[name] = scores

    per_sub_thresholds = {}
    for name, scores in smoothed.items():
        normal_sc = scores[y_true == 0]
        if len(normal_sc) > 0:
            # Subsystem Independent Threshold based on normal distribution
            best_t = np.percentile(normal_sc, 99.8)
        else:
            best_t = np.percentile(scores, 99.8)
        per_sub_thresholds[name] = best_t

    combined_preds = np.zeros(n_samples, dtype=int)
    for name, scores in smoothed.items():
        t = per_sub_thresholds[name]
        sub_preds = (scores > t).astype(int)
        combined_preds = np.maximum(combined_preds, sub_preds)

    if y_true.sum() > 0:
        cm = confusion_matrix(y_true, combined_preds)
        if cm.shape == (2,2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = cm[0,0], 0, 0, 0
            
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = f1_score(y_true, combined_preds, zero_division=0)
        false_alarm = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    else:
        accuracy, precision, recall, f1, false_alarm = 0, 0, 0, 0, 0

    print(f"  Accuracy:                {accuracy:.4%}")
    print(f"  F1-Score:                {f1:.4%}")
    print(f"  Recall (Detection Rate): {recall:.4%}")
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'false_alarm': false_alarm}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['swat', 'wadi', 'hai'])
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--adversarial', action='store_true')
    parser.add_argument('--seed', type=int, default=None,
                        help='Pin the training seed (weight init, shuffling, split). The shipped '
                             'SWaT bundles are seed 4; see the Reproducibility section of the README.')
    args = parser.parse_args()
    
    dataset_name = args.dataset
    if dataset_name == 'swat':
        train_path = r"Datasets/swat_train_preprocessed.csv"
        test_path = r"Datasets/swat_test_preprocessed.csv"
        sub_json = r"Datasets/swat_subsystems.json"
        out_dir = r"Oracles/swat"
    elif dataset_name == 'wadi':
        train_path = r"Datasets/wadi_train_preprocessed.csv"
        test_path = r"Datasets/wadi_test_preprocessed.csv"
        sub_json = r"Datasets/wadi_subsystems.json"
        out_dir = r"Oracles/wadi"
    else:
        train_path = r"Datasets/hai_train_preprocessed.csv"
        test_path = r"Datasets/hai_test_preprocessed.csv"
        sub_json = r"Datasets/hai_subsystems.json"
        out_dir = r"Oracles/hai"
        
    with open(sub_json, 'r') as f:
        subsystems = json.load(f)
        
    if args.train:
        for name, features in subsystems.items():
            train_subsystem(name, features, train_path, out_dir, seed=args.seed)
            
    if args.eval:
        evaluate_subsystems_or_voting(subsystems, test_path, out_dir, run_adversarial=args.adversarial)
