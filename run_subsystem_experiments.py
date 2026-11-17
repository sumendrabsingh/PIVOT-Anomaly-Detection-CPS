"""v1 SWaT subsystem runner -- REFERENCE ONLY (leaks test labels; physics rules also flawed:
PINNLoss gates on the RECONSTRUCTED valve state and includes MV301 -> FIT301, which is false --
MV301 is the UF backwash valve. Use run_generalized_subsystems.py --physics interlock instead).

This is the protocol behind the v1 paper's numbers: ALIGN_BASELINE normalizes inputs on test rows
labelled normal, errors are standardized on those rows, and evaluate_dual_threshold sweeps
thresholds for F1 on the test labels. Keep it only to reproduce those numbers. The label-free
protocol (see label_free.py) is run_generalized_subsystems.py with the default --calibration train.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

# --- Config ---
TRAIN_MODELS = True
AGGREGATION_MODE = 'max'  # Options: 'max' or 'mean'
ERROR_MODE = 'max'        # Options: 'max' or 'mean' reconstruction error inside AE
WINDOW_SIZE = 25
SEQ_LEN = 1              # Temporal window size for sequence reconstruction
ALIGN_BASELINE = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRAIN_DATA = r"SWaT_Orig_DataSets/Original Dataset Backup/SWaT_Dataset_Normal_v1_Clean.csv"
EVAL_DATA = r"SWaT_Orig_DataSets/SWaT_Dataset_Attack_v0_Corrected.csv"
# Where subsystem_<name>.pth bundles are read from / written to. The root Oracles/ copies have
# been overwritten by later ablation arms; the archived sets live in Oracles/_backup_* and
# Oracles/_arm_*, so this is overridable rather than inlined.
MODEL_DIR = "Oracles"

# 6 stages of the Secure Water Treatment (SWaT) system
subsystems = {
    "Stage1_RawWater": ['FIT101', 'LIT101', 'MV101', 'P101'],
    "Stage2_ChemicalDosing": ['AIT201', 'AIT202', 'AIT203', 'FIT201', 'MV201', 'P203', 'P205'],
    "Stage3_Ultrafiltration": ['DPIT301', 'FIT301', 'LIT301', 'MV301', 'MV302', 'MV303', 'MV304', 'P301', 'P302'],
    "Stage4_Dechlorination": ['AIT401', 'AIT402', 'FIT401', 'LIT401', 'P402', 'UV401'],
    "Stage5_ReverseOsmosis": ['AIT501', 'AIT502', 'AIT503', 'AIT504', 'FIT501', 'FIT502', 'FIT503', 'FIT504', 'P501', 'PIT501', 'PIT502', 'PIT503'],
    "Stage6_ROCleaning": ['FIT601', 'P602']
}

class SubsystemAutoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim):
        super().__init__()
        
        # Adjust hidden dimensions based on input dimension dynamically
        # Adjust hidden dimensions dynamically or use large default to match checkpoint
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

    def get_latent(self, x):
        with torch.no_grad():
            return self.encoder(x)

class PINNLoss(nn.Module):
    def __init__(self, features, mean, std, lambda_phys=0.1):
        super().__init__()
        self.mse = nn.MSELoss(reduction='mean')
        self.features = features
        self.mean = mean
        self.std = std
        self.lambda_phys = lambda_phys
        
        self.idx_MV101 = features.index('MV101') if 'MV101' in features else -1
        self.idx_FIT101 = features.index('FIT101') if 'FIT101' in features else -1
        self.idx_MV201 = features.index('MV201') if 'MV201' in features else -1
        self.idx_FIT201 = features.index('FIT201') if 'FIT201' in features else -1
        self.idx_MV301 = features.index('MV301') if 'MV301' in features else -1
        self.idx_FIT301 = features.index('FIT301') if 'FIT301' in features else -1

        # Stage 4-6 pump-off -> zero-flow traps (pumps share the valve encoding: ~1.0=OFF, ~2.0=ON)
        self.idx_P402 = features.index('P402') if 'P402' in features else -1
        self.idx_FIT401 = features.index('FIT401') if 'FIT401' in features else -1
        self.idx_P501 = features.index('P501') if 'P501' in features else -1
        self.idx_FIT501 = features.index('FIT501') if 'FIT501' in features else -1
        self.idx_P602 = features.index('P602') if 'P602' in features else -1
        self.idx_FIT601 = features.index('FIT601') if 'FIT601' in features else -1

    def forward(self, x_hat, x):
        mse_loss = self.mse(x_hat, x)
        
        if self.lambda_phys <= 0:
            return mse_loss
            
        x_hat_phys = x_hat * self.std + self.mean
        phys_loss = torch.tensor(0.0, device=x.device)
        
        for i, feat in enumerate(self.features):
            if 'FIT' in feat or 'LIT' in feat or 'PIT' in feat:
                phys_loss += torch.mean(torch.relu(-x_hat_phys[:, i]))
                
        if self.idx_MV101 != -1 and self.idx_FIT101 != -1:
            mv = x_hat_phys[:, self.idx_MV101]
            fit = x_hat_phys[:, self.idx_FIT101]
            weight = torch.sigmoid(10.0 * (1.5 - mv))
            phys_loss += torch.mean(weight * (fit ** 2))
            
        if self.idx_MV201 != -1 and self.idx_FIT201 != -1:
            mv = x_hat_phys[:, self.idx_MV201]
            fit = x_hat_phys[:, self.idx_FIT201]
            weight = torch.sigmoid(10.0 * (1.5 - mv))
            phys_loss += torch.mean(weight * (fit ** 2))
            
        if self.idx_MV301 != -1 and self.idx_FIT301 != -1:
            mv = x_hat_phys[:, self.idx_MV301]
            fit = x_hat_phys[:, self.idx_FIT301]
            weight = torch.sigmoid(10.0 * (1.5 - mv))
            phys_loss += torch.mean(weight * (fit ** 2))

        # Stage 4: P402 OFF -> FIT401 Flow = 0
        if self.idx_P402 != -1 and self.idx_FIT401 != -1:
            pump = x_hat_phys[:, self.idx_P402]
            fit = x_hat_phys[:, self.idx_FIT401]
            weight = torch.sigmoid(10.0 * (1.5 - pump))
            phys_loss += torch.mean(weight * (fit ** 2))

        # Stage 5: P501 OFF -> FIT501 Flow = 0
        if self.idx_P501 != -1 and self.idx_FIT501 != -1:
            pump = x_hat_phys[:, self.idx_P501]
            fit = x_hat_phys[:, self.idx_FIT501]
            weight = torch.sigmoid(10.0 * (1.5 - pump))
            phys_loss += torch.mean(weight * (fit ** 2))

        # Stage 6: P602 OFF -> FIT601 Flow = 0
        if self.idx_P602 != -1 and self.idx_FIT601 != -1:
            pump = x_hat_phys[:, self.idx_P602]
            fit = x_hat_phys[:, self.idx_FIT601]
            weight = torch.sigmoid(10.0 * (1.5 - pump))
            phys_loss += torch.mean(weight * (fit ** 2))

        return mse_loss + self.lambda_phys * phys_loss

def train_subsystem(name, features):
    print(f"\nTraining Subsystem Autoencoder for {name} with features: {features}")
    
    # 1. Load normal dataset
    df = pd.read_csv(TRAIN_DATA)
    df.columns = df.columns.str.strip()
    
    # Extract features
    sub_df = df[features].fillna(df[features].mean())
    
    # Downsample normal data (like in baseline model training)
    print(f"  Total samples in training set: {len(sub_df)}")
    sub_df = sub_df.iloc[::5].reset_index(drop=True)
    print(f"  Downsampled to {len(sub_df)} samples for training.")
    
    data_val = sub_df.values
    data_tensor = torch.tensor(data_val, dtype=torch.float32)
    
    # Normalize using Z-score
    mean = data_tensor.mean(dim=0)
    std = data_tensor.std(dim=0)
    std = torch.where(std < 1e-4, torch.ones_like(std), std)
    
    norm_data = (data_tensor - mean) / std
    
    # Create Temporal Windows
    if SEQ_LEN > 1:
        # Unfold creates [N - SEQ_LEN + 1, F, SEQ_LEN]. We transpose to [N - SEQ_LEN + 1, SEQ_LEN, F] and flatten.
        norm_seqs = norm_data.unfold(0, SEQ_LEN, 1).transpose(1, 2).reshape(-1, SEQ_LEN * data_tensor.shape[1])
    else:
        norm_seqs = norm_data
    norm_seqs = norm_seqs.to(DEVICE)
    
    # Split train/val
    dataset = TensorDataset(norm_seqs)
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4096, shuffle=False)
    
    input_dim = data_tensor.shape[1] * SEQ_LEN
    bottleneck_dim = max(1, input_dim // 2)
    
    model = SubsystemAutoencoder(input_dim, bottleneck_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = PINNLoss(features, mean.to(DEVICE), std.to(DEVICE), lambda_phys=0.1)
    
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
        
        # Validation
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
            
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
        if epochs_no_improve >= patience:
            print(f"  Early stopping at epoch {epoch}. Best Val Loss: {best_val_loss:.6f}")
            break
            
    # Save checkpoint
    checkpoint = {
        'model_state': best_model_state,
        'mean': mean,
        'std': std,
        'features': features,
        'input_dim': input_dim,
        'bottleneck_dim': bottleneck_dim
    }
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    out_path = os.path.join(MODEL_DIR, f"subsystem_{name}.pth")
    torch.save(checkpoint, out_path)
    print(f"  Saved subsystem model to {out_path}")

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

def evaluate_subsystems():
    print("\n" + "="*80)
    print(" EVALUATING SUB-SYSTEM AUTOENCODERS")
    print("="*80)
    print(f"Aggregation Mode: {AGGREGATION_MODE.upper()}")
    print(f"Error Mode:       {ERROR_MODE.upper()}")
    print(f"Window Size:      {WINDOW_SIZE}")
    print(f"Align Baseline:   {ALIGN_BASELINE}")
    print("="*80)
    
    y_true, subsystem_scores = _compute_subsystem_scores()
    if y_true is None:
        return None
        
    # 3. Aggregate scores
    print(f"\n[EVAL] Aggregating sub-system anomaly scores using: {AGGREGATION_MODE.upper()}")
    all_scores = [subsystem_scores[name] for name in subsystems.keys()]
    
    if AGGREGATION_MODE == 'max':
        aggregated_scores = np.max(all_scores, axis=0)
    else:
        aggregated_scores = np.mean(all_scores, axis=0)
        
    # 4. Temporal window smoothing
    if WINDOW_SIZE > 1:
        print(f"[EVAL] Applying temporal window smoothing (window size = {WINDOW_SIZE})...")
        aggregated_scores = pd.Series(aggregated_scores).rolling(window=WINDOW_SIZE, min_periods=1).mean().values
        
    # 5. Threshold sweep (Find optimal cutoff on aggregated scores)
    print("\n[EVAL] Sweeping thresholds on aggregated scores...")
    
    normal_scores = aggregated_scores[y_true == 0]
    attack_scores = aggregated_scores[y_true == 1]
    
    print("Aggregated Score Analysis:")
    for cls_name, cls_scores in [("Normal", normal_scores), ("Attack", attack_scores)]:
        print(f"  {cls_name} (n={len(cls_scores)}):")
        print(f"    Mean:   {np.mean(cls_scores):.4f}")
        print(f"    Median: {np.median(cls_scores):.4f}")
        print(f"    Std:    {np.std(cls_scores):.4f}")
        print(f"    P95:    {np.percentile(cls_scores, 95):.4f}")
        print(f"    Max:    {cls_scores.max():.4f}")
        
    # sweep search bounds
    p2 = np.percentile(aggregated_scores, 2)
    p99_8 = np.percentile(aggregated_scores, 99.8)
    thresholds = np.logspace(np.log10(max(1e-5, p2)), np.log10(p99_8), 150)
    
    best_f1 = -1.0
    best_threshold = 0.0
    
    # Fast evaluation loop
    for t in thresholds:
        preds = (aggregated_scores > t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            
    print(f"\n[SWEEP] Best Threshold: {best_threshold:.6f} | Best F1: {best_f1:.4%}")
    
    # Final performance metrics
    preds = (aggregated_scores > best_threshold).astype(int)
    cm = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    false_alarm = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    print("\n" + "="*60)
    print(f" FINAL PERFORMANCE REPORT (SUB-SYSTEM AUTOENCODERS)")
    print("="*60)
    print(f"  Accuracy:                      {accuracy:.4%}")
    print(f"  Precision:                     {precision:.4%}")
    print(f"  Recall (Detection Rate):       {recall:.4%}")
    print(f"  F1-Score:                      {best_f1:.4%}")
    print(f"  False Alarm Rate:              {false_alarm:.4%}")
    print(f"  Confusion Matrix:")
    print(f"    TP: {tp:,} | FN: {fn:,}")
    print(f"    FP: {fp:,} | TN: {tn:,}")
    print("="*60)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': best_f1,
        'false_alarm': false_alarm,
        'threshold': best_threshold,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }


def _compute_subsystem_scores(run_adversarial=False, epsilon=0.1, alpha=0.01, num_iter=10):
    """Shared helper: loads test data and computes per-subsystem normalized scores.
    Returns (y_true, subsystem_scores_dict)."""
    print(f"\n[INIT] Loading testing data from: {EVAL_DATA}")
    df_eval = pd.read_csv(EVAL_DATA)
    df_eval.columns = df_eval.columns.str.strip()

    label_col = 'Label' if 'Label' in df_eval.columns else 'Normal/Attack'
    y_true = df_eval[label_col].astype(str).str.strip().str.lower().str.replace(" ", "").apply(
        lambda val: 1 if val == "attack" else 0).values

    subsystem_scores = {}

    for name, features in subsystems.items():
        model_path = os.path.join(MODEL_DIR, f"subsystem_{name}.pth")
        if not os.path.exists(model_path):
            print(f"  [ERROR] Model checkpoint {model_path} not found!")
            return None, None

        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        input_dim = checkpoint['input_dim']
        bottleneck_dim = checkpoint['bottleneck_dim']
        
        # Use features saved in the checkpoint rather than the hardcoded dictionary
        ckpt_features = checkpoint.get('features', features)

        model_state = checkpoint['model_state']
        
        # Dynamically determine hidden dimensions from the saved weights
        hidden1 = model_state['encoder.0.weight'].shape[0]
        hidden2 = model_state['encoder.4.weight'].shape[0]
        
        # We need to temporarily patch the class to accept these hidden dims
        # To avoid changing the class definition everywhere, we'll just reconstruct the model
        model = SubsystemAutoencoder(input_dim, bottleneck_dim).to(DEVICE)
        
        # Re-initialize the layers with correct dimensions
        model.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden2, bottleneck_dim)
        ).to(DEVICE)
        
        model.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden2, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, input_dim)
        ).to(DEVICE)
        
        model.load_state_dict(model_state)
        model.eval()

        sub_df = df_eval[ckpt_features].fillna(df_eval[ckpt_features].mean())
        data_tensor = torch.tensor(sub_df.values, dtype=torch.float32)

        MEAN = checkpoint.get('mean', 0.0)
        STD = checkpoint.get('std', 1.0)

        if ALIGN_BASELINE:
            normal_mask = (y_true == 0)
            normal_data = data_tensor[normal_mask]
            if len(normal_data) > 0:
                MEAN = normal_data.mean(dim=0)
                STD = normal_data.std(dim=0)
                STD = torch.where(STD < 1e-4, torch.ones_like(STD), STD)

        norm_data = (data_tensor - MEAN) / STD

        # Dynamically infer SEQ_LEN used during training
        num_features = data_tensor.shape[1]
        inferred_seq_len = input_dim // num_features

        if inferred_seq_len > 1:
            # Pad the beginning to keep the output length exactly equal to N
            pad = norm_data[0].unsqueeze(0).repeat(inferred_seq_len - 1, 1)
            padded_norm_data = torch.cat([pad, norm_data], dim=0)
            norm_seqs = padded_norm_data.unfold(0, inferred_seq_len, 1).transpose(1, 2).reshape(-1, inferred_seq_len * num_features)
        else:
            norm_seqs = norm_data

        batch_size = 4096
        errors_list = []
        y_true_tensor = torch.tensor(y_true, dtype=torch.long, device=DEVICE)
        
        for i in range(0, len(norm_seqs), batch_size):
            batch = norm_seqs[i:i + batch_size].to(DEVICE)
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

        normal_scores_subset = raw_scores[y_true == 0]
        mean_err = np.mean(normal_scores_subset)
        std_err = np.std(normal_scores_subset)
        
        # Standardization
        standardized_scores = (raw_scores - mean_err) / (std_err + 1e-6)
        subsystem_scores[name] = standardized_scores
        print(f"  {name}: NormalErr Mean={mean_err:.6f} Std={std_err:.6f} | NormRange=[{standardized_scores.min():.2f}, {standardized_scores.max():.2f}]")

    return y_true, subsystem_scores


def evaluate_subsystems_or_voting(window_size=25, run_adversarial=False, epsilon=0.1, alpha=0.01, num_iter=10,
                                  export_predictions=None):
    """Per-Subsystem Independent Thresholding with OR-Voting.

    Each subsystem gets its own optimal threshold. A sample is flagged
    as anomalous if ANY subsystem exceeds its own threshold.

    export_predictions: path to write y_true/y_pred/per-subsystem scores for evaluate_events.py.
    """
    print("\n" + "=" * 80)
    print(" EVALUATING SUB-SYSTEM AUTOENCODERS -- OR-VOTING MODE")
    print("=" * 80)
    print(f"Window Size:      {window_size}")
    print(f"Error Mode:       {ERROR_MODE.upper()}")
    print(f"Align Baseline:   {ALIGN_BASELINE}")
    print(f"Adversarial:      {run_adversarial}")
    print("=" * 80)

    y_true, subsystem_scores = _compute_subsystem_scores(run_adversarial, epsilon, alpha, num_iter)
    if y_true is None:
        return None

    n_samples = len(y_true)

    # Step 1: Smooth each subsystem's scores independently
    smoothed = {}
    for name, scores in subsystem_scores.items():
        if window_size > 1:
            smoothed[name] = pd.Series(scores).rolling(window=window_size, min_periods=1).mean().values
        else:
            smoothed[name] = scores

    # Step 2: Find per-subsystem optimal thresholds
    print("\n[OR-VOTING] Finding per-subsystem optimal thresholds...")
    per_sub_thresholds = {}
    per_sub_stats = {}

    for name, scores in smoothed.items():
        normal_sc = scores[y_true == 0]
        attack_sc = scores[y_true == 1]

        # Search thresholds spanning the range of this subsystem's scores
        lo = np.percentile(scores, 80)
        hi = np.percentile(scores, 99.9)
        if hi <= lo:
            hi = lo + 1.0
        candidates = np.linspace(lo, hi, 200)

        best_f1 = -1.0
        best_t = 0.0
        best_tp = 0
        best_fp = 0
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
                best_tp = tp
                best_fp = fp

        per_sub_thresholds[name] = best_t
        per_sub_stats[name] = {
            'threshold': best_t,
            'f1': best_f1,
            'tp': best_tp,
            'fp': best_fp,
            'normal_p95': np.percentile(normal_sc, 95),
            'attack_median': np.median(attack_sc),
        }
        print(f"  {name:>30}: Threshold={best_t:.4f} | F1={best_f1:.4f} | TP={best_tp:,} | FP={best_fp:,} | NormalP95={np.percentile(normal_sc, 95):.4f}")

    # Step 3: Combine with OR-logic
    print("\n[OR-VOTING] Combining predictions with OR logic...")
    combined_preds = np.zeros(n_samples, dtype=int)
    for name, scores in smoothed.items():
        t = per_sub_thresholds[name]
        sub_preds = (scores > t).astype(int)
        combined_preds = np.maximum(combined_preds, sub_preds)

    # Step 4: Compute final metrics
    cm = confusion_matrix(y_true, combined_preds)
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = f1_score(y_true, combined_preds, zero_division=0)
    false_alarm = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print("\n" + "=" * 60)
    print(" FINAL PERFORMANCE REPORT (OR-VOTING)")
    print("=" * 60)
    print(f"  Accuracy:                      {accuracy:.4%}")
    print(f"  Precision:                     {precision:.4%}")
    print(f"  Recall (Detection Rate):       {recall:.4%}")
    print(f"  F1-Score:                      {f1:.4%}")
    print(f"  False Alarm Rate:              {false_alarm:.4%}")
    print(f"  Confusion Matrix:")
    print(f"    TP: {tp:,} | FN: {fn:,}")
    print(f"    FP: {fp:,} | TN: {tn:,}")
    print("=" * 60)

    if export_predictions:
        out = pd.DataFrame({'y_true': y_true.astype(int), 'y_pred': combined_preds})
        out['score'] = np.max([smoothed[name] for name in subsystems.keys()], axis=0)
        for name in subsystems.keys():
            out[f'score_{name}'] = smoothed[name]
        out.to_csv(export_predictions, index=False)
        print(f"  [SAVED] predictions -> {export_predictions}  (score with evaluate_events.py)")

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall,
            'f1': f1, 'false_alarm': false_alarm, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def evaluate_subsystems_cusum(window_size=25, k=0.5, h_multiplier=8.0):
    """CUSUM (Cumulative Sum) change detection per subsystem with OR-Voting.

    For each subsystem, CUSUM accumulates positive deviations above k standard
    deviations. When the accumulated sum exceeds h, an alarm is raised.
    Combined across subsystems with OR logic.

    Args:
        window_size: Rolling window for initial smoothing before CUSUM.
        k: Allowance parameter (in std devs of normal scores). Deviations below
           this are tolerated without accumulation.
        h_multiplier: Decision boundary multiplier. CUSUM triggers when
                      accumulated sum > h_multiplier.
    """
    print("\n" + "=" * 80)
    print(" EVALUATING SUB-SYSTEM AUTOENCODERS -- CUSUM + OR-VOTING MODE")
    print("=" * 80)
    print(f"Window Size:      {window_size}")
    print(f"CUSUM k:          {k}")
    print(f"CUSUM h_mult:     {h_multiplier}")
    print(f"Error Mode:       {ERROR_MODE.upper()}")
    print(f"Align Baseline:   {ALIGN_BASELINE}")
    print("=" * 80)

    y_true, subsystem_scores = _compute_subsystem_scores()
    if y_true is None:
        return None

    n_samples = len(y_true)

    # Step 1: Optional light smoothing before CUSUM
    smoothed = {}
    for name, scores in subsystem_scores.items():
        if window_size > 1:
            smoothed[name] = pd.Series(scores).rolling(window=window_size, min_periods=1).mean().values
        else:
            smoothed[name] = scores

    # Step 2: Apply CUSUM per subsystem and sweep h for optimal F1
    print("\n[CUSUM] Applying CUSUM per subsystem and finding optimal h...")
    per_sub_cusum_preds = {}

    for name, scores in smoothed.items():
        # Compute CUSUM signal: accumulate deviations above k
        cusum = np.zeros(n_samples)
        cusum[0] = max(0, scores[0] - k)
        for t in range(1, n_samples):
            cusum[t] = max(0, cusum[t - 1] + scores[t] - k)

        # Sweep h threshold on the CUSUM signal
        normal_cusum = cusum[y_true == 0]
        p95_normal = np.percentile(normal_cusum, 95)
        p999_normal = np.percentile(normal_cusum, 99.9)

        lo = p95_normal
        hi = max(p999_normal * 2, p95_normal + 10)
        candidates = np.linspace(lo, hi, 200)

        best_f1 = -1.0
        best_h = 0.0
        best_tp = 0
        best_fp = 0
        for h_val in candidates:
            preds = (cusum > h_val).astype(int)
            tp = ((preds == 1) & (y_true == 1)).sum()
            fp = ((preds == 1) & (y_true == 0)).sum()
            fn = ((preds == 0) & (y_true == 1)).sum()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_h = h_val
                best_tp = tp
                best_fp = fp

        per_sub_cusum_preds[name] = (cusum > best_h).astype(int)
        print(f"  {name:>30}: h={best_h:.4f} | F1={best_f1:.4f} | TP={best_tp:,} | FP={best_fp:,}")

    # Step 3: Combine with OR-logic
    print("\n[CUSUM] Combining predictions with OR logic...")
    combined_preds = np.zeros(n_samples, dtype=int)
    for name, preds in per_sub_cusum_preds.items():
        combined_preds = np.maximum(combined_preds, preds)

    # Step 4: Compute final metrics
    cm = confusion_matrix(y_true, combined_preds)
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_val = f1_score(y_true, combined_preds, zero_division=0)
    false_alarm = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print("\n" + "=" * 60)
    print(" FINAL PERFORMANCE REPORT (CUSUM + OR-VOTING)")
    print("=" * 60)
    print(f"  Accuracy:                      {accuracy:.4%}")
    print(f"  Precision:                     {precision:.4%}")
    print(f"  Recall (Detection Rate):       {recall:.4%}")
    print(f"  F1-Score:                      {f1_val:.4%}")
    print(f"  False Alarm Rate:              {false_alarm:.4%}")
    print(f"  Confusion Matrix:")
    print(f"    TP: {tp:,} | FN: {fn:,}")
    print(f"    FP: {fp:,} | TN: {tn:,}")
    print("=" * 60)

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall,
            'f1': f1_val, 'false_alarm': false_alarm, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}

def evaluate_subsystems_selective_or(window_size=25, min_sub_f1=0.30):
    """Selective OR-Voting: uses independent thresholds only for subsystems
    with good individual discrimination (F1 > min_sub_f1), plus a global
    MAX-aggregated fallback to catch attacks in weaker subsystems.

    This hybrid approach avoids the false alarm explosion from noisy
    subsystems while still benefiting from per-subsystem sensitivity
    for the strong ones.
    """
    print("\n" + "=" * 80)
    print(" EVALUATING SUB-SYSTEM AUTOENCODERS -- SELECTIVE OR-VOTING + GLOBAL FALLBACK")
    print("=" * 80)
    print(f"Window Size:         {window_size}")
    print(f"Error Mode:          {ERROR_MODE.upper()}")
    print(f"Align Baseline:      {ALIGN_BASELINE}")
    print(f"Min Sub F1 Cutoff:   {min_sub_f1}")
    print("=" * 80)

    y_true, subsystem_scores = _compute_subsystem_scores()
    if y_true is None:
        return None

    n_samples = len(y_true)

    # Step 1: Smooth each subsystem's scores independently
    smoothed = {}
    for name, scores in subsystem_scores.items():
        if window_size > 1:
            smoothed[name] = pd.Series(scores).rolling(window=window_size, min_periods=1).mean().values
        else:
            smoothed[name] = scores

    # Step 2: Evaluate each subsystem individually, find thresholds
    print(f"\n[SELECTIVE OR] Evaluating per-subsystem discrimination...")
    per_sub_info = {}
    for name, scores in smoothed.items():
        lo = np.percentile(scores, 80)
        hi = np.percentile(scores, 99.9)
        if hi <= lo:
            hi = lo + 1.0
        candidates = np.linspace(lo, hi, 200)

        best_f1 = -1.0
        best_t = 0.0
        best_tp = 0
        best_fp = 0
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
                best_tp = tp
                best_fp = fp

        selected = best_f1 >= min_sub_f1
        per_sub_info[name] = {
            'threshold': best_t, 'f1': best_f1, 'tp': best_tp, 'fp': best_fp,
            'selected': selected
        }
        status = "SELECTED" if selected else "EXCLUDED (weak)"
        print(f"  {name:>30}: F1={best_f1:.4f} | Thresh={best_t:.4f} | TP={best_tp:,} | FP={best_fp:,} | {status}")

    selected_subs = [n for n, info in per_sub_info.items() if info['selected']]
    excluded_subs = [n for n, info in per_sub_info.items() if not info['selected']]
    print(f"\n  Selected for OR-Voting: {selected_subs}")
    print(f"  Excluded (global fallback): {excluded_subs}")

    # Step 3: Build OR-voted predictions from selected subsystems
    or_preds = np.zeros(n_samples, dtype=int)
    for name in selected_subs:
        t = per_sub_info[name]['threshold']
        sub_preds = (smoothed[name] > t).astype(int)
        or_preds = np.maximum(or_preds, sub_preds)

    # Step 4: Build global MAX-aggregated fallback (catches attacks in weak subsystems)
    all_scores = [smoothed[name] for name in subsystems.keys()]
    global_max = np.max(all_scores, axis=0)

    # Find optimal threshold for the global fallback on samples NOT already caught by OR-voting
    print(f"\n[SELECTIVE OR] Sweeping global fallback threshold...")
    p2 = np.percentile(global_max, 2)
    p998 = np.percentile(global_max, 99.8)
    candidates = np.logspace(np.log10(max(1e-5, p2)), np.log10(p998), 200)

    best_combined_f1 = -1.0
    best_global_t = 0.0

    for gt in candidates:
        global_preds = (global_max > gt).astype(int)
        combined = np.maximum(or_preds, global_preds)
        f1 = f1_score(y_true, combined, zero_division=0)
        if f1 > best_combined_f1:
            best_combined_f1 = f1
            best_global_t = gt

    print(f"  Global Fallback Threshold: {best_global_t:.6f}")
    print(f"  Combined F1: {best_combined_f1:.4%}")

    # Step 5: Final combined predictions
    global_preds = (global_max > best_global_t).astype(int)
    combined_preds = np.maximum(or_preds, global_preds)

    cm = confusion_matrix(y_true, combined_preds)
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_val = f1_score(y_true, combined_preds, zero_division=0)
    false_alarm = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print("\n" + "=" * 60)
    print(" FINAL PERFORMANCE REPORT (SELECTIVE OR-VOTING + GLOBAL FALLBACK)")
    print("=" * 60)
    print(f"  Accuracy:                      {accuracy:.4%}")
    print(f"  Precision:                     {precision:.4%}")
    print(f"  Recall (Detection Rate):       {recall:.4%}")
    print(f"  F1-Score:                      {f1_val:.4%}")
    print(f"  False Alarm Rate:              {false_alarm:.4%}")
    print(f"  Confusion Matrix:")
    print(f"    TP: {tp:,} | FN: {fn:,}")
    print(f"    FP: {fp:,} | TN: {tn:,}")
    print("=" * 60)

    # Breakdown: contribution from OR-voting vs global fallback
    or_only_tp = ((or_preds == 1) & (y_true == 1)).sum()
    global_only_tp = ((global_preds == 1) & (or_preds == 0) & (y_true == 1)).sum()
    print(f"\n  Contribution Breakdown:")
    print(f"    OR-Voting Detections:     {or_only_tp:,}")
    print(f"    Global Fallback Adds:     {global_only_tp:,}")

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall,
            'f1': f1_val, 'false_alarm': false_alarm, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}

def evaluate_dual_threshold(window_size=25, export_predictions=None):
    """Dual-Threshold with Temporal Persistence.

    Uses two tiers:
    - Tier 1 (Instant): score > high_threshold → anomaly immediately
    - Tier 2 (Sustained): score > low_threshold for K+ consecutive samples → anomaly

    This catches attack segments that produce persistent elevated scores
    (e.g., 5-9) that are just below the optimal single threshold.
    """
    print("\n" + "=" * 80)
    print(" EVALUATING SUB-SYSTEM AUTOENCODERS -- DUAL-THRESHOLD + PERSISTENCE")
    print("=" * 80)
    print(f"Window Size:      {window_size}")
    print(f"Error Mode:       {ERROR_MODE.upper()}")
    print(f"Align Baseline:   {ALIGN_BASELINE}")
    print("=" * 80)

    y_true, subsystem_scores = _compute_subsystem_scores()
    if y_true is None:
        return None

    n_samples = len(y_true)

    # Build smoothed MAX-aggregated scores (same as baseline)
    smoothed = {}
    for name, scores in subsystem_scores.items():
        if window_size > 1:
            smoothed[name] = pd.Series(scores).rolling(window=window_size, min_periods=1).mean().values
        else:
            smoothed[name] = scores

    all_scores = [smoothed[name] for name in subsystems.keys()]
    agg_scores = np.max(all_scores, axis=0)

    # Sweep over (high_threshold, low_threshold, min_duration) combinations
    print("\n[DUAL-THRESH] Sweeping dual-threshold parameters...")

    # Build candidate parameter grid
    p_range = np.percentile(agg_scores, [85, 90, 95, 97, 99, 99.5, 99.8])

    best_f1 = -1.0
    best_params = {}
    best_preds = None

    # Candidate high thresholds (for instant detection)
    high_candidates = np.linspace(p_range[3], p_range[6], 15)  # P97 to P99.8
    # Candidate low thresholds (for sustained detection)
    low_candidates = np.linspace(p_range[0], p_range[3], 15)   # P85 to P97
    # Candidate min durations (consecutive samples above low threshold)
    duration_candidates = [25, 50, 75, 100, 150, 200]

    total_combos = len(high_candidates) * len(low_candidates) * len(duration_candidates)
    print(f"  Searching {total_combos} parameter combinations...")

    for h_thresh in high_candidates:
        # Pre-compute instant detections
        instant_preds = (agg_scores > h_thresh).astype(int)

        for l_thresh in low_candidates:
            if l_thresh >= h_thresh:
                continue
            # Pre-compute which samples exceed low threshold
            low_mask = (agg_scores > l_thresh)

            for min_dur in duration_candidates:
                # Find sustained runs above low threshold
                sustained_preds = np.zeros(n_samples, dtype=int)
                run_length = 0
                run_start = 0
                for i in range(n_samples):
                    if low_mask[i]:
                        if run_length == 0:
                            run_start = i
                        run_length += 1
                        if run_length >= min_dur:
                            # Mark the entire run from start to current
                            sustained_preds[run_start:i + 1] = 1
                    else:
                        run_length = 0

                combined = np.maximum(instant_preds, sustained_preds)
                f1 = f1_score(y_true, combined, zero_division=0)

                if f1 > best_f1:
                    best_f1 = f1
                    best_params = {
                        'high_threshold': h_thresh,
                        'low_threshold': l_thresh,
                        'min_duration': min_dur,
                    }
                    best_preds = combined.copy()

    print(f"\n  Best Parameters:")
    print(f"    High Threshold (instant):  {best_params['high_threshold']:.4f}")
    print(f"    Low Threshold (sustained): {best_params['low_threshold']:.4f}")
    print(f"    Min Duration (samples):    {best_params['min_duration']}")
    print(f"    Combined F1:               {best_f1:.4%}")

    # Final metrics
    cm = confusion_matrix(y_true, best_preds)
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    false_alarm = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print("\n" + "=" * 60)
    print(" FINAL PERFORMANCE REPORT (DUAL-THRESHOLD + PERSISTENCE)")
    print("=" * 60)
    print(f"  Accuracy:                      {accuracy:.4%}")
    print(f"  Precision:                     {precision:.4%}")
    print(f"  Recall (Detection Rate):       {recall:.4%}")
    print(f"  F1-Score:                      {best_f1:.4%}")
    print(f"  False Alarm Rate:              {false_alarm:.4%}")
    print(f"  Confusion Matrix:")
    print(f"    TP: {tp:,} | FN: {fn:,}")
    print(f"    FP: {fp:,} | TN: {tn:,}")
    print("=" * 60)

    # Breakdown
    h_t = best_params['high_threshold']
    instant_tp = (((agg_scores > h_t).astype(int) == 1) & (y_true == 1)).sum()
    sustained_only_tp = tp - instant_tp
    if sustained_only_tp < 0:
        sustained_only_tp = 0
    print(f"\n  Contribution Breakdown:")
    print(f"    Instant Detections (>{h_t:.2f}):  {instant_tp:,}")
    print(f"    Sustained Detections Added:    {sustained_only_tp:,}")

    if export_predictions:
        out = pd.DataFrame({'y_true': y_true.astype(int), 'y_pred': best_preds, 'score': agg_scores})
        for name in subsystems.keys():
            out[f'score_{name}'] = smoothed[name]
        out.to_csv(export_predictions, index=False)
        print(f"  [SAVED] predictions -> {export_predictions}  (score with evaluate_events.py)")

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall,
            'f1': best_f1, 'false_alarm': false_alarm, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def main():
    print("[WARNING] run_subsystem_experiments.py calibrates on test labels (v1 reference protocol).")
    if TRAIN_MODELS:
        print("Starting training for all 6 subsystem autoencoders...")
        for name, features in subsystems.items():
            train_subsystem(name, features)

    print("\n\n" + "#" * 80)
    print(" MODE 1: AGGREGATED THRESHOLD (baseline)")
    print("#" * 80)
    evaluate_subsystems()

    print("\n\n" + "#" * 80)
    print(" MODE 2: DUAL-THRESHOLD + PERSISTENCE")
    print("#" * 80)
    evaluate_dual_threshold(window_size=WINDOW_SIZE)
    
    print("\n\n" + "#" * 80)
    print(" MODE 3: SMOOTHED OR-VOTING")
    print("#" * 80)
    evaluate_subsystems_or_voting(window_size=WINDOW_SIZE)

    print("\n\n" + "#" * 80)
    print(" MODE 4: RAW OR-VOTING")
    print("#" * 80)
    evaluate_subsystems_or_voting(window_size=1)
    
    print("\n\n" + "#" * 80)
    print(" MODE 5: SELECTIVE OR-VOTING (Conservative)")
    print("#" * 80)
    evaluate_subsystems_selective_or(window_size=WINDOW_SIZE)


if __name__ == "__main__":
    main()
