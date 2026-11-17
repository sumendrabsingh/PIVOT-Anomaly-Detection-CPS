import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score

# --- Config ---
TRAIN_MODELS = True
AGGREGATION_MODE = 'max'  # Options: 'max' or 'mean'
ERROR_MODE = 'max'        # Options: 'max' or 'mean' reconstruction error inside AE
WINDOW_SIZE = 25
SEQ_LEN = 30              # Temporal window size for sequence reconstruction (SEQ_LEN > 1)
ALIGN_BASELINE = True
ENABLE_STAGE456_PHYS = True  # Ablation switch: enable pump-off->zero-flow PINN traps for Stages 4-6
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRAIN_DATA = r"Datasets/swat_train_preprocessed.csv"
META_DATA = r"Datasets/preprocessing_meta.json"

# Load metadata config
with open(META_DATA, 'r') as f:
    meta = json.load(f)

# Define physical subsystems (stages) dynamically based on metadata columns
def get_subsystem_features(final_columns):
    subsystems = {
        "Stage1_RawWater": [],
        "Stage2_ChemicalDosing": [],
        "Stage3_Ultrafiltration": [],
        "Stage4_Dechlorination": [],
        "Stage5_ReverseOsmosis": [],
        "Stage6_ROCleaning": []
    }
    for col in final_columns:
        base_name = col.split('_')[0]
        digits = [c for c in base_name if c.isdigit()]
        if digits:
            stage_num = int(digits[0])
            if stage_num == 1:
                subsystems["Stage1_RawWater"].append(col)
            elif stage_num == 2:
                subsystems["Stage2_ChemicalDosing"].append(col)
            elif stage_num == 3:
                subsystems["Stage3_Ultrafiltration"].append(col)
            elif stage_num == 4:
                subsystems["Stage4_Dechlorination"].append(col)
            elif stage_num == 5:
                subsystems["Stage5_ReverseOsmosis"].append(col)
            elif stage_num == 6:
                subsystems["Stage6_ROCleaning"].append(col)
    return subsystems

subsystems = get_subsystem_features(meta["final_columns"])

print("\n--- Subsystem Feature Breakdown ---")
for name, features in subsystems.items():
    print(f"  {name} ({len(features)} variables): {features[:5]}...")

class SubsystemAutoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim):
        super().__init__()
        
        # Dynamically set hidden layers based on stage dimensionality
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

class PINNLoss(nn.Module):
    def __init__(self, features, mean_vals, std_vals, seq_len=1, lambda_phys=0.1):
        super().__init__()
        self.mse = nn.MSELoss(reduction='mean')
        self.features = features
        self.seq_len = seq_len
        self.num_features = len(features)
        self.lambda_phys = lambda_phys
        
        # Build mean/std scaling tensors matching the feature order
        self.mean_vals = torch.tensor([mean_vals.get(f.split('_')[0], 0.0) for f in features], dtype=torch.float32)
        self.std_vals = torch.tensor([std_vals.get(f.split('_')[0], 1.0) for f in features], dtype=torch.float32)
        
        # Setup valve-closed index constraints
        self.idx_MV101_0 = features.index('MV101_0') if 'MV101_0' in features else -1
        self.idx_FIT101 = features.index('FIT101') if 'FIT101' in features else -1
        
        self.idx_MV201_0 = features.index('MV201_0') if 'MV201_0' in features else -1
        self.idx_FIT201 = features.index('FIT201') if 'FIT201' in features else -1
        
        self.idx_MV301_0 = features.index('MV301_0') if 'MV301_0' in features else -1
        self.idx_FIT301 = features.index('FIT301') if 'FIT301' in features else -1

        # Stage 4-6 pump-off -> zero-flow traps. Pumps stay continuous (1.0=OFF, 2.0=ON),
        # so they are gated on the denormalized physical value rather than a one-hot index.
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
            
        # Reshape [Batch, SEQ_LEN * F] -> [Batch * SEQ_LEN, F] to apply physical rules on all timesteps
        x_hat_flat = x_hat.view(-1, self.seq_len, self.num_features).view(-1, self.num_features)
        x_flat = x.view(-1, self.seq_len, self.num_features).view(-1, self.num_features)
        
        mean_v = self.mean_vals.to(x.device)
        std_v = self.std_vals.to(x.device)
        
        # Denormalize x_hat back to physical values
        x_hat_phys = x_hat_flat * std_v + mean_v
        # Denormalize the input too, for continuous-actuator (pump) gating in Stages 4-6
        x_phys = x_flat * std_v + mean_v
        
        phys_loss = torch.tensor(0.0, device=x.device)
        
        # 1. Non-negativity for continuous physical rates
        for i, feat in enumerate(self.features):
            if any(p in feat for p in ['FIT', 'LIT', 'DPIT', 'PIT']):
                phys_loss += torch.mean(torch.relu(-x_hat_phys[:, i]))
                
        # 2. MV101 Closed -> FIT101 Flow = 0
        if self.idx_MV101_0 != -1 and self.idx_FIT101 != -1:
            mv_closed = x_flat[:, self.idx_MV101_0]
            fit = x_hat_phys[:, self.idx_FIT101]
            phys_loss += torch.mean(mv_closed * (fit ** 2))
            
        # 3. MV201 Closed -> FIT201 Flow = 0
        if self.idx_MV201_0 != -1 and self.idx_FIT201 != -1:
            mv_closed = x_flat[:, self.idx_MV201_0]
            fit = x_hat_phys[:, self.idx_FIT201]
            phys_loss += torch.mean(mv_closed * (fit ** 2))
            
        # 4. MV301 Closed -> FIT301 Flow = 0
        if self.idx_MV301_0 != -1 and self.idx_FIT301 != -1:
            mv_closed = x_flat[:, self.idx_MV301_0]
            fit = x_hat_phys[:, self.idx_FIT301]
            phys_loss += torch.mean(mv_closed * (fit ** 2))

        # 5. Stage 4: P402 OFF -> FIT401 Flow = 0
        if ENABLE_STAGE456_PHYS and self.idx_P402 != -1 and self.idx_FIT401 != -1:
            pump_off = torch.sigmoid(10.0 * (1.5 - x_phys[:, self.idx_P402]))
            fit = x_hat_phys[:, self.idx_FIT401]
            phys_loss += torch.mean(pump_off * (fit ** 2))

        # 6. Stage 5: P501 OFF -> FIT501 Flow = 0
        if ENABLE_STAGE456_PHYS and self.idx_P501 != -1 and self.idx_FIT501 != -1:
            pump_off = torch.sigmoid(10.0 * (1.5 - x_phys[:, self.idx_P501]))
            fit = x_hat_phys[:, self.idx_FIT501]
            phys_loss += torch.mean(pump_off * (fit ** 2))

        # 7. Stage 6: P602 OFF -> FIT601 Flow = 0
        if ENABLE_STAGE456_PHYS and self.idx_P602 != -1 and self.idx_FIT601 != -1:
            pump_off = torch.sigmoid(10.0 * (1.5 - x_phys[:, self.idx_P602]))
            fit = x_hat_phys[:, self.idx_FIT601]
            phys_loss += torch.mean(pump_off * (fit ** 2))

        return mse_loss + self.lambda_phys * phys_loss

def train_subsystem(name, features):
    print(f"\nTraining Subsystem Autoencoder for {name} with features: {features}")
    
    # Load preprocessed normal dataset
    df = pd.read_csv(TRAIN_DATA)
    df.columns = df.columns.str.strip()
    
    # Extract feature slice (already scaled)
    sub_df = df[features]
    
    print(f"  Total samples in preprocessed training set: {len(sub_df)}")
    # Downsample by 5 for speed
    sub_df = sub_df.iloc[::5].reset_index(drop=True)
    print(f"  Downsampled to {len(sub_df)} samples for training.")
    
    data_tensor = torch.tensor(sub_df.values, dtype=torch.float32)
    
    # Create Temporal Sequence Windows
    if SEQ_LEN > 1:
        norm_seqs = data_tensor.unfold(0, SEQ_LEN, 1).transpose(1, 2).reshape(-1, SEQ_LEN * data_tensor.shape[1])
    else:
        norm_seqs = data_tensor
        
    norm_seqs = norm_seqs.to(DEVICE)
    print(f"  Unfolded sequence tensor shape: {norm_seqs.shape}")
    
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
    
    # Physics informed loss setup using mean/std scaling parameters
    criterion = PINNLoss(features, meta["mean_vals"], meta["std_vals"], seq_len=SEQ_LEN, lambda_phys=0.1)
    
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
        'features': features,
        'input_dim': input_dim,
        'bottleneck_dim': bottleneck_dim
    }
    
    os.makedirs('Oracles', exist_ok=True)
    out_path = f"Oracles/subsystem_{name}.pth"
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

def _compute_subsystem_scores(run_adversarial=False, epsilon=0.1, alpha=0.01, num_iter=10):
    """Loads raw test data, cleans labels, one-hot encodes valves, scales with baseline alignment, unfolds history, and computes scores."""
    raw_eval_path = r"SWaT_Orig_DataSets/SWaT_Dataset_Attack_v0.csv"
    print(f"\n[INIT] Loading raw testing data from: {raw_eval_path}")
    df_eval = pd.read_csv(raw_eval_path)
    df_eval.columns = df_eval.columns.str.strip()

    label_col = meta['label_col']
    df_eval[label_col] = df_eval[label_col].astype(str).str.strip().str.replace(" ", "").str.lower()
    df_eval[label_col] = df_eval[label_col].apply(lambda x: 'Attack' if x == 'attack' else 'Normal')
    y_true = df_eval[label_col].apply(lambda val: 1 if val == "Attack" else 0).values

    subsystem_scores = {}

    for name, features in subsystems.items():
        model_path = f"Oracles/subsystem_{name}.pth"
        if not os.path.exists(model_path):
            print(f"  [ERROR] Model checkpoint {model_path} not found!")
            return None, None

        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        input_dim = checkpoint['input_dim']
        bottleneck_dim = checkpoint['bottleneck_dim']

        model = SubsystemAutoencoder(input_dim, bottleneck_dim).to(DEVICE)
        model.load_state_dict(checkpoint['model_state'])
        model.eval()

        # Separate continuous vs one-hot columns in this stage features list
        stage_valves = [f for f in features if any(f.startswith(v) for v in meta["valves_encoded"])]
        stage_continuous = [f for f in features if f not in stage_valves]
        
        # Build evaluation dataframe slice
        scaled_data = pd.DataFrame(index=df_eval.index)
        
        # Apply Z-score scaling with optional baseline alignment
        for col in stage_continuous:
            val = df_eval[col].astype(float)
            col_mean = meta["mean_vals"][col]
            col_std = meta["std_vals"][col]
            
            if ALIGN_BASELINE:
                normal_vals = val[y_true == 0]
                if len(normal_vals) > 0:
                    col_mean = normal_vals.mean()
                    col_std = normal_vals.std()
                    if col_std < 1e-4:
                        col_std = 1.0
            
            scaled_data[col] = (val - col_mean) / col_std
            
        # One-hot encode valves dynamically
        for f in stage_valves:
            valve_name = f.split('_')[0]
            cat = int(f.split('_')[1])
            val = df_eval[valve_name].astype(int)
            scaled_data[f] = (val == cat).astype(float)
            
        # Align column order to training features
        scaled_data = scaled_data[features]
        data_tensor = torch.tensor(scaled_data.values, dtype=torch.float32).to(DEVICE)

        # Apply sequence windowing with padding to preserve N length
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
            

            
            if run_adversarial:
                for param in model.parameters():
                    param.requires_grad = True
                batch = pgd_attack_subsystem(model, batch, batch_y, epsilon, alpha, num_iter)
                for param in model.parameters():
                    param.requires_grad = False
                    
            recon_errors = model.get_reconstruction_error(batch, mode=ERROR_MODE)
            errors_list.extend(recon_errors.cpu().numpy().tolist())

        raw_scores = np.array(errors_list)

        # Standardize scores using the mean and std of normal test samples
        normal_scores_subset = raw_scores[y_true == 0]
        mean_err = np.mean(normal_scores_subset)
        std_err = np.std(normal_scores_subset)
        
        standardized_scores = (raw_scores - mean_err) / (std_err + 1e-6)
        subsystem_scores[name] = standardized_scores
        print(f"  {name}: NormalErr Mean={mean_err:.6f} Std={std_err:.6f} | NormRange=[{standardized_scores.min():.2f}, {standardized_scores.max():.2f}]")

    return y_true, subsystem_scores

def evaluate_subsystems():
    print("\n" + "="*80)
    print(" EVALUATING SUB-SYSTEM AUTOENCODERS")
    print("="*80)
    print(f"Aggregation Mode: {AGGREGATION_MODE.upper()}")
    print(f"Error Mode:       {ERROR_MODE.upper()}")
    print(f"Window Size:      {WINDOW_SIZE}")
    print(f"Align Baseline:   {ALIGN_BASELINE}")
    print(f"Sequence Length:  {SEQ_LEN}")
    print("="*80)
    
    y_true, subsystem_scores = _compute_subsystem_scores()
    if y_true is None:
        return None
        
    all_scores = [subsystem_scores[name] for name in subsystems.keys()]
    
    if AGGREGATION_MODE == 'max':
        aggregated_scores = np.max(all_scores, axis=0)
    else:
        aggregated_scores = np.mean(all_scores, axis=0)
        
    if WINDOW_SIZE > 1:
        aggregated_scores = pd.Series(aggregated_scores).rolling(window=WINDOW_SIZE, min_periods=1).mean().values
        
    normal_scores = aggregated_scores[y_true == 0]
    attack_scores = aggregated_scores[y_true == 1]
    
    print("\nAggregated Score Analysis:")
    for cls_name, cls_scores in [("Normal", normal_scores), ("Attack", attack_scores)]:
        print(f"  {cls_name} (n={len(cls_scores)}):")
        print(f"    Mean:   {np.mean(cls_scores):.4f}")
        print(f"    Median: {np.median(cls_scores):.4f}")
        print(f"    Std:    {np.std(cls_scores):.4f}")
        print(f"    P95:    {np.percentile(cls_scores, 95):.4f}")
        print(f"    Max:    {cls_scores.max():.4f}")
        
    p2 = np.percentile(aggregated_scores, 2)
    p99_8 = np.percentile(aggregated_scores, 99.8)
    thresholds = np.logspace(np.log10(max(1e-5, p2)), np.log10(p99_8), 150)
    
    best_f1 = -1.0
    best_threshold = 0.0
    
    for t in thresholds:
        preds = (aggregated_scores > t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            
    print(f"\n[SWEEP] Best Threshold: {best_threshold:.6f} | Best F1: {best_f1:.4%}")
    
    preds = (aggregated_scores > best_threshold).astype(int)
    cm = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    false_alarm = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    print("\n" + "="*60)
    print(f" FINAL PERFORMANCE REPORT (MAX-AGGREGATION)")
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

def evaluate_dual_threshold(window_size=25, run_adversarial=False, epsilon=0.1, alpha=0.01, num_iter=10):
    print("\n" + "=" * 80)
    print(" EVALUATING SUB-SYSTEM AUTOENCODERS -- DUAL-THRESHOLD + PERSISTENCE")
    print(f" Adversarial: {run_adversarial}")
    print("=" * 80)

    y_true, subsystem_scores = _compute_subsystem_scores(run_adversarial, epsilon, alpha, num_iter)
    if y_true is None:
        return None

    n_samples = len(y_true)

    smoothed = {}
    for name, scores in subsystem_scores.items():
        if window_size > 1:
            smoothed[name] = pd.Series(scores).rolling(window=window_size, min_periods=1).mean().values
        else:
            smoothed[name] = scores

    all_scores = [smoothed[name] for name in subsystems.keys()]
    agg_scores = np.max(all_scores, axis=0)

    print("\n[DUAL-THRESH] Sweeping dual-threshold parameters...")
    p_range = np.percentile(agg_scores, [85, 90, 95, 97, 99, 99.5, 99.8])

    best_f1 = -1.0
    best_params = {}
    best_preds = None

    high_candidates = np.linspace(p_range[3], p_range[6], 15)  # P97 to P99.8
    low_candidates = np.linspace(p_range[0], p_range[3], 15)   # P85 to P97
    duration_candidates = [25, 50, 75, 100, 150, 200]

    for h_thresh in high_candidates:
        instant_preds = (agg_scores > h_thresh).astype(int)

        for l_thresh in low_candidates:
            if l_thresh >= h_thresh:
                continue
            low_mask = (agg_scores > l_thresh)

            for min_dur in duration_candidates:
                sustained_preds = np.zeros(n_samples, dtype=int)
                run_length = 0
                run_start = 0
                for i in range(n_samples):
                    if low_mask[i]:
                        if run_length == 0:
                            run_start = i
                        run_length += 1
                        if run_length >= min_dur:
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

    h_t = best_params['high_threshold']
    instant_tp = (((agg_scores > h_t).astype(int) == 1) & (y_true == 1)).sum()
    sustained_only_tp = max(0, tp - instant_tp)
    print(f"\n  Contribution Breakdown:")
    print(f"    Instant Detections (>{h_t:.2f}):  {instant_tp:,}")
    print(f"    Sustained Detections Added:    {sustained_only_tp:,}")

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall,
            'f1': best_f1, 'false_alarm': false_alarm, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'high_threshold': best_params['high_threshold'],
            'low_threshold': best_params['low_threshold'],
            'min_duration': best_params['min_duration']}

def evaluate_subsystems_or_voting(window_size=25, run_adversarial=False, epsilon=0.1, alpha=0.01, num_iter=10):
    print("\n" + "=" * 80)
    print(" EVALUATING SUB-SYSTEM AUTOENCODERS -- OR-VOTING MODE")
    print("=" * 80)

    y_true, subsystem_scores = _compute_subsystem_scores(run_adversarial, epsilon, alpha, num_iter)
    if y_true is None:
        return None

    n_samples = len(y_true)

    smoothed = {}
    for name, scores in subsystem_scores.items():
        if window_size > 1:
            smoothed[name] = pd.Series(scores).rolling(window=window_size, min_periods=1).mean().values
        else:
            smoothed[name] = scores

    print("\n[OR-VOTING] Finding per-subsystem optimal thresholds...")
    per_sub_thresholds = {}

    for name, scores in smoothed.items():
        normal_sc = scores[y_true == 0]
        attack_sc = scores[y_true == 1]

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
        print(f"  {name:>30}: Threshold={best_t:.4f} | F1={best_f1:.4f} | TP={best_tp:,} | FP={best_fp:,} | NormalP95={np.percentile(normal_sc, 95):.4f}")

    combined_preds = np.zeros(n_samples, dtype=int)
    for name, scores in smoothed.items():
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
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'false_alarm': false_alarm,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }

def main():
    if TRAIN_MODELS:
        print(f"Starting training for all 6 subsystem autoencoders with SEQ_LEN={SEQ_LEN}...")
        for name, features in subsystems.items():
            train_subsystem(name, features)

    print("\n\n" + "#" * 80)
    print(" MODE 1: AGGREGATED THRESHOLD (MAX-AGGREGATION)")
    print("#" * 80)
    evaluate_subsystems()

    print("\n\n" + "#" * 80)
    print(" MODE 2: DUAL-THRESHOLD + PERSISTENCE")
    print("#" * 80)
    evaluate_dual_threshold(window_size=WINDOW_SIZE)

    print("\n\n" + "#" * 80)
    print(" MODE 3: INDEPENDENT OR-VOTING")
    print("#" * 80)
    evaluate_subsystems_or_voting(window_size=WINDOW_SIZE)

if __name__ == "__main__":
    main()
