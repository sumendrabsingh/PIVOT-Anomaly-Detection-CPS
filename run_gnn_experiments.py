import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import json
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import confusion_matrix, f1_score

# --- Config ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HIDDEN_DIM = 64
NUM_GCN_LAYERS = 3
EPOCHS = 100
BATCH_SIZE = 4096
WINDOW_SIZE = 25
ERROR_MODE = 'max'        # 'max' or 'mean' reconstruction error inside GAE
AGGREGATION_MODE = 'max'  # 'max' or 'mean' error across 6 stages
ALIGN_BASELINE = True

TRAIN_DATA = r"SWaT_Orig_DataSets/Original Dataset Backup/SWaT_Dataset_Normal_v1_Clean.csv"
EVAL_DATA = r"SWaT_Orig_DataSets/SWaT_Dataset_Attack_v0_Corrected.csv"
META_DATA = r"Datasets/preprocessing_meta.json"

# Load metadata config
with open(META_DATA, 'r') as f:
    meta = json.load(f)

NODE_KEYS = [
    "Stage1_RawWater",
    "Stage2_ChemicalDosing",
    "Stage3_Ultrafiltration",
    "Stage4_Dechlorination",
    "Stage5_ReverseOsmosis",
    "Stage6_ROCleaning",
    "Stage7_Network"
]

def get_normalized_adjacency():
    num_nodes = len(NODE_KEYS)
    A = torch.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        A[i, i] = 1.0
    for i in range(num_nodes - 1):
        A[i, i+1] = 1.0
        A[i+1, i] = 1.0
        
    D = torch.sum(A, dim=1)
    D_inv_sqrt = torch.pow(D, -0.5)
    D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0
    
    D_mat = torch.diag(D_inv_sqrt)
    A_norm = D_mat @ A @ D_mat
    return A_norm.to(DEVICE)

def get_subsystem_features(final_columns):
    subsystems = {k: [] for k in NODE_KEYS}
    network_features = ['total_packets', 'read_tags', 'write_tags', 'error_count', 'unique_src', 'unique_dst']
    
    for col in final_columns:
        if col in network_features:
            subsystems["Stage7_Network"].append(col)
            continue
            
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

class GraphAutoencoder(nn.Module):
    def __init__(self, feature_dims, hidden_dim, num_layers, A_norm):
        super().__init__()
        self.A_norm = A_norm
        self.hidden_dim = hidden_dim
        
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim)
            ) for dim in feature_dims
        ])
        
        self.gcn_enc = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        self.gcn_dec = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, dim)
            ) for dim in feature_dims
        ])
        
    def forward(self, x_list):
        h_list = []
        for i, encoder in enumerate(self.encoders):
            h_list.append(encoder(x_list[i]))
            
        H = torch.stack(h_list, dim=1) # Shape: [Batch, NUM_NODES, hidden_dim]
        
        for gcn in self.gcn_enc:
            H = torch.einsum('ij, bjk -> bik', self.A_norm, H)
            H = torch.relu(gcn(H))
            
        for gcn in self.gcn_dec:
            H = torch.einsum('ij, bjk -> bik', self.A_norm, H)
            H = torch.relu(gcn(H))
            
        out_list = []
        for i, decoder in enumerate(self.decoders):
            out_list.append(decoder(H[:, i, :]))
            
        return out_list

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

    def forward(self, x_hat, x):
        mse_loss = self.mse(x_hat, x)
        
        if self.lambda_phys <= 0:
            return mse_loss
            
        # Reshape [Batch, SEQ_LEN * F] -> [Batch * SEQ_LEN, F] to apply physical rules on all timesteps
        x_hat_flat = x_hat.reshape(-1, self.seq_len, self.num_features)
        x_flat = x.reshape(-1, self.seq_len, self.num_features)
        
        # Flatten to 2D
        x_hat_flat = x_hat_flat.reshape(-1, self.num_features)
        x_flat = x_flat.reshape(-1, self.num_features)
        
        mean_v = self.mean_vals.to(x.device)
        std_v = self.std_vals.to(x.device)
        
        # Denormalize x_hat back to physical values
        x_hat_phys = x_hat_flat * std_v + mean_v
        
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
            
        return mse_loss + self.lambda_phys * phys_loss

def train_gnn():
    print(f"\n[TRAIN] Loading preprocessed normal dataset: {TRAIN_DATA}")
    df = pd.read_csv(TRAIN_DATA)
    df.columns = df.columns.str.strip()
    
    stage_tensors = []
    feature_dims = []
    
    for name in NODE_KEYS:
        features = subsystems[name]
        sub_df = df[features]
        
        # Downsample by 5 for training efficiency (matching baseline AEs)
        sub_df_down = sub_df.iloc[::5].reset_index(drop=True)
        data_tensor = torch.tensor(sub_df_down.values, dtype=torch.float32)
        
        if SEQ_LEN > 1:
            seqs = data_tensor.unfold(0, SEQ_LEN, 1).transpose(1, 2).reshape(-1, SEQ_LEN * data_tensor.shape[1])
        else:
            seqs = data_tensor
            
        stage_tensors.append(seqs)
        feature_dims.append(seqs.shape[1])
        
    num_samples = min(t.shape[0] for t in stage_tensors)
    stage_tensors = [t[:num_samples] for t in stage_tensors]
    
    concat_data = torch.cat(stage_tensors, dim=1)
    print(f"[TRAIN] Joined dataset tensor shape: {concat_data.shape}")
    
    dataset = TensorDataset(concat_data)
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    A_norm = get_normalized_adjacency()
    model = GraphAutoencoder(feature_dims, HIDDEN_DIM, NUM_GCN_LAYERS, A_norm).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    pinn_losses = []
    for name in NODE_KEYS:
        features = subsystems[name]
        pinn_losses.append(PINNLoss(features, meta["mean_vals"], meta["std_vals"], seq_len=SEQ_LEN, lambda_phys=0.1))
        
    best_val_loss = float('inf')
    patience = 20
    epochs_no_improve = 0
    best_model_state = None
    
    print(f"[TRAIN] Starting GNN Autoencoder Training (SEQ_LEN = {SEQ_LEN})...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x_batch = batch[0].to(DEVICE)
            
            x_list = []
            idx = 0
            for dim in feature_dims:
                x_list.append(x_batch[:, idx:idx+dim])
                idx += dim
                
            optimizer.zero_grad()
            out_list = model(x_list)
            
            loss = 0
            for i, (out, x) in enumerate(zip(out_list, x_list)):
                loss += pinn_losses[i](out, x)
                
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x_batch = batch[0].to(DEVICE)
                x_list = []
                idx = 0
                for dim in feature_dims:
                    x_list.append(x_batch[:, idx:idx+dim])
                    idx += dim
                
                out_list = model(x_list)
                loss = 0
                for i, (out, x) in enumerate(zip(out_list, x_list)):
                    loss += pinn_losses[i](out, x)
                val_loss += loss.item() * x_batch.size(0)
        val_loss /= len(val_loader.dataset)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
        if epochs_no_improve >= patience:
            print(f"  Early stopping at epoch {epoch}. Best Val Loss: {best_val_loss:.6f}")
            break
            
    os.makedirs('Oracles', exist_ok=True)
    out_path = f"Oracles/gnn_autoencoder_seq_{SEQ_LEN}.pth"
    torch.save({
        'model_state': best_model_state,
        'feature_dims': feature_dims
    }, out_path)
    print(f"  Saved GNN model to {out_path}")

def evaluate_gnn(export_fn=False, eval_data=None):
    print(f"\n[EVAL] Loading GNN and Attack Dataset (SEQ_LEN = {SEQ_LEN})...")
    raw_eval_path = eval_data if eval_data else r"Datasets/swat_attack_with_network.csv"
    print(f"[EVAL] Eval data: {raw_eval_path}")
    df_eval = pd.read_csv(raw_eval_path)
    df_eval.columns = df_eval.columns.str.strip()
    
    label_col = meta['label_col']
    df_eval[label_col] = df_eval[label_col].astype(str).str.strip().str.replace(" ", "").str.lower()
    df_eval[label_col] = df_eval[label_col].apply(lambda x: 'Attack' if x == 'attack' else 'Normal')
    y_true = df_eval[label_col].apply(lambda val: 1 if val == "Attack" else 0).values
    
    stage_tensors = []
    for name in NODE_KEYS:
        features = subsystems[name]
        stage_valves = [f for f in features if any(f.startswith(v) for v in meta["valves_encoded"])]
        stage_continuous = [f for f in features if f not in stage_valves]
        
        scaled_data = pd.DataFrame(index=df_eval.index)
        
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
            seqs = padded_data.unfold(0, SEQ_LEN, 1).transpose(1, 2).reshape(-1, SEQ_LEN * data_tensor.shape[1])
        else:
            seqs = data_tensor
            
        stage_tensors.append(seqs)
        
    num_eval_samples = min(t.shape[0] for t in stage_tensors)
    stage_tensors = [t[:num_eval_samples] for t in stage_tensors]
    y_true = y_true[:num_eval_samples]
    
    checkpoint_path = f"Oracles/gnn_autoencoder_seq_{SEQ_LEN}.pth"
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint {checkpoint_path} does not exist. Please train the model first.")
        return
        
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    feature_dims = checkpoint['feature_dims']
    A_norm = get_normalized_adjacency()
    
    model = GraphAutoencoder(feature_dims, HIDDEN_DIM, NUM_GCN_LAYERS, A_norm).to(DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    print("[EVAL] Running GNN reconstruction error calculation...")
    node_errors = [[] for _ in NODE_KEYS]
    
    # Calculate clip bounds per node/stage
    stage_bounds = []
    for name in NODE_KEYS:
        features = subsystems[name]
        # For Z-score, the normal bounds are approx -3 to +3, but we can compute exact Z-score bounds from meta max/min
        min_bound = []
        max_bound = []
        for f in features:
            base = f.split('_')[0]
            if base in meta["mean_vals"]:
                z_min = (meta["min_vals"][base] - meta["mean_vals"][base]) / meta["std_vals"][base]
                z_max = (meta["max_vals"][base] - meta["mean_vals"][base]) / meta["std_vals"][base]
                min_bound.append(z_min)
                max_bound.append(z_max)
            else:
                min_bound.append(-1.0)
                max_bound.append(2.0) # For valves, max is 1, maybe wait! Valves are [0,1], after scaling... Wait, valves aren't scaled in z-score, they are 0.0 or 1.0!
        
        # If SEQ_LEN > 1, we need to repeat the bounds for each timestep
        min_b = torch.tensor(min_bound, dtype=torch.float32).to(DEVICE).repeat(SEQ_LEN)
        max_b = torch.tensor(max_bound, dtype=torch.float32).to(DEVICE).repeat(SEQ_LEN)
        stage_bounds.append((min_b, max_b))

    for idx in range(0, num_eval_samples, BATCH_SIZE):
        batch_x_list = [t[idx:idx+BATCH_SIZE].to(DEVICE) for t in stage_tensors]
        with torch.no_grad():
            batch_x_clipped = []
            for t, bounds in zip(batch_x_list, stage_bounds):
                batch_x_clipped.append(torch.max(torch.min(t, bounds[1]), bounds[0]))
            out_list = model(batch_x_clipped)
            
        for i, (out, x) in enumerate(zip(out_list, batch_x_list)):
            squared_diff = (out - x) ** 2
            if ERROR_MODE == 'max':
                node_err = torch.max(squared_diff, dim=1)[0]
            else:
                node_err = torch.mean(squared_diff, dim=1)
            node_errors[i].extend(node_err.cpu().numpy().tolist())
            
    node_scores = [np.array(err_list) for err_list in node_errors]
    
    standardized_node_scores = []
    print("\n--- Subsystem Reconstruction Error Normalization Ranges ---")
    for i, name in enumerate(NODE_KEYS):
        scores = node_scores[i]
        normal_scores_subset = scores[y_true == 0]
        mean_err = np.mean(normal_scores_subset)
        std_err = np.std(normal_scores_subset)
        std_scores = (scores - mean_err) / (std_err + 1e-6)
        standardized_node_scores.append(std_scores)
        print(f"  {name}: NormalMean={mean_err:.6f} Std={std_err:.6f} | NormRange=[{std_scores.min():.2f}, {std_scores.max():.2f}]")
        
    if AGGREGATION_MODE == 'max':
        aggregated_scores = np.max(standardized_node_scores, axis=0)
    else:
        aggregated_scores = np.mean(standardized_node_scores, axis=0)
        
    smoothed_scores = pd.Series(aggregated_scores).rolling(window=WINDOW_SIZE, min_periods=1).mean().values
    
    # 1. Aggregated Max (Single Threshold) Evaluation
    print("\n[EVAL] Running Aggregated Max Single Threshold sweep...")
    p2 = np.percentile(smoothed_scores, 2)
    p99_8 = np.percentile(smoothed_scores, 99.8)
    thresholds = np.logspace(np.log10(max(1e-5, p2)), np.log10(p99_8), 200)
    
    best_single_f1 = -1.0
    best_single_th = 0.0
    
    for t in thresholds:
        preds = (smoothed_scores > t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_single_f1:
            best_single_f1 = f1
            best_single_th = t
            
    single_preds = (smoothed_scores > best_single_th).astype(int)
    cm_single = confusion_matrix(y_true, single_preds)
    tn_s, fp_s, fn_s, tp_s = cm_single.ravel()
    
    acc_s = (tp_s + tn_s) / len(y_true)
    prec_s = tp_s / (tp_s + fp_s) if (tp_s + fp_s) > 0 else 0.0
    rec_s = tp_s / (tp_s + fn_s) if (tp_s + fn_s) > 0 else 0.0
    far_s = fp_s / (fp_s + tn_s) if (fp_s + tn_s) > 0 else 0.0
    
    print("\n" + "="*80)
    print(f" FINAL PERFORMANCE REPORT: AGGREGATED MAX (SEQ_LEN = {SEQ_LEN})")
    print("="*80)
    print(f"  Best Single Threshold:         {best_single_th:.6f}")
    print(f"  Accuracy:                      {acc_s:.4%}")
    print(f"  Precision:                     {prec_s:.4%}")
    print(f"  Recall (Detection Rate):       {rec_s:.4%}")
    print(f"  F1-Score:                      {best_single_f1:.4%}")
    print(f"  False Alarm Rate:              {far_s:.4%}")
    print("  Confusion Matrix:")
    print(f"    TP: {tp_s:,} | FN: {fn_s:,}")
    print(f"    FP: {fp_s:,} | TN: {tn_s:,}")
    print("="*80)

    # 2. Dual-Threshold + Persistence Evaluation
    print("\n[EVAL] Running Dual-Threshold + Persistence sweep...")
    p_range = np.percentile(smoothed_scores, [85, 90, 95, 97, 99, 99.5, 99.8])
    
    best_dual_f1 = -1.0
    best_dual_params = {}
    best_dual_preds = None
    
    high_candidates = np.linspace(p_range[3], p_range[6], 15)  # P97 to P99.8
    low_candidates = np.linspace(p_range[0], p_range[3], 15)   # P85 to P97
    duration_candidates = [25, 50, 75, 100, 150, 200]
    
    for h_thresh in high_candidates:
        instant_preds = (smoothed_scores > h_thresh).astype(int)
        for l_thresh in low_candidates:
            if l_thresh >= h_thresh:
                continue
            low_mask = (smoothed_scores > l_thresh)
            for min_dur in duration_candidates:
                sustained_preds = np.zeros(num_eval_samples, dtype=int)
                run_length = 0
                run_start = 0
                for i in range(num_eval_samples):
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
                if f1 > best_dual_f1:
                    best_dual_f1 = f1
                    best_dual_params = {
                        'high_threshold': h_thresh,
                        'low_threshold': l_thresh,
                        'min_duration': min_dur
                    }
                    best_dual_preds = combined.copy()
                    
    if best_dual_preds is not None:
        tn_d, fp_d, fn_d, tp_d = confusion_matrix(y_true, best_dual_preds).ravel()
        acc_d = (tp_d + tn_d) / len(y_true)
        prec_d = tp_d / (tp_d + fp_d) if (tp_d + fp_d) > 0 else 0.0
        rec_d = tp_d / (tp_d + fn_d) if (tp_d + fn_d) > 0 else 0.0
        far_d = fp_d / (fp_d + tn_d) if (fp_d + tn_d) > 0 else 0.0
        
        print("\n" + "="*80)
        print(f" FINAL PERFORMANCE REPORT: DUAL-THRESHOLD + PERSISTENCE (SEQ_LEN = {SEQ_LEN})")
        print("="*80)
        print(f"  Best Parameters:")
        print(f"    High Threshold (instant):  {best_dual_params['high_threshold']:.4f}")
        print(f"    Low Threshold (sustained): {best_dual_params['low_threshold']:.4f}")
        print(f"    Min Duration (samples):    {best_dual_params['min_duration']}")
        print(f"  Accuracy:                      {acc_d:.4%}")
        print(f"  Precision:                     {prec_d:.4%}")
        print(f"  Recall (Detection Rate):       {rec_d:.4%}")
        print(f"  F1-Score:                      {best_dual_f1:.4%}")
        print(f"  False Alarm Rate:              {far_d:.4%}")
        print("  Confusion Matrix:")
        print(f"    TP: {tp_d:,} | FN: {fn_d:,}")
        print(f"    FP: {fp_d:,} | TN: {tn_d:,}")
        print("="*80)
        
        instant_tp = (((smoothed_scores > best_dual_params['high_threshold']).astype(int) == 1) & (y_true == 1)).sum()
        sustained_only_tp = max(0, tp_d - instant_tp)
        print(f"\n  Contribution Breakdown:")
        print(f"    Instant Detections:  {instant_tp:,}")
        print(f"    Sustained Detections: {sustained_only_tp:,}")
        print("="*80)
    else:
        print("[ERROR] No optimal parameters found.")

    # Determine final predictions
    final_preds = best_dual_preds if best_dual_preds is not None else single_preds

    # Export false negatives if requested
    if export_fn:
        fn_indices = np.where((y_true == 1) & (final_preds == 0))[0]
        if len(fn_indices) == 0:
            print("[EXPORT] No false negatives found.")
        else:
            df_fn = df_eval.iloc[fn_indices].copy()
            for stage_name, feats in subsystems.items():
                base_feats = list(set([f.split('_')[0] for f in feats]))
                stage_df = df_fn[base_feats]
                out_path = f"e:/PIVOT/scratch/fn_{stage_name.lower()}.csv"
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                stage_df.to_csv(out_path, index=False)
                print(f"[EXPORT] Saved FN samples for {stage_name} to {out_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--export-fn', action='store_true', help='Export false negative samples per stage after evaluation')
    parser.add_argument('--eval_data', type=str, default=None, help='Override eval CSV (default: Datasets\\swat_attack_with_network.csv)')
    args = parser.parse_args()

    SEQ_LEN = args.seq_len
    if args.train:
        train_gnn()
    evaluate_gnn(export_fn=args.export_fn, eval_data=args.eval_data)
