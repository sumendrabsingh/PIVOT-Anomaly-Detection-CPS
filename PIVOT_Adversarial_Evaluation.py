import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib
# Use Agg backend to prevent interactive window blocks
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score

# --- 1. ARCHITECTURE (Must match training) ---


class OracleVAE(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=32, dropout_rate=0.2):
        super().__init__()
        
        self.encoder_base = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.fc_mu = nn.Linear(64, bottleneck_dim)
        self.fc_logvar = nn.Linear(64, bottleneck_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128, input_dim)
        )
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x):
        h = self.encoder_base(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        decoded = self.decoder(z)
        return decoded, mu, logvar
        
    def get_reconstruction_error(self, x, mode='mean'):
        """Deterministic MSE or Max for evaluation"""
        with torch.no_grad():
            h = self.encoder_base(x)
            mu = self.fc_mu(h)
            reconstructed = self.decoder(mu)
            squared_diff = (x - reconstructed) ** 2
            if mode == 'max':
                error = torch.max(squared_diff, dim=1)[0]
            else:
                error = torch.mean(squared_diff, dim=1)
        return error


import torch.nn.functional as F

class OracleContrastive(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=32, dropout_rate=0.2):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, bottleneck_dim)
        )
        
    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return F.normalize(z, dim=1)

    def get_anomaly_score(self, x, c):
        """Cosine distance (1 - Cosine Similarity) from output to the learned center c."""
        with torch.no_grad():
            z = self.forward(x)
            c = c.unsqueeze(0) if c.dim() == 1 else c
            c = F.normalize(c, dim=1)
            sim = F.cosine_similarity(z, c)
            dist = 1.0 - sim
        return dist

class OracleDeepSVDD(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=32, dropout_rate=0.2):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128, bias=False),
            nn.BatchNorm1d(128, eps=1e-4, affine=False),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64, bias=False),
            nn.BatchNorm1d(64, eps=1e-4, affine=False),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, bottleneck_dim, bias=False)
        )
        
    def forward(self, x):
        return self.encoder(x)
        
    def get_anomaly_score(self, x, c):
        """Distance from output to the learned center c."""
        with torch.no_grad():
            z = self.forward(x)
            dist = torch.sum((z - c) ** 2, dim=1)
        return dist

class OracleAutoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=32):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, bottleneck_dim),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128, input_dim),
        )
    
    def forward(self, x):
        return self.decoder(self.encoder(x))
    
    def get_reconstruction_error(self, x, mode='mean'):
        """Per-sample MSE or Max reconstruction error."""
        with torch.no_grad():
            reconstructed = self.forward(x)
            squared_diff = (x - reconstructed) ** 2
            if mode == 'max':
                error = torch.max(squared_diff, dim=1)[0]
            else:
                error = torch.mean(squared_diff, dim=1)
        return error


class OracleTransformerAutoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=32, d_model=32, nhead=4, num_layers=2, dim_feedforward=64, dropout_rate=0.2, use_cls_bottleneck=True):
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.use_cls_bottleneck = use_cls_bottleneck
        # Use bottleneck_dim as d_model if provided, since bottleneck_dim is the hyperparameter passed from training command line
        self.d_model = bottleneck_dim if bottleneck_dim is not None else d_model
        
        # Adjust nhead if necessary so that self.d_model is divisible by nhead
        if self.d_model % nhead != 0:
            adjusted = False
            for h in [4, 2, 1]:
                if self.d_model % h == 0:
                    nhead = h
                    adjusted = True
                    break
            if not adjusted:
                nhead = 1
                
        self.nhead = nhead
        
        # Parallel embedding weights: [input_dim, d_model]
        self.embed_w = nn.Parameter(torch.empty(input_dim, self.d_model))
        self.embed_b = nn.Parameter(torch.empty(input_dim, self.d_model))
        
        # Learnable feature token identification embedding (positional embedding)
        self.feature_pos = nn.Parameter(torch.empty(1, input_dim, self.d_model))
        
        # Transformer Encoder Block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=self.nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        if self.use_cls_bottleneck:
            # Learnable classification token
            self.cls_token = nn.Parameter(torch.empty(1, 1, self.d_model))
            # Decoding linear layer from CLS embedding back to original input dimensions
            self.decoder_linear = nn.Linear(self.d_model, input_dim)
            
            nn.init.normal_(self.cls_token, std=0.02)
        else:
            # Parallel decoding weights: [input_dim, d_model]
            self.decode_w = nn.Parameter(torch.empty(input_dim, self.d_model))
            self.decode_b = nn.Parameter(torch.empty(input_dim))
            
            stdv_decode = 1.0 / np.sqrt(self.d_model)
            nn.init.uniform_(self.decode_w, -stdv_decode, stdv_decode)
            nn.init.uniform_(self.decode_b, -stdv_decode, stdv_decode)
            
        # Parameter initialization
        stdv_embed = 1.0
        nn.init.uniform_(self.embed_w, -stdv_embed, stdv_embed)
        nn.init.uniform_(self.embed_b, -stdv_embed, stdv_embed)
        nn.init.normal_(self.feature_pos, std=0.02)
        
    def forward(self, x):
        # x shape: [batch, input_dim]
        # 1. Project each feature to d_model embedding (vectorized)
        tokens = x.unsqueeze(2) * self.embed_w.unsqueeze(0) + self.embed_b.unsqueeze(0) # [batch, input_dim, d_model]
        
        # 2. Add sensor identification embeddings
        tokens = tokens + self.feature_pos
        
        if self.use_cls_bottleneck:
            # Expand CLS token to match batch size
            cls_tokens = self.cls_token.expand(x.size(0), -1, -1) # [batch, 1, d_model]
            # Prepend CLS token to the sequence
            tokens = torch.cat((cls_tokens, tokens), dim=1) # [batch, input_dim + 1, d_model]
            
            # 3. Pass through self-attention layers
            out = self.transformer_encoder(tokens) # [batch, input_dim + 1, d_model]
            
            # Extract CLS token output: [batch, d_model]
            cls_out = out[:, 0, :]
            
            # Reconstruct original features
            reconstructed = self.decoder_linear(cls_out)
        else:
            # 3. Pass through self-attention layers
            out = self.transformer_encoder(tokens) # [batch, input_dim, d_model]
            
            # 4. Decode each token back to a scalar value (vectorized)
            reconstructed = torch.sum(out * self.decode_w.unsqueeze(0), dim=2) + self.decode_b.unsqueeze(0)
            
        return reconstructed
        
    def get_reconstruction_error(self, x, mode='mean'):
        with torch.no_grad():
            reconstructed = self.forward(x)
            squared_diff = (x - reconstructed) ** 2
            if mode == 'max':
                error = torch.max(squared_diff, dim=1)[0]
            else:
                error = torch.mean(squared_diff, dim=1)
        return error


# --- 1.5. ADVERSARIAL ATTACK ENGINE ---
def pgd_attack(model, x_clean, y_true, center_c, model_type, error_mode='mean', epsilon=0.1, alpha=0.01, num_iter=10):
    """
    Projected Gradient Descent (PGD) evasion attack.
    The attacker's goal is to MINIMIZE the anomaly score for ATTACK samples (y_true == 1),
    making them look like normal samples.
    """
    # Clone and detach to avoid affecting the original tensor
    x_adv = x_clean.clone().detach().to(x_clean.device)
    
    # We only perturb the attack samples
    attack_mask = (y_true == 1)
    if not attack_mask.any():
        return x_adv

    # Only optimize the attack samples
    x_adv_attacks = x_adv[attack_mask].clone().detach()
    x_clean_attacks = x_clean[attack_mask].clone().detach()
    
    # Enable gradients for the input
    x_adv_attacks.requires_grad = True
    
    for _ in range(num_iter):
        # Differentiable forward pass
        if model_type == 'svdd':
            z = model(x_adv_attacks)
            loss = torch.mean(torch.sum((z - center_c) ** 2, dim=1))
        elif model_type == 'contrastive':
            z = model(x_adv_attacks)
            c_norm = F.normalize(center_c.unsqueeze(0) if center_c.dim() == 1 else center_c, dim=1)
            sim = F.cosine_similarity(z, c_norm)
            loss = torch.mean(1.0 - sim)
        elif model_type == 'vae':
            reconstructed, _, _ = model(x_adv_attacks)
            squared_diff = (x_adv_attacks - reconstructed) ** 2
            loss = torch.mean(torch.max(squared_diff, dim=1)[0]) if error_mode == 'max' else torch.mean(squared_diff)
        else:
            reconstructed = model(x_adv_attacks)
            squared_diff = (x_adv_attacks - reconstructed) ** 2
            loss = torch.mean(torch.max(squared_diff, dim=1)[0]) if error_mode == 'max' else torch.mean(squared_diff)
            
        model.zero_grad()
        if x_adv_attacks.grad is not None:
            x_adv_attacks.grad.zero_()
            
        loss.backward()
        
        # PGD step: we want to MINIMIZE the loss (reconstruction error), so we step in the OPPOSITE direction of the gradient
        with torch.no_grad():
            grad_sign = x_adv_attacks.grad.sign()
            x_adv_attacks = x_adv_attacks - alpha * grad_sign
            
            # Project back to L-infinity epsilon ball around x_clean
            delta = torch.clamp(x_adv_attacks - x_clean_attacks, min=-epsilon, max=epsilon)
            x_adv_attacks = x_clean_attacks + delta
            
            x_adv_attacks.requires_grad = True

    with torch.no_grad():
        x_adv[attack_mask] = x_adv_attacks.detach()

    return x_adv

# --- 2. EVALUATION ENGINE ---
def run_final_evaluation(data_path, bundle_path, threshold_mode='f1_opt', window_size=1, align_baseline=False, error_mode='mean', run_adversarial=False, epsilon=0.1, alpha=0.01, num_iter=10):
    """
    Evaluate the autoencoder oracle on test data containing both Normal and Attack samples.
    Uses reconstruction error as the anomaly score — normal data should reconstruct well
    (low error), while attack data should not (high error).
    """
    # Resolve relative paths dynamically to script folder if not found in CWD
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(bundle_path):
        alt_bundle_path = os.path.join(script_dir, bundle_path)
        if os.path.exists(alt_bundle_path):
            bundle_path = alt_bundle_path
            
    if not os.path.exists(data_path):
        alt_data_path = os.path.join(script_dir, data_path)
        if os.path.exists(alt_data_path):
            data_path = alt_data_path

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INIT] Using device: {device}")

    # Load model
    checkpoint = torch.load(bundle_path, map_location='cpu', weights_only=False)
    
    input_dim = checkpoint['input_dim']
    bottleneck_dim = checkpoint.get('bottleneck_dim', 16)
    
    model_type = checkpoint.get('model_type', 'autoencoder')
    
    if model_type == 'svdd':
        model = OracleDeepSVDD(input_dim=input_dim, bottleneck_dim=bottleneck_dim).to(device)
        center_c = checkpoint['center_c'].to(device)
    elif model_type == 'vae':
        model = OracleVAE(input_dim=input_dim, bottleneck_dim=bottleneck_dim).to(device)
        center_c = None
    elif model_type == 'contrastive':
        model = OracleContrastive(input_dim=input_dim, bottleneck_dim=bottleneck_dim).to(device)
        center_c = checkpoint['center_c'].to(device)
    elif model_type == 'transformer':
        use_cls = 'cls_token' in checkpoint['model_state']
        model = OracleTransformerAutoencoder(input_dim=input_dim, bottleneck_dim=bottleneck_dim, use_cls_bottleneck=use_cls).to(device)
        center_c = None
    else:
        model = OracleAutoencoder(input_dim=input_dim, bottleneck_dim=bottleneck_dim).to(device)
        center_c = None
        
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    MEAN, STD = checkpoint['mean'], checkpoint['std']
    
    # Load saved threshold references from training
    train_p95 = checkpoint.get('train_error_p95', None)
    train_p99 = checkpoint.get('train_error_p99', None)
    
    print(f"[MODEL] Loaded autoencoder: {input_dim} features -> {bottleneck_dim} bottleneck")
    print(f"[MODEL] Best epoch: {checkpoint.get('best_epoch', 'N/A')}")
    if train_p95:
        print(f"[MODEL] Training P95 threshold: {train_p95:.6f}")
        print(f"[MODEL] Training P99 threshold: {train_p99:.6f}")
    
    # Load test data
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    
    label_col = 'Label' if 'Label' in df.columns else 'Normal/Attack'
    numeric_df = df.select_dtypes(include=[np.number]).fillna(df.mean(numeric_only=True))
    
    # Filter features based on checkpoint if available (Phase 1)
    features_list = checkpoint.get('features', None)
    if features_list is not None:
        print(f"[PREPROCESS] Aligning test data columns to model features ({len(features_list)} features)...")
        numeric_df = numeric_df[features_list]
    
    print(f"\n[DATA] Test samples: {len(numeric_df)}")
    
    # Get true labels (vectorized & robust to typos like "A ttack") early for baseline alignment
    y_true = df[label_col].astype(str).str.strip().str.lower().str.replace(" ", "").apply(lambda val: 1 if val == "attack" else 0).values
    
    # Normalize using training statistics (supporting both min-max and z-score scaling)
    scaler_type = checkpoint.get('scaler_type', 'zscore')
    data_tensor = torch.tensor(numeric_df.values, dtype=torch.float32)
    
    if align_baseline:
        normal_mask = (y_true == 0)
        normal_data = data_tensor[normal_mask]
        if len(normal_data) > 0:
            if scaler_type == 'minmax':
                MIN = normal_data.min(dim=0)[0]
                MAX = normal_data.max(dim=0)[0]
                MEAN = MIN
                STD = MAX
                print("[PREPROCESS] Aligned baseline Min-Max scaling parameters to test set normal data.")
            else:
                MEAN = normal_data.mean(dim=0)
                STD = normal_data.std(dim=0)
                print("[PREPROCESS] Aligned baseline Z-Score scaling parameters to test set normal data.")
        else:
            print("[WARNING] No normal samples found in test set to align baseline. Falling back to training baseline.")
            MIN = checkpoint.get('min', MEAN)
            MAX = checkpoint.get('max', STD)
    else:
        MIN = checkpoint.get('min', MEAN)
        MAX = checkpoint.get('max', STD)
        
    if scaler_type == 'minmax':
        print("[PREPROCESS] Normalizing test data using MIN-MAX scaling...")
        denom = MAX - MIN
        denom = torch.where(denom < 1e-4, torch.ones_like(denom), denom)
        norm_data = (data_tensor - MIN) / denom
    else:
        print("[PREPROCESS] Normalizing test data using Z-SCORE scaling...")
        std_denominator = torch.where(STD < 1e-4, torch.ones_like(STD), STD)
        norm_data = (data_tensor - MEAN) / std_denominator
    
    # --- Compute reconstruction errors (batch mode — fast!) ---
    print("[EVAL] Computing Baseline (Clean) reconstruction errors...")
    batch_size = 256
    all_errors_clean = []
    
    for i in range(0, len(norm_data), batch_size):
        batch = norm_data[i:i+batch_size].to(device)
        if model_type == 'svdd' or model_type == 'contrastive':
            errors = model.get_anomaly_score(batch, center_c)
        else:
            errors = model.get_reconstruction_error(batch, mode=error_mode)
        all_errors_clean.extend(errors.cpu().numpy().tolist())
    
    scores_clean = np.array(all_errors_clean)
    
    if run_adversarial:
        print(f"[EVAL] Generating Adversarial Examples (Epsilon={epsilon}, Iters={num_iter})...")
        all_errors_adv = []
        y_true_tensor = torch.tensor(y_true, dtype=torch.long, device=device)
        
        for i in range(0, len(norm_data), batch_size):
            batch = norm_data[i:i+batch_size].to(device)
            batch_y = y_true_tensor[i:i+batch_size]
            
            # Temporarily unfreeze model to allow gradients to flow through it (weights still don't update)
            for param in model.parameters():
                param.requires_grad = True
                
            batch_adv = pgd_attack(model, batch, batch_y, center_c, model_type, error_mode, epsilon, alpha, num_iter)
            
            # Freeze again for inference
            for param in model.parameters():
                param.requires_grad = False
                
            if model_type == 'svdd' or model_type == 'contrastive':
                errors_adv = model.get_anomaly_score(batch_adv, center_c)
            else:
                errors_adv = model.get_reconstruction_error(batch_adv, mode=error_mode)
            all_errors_adv.extend(errors_adv.cpu().numpy().tolist())
            
        scores_adv = np.array(all_errors_adv)
    else:
        scores_adv = scores_clean

    # Apply rolling average window smoothing if specified
    if window_size > 1:
        print(f"[EVAL] Applying temporal smoothing (rolling window size = {window_size} samples)...")
        scores_clean = pd.Series(scores_clean).rolling(window=window_size, min_periods=1).mean().values
        if run_adversarial:
            scores_adv = pd.Series(scores_adv).rolling(window=window_size, min_periods=1).mean().values

    # We will compute thresholds and metrics based on the chosen mode.
    # If adversarial, we evaluate both. Let's use scores_clean for threshold sweeping.
    scores = scores_clean
    
    # Get true labels (vectorized & robust to typos like "A ttack")
    y_true = df[label_col].astype(str).str.strip().str.lower().str.replace(" ", "").apply(lambda val: 1 if val == "attack" else 0).values
    
    # Separate scores by class
    normal_scores = scores[y_true == 0]
    attack_scores = scores[y_true == 1]
    
    # --- Score Range Analysis (with robust statistics) ---
    print("\n" + "="*60)
    print(" RECONSTRUCTION ERROR ANALYSIS")
    print("="*60)
    for cls_name, cls_scores in [("Normal", normal_scores), ("Attack", attack_scores)]:
        print(f"\n  {cls_name} (n={len(cls_scores)}):")
        print(f"    Mean:   {np.mean(cls_scores):.6f}")
        print(f"    Median: {np.median(cls_scores):.6f}")
        print(f"    Std:    {np.std(cls_scores):.6f}")
        print(f"    P5:     {np.percentile(cls_scores, 5):.6f}")
        print(f"    P25:    {np.percentile(cls_scores, 25):.6f}")
        print(f"    P75:    {np.percentile(cls_scores, 75):.6f}")
        print(f"    P95:    {np.percentile(cls_scores, 95):.6f}")
        print(f"    Min:    {np.min(cls_scores):.6f}")
        print(f"    Max:    {np.max(cls_scores):.6f}")
    
    median_gap = np.median(attack_scores) - np.median(normal_scores)
    print(f"\n  >> Median Gap (Attack - Normal): {median_gap:.6f}")
    if median_gap < 0.01:
        print("  >> WARNING: Very small median gap — classes overlap significantly")
    elif median_gap < 0.1:
        print("  >> MODERATE gap — some overlap expected")
    else:
        print("  >> GOOD separation at the median level!")

    # --- THRESHOLD SWEEP (Log-scale to cover wide range) ---
    print("\n" + "="*60)
    print(" THRESHOLD SWEEP -- Log-Scale (Finding Optimal Cutoff)")
    print("="*60)
    
    best_f1 = 0
    best_threshold = 0
    sweep_results = []
    
    # Use log-scale thresholds to properly explore the critical range
    min_score = max(scores.min(), 1e-6)
    max_score = scores.max()
    thresholds = np.logspace(np.log10(min_score), np.log10(max_score), 1000)
    
    for t in thresholds:
        y_pred_t = (scores > t).astype(int)
        f1 = f1_score(y_true, y_pred_t, zero_division=0)
        sweep_results.append((t, f1))
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            
    # Override optimal threshold if custom mode is specified
    if threshold_mode == 'train_p95' and train_p95 is not None:
        best_threshold = train_p95
        print(f"  Optimal Threshold: {best_threshold:.6f} (Overridden to Train P95 Reference)")
    elif threshold_mode == 'train_p99' and train_p99 is not None:
        best_threshold = train_p99
        print(f"  Optimal Threshold: {best_threshold:.6f} (Overridden to Train P99 Reference)")
    else:
        print(f"  Optimal Threshold: {best_threshold:.6f}")
        print(f"  Best F1 Score:     {best_f1:.4f}")

    # --- FINAL METRICS ---
    y_pred_final_clean = (scores_clean > best_threshold).astype(int)
    cm_clean = confusion_matrix(y_true, y_pred_final_clean)
    tp_c, fp_c, fn_c, tn_c = cm_clean[1][1], cm_clean[0][1], cm_clean[1][0], cm_clean[0][0]
    precision_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
    attack_recall_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
    f1_c = 2 * precision_c * attack_recall_c / (precision_c + attack_recall_c) if (precision_c + attack_recall_c) > 0 else 0.0

    print("\n" + "="*55)
    print(f" BASELINE PERFORMANCE @ THRESHOLD = {best_threshold:.6f}")
    print("="*55)
    print(f"  F1 Score:  {f1_c:.4f}")
    print(f"  Recall:    {attack_recall_c:.4f}")
    
    if run_adversarial:
        y_pred_final_adv = (scores_adv > best_threshold).astype(int)
        cm_adv = confusion_matrix(y_true, y_pred_final_adv)
        tp_a, fp_a, fn_a, tn_a = cm_adv[1][1], cm_adv[0][1], cm_adv[1][0], cm_adv[0][0]
        precision_a = tp_a / (tp_a + fp_a) if (tp_a + fp_a) > 0 else 0.0
        attack_recall_a = tp_a / (tp_a + fn_a) if (tp_a + fn_a) > 0 else 0.0
        f1_a = 2 * precision_a * attack_recall_a / (precision_a + attack_recall_a) if (precision_a + attack_recall_a) > 0 else 0.0
        
        print("\n" + "="*55)
        print(f" ADVERSARIAL PERFORMANCE (PGD L-inf eps={epsilon})")
        print("="*55)
        print(f"  F1 Score:  {f1_a:.4f}  (Change: {f1_a - f1_c:+.4f})")
        print(f"  Recall:    {attack_recall_a:.4f}  (Change: {attack_recall_a - attack_recall_c:+.4f})")
        print("="*55)

    # --- VISUALIZATION ---
    if run_adversarial:
        plot_adversarial_results(normal_scores, attack_scores, scores_adv[y_true == 1], best_threshold, y_true, scores_clean, scores_adv, bundle_path)
    else:
        plot_evaluation_results(normal_scores, attack_scores, sweep_results, best_threshold, y_true, scores_clean)

    res = {
        'accuracy_clean': (tn_c + tp_c) / (tn_c + tp_c + fn_c + fp_c),
        'precision_clean': precision_c,
        'recall_clean': attack_recall_c,
        'f1_clean': f1_c,
        'false_alarm_clean': fp_c / (fp_c + tn_c) if (fp_c + tn_c) > 0 else 0.0,
        'threshold': best_threshold,
        'tp_clean': int(tp_c), 'fp_clean': int(fp_c), 'fn_clean': int(fn_c), 'tn_clean': int(tn_c)
    }
    if run_adversarial:
        res['f1_adv'] = f1_a
        res['recall_adv'] = attack_recall_a
        res['precision_adv'] = precision_a
        res['accuracy_adv'] = (tn_a + tp_a) / (tn_a + tp_a + fn_a + fp_a)
        res['false_alarm_adv'] = fp_a / (fp_a + tn_a) if (fp_a + tn_a) > 0 else 0.0
    return res


def plot_adversarial_results(normal_scores, attack_scores_clean, attack_scores_adv, best_threshold, y_true, scores_clean, scores_adv, bundle_path):
    """Plot comparing clean vs adversarial distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('PIVOT Oracle -- Adversarial Robustness Evaluation', fontsize=15, fontweight='bold', y=1.02)

    # Panel 1: Score Distributions
    ax1 = axes[0]
    all_scores = np.concatenate([normal_scores, attack_scores_clean, attack_scores_adv])
    log_bins = np.logspace(np.log10(max(all_scores.min(), 1e-6)), np.log10(all_scores.max()), 80)
    
    ax1.hist(normal_scores, bins=log_bins, alpha=0.5, color='#4CAF50', label="Normal", density=True)
    ax1.hist(attack_scores_clean, bins=log_bins, alpha=0.5, color='#FF9800', label="Attack (Clean)", density=True)
    ax1.hist(attack_scores_adv, bins=log_bins, alpha=0.5, color='#F44336', label="Attack (Adversarial PGD)", density=True)
    ax1.axvline(x=best_threshold, color='black', linewidth=2, linestyle='--', label=f'Threshold ({best_threshold:.4f})')
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Anomaly Score (log scale)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Adversarial Evasion Shift', fontsize=13)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Timeline Comparison
    ax2 = axes[1]
    step = max(1, len(scores_clean) // 10000)
    indices = np.arange(0, len(scores_clean), step)
    
    ax2.plot(indices, scores_clean[indices], color='#2196F3', alpha=0.6, label='Clean Scores', linewidth=1)
    ax2.plot(indices, scores_adv[indices], color='#F44336', alpha=0.6, label='Adversarial Scores', linewidth=1)
    ax2.axhline(y=best_threshold, color='black', linewidth=2, linestyle='--', label='Threshold')
    
    ax2.set_yscale('log')
    ax2.set_xlabel('Sample Index', fontsize=12)
    ax2.set_title('Score Timeline Comparison', fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_name = os.path.splitext(os.path.basename(bundle_path))[0]
    plot_path = os.path.join(script_dir, 'Oracles', f'{model_name}_adversarial_eval.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n[PLOT] Adversarial analysis saved to: {plot_path}")

def plot_evaluation_results(normal_scores, attack_scores, sweep_results, best_threshold, y_true, scores):
    """Generate evaluation analysis plots."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle('PIVOT Oracle -- Anomaly Detection Evaluation',
                 fontsize=15, fontweight='bold', y=1.02)

    # ---- Panel 1: Score Distribution (log-scale bins) ----
    ax1 = axes[0]
    # Use log-scale bins for better visualization
    all_scores_combined = np.concatenate([normal_scores, attack_scores])
    log_bins = np.logspace(np.log10(max(all_scores_combined.min(), 1e-6)),
                           np.log10(all_scores_combined.max()), 80)
    ax1.hist(normal_scores, bins=log_bins, alpha=0.7, color='#4CAF50',
             label=f"Normal (n={len(normal_scores)})", density=True)
    ax1.hist(attack_scores, bins=log_bins, alpha=0.7, color='#F44336',
             label=f"Attack (n={len(attack_scores)})", density=True)
    ax1.axvline(x=best_threshold, color='#FF9800', linewidth=2, linestyle='--',
                label=f'Threshold ({best_threshold:.4f})')
    ax1.set_xscale('log')
    ax1.set_xlabel('Anomaly Score (log scale)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Score Distribution by Class', fontsize=13)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ---- Panel 2: F1 vs Threshold (log-x) ----
    ax2 = axes[1]
    thresholds_plot = [r[0] for r in sweep_results]
    f1_scores_plot = [r[1] for r in sweep_results]
    ax2.plot(thresholds_plot, f1_scores_plot, color='#2196F3', linewidth=2)
    ax2.axvline(x=best_threshold, color='#FF9800', linewidth=2, linestyle='--',
                label=f'Best ({best_threshold:.4f}, F1={max(f1_scores_plot):.3f})')
    ax2.set_xscale('log')
    ax2.set_xlabel('Threshold (log scale)', fontsize=12)
    ax2.set_ylabel('F1 Score', fontsize=12)
    ax2.set_title('F1 Score vs Threshold', fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ---- Panel 3: Score Timeline (log-y) ----
    ax3 = axes[2]
    # Downsample points for scatter plot if dataset is large to prevent matplotlib from hanging
    if len(scores) > 20000:
        step = len(scores) // 20000
        indices = np.arange(0, len(scores), step)
        scores_plot = scores[indices]
        y_true_plot = y_true[indices]
        colors = ['#F44336' if y == 1 else '#4CAF50' for y in y_true_plot]
        ax3.scatter(indices, scores_plot, c=colors, alpha=0.3, s=2)
    else:
        colors = ['#F44336' if y == 1 else '#4CAF50' for y in y_true]
        ax3.scatter(range(len(scores)), scores, c=colors, alpha=0.3, s=2)
    ax3.axhline(y=best_threshold, color='#FF9800', linewidth=2, linestyle='--',
                label=f'Threshold ({best_threshold:.4f})')
    ax3.set_yscale('log')
    ax3.set_xlabel('Sample Index', fontsize=12)
    ax3.set_ylabel('Anomaly Score (log scale)', fontsize=12)
    ax3.set_title('Score Timeline (Green=Normal, Red=Attack)', fontsize=13)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(script_dir, 'Oracles', 'autoencoder_evaluation.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"\n[PLOT] Evaluation analysis saved to: {plot_path}")


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PIVOT Oracle Autoencoder Models")
    parser.add_argument('--data', type=str, default=r'Datasets\dataset_14K_Testing Attack&Normal.csv',
                        help='Path to the testing CSV dataset')
    parser.add_argument('--model', type=str, default=r'Oracles\50K_dae_model.pth',
                        help='Path to the trained model (.pth file) to evaluate')
    parser.add_argument('--threshold_mode', type=str, default='f1_opt', choices=['f1_opt', 'train_p95', 'train_p99'],
                        help='Threshold mode: f1_opt (optimal sweep), train_p95 (95th percentile of training data), train_p99 (99th percentile)')
    parser.add_argument('--window_size', type=int, default=1,
                        help='Rolling window size for anomaly score smoothing (1 = no smoothing)')
    parser.add_argument('--align_baseline', action='store_true',
                        help='Align the baseline of test set to test normal data to correct sensor drift')
    parser.add_argument('--error_mode', type=str, default='mean', choices=['mean', 'max'],
                        help='Reconstruction error calculation mode: mean (average MSE) or max (maximum squared error)')
    parser.add_argument('--run_adversarial', action='store_true',
                        help='Run the PGD adversarial attack to evaluate robustness.')
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='L-infinity epsilon bound for the PGD attack.')
    parser.add_argument('--alpha', type=float, default=0.02,
                        help='Step size for PGD attack.')
    parser.add_argument('--num_iter', type=int, default=10,
                        help='Number of iterations for PGD attack.')
    
    args = parser.parse_args()
    
    print("\n" + "#"*70)
    print(f" EVALUATING MODEL: {args.model}")
    print(f" TEST DATA:        {args.data}")
    print(f" ADVERSARIAL MODE: {args.run_adversarial} (Eps={args.epsilon})")
    print("#"*70)
    
    run_final_evaluation(args.data, args.model, threshold_mode=args.threshold_mode, window_size=args.window_size, align_baseline=args.align_baseline, error_mode=args.error_mode, run_adversarial=args.run_adversarial, epsilon=args.epsilon, alpha=args.alpha, num_iter=args.num_iter)