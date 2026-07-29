import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
import matplotlib
# Use Agg backend for headless/non-interactive plotting
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# --- Autoencoder Architecture ---
# The model learns to RECONSTRUCT normal data through a bottleneck.
# Normal data -> low reconstruction error
# Attack data -> high reconstruction error (never seen during training)

class OracleAutoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=32, dropout_rate=0.2):
        super().__init__()
        
        # Encoder: compress input to bottleneck
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, bottleneck_dim),  # Bottleneck
        )
        
        # Decoder: reconstruct from bottleneck
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128, input_dim),  # Reconstruct original dimensions
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def get_reconstruction_error(self, x):
        """Per-sample MSE reconstruction error."""
        with torch.no_grad():
            reconstructed = self.forward(x)
            error = torch.mean((x - reconstructed) ** 2, dim=1)
        return error




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
        
    def get_reconstruction_error(self, x):
        """Deterministic MSE for evaluation"""
        with torch.no_grad():
            h = self.encoder_base(x)
            mu = self.fc_mu(h)
            reconstructed = self.decoder(mu)
            error = torch.mean((x - reconstructed) ** 2, dim=1)
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
        # L2 Normalize so all vectors live on unit hypersphere
        return F.normalize(z, dim=1)

    def get_anomaly_score(self, x, c):
        """Cosine distance (1 - Cosine Similarity) from output to the learned center c."""
        with torch.no_grad():
            z = self.forward(x)
            c = c.unsqueeze(0) if c.dim() == 1 else c
            c = F.normalize(c, dim=1) # Ensure center is normalized
            # Cosine similarity
            sim = F.cosine_similarity(z, c)
            dist = 1.0 - sim # Distance is 1 - similarity
        return dist

def compute_center_c(train_loader, model, device):
    """Compute the mean representation of the normal data after training."""
    model.eval()
    n_samples = 0
    c = torch.zeros(model.projector[-1].out_features, device=device)

    with torch.no_grad():
        for batch in train_loader:
            x = batch[0].to(device)
            outputs = model(x)
            n_samples += outputs.shape[0]
            c += torch.sum(outputs, dim=0)

    c /= n_samples
    c = F.normalize(c, dim=0) # Normalize the center itself
    return c

class OracleDeepSVDD(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=32, dropout_rate=0.2):
        super().__init__()
        
        # Deep SVDD Encoder (No bias to prevent hypersphere collapse)
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

def init_center_c(train_loader, model, device, eps=0.1):
    """Initialize hypersphere center c as the mean from an initial forward pass."""
    model.eval()
    n_samples = 0
    c = torch.zeros(model.encoder[-1].out_features, device=device)

    with torch.no_grad():
        for batch in train_loader:
            x = batch[0].to(device)
            outputs = model(x)
            n_samples += outputs.shape[0]
            c += torch.sum(outputs, dim=0)

    c /= n_samples
    
    # If c_i is too close to 0, set to +-eps to avoid degenerate trivial solutions
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps
    return c

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
        
    def get_reconstruction_error(self, x):
        with torch.no_grad():
            reconstructed = self.forward(x)
            error = torch.mean((x - reconstructed) ** 2, dim=1)
        return error



def train_oracle_autoencoder(data_path, output_name='Oracles//autoencoder_model.pth', bottleneck_dim=32, model_type='dae', val_split=0.2, patience=30, max_epochs=300, scaler='minmax', use_cls_bottleneck=True, noise_factor=0.0, no_downsample=False, batch_size=256):
    """
    Train an Autoencoder Oracle on normal-only data.
    
    The autoencoder learns to reconstruct normal patterns through a bottleneck.
    Attack data, which deviates from normal patterns, will have higher
    reconstruction error — enabling anomaly detection.
    """
    # Setup Hyperparameters based on type
    is_dae = (model_type == 'dae')
    is_svdd = (model_type == 'svdd')
    is_vae = (model_type == 'vae')
    is_contrastive = (model_type == 'contrastive')
    is_transformer = (model_type == 'transformer')
    
    dropout_rate = 0.3 if is_dae else 0.2
    weight_decay = 1e-3 if is_dae else 1e-4
    noise_factor = noise_factor if noise_factor > 0.0 else (0.05 if is_dae else 0.0)
    
    # Resolve relative paths dynamically to script folder if not found in CWD
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(data_path):
        alt_data_path = os.path.join(script_dir, data_path)
        if os.path.exists(alt_data_path):
            data_path = alt_data_path
            
    if not os.path.isabs(output_name):
        dir_name = os.path.dirname(output_name)
        if dir_name:
            alt_dir = os.path.join(script_dir, dir_name)
            os.makedirs(alt_dir, exist_ok=True)
            output_name = os.path.join(script_dir, output_name)

    print(f"\n[INIT] Loading data from: {data_path}")
    print(f"[INIT] Model Type: {model_type.upper()}")
    
    # Load dataset
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    
    # If the dataset is large (e.g. full SWaT dataset), downsample for CPU training efficiency
    if len(df) > 100000 and not no_downsample:
        print(f"[PREPROCESS] Downsampling dataset from {len(df)} to {len(df)//5} samples for CPU efficiency...")
        df = df.iloc[::5].reset_index(drop=True)
        
    # Exclude non-numeric columns
    numeric_df = df.select_dtypes(include=[np.number]).fillna(df.mean(numeric_only=True))
    
    # Identify and drop constant columns (Phase 1)
    stds = numeric_df.std(axis=0)
    kept_columns = stds[stds >= 1e-4].index.tolist()
    dropped_columns = stds[stds < 1e-4].index.tolist()
    if len(dropped_columns) > 0:
        print(f"[PREPROCESS] Dropped constant columns: {dropped_columns}")
    print(f"[PREPROCESS] Kept {len(kept_columns)} columns out of {len(numeric_df.columns)}")
    
    numeric_df = numeric_df[kept_columns]
    data_tensor = torch.tensor(numeric_df.values, dtype=torch.float32)
    
    # Preprocess scaling based on selection (Phase 2)
    if scaler == 'minmax':
        print("[PREPROCESS] Using MIN-MAX scaling normalization...")
        MEAN = data_tensor.min(dim=0)[0]
        STD = data_tensor.max(dim=0)[0]
        # Avoid division by zero if min and max are equal (though constant columns are already dropped)
        denom = STD - MEAN
        denom = torch.where(denom < 1e-4, torch.ones_like(denom), denom)
        norm_data = (data_tensor - MEAN) / denom
    else:
        print("[PREPROCESS] Using Z-SCORE scaling normalization...")
        # Calculate dataset statistics for normalization
        MEAN = data_tensor.mean(dim=0)
        STD = data_tensor.std(dim=0)
        # Normalize data robustly (handling zero std to prevent value explosion)
        std_denominator = torch.where(STD < 1e-4, torch.ones_like(STD), STD)
        norm_data = (data_tensor - MEAN) / std_denominator
    
    # Split into train/validation
    dataset = TensorDataset(norm_data)
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    input_dim = data_tensor.shape[1]
    
    print(f"[DATA] Total: {len(dataset)} | Train: {train_size} | Val: {val_size}")
    print(f"[DATA] Features: {input_dim} | Bottleneck: {bottleneck_dim}")
    print(f"[DATA] Compression ratio: {input_dim}:{bottleneck_dim} ({input_dim/bottleneck_dim:.1f}x)")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INIT] Using device: {device}")

    if is_svdd:
        model = OracleDeepSVDD(input_dim=input_dim, bottleneck_dim=bottleneck_dim, dropout_rate=dropout_rate).to(device)
        print("[INIT] Initializing Deep SVDD Center 'c'...")
        center_c = init_center_c(train_loader, model, device)
    elif is_vae:
        model = OracleVAE(input_dim=input_dim, bottleneck_dim=bottleneck_dim, dropout_rate=dropout_rate).to(device)
        center_c = None
    elif is_contrastive:
        model = OracleContrastive(input_dim=input_dim, bottleneck_dim=bottleneck_dim, dropout_rate=dropout_rate).to(device)
        center_c = None
    elif is_transformer:
        model = OracleTransformerAutoencoder(input_dim=input_dim, bottleneck_dim=bottleneck_dim, dropout_rate=dropout_rate, use_cls_bottleneck=use_cls_bottleneck).to(device)
        center_c = None
    else:
        model = OracleAutoencoder(input_dim=input_dim, bottleneck_dim=bottleneck_dim, dropout_rate=dropout_rate).to(device)
        center_c = None
        
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    criterion = nn.MSELoss()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] Total parameters: {total_params:,}")

    # --- Early Stopping Setup ---
    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0
    best_epoch = 0

    train_losses = []
    val_losses = []

    print(f"\n[START] Training Autoencoder Oracle (max {max_epochs} epochs, patience={patience})...")
    
    for epoch in range(max_epochs + 1):
        # ---- Training Phase ----
        model.train()
        epoch_train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            x = batch[0].to(device)
            
            if is_svdd:
                outputs = model(x)
                loss = torch.mean(torch.sum((outputs - center_c) ** 2, dim=1))
            elif is_vae:
                reconstructed, mu, logvar = model(x)
                recon_loss = criterion(reconstructed, x)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                kl_loss = kl_loss / (batch[0].shape[0] * input_dim)
                loss = recon_loss + (0.001 * kl_loss)
            elif is_contrastive:
                temp = 0.1
                view1 = x + torch.randn_like(x) * 0.05
                view2 = x + torch.randn_like(x) * 0.05
                z1 = model(view1)
                z2 = model(view2)
                bsz = x.shape[0]
                out = torch.cat([z1, z2], dim=0)
                sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temp)
                mask = (torch.ones_like(sim_matrix) - torch.eye(2 * bsz, device=sim_matrix.device)).bool()
                sim_matrix = sim_matrix.masked_select(mask).view(2 * bsz, -1)
                pos_sim = torch.exp(torch.sum(z1 * z2, dim=-1) / temp)
                pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
                loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
            else:
                if is_dae or (is_transformer and noise_factor > 0.0):
                    noisy_x = x + torch.randn_like(x) * noise_factor
                    reconstructed = model(noisy_x)
                else:
                    reconstructed = model(x)
                loss = criterion(reconstructed, x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # ---- Validation Phase ----
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                if is_svdd:
                    outputs = model(x)
                    loss = torch.mean(torch.sum((outputs - center_c) ** 2, dim=1))
                elif is_vae:
                    reconstructed, mu, logvar = model(x)
                    recon_loss = criterion(reconstructed, x)
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    kl_loss = kl_loss / (batch[0].shape[0] * input_dim)
                    loss = recon_loss + (0.001 * kl_loss)
                elif is_contrastive:
                    temp = 0.1
                    view1 = x + torch.randn_like(x) * 0.05
                    view2 = x + torch.randn_like(x) * 0.05
                    z1 = model(view1)
                    z2 = model(view2)
                    bsz = x.shape[0]
                    out = torch.cat([z1, z2], dim=0)
                    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temp)
                    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * bsz, device=sim_matrix.device)).bool()
                    sim_matrix = sim_matrix.masked_select(mask).view(2 * bsz, -1)
                    pos_sim = torch.exp(torch.sum(z1 * z2, dim=-1) / temp)
                    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
                    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
                else:
                    reconstructed = model(x)
                    loss = criterion(reconstructed, x)
                epoch_val_loss += loss.item()
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # ---- Early Stopping Check ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        status = ""
        if epochs_without_improvement > 0:
            status = f" | No improvement for {epochs_without_improvement} epochs"
        print(f"Epoch {epoch:3d} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | LR: {current_lr:.6f}{status}", flush=True)
        
        if epochs_without_improvement >= patience:
            print(f"\n[EARLY STOP] No improvement for {patience} epochs. Stopping at epoch {epoch}.")
            print(f"[EARLY STOP] Best epoch: {best_epoch} | Best Val Loss: {best_val_loss:.6f}")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"[RESTORE] Loaded best model from epoch {best_epoch}")
        
    if is_contrastive:
        print("[INIT] Computing final Contrastive Center 'c' for anomaly scoring...")
        center_c = compute_center_c(train_loader, model, device)

    # --- Compute training set reconstruction error stats (for threshold reference) ---
    model.eval()
    all_train_errors = []
    with torch.no_grad():
        for batch in train_loader:
            x = batch[0].to(device)
            if is_svdd or is_contrastive:
                errors = model.get_anomaly_score(x, center_c)
            else:
                errors = model.get_reconstruction_error(x)
            all_train_errors.extend(errors.cpu().numpy().tolist())
    
    train_error_mean = np.mean(all_train_errors)
    train_error_std = np.std(all_train_errors)
    train_error_p95 = np.percentile(all_train_errors, 95)
    train_error_p99 = np.percentile(all_train_errors, 99)
    
    print(f"\n[STATS] Training Reconstruction Error:")
    print(f"  Mean: {train_error_mean:.6f}")
    print(f"  Std:  {train_error_std:.6f}")
    print(f"  P95:  {train_error_p95:.6f} (suggested threshold)")
    print(f"  P99:  {train_error_p99:.6f} (strict threshold)")

    checkpoint_dict = {
        'model_state': model.state_dict(),
        'model_type': model_type,
        'scaler_type': scaler,
        'mean': MEAN, 'std': STD,  # MEAN holds MIN and STD holds MAX if minmax
        'min': MEAN, 'max': STD,
        'input_dim': input_dim,
        'bottleneck_dim': bottleneck_dim,
        'use_cls_bottleneck': use_cls_bottleneck,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'train_error_mean': train_error_mean,
        'train_error_std': train_error_std,
        'train_error_p95': train_error_p95,
        'train_error_p99': train_error_p99,
        'total_epochs_trained': len(train_losses),
        'features': kept_columns
    }
    if is_svdd or is_contrastive:
        checkpoint_dict['center_c'] = center_c
        
    torch.save(checkpoint_dict, output_name)
    print(f"[DONE] Oracle Model Saved!")

    # --- Training Visualization ---
    plot_training_curves(train_losses, val_losses, best_epoch, all_train_errors, output_name)


def plot_training_curves(train_losses, val_losses, best_epoch, train_errors, output_name):
    """Generate training analysis plots."""
    plot_path = os.path.splitext(output_name)[0] + '_training_loss.png'
    
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle('Autoencoder Oracle — Training Analysis',
                 fontsize=15, fontweight='bold', y=1.02)

    epochs = list(range(len(train_losses)))

    # ---- Panel 1: Linear Scale ----
    ax1 = axes[0]
    ax1.plot(epochs, train_losses, color='#2196F3', linewidth=1.8, alpha=0.9, label='Train Loss')
    ax1.plot(epochs, val_losses, color='#FF9800', linewidth=1.8, alpha=0.9, label='Val Loss', linestyle='--')
    ax1.fill_between(epochs, train_losses, alpha=0.1, color='#2196F3')
    ax1.fill_between(epochs, val_losses, alpha=0.1, color='#FF9800')
    ax1.axvline(x=best_epoch, color='#4CAF50', linestyle=':', linewidth=2, alpha=0.7, label=f'Best Epoch ({best_epoch})')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Reconstruction MSE', fontsize=12)
    ax1.set_title('Train vs Val Loss (Linear)', fontsize=13)
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, len(train_losses) - 1)

    # ---- Panel 2: Log Scale ----
    ax2 = axes[1]
    ax2.plot(epochs, train_losses, color='#2196F3', linewidth=1.8, alpha=0.9, label='Train Loss')
    ax2.plot(epochs, val_losses, color='#FF9800', linewidth=1.8, alpha=0.9, label='Val Loss', linestyle='--')
    ax2.axvline(x=best_epoch, color='#4CAF50', linestyle=':', linewidth=2, alpha=0.7, label=f'Best Epoch ({best_epoch})')
    ax2.set_yscale('log')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Reconstruction MSE (log)', fontsize=12)
    ax2.set_title('Train vs Val Loss (Log)', fontsize=13)
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_xlim(0, len(train_losses) - 1)

    # ---- Panel 3: Training Error Distribution ----
    ax3 = axes[2]
    ax3.hist(train_errors, bins=100, alpha=0.7, color='#4CAF50', density=True, label='Normal (train)')
    p95 = np.percentile(train_errors, 95)
    p99 = np.percentile(train_errors, 99)
    ax3.axvline(x=p95, color='#FF9800', linewidth=2, linestyle='--', label=f'P95 ({p95:.4f})')
    ax3.axvline(x=p99, color='#F44336', linewidth=2, linestyle='--', label=f'P99 ({p99:.4f})')
    ax3.set_xlabel('Reconstruction Error (MSE)', fontsize=12)
    ax3.set_ylabel('Density', fontsize=12)
    ax3.set_title('Normal Data Error Distribution', fontsize=13)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Generalization gap annotation
    gap = [v - t for t, v in zip(train_losses, val_losses)]
    avg_gap = np.mean(gap[-20:]) if len(gap) >= 20 else np.mean(gap)
    if avg_gap > 0.01:
        diagnosis = "OVERFITTING detected"
        diag_color = '#F44336'
    elif avg_gap < -0.01:
        diagnosis = "UNUSUAL gap"
        diag_color = '#FF9800'
    else:
        diagnosis = "HEALTHY"
        diag_color = '#4CAF50'
    
    ax1.text(0.5, 0.95, diagnosis, transform=ax1.transAxes, fontsize=10,
             fontweight='bold', color=diag_color, ha='center', va='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=diag_color, alpha=0.8))

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"[PLOT] Training analysis saved to: {plot_path}")


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PIVOT Oracle Autoencoder")
    parser.add_argument('--data', type=str, default=r'Datasets\Trainning_normal_only.csv',
                        help='Path to the training CSV dataset (Normal data only)')
    parser.add_argument('--output', type=str, default=r'Oracles\oracle.pth',
                        help='Path where the trained model will be saved')
    parser.add_argument('--bottleneck', type=int, default=32,
                        help='Dimension of the autoencoder bottleneck')
    parser.add_argument('--type', type=str, default='dae', choices=['standard', 'dae', 'svdd', 'vae', 'contrastive', 'transformer'],
                        help='Type of model to train: "standard", "dae", "svdd", "vae", "contrastive", or "transformer"')
    parser.add_argument('--scaler', type=str, default='minmax', choices=['zscore', 'minmax'],
                        help='Feature scaling method: "zscore" (mean/std) or "minmax" (0 to 1 scaling)')
    parser.add_argument('--use_cls', type=str, default='true', choices=['true', 'false'],
                        help='Whether to use CLS token bottleneck for Transformer (default: "true")')
    parser.add_argument('--noise', type=float, default=0.0,
                        help='Optional noise factor for DAE or Transformer (default: 0.0)')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Maximum number of training epochs')
    parser.add_argument('--patience', type=int, default=30,
                        help='Early stopping patience')
    parser.add_argument('--no_downsample', action='store_true',
                        help='Disable automatic downsampling of large datasets')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training')
    
    args = parser.parse_args()

    model_type = args.type.lower()
    use_cls = args.use_cls.lower() == 'true'

    print("\n" + "="*50)
    print(f" STARTING TRAINING SESSION")
    print(f"  -> DATASET:    {args.data}")
    print(f"  -> OUTPUT:     {args.output}")
    print(f"  -> BOTTLENECK: {args.bottleneck}")
    print(f"  -> TYPE:       {model_type.upper()}")
    print(f"  -> SCALER:     {args.scaler.upper()}")
    print(f"  -> USE CLS:    {use_cls}")
    print(f"  -> NOISE FACTOR: {args.noise}")
    print(f"  -> EPOCHS:     {args.epochs}")
    print(f"  -> PATIENCE:   {args.patience}")
    print(f"  -> NO DOWNSAMPLE: {args.no_downsample}")
    print(f"  -> BATCH SIZE: {args.batch_size}")
    print("="*50)
    
    train_oracle_autoencoder(args.data, 
                             output_name=args.output, 
                             bottleneck_dim=args.bottleneck,
                             model_type=model_type,
                             scaler=args.scaler,
                             use_cls_bottleneck=use_cls,
                             noise_factor=args.noise,
                             max_epochs=args.epochs,
                             patience=args.patience,
                             no_downsample=args.no_downsample,
                             batch_size=args.batch_size)