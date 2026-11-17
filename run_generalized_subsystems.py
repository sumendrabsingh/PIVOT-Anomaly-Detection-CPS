import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
import json
import argparse

import label_free as lfp
from evaluate_events import evaluate, to_binary

# --- Config ---
AGGREGATION_MODE = 'max'
ERROR_MODE = 'max'
WINDOW_SIZE = 25
SEQ_LEN = 1
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

# GenericPINNLoss above is kept only to reproduce models trained before 2026-10-05 ('legacy'). It is
# mis-specified: the preprocessed CSVs are already z-scored, so undoing this model's own scaling lands in
# z-scores, not physical units, and ReLU(-x) penalizes every below-mean flow/level (27% of normal FIT101,
# 66% of LIT101) -- a bias toward the mean, not non-negativity.

# Actuator -> flow interlocks inside one SWaT stage: a closed valve (one-hot state 1; 0 = transitioning,
# 2 = open) or a stopped pump (raw state 1 = off, 2 = on) means ~zero flow. MV301 -> FIT301 is NOT a rule:
# MV301 is the UF backwash valve, closed 98.8% of the time while FIT301 flows (~1.86 m^3/h).
SWAT_INTERLOCKS = [('MV101_1', 'FIT101', 'valve'), ('MV201_1', 'FIT201', 'valve'),
                   ('P402', 'FIT401', 'pump'), ('P501', 'FIT501', 'pump'), ('P602', 'FIT601', 'pump')]
NONNEG_PREFIXES = ('FIT', 'LIT', 'DPIT', 'PIT')


class PhysicsLoss(nn.Module):
    """MSE + lambda * physics, with physics evaluated in PHYSICAL units.

    model space --(this model's mean/std)--> preprocessed z-scores --(raw_mean/raw_std from
    preprocessing_meta.json)--> physical units. Non-negativity on flows, levels and pressures, in units
    of each sensor's raw SD; interlocks gate on the OBSERVED actuator state, so the model cannot dodge
    the penalty by reconstructing the actuator."""

    def __init__(self, features, mean, std, raw_mean, raw_std, interlocks=False, lambda_phys=0.1):
        super().__init__()
        self.mse = nn.MSELoss(reduction='mean')
        self.lambda_phys = lambda_phys
        rm = torch.tensor([float(raw_mean.get(f, 0.0)) for f in features], device=mean.device)
        rs = torch.tensor([float(raw_std.get(f, 1.0)) for f in features], device=mean.device)
        self.scale, self.shift, self.raw_std = std * rs, mean * rs + rm, rs
        self.nonneg = [i for i, f in enumerate(features) if f.startswith(NONNEG_PREFIXES) and f in raw_mean]
        self.rules = [(features.index(a), features.index(f), kind) for a, f, kind in SWAT_INTERLOCKS
                      if interlocks and a in features and f in features]

    def forward(self, x_hat, x):
        mse_loss = self.mse(x_hat, x)
        if self.lambda_phys <= 0 or not (self.nonneg or self.rules):
            return mse_loss
        x_hat_phys = x_hat * self.scale + self.shift
        x_phys = x * self.scale + self.shift
        phys_loss = torch.tensor(0.0, device=x.device)
        for i in self.nonneg:
            phys_loss += torch.mean(torch.relu(-x_hat_phys[:, i] / self.raw_std[i]))
        for a, f, kind in self.rules:
            # observed states are discrete (valve one-hot 0/1, pump 1 = off / 2 = on): use exact
            # indicators; a soft sigmoid gate leaks ~0.7% at 'on' and penalizes normal running flow
            off = x_phys[:, a] if kind == 'valve' else (x_phys[:, a] < 1.5).float()
            phys_loss += torch.mean(off * (x_hat_phys[:, f] / self.raw_std[f]) ** 2)
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
    cols = list(columns)
    matched = [c for c in cols
               if any(c == f or c.startswith(f + '_') for f in features)]
    return matched


def train_subsystem(name, features, train_data_path, out_dir, seed=None, physics='legacy',
                    meta_path='Datasets/preprocessing_meta.json'):
    """physics: 'none' (lambda = 0), 'nonneg' / 'interlock' (PhysicsLoss, SWaT preprocessing meta
    required), or 'legacy' (the mis-specified GenericPINNLoss, kept to reproduce older models)."""
    print(f"\nTraining Subsystem Autoencoder for {name} with {len(features)} features")
    if seed is not None:
        # v1 seeded only the train/val split (generator below); weight init and DataLoader
        # shuffling were left unseeded, so every run drifted by ~1 F1 point. Pin all three.
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
    df = pd.read_csv(train_data_path)
    df.columns = df.columns.str.strip()
    
    valid_features = expand_features(features, df.columns)
    if len(valid_features) == 0:
        print(f"Skipping {name}: No matching features found in dataset.")
        return
    dropped = [f for f in features
               if not any(c == f or c.startswith(f + '_') for c in df.columns)]
    if dropped:
        print(f"  [WARNING] {name}: no column for {dropped} -- excluded from the model")
        
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
    
    # A final batch of one sample breaks BatchNorm in training mode; drop it only in that case (no model
    # trained before 2026-10-07 had one, since it would have crashed), so every earlier run is unchanged.
    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True, drop_last=train_size % 4096 == 1)
    val_loader = DataLoader(val_dataset, batch_size=4096, shuffle=False)
    
    input_dim = data_tensor.shape[1]
    bottleneck_dim = max(1, input_dim // 2)
    
    model = SubsystemAutoencoder(input_dim, bottleneck_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    if physics in ('nonneg', 'interlock'):
        with open(meta_path) as f:
            meta = json.load(f)
        criterion = PhysicsLoss(valid_features, mean.to(DEVICE), std.to(DEVICE), meta['mean_vals'],
                                meta['std_vals'], interlocks=(physics == 'interlock'))
    else:
        criterion = GenericPINNLoss(valid_features, mean.to(DEVICE), std.to(DEVICE),
                                    lambda_phys=0.0 if physics == 'none' else 0.1)
    
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
        'bottleneck_dim': bottleneck_dim,
        'physics': physics
    }
    
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"subsystem_{name}.pth")
    torch.save(checkpoint, out_path)
    print(f"  Saved subsystem model to {out_path} (Best Val: {best_val_loss:.6f})")

def _load_subsystem_model(model_path):
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    input_dim = checkpoint['input_dim']
    bottleneck_dim = checkpoint['bottleneck_dim']
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
    return model, checkpoint


def _reconstruction_errors(model, checkpoint, df, y_true=None, run_adversarial=False,
                           epsilon=0.1, alpha=0.01, num_iter=10):
    features = checkpoint['features']
    data_tensor = torch.tensor(df[features].fillna(df[features].mean()).values, dtype=torch.float32)

    # Bundles may store per-feature mean/std as lists (v1 per-timestep bundles) or omit
    # them entirely (windowed arm models); coerce either into a broadcastable tensor.
    MEAN = torch.as_tensor(checkpoint.get('mean', 0.0), dtype=torch.float32)
    STD = torch.as_tensor(checkpoint.get('std', 1.0), dtype=torch.float32)
    STD = torch.where(STD == 0, torch.ones_like(STD), STD)
    norm_data = (data_tensor - MEAN) / STD

    batch_size = 4096
    errors_list = []
    y_true_tensor = None if y_true is None else torch.tensor(y_true, dtype=torch.long, device=DEVICE)
    for i in range(0, len(norm_data), batch_size):
        batch = norm_data[i:i + batch_size].to(DEVICE)
        if run_adversarial:
            for param in model.parameters():
                param.requires_grad = True
            batch = pgd_attack_subsystem(model, batch, y_true_tensor[i:i + batch_size], epsilon, alpha, num_iter)
            for param in model.parameters():
                param.requires_grad = False
        errors_list.append(model.get_reconstruction_error(batch, mode=ERROR_MODE).cpu().numpy())
    return np.concatenate(errors_list)


def _compute_subsystem_scores(subsystems, df_eval, y_true, out_dir, calibration='train', df_train=None,
                              run_adversarial=False, epsilon=0.1, alpha=0.01, num_iter=10):
    """Per-subsystem standardized, smoothed scores.

    calibration='train'  : errors standardized on the TRAINING errors (label-free); also returns the
                           training scores so thresholds can be calibrated on them.
    calibration='oracle' : errors standardized on test rows labelled normal (the v1/v2 paper protocol;
                           leaks test labels -- reference only). No training scores are returned.
    """
    scores, train_scores = {}, {}
    for name in subsystems:
        model_path = os.path.join(out_dir, f"subsystem_{name}.pth")
        if not os.path.exists(model_path):
            print(f"  [WARNING] no model for subsystem '{name}' at {model_path} -- skipping it")
            continue
        model, checkpoint = _load_subsystem_model(model_path)
        raw = _reconstruction_errors(model, checkpoint, df_eval, y_true, run_adversarial, epsilon, alpha, num_iter)
        if calibration == 'train':
            ref = _reconstruction_errors(model, checkpoint, df_train)
            mean_err, std_err = ref.mean(), ref.std()
            train_scores[name] = lfp.smooth((ref - mean_err) / (std_err + 1e-6), WINDOW_SIZE)
        else:
            ref = raw[y_true == 0] if (y_true == 0).any() else raw
            mean_err, std_err = ref.mean(), ref.std()
        scores[name] = lfp.smooth((raw - mean_err) / (std_err + 1e-6), WINDOW_SIZE)
    return scores, (train_scores if calibration == 'train' else None)


def evaluate_subsystems(subsystems, test_data_path, out_dir, calibration='train', train_data_path=None,
                        decision='orvote', analyzers='keep', run_adversarial=False, export_predictions=None):
    """Evaluate the subsystem array.

    decision='orvote' : per-subsystem threshold P99.8, OR across subsystems
    decision='dual'   : max across subsystems; Tier 1 > P99.9, Tier 2 > P99 for 50 samples (causal)
    Thresholds come from the training scores (calibration='train') or from test rows labelled normal
    (calibration='oracle', reference only). analyzers='split' ORs in the analyzer arm; the subsystem
    models must then have been trained with --analyzers split.
    """
    df_eval = pd.read_csv(test_data_path)
    df_eval.columns = df_eval.columns.str.strip()
    label_col = 'Label' if 'Label' in df_eval.columns else 'Normal/Attack'
    y_true = to_binary(df_eval[label_col]) if label_col in df_eval.columns else np.zeros(len(df_eval), dtype=int)

    df_train = None
    if calibration == 'train':
        df_train = pd.read_csv(train_data_path)
        df_train.columns = df_train.columns.str.strip()
        df_eval, fired = lfp.fault_rule(df_train, df_eval)
        if fired:
            print(f"  [FAULT RULE] replaced impossible values in: "
                  f"{ {c: f'{v:.1%}' for c, v in fired.items()} }")
    else:
        if analyzers == 'split' or decision == 'dual':
            raise SystemExit("--calibration oracle supports only --decision orvote --analyzers keep")
        print("[WARNING] --calibration oracle standardizes and thresholds on test rows labelled normal. "
              "This leaks test labels; use it only to reproduce the paper's reference numbers.")

    print(f"\n[EVAL] calibration={calibration}  decision={decision}  analyzers={analyzers}")
    scores, train_scores = _compute_subsystem_scores(subsystems, df_eval, y_true, out_dir, calibration,
                                                     df_train, run_adversarial)
    if not scores:
        return None
    names = list(scores)

    if decision == 'orvote':
        preds = np.zeros(len(y_true), dtype=int)
        for name in names:
            ref = train_scores[name] if calibration == 'train' else scores[name][y_true == 0]
            preds = np.maximum(preds, (scores[name] > np.percentile(ref, 99.8)).astype(int))
    else:
        agg = np.max([scores[n] for n in names], axis=0)
        cal = np.max([train_scores[n] for n in names], axis=0)
        preds = lfp.dual_threshold(agg, np.percentile(cal, 99.9), np.percentile(cal, 99.0), 50).astype(int)

    if analyzers == 'split':
        cols = [c for c in df_eval.columns if lfp.is_analyzer(c) and c in df_train.columns]
        preds = np.maximum(preds, lfp.analyzer_arm_predict(df_train, df_eval, cols))

    s, _ = evaluate(y_true, preds)
    s['chance_mean'], s['chance_p95'] = lfp.chance_events(preds, y_true) if y_true.any() else (0.0, 0.0)
    print(f"  F1 {s['f1']:.2%} | Precision {s['precision']:.2%} | Recall {s['recall']:.2%} | "
          f"FAR {s['false_alarm']:.2%} | attacks caught {s['n_detected']}/{s['n_events']} "
          f"(by chance {s['chance_mean']:.1f}, 95th pct {s['chance_p95']:.0f}) | "
          f"false-alarm episodes {s['false_alarm_events']}")

    if export_predictions:
        # Per-subsystem scores are kept alongside the OR-vote so the event-level harness can
        # attribute each detection to the subsystem that fired it.
        out = pd.DataFrame({'y_true': y_true.astype(int), 'y_pred': preds})
        out['score'] = np.max([scores[n] for n in names], axis=0)
        for name in names:
            out[f'score_{name}'] = scores[name]
        out.to_csv(export_predictions, index=False)
        print(f"  [SAVED] predictions -> {export_predictions}  (score with evaluate_events.py)")

    return s

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['swat', 'wadi', 'hai'])
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--adversarial', action='store_true')
    parser.add_argument('--calibration', choices=['train', 'oracle'], default='train',
                        help="'train' (default): label-free, calibrated on normal training data. "
                             "'oracle': the paper protocol, calibrated on test rows labelled normal "
                             "(leaks labels; reference only)")
    parser.add_argument('--decision', choices=['orvote', 'dual'], default='orvote',
                        help="'orvote': per-subsystem P99.8 OR-vote. 'dual': dual threshold with "
                             "causal persistence (label-free only)")
    parser.add_argument('--analyzers', choices=['keep', 'split'], default='keep',
                        help="'split': train/evaluate the array without *AIT* analyzers and OR in the "
                             "drift-robust analyzer arm (recommended for SWaT)")
    parser.add_argument('--export_predictions', type=str, default=None,
                        help='Write y_true/y_pred/scores to this CSV for evaluate_events.py')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Override the model directory (default Oracles/<dataset>). The v1 '
                             'SWaT per-timestep bundles live in Oracles/_backup_3stage_physics.')
    parser.add_argument('--eval_data', type=str, default=None,
                        help='Override the evaluation CSV (e.g. corrected-label variants)')
    parser.add_argument('--sub_json', type=str, default=None,
                        help='Override the subsystem-definition JSON (e.g. a WS4 learned decomposition)')
    parser.add_argument('--physics', choices=['none', 'nonneg', 'interlock', 'legacy'], default='none',
                        help="training physics term. 'nonneg'/'interlock' work in physical units (SWaT only: "
                             "needs Datasets/preprocessing_meta.json); 'legacy' is the mis-specified term "
                             "behind models trained before 2026-10-05")
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed weight init / shuffling for reproducible, seed-matched runs')
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
        
    if args.out_dir:
        out_dir = args.out_dir
    if args.eval_data:
        test_path = args.eval_data
    if args.sub_json:
        sub_json = args.sub_json

    with open(sub_json, 'r') as f:
        subsystems = json.load(f)
    if args.analyzers == 'split':
        subsystems = {name: [f for f in feats if not lfp.is_analyzer(f)] for name, feats in subsystems.items()}
        subsystems = {name: feats for name, feats in subsystems.items() if feats}
        
    if args.train:
        if args.physics in ('nonneg', 'interlock') and dataset_name != 'swat':
            raise SystemExit("--physics nonneg/interlock needs raw-unit statistics; only SWaT has them "
                             "(Datasets/preprocessing_meta.json)")
        for name, features in subsystems.items():
            train_subsystem(name, features, train_path, out_dir, seed=args.seed, physics=args.physics)
            
    if args.eval:
        evaluate_subsystems(subsystems, test_path, out_dir, calibration=args.calibration,
                            train_data_path=train_path, decision=args.decision, analyzers=args.analyzers,
                            run_adversarial=args.adversarial, export_predictions=args.export_predictions)
