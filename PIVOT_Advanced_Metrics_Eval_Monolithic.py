import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix
import sys

from PIVOT_Adversarial_Evaluation import (
    OracleAutoencoder, OracleDeepSVDD, OracleVAE, OracleContrastive, OracleTransformerAutoencoder
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- ADVANCED ATTACKS FOR MONOLITHIC MODELS ---

def compute_loss(model, x_adv, model_type, center_c, error_mode='mean'):
    if model_type == 'svdd':
        z = model(x_adv)
        return torch.mean(torch.sum((z - center_c) ** 2, dim=1))
    elif model_type == 'contrastive':
        z = model(x_adv)
        c_norm = F.normalize(center_c.unsqueeze(0) if center_c.dim() == 1 else center_c, dim=1)
        sim = F.cosine_similarity(z, c_norm)
        return torch.mean(1.0 - sim)
    elif model_type == 'vae':
        reconstructed, _, _ = model(x_adv)
        squared_diff = (x_adv - reconstructed) ** 2
        return torch.mean(torch.max(squared_diff, dim=1)[0]) if error_mode == 'max' else torch.mean(squared_diff)
    else:
        reconstructed = model(x_adv)
        squared_diff = (x_adv - reconstructed) ** 2
        return torch.mean(torch.max(squared_diff, dim=1)[0]) if error_mode == 'max' else torch.mean(squared_diff)

def cw_attack_monolithic(model, x_clean, y_true, model_type, center_c, epsilon=0.1, num_iter=50, lr=0.01):
    attack_mask = (y_true == 1)
    if not attack_mask.any(): return x_clean.clone()
    
    x_clean_attacks = x_clean[attack_mask].clone().detach()
    delta = torch.zeros_like(x_clean_attacks, requires_grad=True)
    
    optimizer = torch.optim.Adam([delta], lr=lr)
    
    for _ in range(num_iter):
        optimizer.zero_grad()
        x_adv = x_clean_attacks + delta
        loss = compute_loss(model, x_adv, model_type, center_c)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            delta.clamp_(-epsilon, epsilon)
            
    x_adv_final = x_clean.clone().detach()
    x_adv_final[attack_mask] = (x_clean_attacks + delta).detach()
    return x_adv_final

def apgd_attack_monolithic(model, x_clean, y_true, model_type, center_c, epsilon=0.1, num_iter=20):
    attack_mask = (y_true == 1)
    if not attack_mask.any(): return x_clean.clone()
    
    x_clean_attacks = x_clean[attack_mask].clone().detach()
    x_adv = x_clean_attacks.clone().detach()
    x_adv.requires_grad = True
    
    alpha = 1.5 * epsilon / (num_iter * 0.25)
    momentum = torch.zeros_like(x_adv)
    
    for i in range(num_iter):
        loss = compute_loss(model, x_adv, model_type, center_c)
        
        if i % 5 == 0 and i > 0:
            alpha = alpha / 1.5
            
        model.zero_grad()
        if x_adv.grad is not None: x_adv.grad.zero_()
        loss.backward()
        
        with torch.no_grad():
            grad = x_adv.grad
            momentum = 0.75 * momentum + grad / (torch.norm(grad, p=1) + 1e-12)
            x_adv = x_adv - alpha * momentum.sign()
            delta = torch.clamp(x_adv - x_clean_attacks, min=-epsilon, max=epsilon)
            x_adv = x_clean_attacks + delta
            x_adv.requires_grad = True
            
    x_adv_final = x_clean.clone().detach()
    x_adv_final[attack_mask] = x_adv.detach()
    return x_adv_final

def evaluate_advanced_monolithic(data_path, bundle_path, attack_type='cw', epsilon=0.1):
    checkpoint = torch.load(bundle_path, map_location=DEVICE, weights_only=False)
    input_dim = checkpoint['input_dim']
    bottleneck_dim = checkpoint.get('bottleneck_dim', 16)
    model_type = checkpoint.get('model_type', 'autoencoder')
    
    if model_type == 'svdd':
        model = OracleDeepSVDD(input_dim=input_dim, bottleneck_dim=bottleneck_dim).to(DEVICE)
        center_c = checkpoint['center_c'].to(DEVICE)
    elif model_type == 'vae':
        model = OracleVAE(input_dim=input_dim, bottleneck_dim=bottleneck_dim).to(DEVICE)
        center_c = None
    elif model_type == 'contrastive':
        model = OracleContrastive(input_dim=input_dim, bottleneck_dim=bottleneck_dim).to(DEVICE)
        center_c = checkpoint['center_c'].to(DEVICE)
    elif model_type == 'transformer':
        use_cls = 'cls_token' in checkpoint['model_state']
        model = OracleTransformerAutoencoder(input_dim=input_dim, bottleneck_dim=bottleneck_dim, use_cls_bottleneck=use_cls).to(DEVICE)
        center_c = None
    else:
        model = OracleAutoencoder(input_dim=input_dim, bottleneck_dim=bottleneck_dim).to(DEVICE)
        center_c = None
        
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    MEAN, STD = checkpoint['mean'].cpu().numpy(), checkpoint['std'].cpu().numpy()
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    
    label_col = 'Label' if 'Label' in df.columns else 'Normal/Attack'
    numeric_df = df.select_dtypes(include=[np.number]).fillna(df.mean(numeric_only=True))
    
    features_list = checkpoint.get('features', None)
    if features_list is not None:
        numeric_df = numeric_df[features_list]
        
    scaled_data = (numeric_df - MEAN) / STD
    y_true = df[label_col].astype(str).str.strip().str.replace(" ", "").str.lower()
    y_true = y_true.apply(lambda val: 1 if val == "attack" else 0).values

    batch_size = 4096
    tensor_data = torch.tensor(scaled_data.values, dtype=torch.float32).to(DEVICE)
    y_true_tensor = torch.tensor(y_true, dtype=torch.long, device=DEVICE)

    # Calculate Clean Threshold natively
    clean_scores = []
    with torch.no_grad():
        for i in range(0, len(tensor_data), batch_size):
            batch = tensor_data[i:i + batch_size]
            if model_type == 'svdd':
                z = model(batch)
                batch_scores = torch.sum((z - center_c) ** 2, dim=1)
            elif model_type == 'contrastive':
                z = model(batch)
                c_norm = F.normalize(center_c.unsqueeze(0) if center_c.dim() == 1 else center_c, dim=1)
                sim = F.cosine_similarity(z, c_norm)
                batch_scores = 1.0 - sim
            else:
                batch_scores = model.get_reconstruction_error(batch, mode='mean')
            clean_scores.extend(batch_scores.cpu().numpy().tolist())
    
    clean_scores = np.array(clean_scores)
    normal_sc = clean_scores[y_true == 0]
    
    lo, hi = max(clean_scores.min(), 1e-6), clean_scores.max()
    thresholds = np.logspace(np.log10(lo), np.log10(hi), 400)
    best_f1, best_t = 0, 0
    for t in thresholds:
        preds = (clean_scores > t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    # Execute Adv Attack
    adv_scores = []
    for i in range(0, len(tensor_data), batch_size):
        batch = tensor_data[i:i + batch_size]
        batch_y = y_true_tensor[i:i + batch_size]
        
        for param in model.parameters(): param.requires_grad = True
            
        if attack_type == 'cw':
            batch_adv = cw_attack_monolithic(model, batch, batch_y, model_type, center_c, epsilon=epsilon)
        else:
            batch_adv = apgd_attack_monolithic(model, batch, batch_y, model_type, center_c, epsilon=epsilon)
            
        for param in model.parameters(): param.requires_grad = False
            
        with torch.no_grad():
            if model_type == 'svdd':
                z = model(batch_adv)
                batch_s = torch.sum((z - center_c) ** 2, dim=1)
            elif model_type == 'contrastive':
                z = model(batch_adv)
                c_norm = F.normalize(center_c.unsqueeze(0) if center_c.dim() == 1 else center_c, dim=1)
                sim = F.cosine_similarity(z, c_norm)
                batch_s = 1.0 - sim
            else:
                batch_s = model.get_reconstruction_error(batch_adv, mode='mean')
            adv_scores.extend(batch_s.cpu().numpy().tolist())
            
    adv_scores = np.array(adv_scores)
    y_pred_adv = (adv_scores > best_t).astype(int)
    cm = confusion_matrix(y_true, y_pred_adv)
    tn, fp, fn, tp = cm.ravel()
    
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = f1_score(y_true, y_pred_adv, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    return {
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'false_alarm': fpr
    }

def run_advanced_monolithic_batch():
    data_path = r'SWaT_Orig_DataSets/Original Dataset Backup/SWaT_Dataset_Attack_v0_Clean.csv'
    models = [
        r'Oracles\clean_standard_model.pth',
        r'Oracles\clean_dae_model.pth',
        r'Oracles\clean_vae_model.pth',
        r'Oracles\clean_svdd_model.pth',
        r'Oracles\clean_contrastive_model.pth',
        r'Oracles\clean_transformer_model.pth'
    ]
    
    results = []
    
    for model_path in models:
        model_name = os.path.basename(model_path)
        print(f"\n========================================================")
        print(f" Evaluating {model_name}")
        print(f"========================================================")
        try:
            print("  -> Running C&W Attack...")
            res_cw = evaluate_advanced_monolithic(data_path, model_path, attack_type='cw')
            print("  -> Running APGD Attack...")
            res_apgd = evaluate_advanced_monolithic(data_path, model_path, attack_type='apgd')
            
            results.append({
                'Model': model_name + "_CW",
                'Accuracy (Adv)': res_cw['accuracy'],
                'Precision (Adv)': res_cw['precision'],
                'Recall (Adv)': res_cw['recall'],
                'F1 (Adv)': res_cw['f1'],
                'False Alarm (Adv)': res_cw['false_alarm']
            })
            results.append({
                'Model': model_name + "_APGD",
                'Accuracy (Adv)': res_apgd['accuracy'],
                'Precision (Adv)': res_apgd['precision'],
                'Recall (Adv)': res_apgd['recall'],
                'F1 (Adv)': res_apgd['f1'],
                'False Alarm (Adv)': res_apgd['false_alarm']
            })
            print(f"  [SUCCESS] {model_name} evaluated.")
        except Exception as e:
            print(f"  [ERROR] Failed to evaluate {model_name}: {e}")
            import traceback; traceback.print_exc()
            
    df = pd.DataFrame(results)
    df.to_csv('monolithic_advanced_metrics.csv', index=False)
    print("\n[DONE] Monolithic Advanced Matrix saved to monolithic_advanced_metrics.csv")

if __name__ == "__main__":
    run_advanced_monolithic_batch()
