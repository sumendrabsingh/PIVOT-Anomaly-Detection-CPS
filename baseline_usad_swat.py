"""USAD baseline (Audibert et al., KDD 2020) on the SWaT split, scored like Table II.

Two autoencoders share an encoder and are trained adversarially: AE2 learns to tell apart real
windows from AE1's reconstructions, AE1 learns to fool it. Anomaly score at test combines AE1's
reconstruction error with the AE2-of-AE1 error. Evaluated point-wise with a best-F1 threshold
(the same f1_opt protocol as the paper's monolithic baselines), plus event-level for context.
"""
import numpy as np, pandas as pd, torch, torch.nn as nn
from evaluate_events import to_binary, find_events

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
K = 12            # window length (USAD SWaT setting)
DS = 5            # downsample factor (USAD down-samples SWaT by 5)
LATENT_DIV = 8
EPOCHS = 60
BS = 512
torch.manual_seed(0); np.random.seed(0)


def load(path, cols, mu, sd):
    df = pd.read_csv(path); df.columns = df.columns.str.strip()
    lab = 'Label' if 'Label' in df.columns else 'Normal/Attack'
    y = to_binary(df[lab])
    X = ((df[cols].values.astype(np.float32) - mu) / sd)
    return X, y


def windows(X, y=None, ds=1):
    Xd = X[::ds]
    n, d = Xd.shape
    idx = np.arange(0, n - K + 1)
    W = np.stack([Xd[i:i + K].ravel() for i in idx]).astype(np.float32)
    yy = None
    if y is not None:
        yd = y[::ds]
        yy = np.array([yd[i:i + K].max() for i in idx])   # window is attack if any frame is
    return W, yy, idx, len(Xd)


class Enc(nn.Module):
    def __init__(s, m, z):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(m, m // 2), nn.ReLU(), nn.Linear(m // 2, m // 4),
                              nn.ReLU(), nn.Linear(m // 4, z), nn.ReLU())
    def forward(s, x): return s.net(x)


class Dec(nn.Module):
    def __init__(s, m, z):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(z, m // 4), nn.ReLU(), nn.Linear(m // 4, m // 2),
                              nn.ReLU(), nn.Linear(m // 2, m), nn.Sigmoid())
    def forward(s, x): return s.net(x)


def main():
    tr = pd.read_csv(r'Datasets\swat_train_preprocessed.csv'); tr.columns = tr.columns.str.strip()
    lab = 'Label' if 'Label' in tr.columns else 'Normal/Attack'
    cols = [c for c in tr.columns if c not in (lab, 'Timestamp')]
    mu = tr[cols].values.astype(np.float32).mean(0); sd = tr[cols].values.astype(np.float32).std(0); sd[sd < 1e-6] = 1
    # USAD uses min-max in [0,1]; we z-score then squash via sigmoid decoder target -> keep z-score
    Xtr = (tr[cols].values.astype(np.float32) - mu) / sd
    Xte, yte = load(r'Datasets\swat_test_preprocessed.csv', cols, mu, sd)

    # rescale to [0,1] for the sigmoid decoders (fit range on train)
    lo, hi = Xtr.min(0), Xtr.max(0); rng = np.where(hi - lo < 1e-6, 1, hi - lo)
    Xtr = (Xtr - lo) / rng; Xte = np.clip((Xte - lo) / rng, -0.5, 1.5).astype(np.float32)

    Wtr, _, _, _ = windows(Xtr, None, ds=DS)
    Wte, ywin, idx, n_te_ds = windows(Xte, yte, ds=DS)
    m = K * len(cols); z = max(20, m // LATENT_DIV)
    print(f"USAD: features {len(cols)}, window {K}, input {m}, latent {z}, "
          f"train windows {len(Wtr)}, test windows {len(Wte)}")

    E, D1, D2 = Enc(m, z).to(DEV), Dec(m, z).to(DEV), Dec(m, z).to(DEV)
    o1 = torch.optim.Adam(list(E.parameters()) + list(D1.parameters()), lr=1e-3)
    o2 = torch.optim.Adam(list(E.parameters()) + list(D2.parameters()), lr=1e-3)
    Wt = torch.tensor(Wtr)

    for ep in range(1, EPOCHS + 1):
        perm = torch.randperm(len(Wt)); tot1 = tot2 = 0.0
        for i in range(0, len(Wt), BS):
            w = Wt[perm[i:i + BS]].to(DEV)
            z0 = E(w); w1 = D1(z0); w2 = D2(z0); w3 = D2(E(w1))
            l1 = (1 / ep) * ((w - w1) ** 2).mean() + (1 - 1 / ep) * ((w - w3) ** 2).mean()
            o1.zero_grad(); l1.backward(retain_graph=True); o1.step()
            z0 = E(w); w1 = D1(z0); w2 = D2(z0); w3 = D2(E(w1))
            l2 = (1 / ep) * ((w - w2) ** 2).mean() - (1 - 1 / ep) * ((w - w3) ** 2).mean()
            o2.zero_grad(); l2.backward(); o2.step()
            tot1 += l1.item(); tot2 += l2.item()
        if ep % 10 == 0 or ep == 1:
            print(f"  epoch {ep:>3}  L1 {tot1:.3f}  L2 {tot2:.3f}")

    # anomaly score per window (alpha*AE1 + beta*AE2oAE1), assign to window's LAST ds-frame
    E.eval(); D1.eval(); D2.eval()
    scores = np.zeros(len(Wte), np.float32)
    with torch.no_grad():
        for i in range(0, len(Wte), 4096):
            w = torch.tensor(Wte[i:i + 4096]).to(DEV)
            z0 = E(w); w1 = D1(z0); w3 = D2(E(w1))
            s = 0.5 * ((w - w1) ** 2).mean(1) + 0.5 * ((w - w3) ** 2).mean(1)
            scores[i:i + len(s)] = s.cpu().numpy()

    # map window scores back to full-resolution test timesteps
    full = np.zeros(len(yte), np.float32)
    last_ds = idx + K - 1              # last down-sampled frame index per window
    ts = last_ds * DS                 # to full-resolution index
    full[:] = full.min()
    for wi, t in enumerate(ts):
        full[min(t, len(full) - 1)] = scores[wi]
    # forward-fill the sparse (every-DS) scores across skipped frames
    s = pd.Series(np.where(full == 0, np.nan, full)).ffill().bfill().values

    # best-F1 (f1_opt) point-wise threshold sweep, same protocol as Table II
    from sklearn.metrics import f1_score, confusion_matrix
    thrs = np.quantile(s, np.linspace(0.90, 0.9995, 60))
    best = (-1, None)
    for t in thrs:
        f = f1_score(yte, (s > t).astype(int), zero_division=0)
        if f > best[0]: best = (f, t)
    pred = (s > best[1]).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    acc = (tp + tn) / len(yte); prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn); f1 = best[0]; far = fp / (fp + tn)
    ev = find_events(yte); det = sum(1 for a, b in ev if pred[a:b + 1].any())
    print("\nUSAD on SWaT (point-wise, best-F1 threshold):")
    print(f"  Acc {acc*100:.2f}  Prec {prec*100:.2f}  Rec {rec*100:.2f}  F1 {f1*100:.2f}  FAR {far*100:.2f}")
    print(f"  event recall {det}/{len(ev)} = {det/len(ev)*100:.1f}%")
    pd.DataFrame({'y_true': yte, 'score': s}).to_csv('baseline_usad_scores.csv', index=False)


if __name__ == "__main__":
    main()
