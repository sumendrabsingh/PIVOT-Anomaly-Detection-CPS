"""Dual-threshold subsystem detector under PGD, APGD, and C&W (released seed-4 model).

Thresholds are selected on CLEAN scores (F1-optimal), then held fixed while each attack is applied
to the y==1 rows. All three attacks minimize the subsystem reconstruction error (the evasion
objective) within an L_inf eps=0.1 ball; they differ only in the optimizer:
  PGD  -- fixed-step sign descent (paper's primary attack)
  APGD -- momentum + decaying step (Croce-Hein style)
  C&W  -- Adam-optimized, L_inf-projected
"""
import numpy as np, pandas as pd, torch
from sklearn.metrics import f1_score, confusion_matrix
import run_subsystem_experiments as v1

v1.EVAL_DATA = r"Datasets\swat_test_preprocessed.csv"
v1.MODEL_DIR = r"Oracles\swat_seeds\seed4"
W, EPS, ITERS = 25, 0.1, 10
DEV = v1.DEVICE


def _loss(model, x):
    rec = model(x); sq = (x - rec) ** 2
    return torch.mean(torch.max(sq, dim=1)[0])          # ERROR_MODE='max'


def pgd(model, x_clean, y, epsilon=EPS, alpha=0.02, num_iter=ITERS):
    m = (y == 1)
    if not m.any(): return x_clean.clone()
    x0 = x_clean[m].clone().detach(); x = x0.clone().detach().requires_grad_(True)
    for _ in range(num_iter):
        g = torch.autograd.grad(_loss(model, x), x)[0]
        with torch.no_grad():
            x = x0 + torch.clamp((x - alpha * g.sign()) - x0, -epsilon, epsilon)
        x.requires_grad_(True)
    out = x_clean.clone(); out[m] = x.detach(); return out


def apgd(model, x_clean, y, epsilon=EPS, alpha=0.02, num_iter=ITERS):
    m = (y == 1)
    if not m.any(): return x_clean.clone()
    x0 = x_clean[m].clone().detach(); x = x0.clone().detach().requires_grad_(True)
    step = 1.5 * epsilon / max(1, 0.25 * num_iter); mom = 0.75
    gprev = torch.zeros_like(x0)
    for it in range(num_iter):
        g = torch.autograd.grad(_loss(model, x), x)[0]
        with torch.no_grad():
            gm = mom * gprev + g
            x = x0 + torch.clamp((x - step * gm.sign()) - x0, -epsilon, epsilon)
            gprev = gm
            if (it + 1) % 5 == 0: step /= 1.5
        x.requires_grad_(True)
    out = x_clean.clone(); out[m] = x.detach(); return out


def cw(model, x_clean, y, epsilon=EPS, alpha=0.02, num_iter=ITERS):
    m = (y == 1)
    if not m.any(): return x_clean.clone()
    x0 = x_clean[m].clone().detach()
    delta = torch.zeros_like(x0, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=alpha)
    for _ in range(num_iter):
        opt.zero_grad(); _loss(model, x0 + delta).backward(); opt.step()
        with torch.no_grad(): delta.clamp_(-epsilon, epsilon)
    out = x_clean.clone(); out[m] = (x0 + delta).detach(); return out


def agg(scores):
    sm = {n: pd.Series(s).rolling(W, min_periods=1).mean().values for n, s in scores.items()}
    return np.max([sm[n] for n in scores], axis=0)


def sustained(mask, d, n):
    out = np.zeros(n, int); run = 0; start = 0
    for i in range(n):
        if mask[i]:
            if run == 0: start = i
            run += 1
            if run >= d: out[start:i + 1] = 1
        else: run = 0
    return out


def met(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return tp / (tp + fn), f1_score(y, pred), fp / (fp + tn)


# clean scores + fixed thresholds
y, sc_clean = v1._compute_subsystem_scores(run_adversarial=False)
n = len(y); ac = agg(sc_clean)
p = np.percentile(ac, [85, 90, 95, 97, 99, 99.5, 99.8])
best = (-1, None)
for h in np.linspace(p[3], p[6], 15):
    inst = (ac > h).astype(int)
    for l in np.linspace(p[0], p[3], 15):
        if l >= h: continue
        lm = ac > l
        for d in [200]:
            comb = np.maximum(inst, sustained(lm, d, n))
            f = f1_score(y, comb, zero_division=0)
            if f > best[0]: best = (f, (h, l, d))
h, l, d = best[1]
rc, fc, farc = met(y, np.maximum((ac > h).astype(int), sustained(ac > l, d, n)))
print(f"CLEAN            recall {rc*100:.2f}  F1 {fc*100:.2f}  FAR {farc*100:.2f}")

for name, fn in [("PGD", pgd), ("APGD", apgd), ("C&W", cw)]:
    v1.pgd_attack_subsystem = fn                      # inject attack into the scorer
    _, sc = v1._compute_subsystem_scores(run_adversarial=True, epsilon=EPS, alpha=0.02, num_iter=ITERS)
    aa = agg(sc)
    r, f, far = met(y, np.maximum((aa > h).astype(int), sustained(aa > l, d, n)))
    print(f"ADV {name:5s}       recall {r*100:.2f}  F1 {f*100:.2f}  FAR {far*100:.2f}   "
          f"(dRec {(rc-r)*100:+.2f}, dF1 {(fc-f)*100:+.2f})")
