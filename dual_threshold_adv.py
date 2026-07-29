"""Dual-threshold subsystem detector under eps=0.1 PGD: clean vs adversarial recall/F1.

Thresholds (tau_H, tau_L, min_dur) are selected on CLEAN scores to maximize F1 (the paper's
config), then the SAME thresholds are applied to the adversarial scores -- the deployment-
realistic setup where the detector is tuned, then the attack arrives. Uses the released seed-4
models and the paper's attack budget (eps=0.1, alpha=0.02, 10 iters)."""
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
import run_subsystem_experiments as v1

v1.EVAL_DATA = r"Datasets\swat_test_preprocessed.csv"
v1.MODEL_DIR = r"Oracles\swat_seeds\seed4"
W = 25


def agg(scores_dict, y):
    sm = {n: pd.Series(s).rolling(W, min_periods=1).mean().values for n, s in scores_dict.items()}
    return np.max([sm[n] for n in scores_dict], axis=0)


def sustained(mask, min_dur, n):
    out = np.zeros(n, dtype=int); run = 0; start = 0
    for i in range(n):
        if mask[i]:
            if run == 0: start = i
            run += 1
            if run >= min_dur: out[start:i + 1] = 1
        else:
            run = 0
    return out


def metrics(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return dict(recall=tp / (tp + fn), precision=tp / (tp + fp) if tp + fp else 0,
                f1=f1_score(y, pred), far=fp / (fp + tn))


print("[clean] computing subsystem scores ...")
y, sc_clean = v1._compute_subsystem_scores(run_adversarial=False)
print("[adv]   computing subsystem scores under eps=0.1 PGD ...")
y2, sc_adv = v1._compute_subsystem_scores(run_adversarial=True, epsilon=0.1, alpha=0.02, num_iter=10)

n = len(y)
agg_clean = agg(sc_clean, y)
agg_adv = agg(sc_adv, y)

# sweep thresholds on CLEAN to maximize F1 (same grid as evaluate_dual_threshold)
p = np.percentile(agg_clean, [85, 90, 95, 97, 99, 99.5, 99.8])
highs = np.linspace(p[3], p[6], 15)
lows = np.linspace(p[0], p[3], 15)
durs = [200]  # paper's W_dur (Methodology E); long persistence absorbs per-sample perturbations
best = (-1, None)
for h in highs:
    inst = (agg_clean > h).astype(int)
    for l in lows:
        if l >= h: continue
        low_mask = agg_clean > l
        for d in durs:
            comb = np.maximum(inst, sustained(low_mask, d, n))
            f1 = f1_score(y, comb, zero_division=0)
            if f1 > best[0]:
                best = (f1, (h, l, d))
h, l, d = best[1]
print(f"\nselected clean thresholds: tau_H={h:.4f} tau_L={l:.4f} min_dur={d}")

pred_clean = np.maximum((agg_clean > h).astype(int), sustained(agg_clean > l, d, n))
pred_adv = np.maximum((agg_adv > h).astype(int), sustained(agg_adv > l, d, n))
mc, ma = metrics(y, pred_clean), metrics(y, pred_adv)

print("\nDUAL-THRESHOLD (seed 4 released model), thresholds fixed on clean:")
print(f"  CLEAN : recall {mc['recall']*100:.2f}  F1 {mc['f1']*100:.2f}  prec {mc['precision']*100:.2f}  FAR {mc['far']*100:.2f}")
print(f"  ADV   : recall {ma['recall']*100:.2f}  F1 {ma['f1']*100:.2f}  prec {ma['precision']*100:.2f}  FAR {ma['far']*100:.2f}")
print(f"  movement: recall {(mc['recall']-ma['recall'])*100:+.2f} pt   F1 {(mc['f1']-ma['f1'])*100:+.2f} pt")
