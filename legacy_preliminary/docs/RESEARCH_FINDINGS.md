# PIVOT — Current Research Findings

Consolidated findings for the **full-dataset** experiments on **SWaT**, **WADI**, and **HAI**.
Models are trained on normal-only data and evaluated on the full attack+normal sets.

---

## 1. Robustness under evasion: graceful degradation, not immunity

Early drafts headlined this as "adversarial immunity" (100% adversarial recall). That claim
was **retired**: it came from the *raw lockdown* OR-voting operating point, which already
alarms on almost everything (~24–30% false-alarm rate), so "100% recall under attack"
reflects a saturated detector, not robustness. The honest, deployment-relevant picture uses
the production **dual-threshold** operating point, under white-box PGD / APGD / C&W at ε=0.1
evaluated *non-adaptively* (from `subsystem_adversarial_benchmark_results.csv` and the
`dual_threshold_adv` reproductions):

| Model under ε=0.1 evasion | Clean Recall | Adversarial Recall | Δ |
|---|---|---|---|
| Monolithic Deep SVDD | 0.92 | **0.36** | −0.56 (collapse) |
| Monolithic Standard AE | 0.62 | 0.62 | ≈0 |
| Monolithic Contrastive | 0.39 | 0.32 | −0.07 |
| **Subsystem array (dual-threshold)** | **0.725** | **0.696** | **−0.03 (graceful)** |

- The subsystem dual-threshold array **degrades gracefully** (~3-point recall drop) where the
  one-class Deep SVDD baseline **collapses** (0.92 → 0.36). A bounded, non-adaptive
  perturbation cannot push every small per-stage detector under its threshold at once.
- This is **not** an immunity claim. We do not evaluate an *adaptive* attacker who optimizes
  directly against the OR-vote or the physics terms — that is explicitly future work.
- The raw lockdown OR-vote can reach ~1.0 recall under attack, but only at ~24–30% FAR — an
  unusable operating point, and not evidence of robustness.

## 2. Subsystem isolation beats monolithic on clean SWaT

From `experiments_comparison.md` (clean normal → attack, full SWaT):

- **Best config: Subsystem AE Dual-Threshold (point-wise) → F1 83.28%, Precision 97.68%,
  FAR 0.24%** in that early run. The **shipped seed-4 bundles** — the run the paper's
  Table II reports — land at **F1 83.35%, Precision 98.04%, FAR 0.20%** (five-seed mean
  83.70 ± 0.24%; see the README reproducibility section).
- **+4.62% absolute F1** over the best global/monolithic model (78.73%) and **+3.81% recall**
  (68.68% → 72.49%) for the shipped run.
- **Subsystem PINN (Static Physics, Dual-Threshold): F1 83.35%** — essentially tied for top.
- Global-model ranking: Standard (78.73%) > DAE > Transformer > VAE > SVDD (74.60%).

## 3. Dual-threshold + temporal persistence mechanism

Best detection configuration: High threshold `41.01` (instant) + Low threshold `7.86`
sustained over `200+` consecutive samples, rolling window `W=25`, MAX error / MAX aggregation
across the 6 subsystems. Catches both spikes (13,389 instant detections) and slow persistent
attacks (26,254 sustained detections).

## 4. Physics catches dataset labeling errors

Correcting SWaT labels for **physical recovery windows** (the plant remains mathematically
broken for up to ~10 min after the attack script stops):

| Metric | Original Labels | Physics-Corrected |
|---|---|---|
| Precision | 98.84% | **99.79%** |
| False Alarm Rate | 0.106% | **0.020%** |
| False Positives | 419 | **77** (eliminated 342) |

The remaining 77 false positives are almost entirely **rare Stage-3 backwash / Stage-6 RO
chemical-cleaning cycles** — correctly flagged deviations from steady state, not true errors.
This ties into the **273 mislabeled-events** XAI study (three patterns: lingering attack
damage, rare maintenance cycles, and PLC deadbands / sensor sloshing).

## 5. GNN (ST-GAE) is competitive but does not surpass subsystem AE

7-node cyber-physical ST-GAE with dual-threshold: F1 ~81.8–82.1%, FAR as low as 0.20%.
Strong, but slightly below the subsystem AE / PINN top line. SEQ_LEN=10 with clipped Z-score
was the best GNN variant.

## 6. Cross-dataset generalization (WADI, HAI) — open gaps

- HAI is decomposed into 4 subsystems (P1–P4); WADI into 3 (P1–P3). Pipelines and XAI
  heatmaps exist for both.
- **HAI results are currently poor**: engineered Subsystem PINN OR-voting F1 ~7.5%,
  engineered GNN F1 ~13.3% (`hai_sub_eng_results.txt`, `hai_gnn_eng_results.txt`). HAI is the
  hardest/unsolved dataset so far and is likely undertuned (only 10 training epochs).
- **WADI** subsystem evaluation infrastructure is in place (`wadi_advanced_metrics.py`,
  `Oracles/wadi/`) but is not yet at the SWaT performance level.

## 7. Physics-coverage ablation: invariants are a false-alarm lever, not an F1/robustness lever

Controlled A/B on the SWaT subsystem array, retrained from scratch with identical seed/config,
varying **only** the PINN physics coverage (toggled via `ENABLE_STAGE456_PHYS` in
`run_preprocessed_subsystem_experiments.py`):

- **Control (3-stage):** valve-closed→zero-flow traps for Stages 1–3 only.
- **Treatment (6-stage):** adds pump-off→zero-flow traps for Stages 4–6
  (`P402→FIT401`, `P501→FIT501`, `P602→FIT601`; pumps gated as continuous 1.0=OFF/2.0=ON).

**Dual-threshold protocol (W=25, the best config), clean / PGD ε=0.1:**

| Arm | Clean FAR | Clean FP | Clean F1 | Adv Recall | Adv F1 |
|---|---|---|---|---|---|
| 3-stage (control) | 0.281% | 1,109 | 0.8077 | 0.6910 | 0.8076 |
| 6-stage (treatment) | **0.252%** | **996** | 0.8055 | 0.6866 | 0.8054 |

Findings:
- **~10% relative false-alarm reduction** (0.281%→0.252%, **113 fewer false positives**),
  precision 97.15%→97.41% — the intended physics effect: tighter normal manifold in Stages 4–6.
- **F1 flat within noise** (−0.22pp); the extra precision is paid for with a slight recall dip,
  as the F1-optimizing threshold sweep shifts higher.
- **No adversarial-robustness gain.** Adversarial ≈ clean in both arms (recall moves <0.0002
  under PGD); temporal persistence already absorbs per-subsystem perturbations.
- In **OR-voting / lockdown** mode the effect vanishes entirely — recall is already saturated
  at 100%, so there is no headroom (deltas ≤0.2pp, within run-to-run noise).

**Defensible paper claim:** extending physical-invariant coverage from 3 to all 6 SWaT stages
cut the dual-threshold false-alarm rate ~10% at equal F1, empirically confirming physics
constraints suppress false positives — while detection recall and adversarial recall were
already saturated by the persistence and OR-voting protocols. Active models in `Oracles/` are
the 6-stage set; the two arms are archived under `Oracles/_arm_control_3stage/` and
`Oracles/_arm_treat_6stage/`.

### 7b. Cross-dataset (WADI, HAI): physics helps *more* when the baseline is weak

The same dual-threshold physics ablation was extended to WADI and HAI. Note two scope facts:
(i) WADI/HAI preprocessed CSVs are z-scored and use a different runner
(`run_generalized_subsystems.py`, `GenericPINNLoss`); (ii) WADI actuator→flow traps were
**discovered empirically** from normal data (no official plant topology) — `1_P_001→1_FIT_001`,
`2_MV_003→2_FIT_001`, `2_P_003→2_FIT_003`, each with strong support (off-state flow pinned to
its minimum). **HAI had no data-supported actuator→flow pair**, so its ablation is
physics on/off (generic non-negativity) rather than plant-specific traps. (HAI also required
re-normalization — its preprocessed data was poorly scaled, max |value| ≈ 5e5.)

| Dataset (ablation) | Clean R | Clean FAR | Clean F1 | Adv R | Adv FAR | Adv F1 |
|---|---|---|---|---|---|---|
| **WADI** control (non-neg only) | 0.194 | 0.43% | 0.299 | 0.154 | 0.43% | 0.244 |
| **WADI** +actuator traps | **0.290** | 2.27% | **0.314** | **0.311** | 4.55% | **0.256** |
| **HAI** control (physics OFF) | 0.210 | 1.22% | 0.178 | 0.194 | 1.20% | 0.167 |
| **HAI** physics ON (non-neg) | **0.308** | 1.27% | **0.246** | **0.274** | 1.69% | **0.191** |

- **WADI traps = a recall / adversarial-robustness lever** (opposite of SWaT): clean recall
  +49% relative (0.194→0.290), **adversarial recall doubled** (0.154→0.311), F1 up — paid for
  with ~5× higher false alarms (0.43%→2.27%).
- **HAI generic physics = near-free recall/F1 gain**: clean F1 +38% relative (0.178→0.246),
  recall +46%, adversarial recall +41% — at **essentially flat FAR** (1.22%→1.27%).

**Unifying interpretation:** the weaker the data-driven baseline, the more physics contributes.
On strong SWaT (F1≈0.81) there is no recall headroom, so physics only trims false alarms. On
weak WADI/HAI (F1≈0.18–0.30) the plain autoencoder misses many attacks, so physical-invariant
penalties substantially lift detection (recall/F1) and, for WADI's sharp actuator traps, restore
adversarial robustness the plain model loses under PGD. The FAR cost scales with constraint
sharpness: HAI's broad non-negativity prior adds ~0 FAR; WADI's hard actuator→flow traps add
the most FAR but also the largest robustness gain.

*Caveats:* WADI/HAI absolute F1 is low (~0.2–0.3) and attack fractions are small (WADI 3.9%,
HAI 1.0%), so these numbers are noisier than SWaT; single seed; transductive thresholding;
WADI traps are empirically discovered, not topology-certified. Results JSON in scratchpad
(`wadi_dual_ablation.json`, `hai_dual_ablation.json`); discovered traps in
`discovered_traps.json`.

---

## Net story

The SWaT results support the paper's two pillars: (1) subsystem decomposition +
dual-threshold beats monolithic detection on clean data (F1 0.787 → 0.834), and (2) under
bounded, non-adaptive evasion the subsystem array degrades **gracefully** (recall 0.725 →
0.696) where a monolithic one-class detector collapses (0.92 → 0.36) — robustness, not
immunity; an adaptive attacker who targets the OR-vote directly is future work. The
label-correction / XAI analysis is a strong supporting narrative. A cross-dataset physics
ablation (§7b) adds a unifying thread: **physical-invariant penalties help more the weaker
the data-driven baseline** — trimming false alarms on strong SWaT, but substantially lifting
recall on weak WADI/HAI. **WADI and HAI
absolute performance (F1 ≈ 0.2–0.3) remain the primary open gaps**, though physics now
demonstrably moves them in the right direction.
