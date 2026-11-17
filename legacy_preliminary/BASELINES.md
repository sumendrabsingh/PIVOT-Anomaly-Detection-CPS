# Baselines and reproduction (SWaT)

All detectors in the paper are scored **point-wise** (best-F1 threshold sweep, *no*
point-adjustment), so the numbers are directly comparable and are not inflated by the
point-adjustment convention that many time-series anomaly-detection papers use. Run
everything from the repository root with the SWaT preprocessed CSVs in `Datasets/`
(the datasets are license-restricted and are **not** shipped here — see the README for
the official sources).

## Clean-data detection (Table II)

The PIVOT subsystem rows and the monolithic baselines are produced by the existing
runners (`run_generalized_subsystems.py`, `run_subsystem_experiments.py`,
`PIVOT_Oracle_*.py`). The two modern deep baselines are added here:

### USAD

Faithful from-scratch reimplementation of Audibert et al. (KDD 2020) — a shared encoder
with two decoders trained adversarially, scored point-wise:

```bash
python baseline_usad_swat.py
# reported: Acc 95.05  Prec 96.08  Rec 61.79  F1 75.21  FAR 0.35
```

### GDN

GDN is evaluated with the **official implementation** (Deng & Hooi, AAAI 2021),
<https://github.com/d-ailin/GDN>, not a re-implementation, to avoid crippling the
baseline. Clone it separately, place the SWaT `train.csv`/`test.csv`/`list.txt` under
its `data/swat/`, train, and evaluate with the paper's settings:

```bash
git clone https://github.com/d-ailin/GDN
cd GDN
# train + test, point-wise best-F1 report:
python main.py -dataset swat -save_path_pattern swat \
  -slide_win 5 -slide_stride 1 -batch 32 -epoch 30 \
  -dim 64 -out_layer_num 1 -out_layer_inter_dim 128 \
  -topk 5 -val_ratio 0.2 -report best -random_seed 5
# reported: F1 73.89  Prec 88.21  Rec 63.57  (Acc 94.55, FAR 1.17 computed from the confusion matrix)
```

GDN's `evaluate.py` (`get_best_performance_data` / `eval_scores`) uses plain
`sklearn` precision/recall/F1 with a threshold search — i.e. point-wise, no
point-adjustment — so its number is comparable to the rest of Table II.

> **Note on point-adjustment.** GDN and USAD are frequently cited with SWaT F1
> ≈ 0.79–0.81; those figures rely on point-adjustment. Under the point-wise protocol
> used throughout this work they drop to the values above — below even the monolithic
> autoencoders — consistent with the rigorous-evaluation literature.

## Adversarial robustness (Table III)

The dual-threshold subsystem detector's recall/F1 before and after bounded
(ε = 0.1) non-adaptive white-box evasion, thresholds fixed on clean data and held
while the attack is applied:

```bash
python dual_threshold_adv.py             # PGD only
python dual_threshold_adv_allattacks.py  # PGD, APGD, and C&W (numerically identical)
# reported: recall 72.49 -> 69.56, F1 83.35 -> 81.35  (graceful degradation, not collapse)
```

## Reproducibility across seeds

The headline clean-data numbers are a single run; the five seeded model sets in
`Oracles/swat_seeds/` back the reported ± spread:

```bash
python reproduce_paper_multiseed.py --seeds 5
# F1 83.70 +/- 0.24% across seeds; the released model (seed 4) is Oracles/subsystem_*.pth
```

## Event-aware metrics

`evaluate_events.py` scores a prediction vector at the event level (per-attack
detection rate, time-to-detect, point-adjusted comparison) in addition to the
point-wise metrics used in the paper.
