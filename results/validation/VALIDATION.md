# PaperV4 validation report

Run 2026-10-09. An independent re-run of everything the evaluation paper relies on, in a separate directory (`PIVOT_repro`):
scripts copied, raw iTrust/HAI files linked read-only, **every dataset rebuilt from the raw files, every model
retrained from scratch (≈200 autoencoders + forecasters), every evaluation script re-run** (50 steps; logs in
`PIVOT_repro/logs/`, status in `PIVOT_repro/logs/STATUS`). Comparison tools: `compare_results.py` (every number of
69 result files → `validation_files.csv`), `validate_claims.py` (key numbers and verdict per claim → `CLAIMS.md`,
`claims.csv`), `validate_data_claims.py` (data claims from the raw files → `validation_data.txt`).

## Bottom line

- **Data: reproduced exactly.** All rebuilt datasets equal the repo's in every value and label (SWaT 2015, WADI A2,
  HAI 23.05, HAI 20.07/21.03/22.04, the three new SWaT releases). Differences are file formatting and six unused
  network columns only.
- **Results: 30 of 69 files bit-identical** (all WADI results; HAI 20.07 and 22.04 H3, mechanism, H4, H5; H1c).
  The other 39 differ only through GPU non-determinism in retraining the SWaT, HAI 23.05 and HAI 21.03 models
  (same seeds, same data).
- **Claims: 22 of 27 reproduced** (6 identical, 16 within tolerance: attack counts ±1.5, percentages ±2 points,
  F1 ±3 points, same verdict). **5 differ**; two of them change a statement the paper must make (C12, C17), three
  are numeric only (C25, C27, C28). Data claims C1–C3 confirmed from the raw files.

## Claims that need a caveat in the paper

| Claim | Pre-registered / repo run | From-scratch re-run | What the paper must say |
|---|---|---|---|
| C12 H2 on SWaT (physics arm OR-ed) | 1% FAR: +5.9 beyond chance, CI −0.17..+12.0 → 1 of 3 levels, not supported | +6.8, CI +0.62..+13.0 → 2 of 3 levels, would be supported | The effect (~+6 attacks at 1–2% FAR, all 5 seeds) is stable; the verdict is not. Report the pre-registered verdict (not supported) and that retraining flips it. WADI and HAI: not supported in both runs. |
| C17 H1 on HAI 23.05 | CI>0 at 0.5% only (1 of 3) | CI>0 at 0.1% only (1 of 3) | Verdict (not supported) stable; which level qualifies is not. Do not single out a level. |
| C25 calibration development | shift best for PIVOT on all 3 dev plants | same choice; HAI values of the rejected methods move by up to 1.7 attacks | No change to the conclusion. |
| C27 H6 FAR levels | Jul 2019 monolithic FAR 37.3% | 30.7% | Quote FAR as a range over both trainings (monolithic 31–37%, PIVOT 62%). |
| C28 HAI forecaster | seeds 1–5: 25.4/52 caught, 22.2 beyond chance, FAR 1.05% (range 23–28) | 26.4/52, 22.5 beyond, FAR 0.98% (range 26–28) | Report seeds 1–5, not one seed (one seed gave 26 or 24 on retraining). |

## Problems found and fixed during validation

1. **Stale WADI baseline rows.** `baselines_simple_generalization.csv` (written 2026-10-07 16:59) predated the
   switch to WADI A2 (19:49); its WADI rows were A1 results. Regenerated; now identical to the re-run (old copy in
   the session scratchpad). Consequence for the paper: on WADI A2 a 10%-tolerance range check (8.6 beyond chance at
   4.3% FAR) roughly matches PIVOT (9.7 at 3.3%); the earlier numbers understated the baselines.
2. **Missing raw file.** `physics_or_test.py` read `SWaT_Orig_DataSets/SWaT_Dataset_Attack_v0.csv`, which no longer
   exists; it now reads the official copy in `Original Dataset Backup/`.
3. **Undocumented label dependency.** `feasible_attack_label_free.py` read labels from
   `swat_test_labels_recovery.csv` (a by-product of `recovery_check.py`); now `label_free.swat_test_labels()`.
   Verified identical (449,919 rows, 54,621 attack rows).
4. **Single-seed forecaster.** The HAI forecaster headline (26/52) was one seed; seeds 1–5 added
   (`Oracles/temporal_hai_W60_seed<k>.pth`, `preds_hai_temporal_seed<k>.csv`).
5. **Overstated leakage claim (outline).** Controlled comparison (`leakage_effect.py`): test-label calibration
   changes F1 by +2.5 (SWaT), +4.8 (HAI), −23 (WADI, where the leaky protocol lacks the fault rule) and leaves
   attacks beyond chance roughly unchanged; the +600 s label convention changes SWaT more (35 → 25 attacks).
   Reproduced in the re-run.

## Data claims (from the raw files, `validation_data.txt`)

- C1 HAI 23.05: `label-test2.csv` timestamps lack seconds; a timestamp join matches 0 of 230,400 rows → 14 attack
  segments instead of 52; positional alignment agrees row by row (100%).
- C2 WADI: labels as preprocessed before the 2026-10-06 fix have 8 events; corrected and A2 have 14.
- C3 SWaT Jul 2019: data runs 105.5 s ahead of the attack PDF (median of 4 second-resolution anchors: −104, −98,
  −110, −107 s).

## Not covered

- PaperV2-era results on legacy models (`v2_label_free.py`, A1 WADI models) — not used by PaperV4.
- WaDi A3: not mapped (67.5% of features), as pre-registered.
- GDN / USAD baselines: not re-run under the full protocol (listed as a limitation in the outline).
