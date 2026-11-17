# Pre-registered confirmation test

Written Oct 7, 2026, before any WADI A2 or multi-seed HAI result exists. On SWaT, six stages with per-sensor scoring beat the strongest monolithic control at 0.1–0.25% FAR (+4.1 and +4.6 attacks beyond chance, CIs above zero, 5 of 5 seeds), but that variant was found after looking at SWaT results, so it is confirmed here on the other two plants with nothing changed.

**Hypothesis H1.** Splitting a plant into its physical stages, with per-sensor error scoring, catches more attacks beyond chance than one autoencoder with the same scoring, at matched FAR.

**Frozen design** (no change after any WADI or HAI result is seen):

- Stage autoencoders exactly as in the paper (λ = 0, same architecture, training and early stopping), stages from `Datasets/wadi_subsystems.json` (A2) and `Datasets/hai2305_subsystems.json`.
- Monolithic control: one autoencoder on the union of the stage features, same code.
- Score: each sensor's squared error standardized on its own training errors, max over sensors, smoothed over 25 samples; for both detectors.
- Seeds 1–5 for both; the fault rule as fixed on 2026-10-07; WADI labels from the maintainers' A2 attack table; HAI 23.05 labels aligned by row (52 segments).

**Primary test.** At matched FAR of 0.1%, 0.25% and 0.5% (one threshold on the smoothed score, set on test normal rows: a comparison, not an operating point), the stage-minus-monolithic difference in attacks beyond chance, averaged over seeds, with a 95% CI from bootstrapping attacks. H1 is supported on a plant if the CI lies above zero at 2 or more of the 3 FAR levels. WADI and HAI are judged separately, and both results are reported whatever they show.

**Secondary, reported but not part of the test:** the label-free operating point (dual threshold P99.9 / P99 of training scores, 50-sample persistence), early detection, and 1–2% FAR.

**Later, separate tests:** H2, invariant detectors added to the OR vote catch more attacks beyond chance at fixed FAR; H3, robustness to the physically valid attack, with no predicted direction.

### Result of the pre-registered test

H1 is not supported: the confidence interval lies above zero at 0 of 3 primary FAR levels on WADI and at 1 of 3 on HAI, and the rule needed 2. The SWaT advantage at strict FAR did not replicate.

| Plant | FAR | Stages, beyond chance | Monolithic, beyond chance | Difference (95% CI) | Seeds where stages win |
| --- | --- | --- | --- | --- | --- |
| WADI (14) | 0.1% | 4.9 | 4.2 | +0.7 (−0.1 to +1.7) | 2 of 5 |
| WADI (14) | 0.25% | 6.9 | 7.2 | −0.2 (−0.6 to −0.0) | 1 of 5 |
| WADI (14) | 0.5% | 7.5 | 7.7 | −0.3 (−1.3 to +0.6) | 0 of 5 |
| HAI (52) | 0.1% | 5.4 | 4.8 | +0.6 (−0.0 to +1.8) | 4 of 5 |
| HAI (52) | 0.25% | 16.9 | 16.8 | +0.0 (−2.6 to +2.6) | 3 of 5 |
| HAI (52) | 0.5% | 22.8 | 20.3 | +2.5 (+0.5 to +5.0) | 5 of 5 |

Secondary results (not part of the test): at 1–2% FAR the stages are ahead on both plants by 0.5–2.0 attacks, significant only on WADI at 2%. At the label-free operating point the stages dominate on HAI (21.5 vs 17.8 beyond chance at 1.67% vs 11.5% FAR, 30 vs 154 false-alarm episodes, F1 41.3 vs 20.9) and are slightly ahead on WADI (9.7 vs 8.9 at 3.3% vs 3.4%). That pattern, stage models whose training-calibrated thresholds transfer better to the test period, was not pre-registered and needs its own held-out test. Per-sensor scoring also improves PIVOT itself at the label-free point: on HAI from 12.2 to 21.5 attacks beyond chance while FAR falls from 2.42% to 1.67%; on SWaT from 6.3 to 9.9, but at 0.71% FAR instead of 0.24%. WADI is not comparable, since its data moved from A1 to A2. Evidence: `decomposition_confirm.py`, `decomposition_confirm_{wadi,hai}_summary.csv`, `decomposition_confirm_{wadi,hai}_labelfree.csv`.
