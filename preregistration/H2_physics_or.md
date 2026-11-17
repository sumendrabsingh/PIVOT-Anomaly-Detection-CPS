# Pre-registered test H2: does physics add detections?

Written Oct 7, 2026, before the physics detectors below have been run on any test data. Building them exposed a bug: on SWaT the interlock residual is `off × |flow|` on standardized flows, so zero flow (about −1.6) scores high and a real violation scores low. The frozen design fixes it.

**Hypothesis H2.** OR-ing physical-invariant detectors into PIVOT's vote catches more attacks beyond chance than PIVOT alone at the same total FAR.

**Frozen design:**

- Base detector: the stage array with per-sensor scoring, seeds 1–5, exactly as in H1 (SWaT: six stages with analyzers split out; WADI A2; HAI 23.05).
- Invariants: SWaT `Datasets/swat_invariants.json` (3 tank mass balances, 5 actuator-to-flow interlocks); WADI A2: interlocks re-derived on A2 training data with the rule in `wadi_interlock_validation.py`, unchanged (first 10 days derive, the rest held out); HAI: `auto_derive_invariants` on training data, as in the paper.
- Residuals in raw units. Mass balance: |Δlevel − (a·inflow + b·outflow + c)| fitted on training data, standardized on training residuals. Interlock: actuator off (state 1 on SWaT and WADI) × max(0, flow − its training median while off), divided by the flow's training SD.
- Each residual: smoothed over 25 samples, PIVOT's dual-threshold rule (P99.9 / P99 of its training residual, 50-sample persistence); the physics arm is the OR over invariants, with fixed, label-free thresholds.

**Primary test.** At total FAR of 0.5%, 1% and 2%, PIVOT alone (threshold set to that FAR) against PIVOT OR physics (PIVOT's threshold raised until the union's FAR equals the target; a level the physics arm alone exceeds counts as not supported). Difference in attacks beyond chance, seed-averaged, 95% CI bootstrapping attacks. H2 is supported on a plant if the CI lies above zero at 2 or more of the 3 levels. All three plants are judged separately and reported whatever they show.

**Secondary:** the label-free operating point (PIVOT's training thresholds, with and without the physics arm), early detection, and each plant's physics arm alone.

### Result of H2

H2 is not supported on any plant: the CI lies above zero at 1 of 3 levels on SWaT and at 0 of 3 on WADI and HAI. On SWaT the physics arm adds about six attacks beyond chance at 1–2% FAR on all five seeds, but the 0.5% level was unreachable (the arm alone runs at 0.81% FAR) and the 1% CI touches zero.

| Plant | FAR | PIVOT alone, beyond chance | PIVOT OR physics | Difference (95% CI) | Seeds where the union wins |
| --- | --- | --- | --- | --- | --- |
| SWaT (35) | 0.5% | – | – | not reachable: physics arm alone at 0.81% | – |
| SWaT (35) | 1% | 9.5 | 15.4 | +5.9 (−0.2 to +12.0) | 5 of 5 |
| SWaT (35) | 2% | 7.4 | 13.4 | +6.0 (+1.2 to +11.2) | 5 of 5 |
| WADI (14) | 0.5% | – | – | not reachable: physics arm alone at 0.55% | – |
| WADI (14) | 1% | 8.9 | 8.7 | −0.2 (−2.1 to +2.1) | 1 of 5 |
| WADI (14) | 2% | 9.1 | 8.2 | −0.9 (−1.9 to +0.3) | 0 of 5 |
| HAI (52) | 0.5% | – | – | not reachable: physics arm alone at 0.86% | – |
| HAI (52) | 1% | 23.5 | 21.2 | −2.4 (−6.6 to +1.8) | 0 of 5 |
| HAI (52) | 2% | 23.7 | 24.0 | +0.3 (−1.8 to +3.1) | 2 of 5 |

Secondary, label-free operating point: on SWaT the union reaches 14.3 attacks beyond chance (early 16.5) against 9.9 for PIVOT alone, at 1.49% vs 0.71% FAR and F1 84.2 vs 81.3, but it raises 355 false-alarm episodes against 6; on WADI and HAI the physics arm adds false alarms and no detections beyond chance. With per-sensor scoring, PIVOT alone already catches what WADI's interlocks caught when they lifted the old A1 array from 6 to 12 of 14. Re-derived on WADI A2 with the frozen rule, the interlocks are the same seven pairs as on A1. Evidence: `physics_or_test.py`, `physics_or_test_{swat,wadi,hai}_summary.csv`, `physics_or_test_{swat,wadi,hai}_labelfree.csv`.
