# Pre-registered test H3: PIVOT's OR-vote rule

Written Oct 7, 2026, before the OR-vote rule has been run on any of the releases below. H1, H1b and H1c used one threshold on the max over sensors; this test uses PIVOT's own rule. HAI 20.07, 21.03 and 22.04 were used in H1b and H1c with the other rule only. On the development plants OR voting won on SWaT at strict FAR, was mixed on WADI and lost on HAI 23.05, so a negative result here is the expected outcome; no SWaT-like held-out plant is available.

**Hypothesis H3.** One autoencoder per stage, frozen, each stage with its own threshold and an alarm when any stage fires, catches more attacks beyond chance than a monolithic autoencoder on the same sensors, at the same total FAR.

**Frozen design:**

- Models as already trained for H1b/H1c, unchanged: stages = the four processes by sensor prefix (`Oracles/hai2007_*`, `hai2103_*`, `hai2204_*`, seeds 1–5); monolithic control on the same sensors.
- Scoring for both: each sensor's squared error standardized on its own training errors; a stage's score is the max over its sensors; smoothed over 25 samples. Chosen because it was best or equal for OR voting on all three development plants.
- OR vote: each stage alarms above the (1 − q) quantile of its own score on test normal rows, with one q for all stages chosen so the OR reaches the target FAR (an equal FAR share per stage); monolithic: one threshold at the target FAR. A comparison, not an operating point, as in H1.

**Primary test.** At total FAR 0.1%, 0.5%, 1% and 2%, OR vote minus monolithic in attacks beyond chance (per-attack chance from circular shifts, seed-averaged), 95% CI bootstrapping attacks. H3 is supported on a release if the CI lies above zero at 2 or more of the 4 levels. The three releases are judged separately and all are reported.

**Secondary:** point-wise F1 at the same levels (the submission's metric), and the label-free version (each stage at P99.8 of its training score, monolithic at P99.8).

### Result of H3

H3 is supported on two of the three held-out releases: on HAI 20.07 and 22.04, OR voting over stage models catches 7–11 more attacks beyond chance than the monolithic model at every FAR level, on 5 of 5 seeds; on HAI 21.03 it wins only at 0.1% FAR and loses at 1–2%.

| Release | 0.1% FAR | 0.5% FAR | 1% FAR | 2% FAR | Verdict |
| --- | --- | --- | --- | --- | --- |
| HAI 20.07 (38) | +7.7 (+2.6 to +13.3) | +11.4 (+6.3 to +16.6) | +11.1 (+5.7 to +16.5) | +8.3 (+3.1 to +13.9) | Supported, 4 of 4 |
| HAI 22.04 (58) | +6.6 (+3.3 to +10.3) | +9.1 (+4.2 to +14.3) | +8.8 (+3.9 to +14.2) | +9.7 (+4.7 to +15.1) | Supported, 4 of 4 |
| HAI 21.03 (50) | +7.1 (−0.4 to +14.5) | +1.1 (−2.7 to +5.1) | −3.6 (−6.8 to −1.1) | −4.6 (−7.6 to −2.3) | Not supported, 0 of 4 |

OR vote minus monolithic, attacks beyond chance, 95% CI over attacks, seeds 1–5. Point-wise F1 agrees (HAI 22.04: 45.7% vs 19.1% at 0.1% FAR). The matched FAR is exact (0.100% for a 0.1% target, each of the four stages at 0.025%). With training-data thresholds both detectors fail on these releases (30–66% FAR), so the result needs on-site calibration to be usable.

**Mechanism (exploratory).** On HAI 22.04 one stage (P2, the turbine) drifts in the test period: its median score on normal test data is 16.8 against a training P99 of 6.6. A single threshold must sit above that drift and misses attacks elsewhere; per-stage thresholds confine it to its own stage. Grouping the monolithic model's own errors by stage with the same OR thresholds recovers most of the gain on HAI 20.07 (within ±2.5 attacks), but on HAI 22.04 separate stage models add a further +3.8 to +4.3 attacks at 0.5–2% FAR (CIs above zero): a single model spreads one stage's drift into its reconstruction of the others. Evidence: `orvote_confirm.py`, `orvote_confirm_{hai2007,hai2103,hai2204}_summary.csv`, `orvote_mechanism.py`.
