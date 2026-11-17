# Pre-registered test H1c: the same comparison with a commissioning window

Written Oct 7, 2026, before any HAI 20.07 model is trained or any detector is calibrated on a commissioning window. H1b showed that thresholds from training data transfer poorly on HAI (22.04: about half of all normal samples alarmed). The paper's HAI forecaster avoids this with a 15-minute commissioning window; H1c asks whether the stage array keeps its advantage when both detectors are calibrated that way.

**Hypothesis H1c.** With thresholds from a 15-minute commissioning window, the stage array with per-sensor scoring raises no more false alarms than the monolithic control and catches at least as many attacks beyond chance.

**Held-out data:** HAI 20.07 (59 sensors, 38 attack segments), not used by any experiment so far. Its training files contain 776 rows labelled as attacks; models are trained on the rows labelled normal only. Its first attack starts 160 minutes into the test stream.

**Frozen design:** stages, models and scoring exactly as in H1b (the four processes by sensor prefix; λ = 0; seeds 1–5; per-sensor standardization on training errors, max, smoothed over 25 samples). Only calibration changes: the dual-threshold rule (P99.9 / P99, 50-sample persistence) takes its thresholds from the first 900 samples (15 minutes) of the test stream instead of from training data. FAR and attacks are counted on the whole test stream.

**Success:** as in H1b. (a) The stage array's FAR is at most the monolithic control's on at least 4 of 5 seeds; (b) the 95% CI of stages minus monolithic in attacks beyond chance has its lower bound above −1 attack (above 0 = the stronger result).

**Secondary, not confirmatory** (data already seen under training calibration): the same commissioning comparison on SWaT, WADI A2, HAI 23.05, 22.04 and 21.03, skipping any plant whose first 15 minutes contain an attack.

### Result of H1c

H1c is not supported on the held-out release: on HAI 20.07 the stage array halves the false-alarm rate but catches 2.7 fewer attacks beyond chance (95% CI −5.0 to −0.6). Calibrated on site, the monolithic model's threshold problem disappears, and with it most of the stage array's advantage.

| Plant | Stage array: FAR, beyond chance | Monolithic: FAR, beyond chance | Difference (95% CI) | Stage FAR lower on | Verdict |
| --- | --- | --- | --- | --- | --- |
| **HAI 20.07 (38), primary** | 2.34%, 15.0 | 4.79%, 17.7 | −2.7 (−5.0 to −0.6) | 4 of 5 seeds | Not supported |
| HAI 22.04 (58) | 0.37%, 21.3 | 4.16%, 17.5 | +3.9 (+1.4 to +6.5) | 5 of 5 | Met (secondary) |
| HAI 23.05 (52) | 0.43%, 20.2 | 0.58%, 20.6 | −0.4 (−1.6 to +0.9) | 3 of 5 | Not met |
| HAI 21.03 (50) | 0.19%, 32.9 | 0.26%, 40.3 | −7.5 (−10.7 to −4.5) | 3 of 5 | Not met |
| WADI A2 (14) | 17.33%, 3.9 | 16.45%, 4.3 | −0.4 (−0.9 to +0.5) | 2 of 5 | Not met |
| SWaT (35) | 7.29%, 5.7 | 9.45%, 8.9 | −3.2 (−6.0 to −0.4) | 5 of 5 | Not met |

Monolithic models were trained on the union of the stage sensors; all thresholds from the first 15 minutes of each test stream. On HAI 21.03 the H1b advantage (training thresholds) reverses under commissioning (−7.5). The stage array's earlier edge was robustness of training-calibrated thresholds to drift, not better detection; where the plant can be calibrated on site, a monolithic model with per-sensor scoring does as well or better. A 15-minute window is too short for WADI and SWaT (FAR 16–17% and 7–9%). Evidence: `deployable_test.py --calibration commission`, `deployable_test_*_commission_summary.csv`.
