# Calibration development and pre-registered test H5

Written Oct 7, 2026. A calibration method was chosen on the development plants only (SWaT, WADI A2, HAI 23.05; `calibration_dev.py`), before any of the methods below touched a held-out release. Four rules were compared, each applied to every score: **window** (thresholds as window quantiles, the H4 rule), **shift** (level from the window: τ = window median + (training quantile − training median)), **ls** (level and spread from the window, tail from training) and **pooled** (stages normalized by their window level and spread, one joint threshold from training). Shift won on development data:

| Development plant | PIVOT, shift: FAR / beyond chance / F1 | Monolithic, shift | Monolithic, window |
| --- | --- | --- | --- |
| SWaT (35) | 0.68% / 10.0 / 81.3 | 3.40% / 12.2 / 74.4 | 5.98% / 12.1 / 70.0 |
| WADI A2 (14) | 2.42% / 10.0 / 61.2 | 2.48% / 8.7 / 53.9 | 3.61% / 7.7 / 52.1 |
| HAI 23.05 (52) | 1.18% / 19.9 / 37.2 | 1.58% / 20.2 / 38.3 | 0.59% / 20.6 / 42.8 |

With the window rule PIVOT's FAR was 69.8%, 34.5% and 7.7% on the same plants.

**Hypothesis H5.** With every threshold calibrated by the shift rule from a 30-minute commissioning window, PIVOT's OR vote raises no more false alarms than the monolithic detector calibrated the same way, and catches more attacks beyond chance.

**Frozen design:** releases, models, scoring, window (first 1,800 samples) and dual-threshold rule exactly as in H4, including the budget split over K stages; only the calibration rule changes from window to shift, for both detectors.

**Success, per release (HAI 20.07, 21.03, 22.04):** supported if (a) PIVOT's FAR is at most the monolithic detector's on at least 4 of 5 seeds and (b) the 95% CI of PIVOT minus monolithic in attacks beyond chance lies above zero; non-inferior (not supported) if (a) holds and the lower bound is above −1 only. All three releases are reported. These releases were used in H3 and H4 with other rules, so this is a test of a calibration method chosen elsewhere, not of fresh data.

**Secondary:** the same comparison against the monolithic detector with the window rule (H4's control), which a deployment might prefer; F1, early detection, false-alarm episodes.

### Result of H5

H5 is not supported on any release: with shift calibration PIVOT catches more attacks beyond chance than the monolithic detector on HAI 20.07 and 22.04, but it raises more false alarms on most seeds everywhere, so condition (a) fails on all three releases.

| Release | PIVOT FAR | Monolithic FAR (shift) | PIVOT minus monolithic, beyond chance (95% CI) | FAR no higher on | Verdict |
| --- | --- | --- | --- | --- | --- |
| HAI 20.07 (38) | 13.87% | 11.41% | +7.7 (+3.3 to +12.2) | 3 of 5 seeds | Not supported |
| HAI 22.04 (58) | 7.34% | 4.27% | +4.1 (+0.1 to +8.4) | 2 of 5 | Not supported |
| HAI 21.03 (50) | 5.71% | 0.24% | −2.5 (−6.3 to +1.6) | 0 of 5 | Not supported |

Against the monolithic detector with the window rule (secondary), PIVOT is +7.1 (CI +2.3 to +12.0) on HAI 20.07 at 13.9% vs 4.9% FAR, +3.4 (CI −0.6 to +7.6) on HAI 22.04 and −2.7 on HAI 21.03. On HAI 22.04 PIVOT's false alarms are fewer but longer (36 episodes against 97 to 127) and its F1 is double (28.6% vs 13–14%); the pre-registered criterion counted alarmed samples, not episodes. Compared with H4, shift calibration cut PIVOT's FAR on HAI 21.03 (16.9% → 5.7%) and 22.04 (12.6% → 7.3%) but raised it on 20.07 (6.9% → 13.9%), where it also raised the monolithic detector's. Conclusion: calibrated on site, PIVOT trades a higher false-alarm rate for more detections on two of three releases; it does not dominate. No held-out data remain for further calibration tuning. Evidence: `calibration_dev.py`, `orvote_deploy.py --calibration shift`, `orvote_deploy_{hai2007,hai2103,hai2204}_shift_{summary,seeds}.csv`.
