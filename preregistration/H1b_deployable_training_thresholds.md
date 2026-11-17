# Pre-registered test H1b: is the stage array the better deployable detector?

Written Oct 7, 2026, before any model is trained or any data prepared for the two held-out releases. The pattern came from the secondary results of H1: at the label-free operating point the stages beat the monolithic control on HAI 23.05 and WADI (in means), but not on SWaT, where they catch fewer attacks (9.9 vs 12.4 beyond chance) at a fifth of the FAR. SWaT is a known counter-example, so even a positive result here supports the claim for the HAI plant family and WADI, not in general.

**Hypothesis H1b.** At the label-free operating point, the stage array with per-sensor scoring raises no more false alarms than the monolithic control with the same scoring, and catches at least as many attacks beyond chance.

**Held-out data:** HAI 22.04 (86 sensors, 58 attack segments) and HAI 21.03 (79 sensors, 50 attack segments), not used by any experiment so far. Training files and test files of each release are concatenated in file order; labels are the release's own attack column.

**Frozen design:** stages = the four processes, every sensor named `P1_` … `P4_` assigned by its prefix; the monolithic control uses the same sensors. Models, scoring and decision rule exactly as in H1 (λ = 0, seeds 1–5, each sensor's squared error standardized on training errors, max, smoothed over 25 samples, dual threshold P99.9 / P99 of training scores, 50-sample persistence, the fault rule fixed on 2026-10-07).

**Success, per release (both conditions):** (a) the stage array's FAR is at most the monolithic control's on at least 4 of 5 seeds; (b) the 95% CI of stages minus monolithic in attacks beyond chance (per-attack chance from circular shifts, seed-averaged, bootstrapping attacks) has its lower bound above −1 attack. A lower bound above 0 is reported as the stronger result. Both releases are reported whatever they show.

### Result of H1b

H1b is supported on one of the two held-out releases: strongly on HAI 21.03, not on HAI 22.04, where training-calibrated thresholds fail for both detectors (about half of all normal samples alarmed), so neither gives a usable operating point.

| Release | Detector | FAR | Caught | Beyond chance | False-alarm episodes | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| HAI 21.03 (50) | Stage array | 6.99% | 48.0 | 40.4 | 147 | Supported: lower FAR on 5 of 5 seeds; +7.1 beyond chance (95% CI +6.7 to +7.5) |
| HAI 21.03 (50) | Monolithic | 19.45% | 48.0 | 33.3 | 297 |  |
| HAI 22.04 (58) | Stage array | 52.63% | 47.0 | 11.2 | 212 | Not supported: lower FAR on 2 of 5 seeds; −0.3 beyond chance (95% CI −5.0 to +4.5) |
| HAI 22.04 (58) | Monolithic | 48.98% | 46.4 | 11.5 | 233 |  |

On 21.03 both detectors catch the same 48 attacks; the stage array's advantage is that it does so with a third of the false-alarm rate and half the false-alarm episodes, so its chance level is lower. Even so, 7% FAR is not deployable as it stands. On both held-out releases, as on HAI 23.05, thresholds from training data transfer poorly to the test period; the paper's 15-minute commissioning window exists for that reason. Evidence: `deployable_test.py`, `deployable_test_{hai2103,hai2204}_seeds.csv`, `deployable_test_{hai2103,hai2204}_summary.csv`.
