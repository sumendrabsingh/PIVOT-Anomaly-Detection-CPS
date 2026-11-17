# Pre-registered test H4: PIVOT's OR vote calibrated on site

Written Oct 7, 2026, before the OR vote has been calibrated on any commissioning window. H3 compared the designs at matched FAR, set on the normal test rows; H4 asks whether PIVOT still wins when both detectors are calibrated the way a deployed system would be, from a short commissioning window at the start of operation.

**Hypothesis H4.** With every threshold taken from a 30-minute commissioning window, PIVOT's OR vote raises no more false alarms than the monolithic detector and catches more attacks beyond chance.

**Frozen design:**

- Releases, models and scoring exactly as in H3 (HAI 20.07, 21.03, 22.04; four stages by sensor prefix; seeds 1–5; per-sensor standardized error, max per stage, smoothed over 25 samples).
- Commissioning window: the first 1,800 samples (30 minutes) of each test stream. 30 minutes is the longest round window free of attacks on all three releases (HAI 21.03's first attack starts at 35 minutes); the attack positions were used only to check that, never to set a threshold.
- Decision rule: PIVOT's dual threshold (spike or 50-sample persistence). The monolithic detector takes τH and τL at the window's 99.9th and 99th percentiles of its score. PIVOT splits the same budget over its K = 4 stages: each stage takes τH and τL at the 1 − 0.001/K and 1 − 0.01/K quantiles of its own window score, and the stages are OR-ed.

**Success, per release:** supported if (a) PIVOT's FAR on the whole test stream is at most the monolithic detector's on at least 4 of 5 seeds, and (b) the 95% CI of PIVOT minus monolithic in attacks beyond chance (seed-averaged, bootstrapping attacks) lies above zero. If (a) holds and the lower bound is above −1 attack only, the result is reported as non-inferior, not supported.

**Secondary:** point-wise F1, early detection, false-alarm episodes, and the same rule with a 15-minute window.

### Result of H4

H4 is not supported on any release: calibrated from a 30-minute commissioning window, PIVOT's OR vote raises 1.4 to 70 times the monolithic detector's false-alarm rate, and it catches more attacks beyond chance only on HAI 22.04.

| Release | PIVOT FAR | Monolithic FAR | PIVOT minus monolithic, beyond chance (95% CI) | PIVOT FAR lower on | Verdict |
| --- | --- | --- | --- | --- | --- |
| HAI 20.07 (38) | 6.92% | 4.87% | −1.9 (−6.9 to +3.3) | 2 of 5 seeds | Not supported |
| HAI 21.03 (50) | 16.94% | 0.24% | −13.8 (−17.3 to −10.1) | 0 of 5 | Not supported |
| HAI 22.04 (58) | 12.64% | 3.82% | +7.2 (+1.7 to +12.7) | 1 of 5 | Not supported |

A 30-minute window gives each stage 1,800 samples, so its threshold at a quarter of a 0.1% budget is in effect the window's maximum, and each stage's later normal variation exceeds that maximum often; four such stages OR-ed multiply the false alarms (up to 1,068 episodes on HAI 21.03, against 4). The monolithic detector, whose one threshold pools all sensors, transfers from the same window far better. The 15-minute window is worse for PIVOT (FAR 15.6–17.9%). PIVOT's advantage at matched FAR (H3) is therefore a property of the design that the obvious on-site calibration does not deliver; per-stage thresholds need more calibration data than one short window provides. Evidence: `orvote_deploy.py`, `orvote_deploy_{hai2007,hai2103,hai2204}_{summary,seeds}.csv`.
