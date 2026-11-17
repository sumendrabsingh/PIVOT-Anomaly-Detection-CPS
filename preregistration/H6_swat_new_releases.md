# Pre-registration H6 — shift-calibrated OR vote on new SWaT releases

Written 2026-10-09, before any new-release data was preprocessed or scored. Only file listings, headers, the
first rows and the iTrust documentation (readme, logs, Umer et al. 2026 attack tables) had been read.

## Question
H5 (HAI 20.07 / 21.03 / 22.04) was not supported. This is the confirmatory replication of the same rule on held-out
SWaT data, years after the 2015 training run: does PIVOT's OR vote, calibrated on site with the shift rule, raise no
more false alarms than the monolithic detector and catch more attacks beyond chance?

## Frozen design (identical to H5 except data)
- Models: trained on SWaT 2015 normal data (A1) only, never retrained. PIVOT = `Oracles/swat_phys_none_seed{1..5}`
  (six stages, analyzers excluded, lambda = 0); monolithic = `Oracles/swat_mono_noAIT_none_seed{1..5}`. These are the
  SWaT models used by `calibration_dev.py`, which chose the shift rule.
- Preprocessing: `preprocess_swat.transform_data` with `Datasets/preprocessing_meta.json` (2015 scaler); tag names
  mapped to the 2015 names (`FIT 101`, `FIT101.Pv`, `P101 Status`, `P101.Status` -> `FIT101`, `P101`); fault rule
  `label_free.fault_rule`. A model feature absent from a release is filled with the 2015 training mean (both detectors).
- Scoring: per-sensor standardized squared error (on 2015 training errors), max per stage / over all sensors,
  smoothed W = 25.
- Calibration: shift rule, tau = window median + (training quantile - training median), window = first 1800
  samples (30 min) of the stream; dual threshold W_dur = 50; PIVOT stages at 1 - 0.001/K and 1 - 0.01/K (K = 6),
  OR-ed; monolithic at 0.999 / 0.99.
- Success, per release: (a) PIVOT FAR <= monolithic FAR on >= 4 of 5 seeds and (b) 95% bootstrap CI (over attacks,
  seed-averaged) of PIVOT minus monolithic in attacks beyond chance lies above 0. (a) with lower bound > -1 only =
  non-inferior, not supported.
- Overall H6 supported if supported on >= 2 of 3 releases.

## Releases and labels (fixed now)
1. **SWaT A4/A5, 20 Jul 2019** (`SWaT_dataset_Jul 19 v2.xlsx`, GMT+0; attack list GMT+8). Stream starts at the
   documented plant start 12:35:00 (GMT+8). Six attacks, labelled from the PDF start/end times
   (to the second; "3:15" = 15:15:00, end "3:56" = 15:56:00).
2. **SWaT A6, 6 Dec 2019** (`Dec2019.xlsx`, GMT+8). Stream 10:05-13:45. Attacks = the five "Disrupt Sensor and
   Actuator" cycles from `Log.docx`: 12:30, 12:43, 12:56, 13:09, 13:22, each 3 min. The four historian-exfiltration
   cycles (10:30-11:20) do not command the process and are labelled normal.
3. **SWaT A9, 11 Nov 2022** (`11-Nov-2022_0900_1143.csv`). Attacks = the 17 launched rules dated that morning in
   Umer et al. Tables 2-3 (nos. 7-9, 11-24), each from start hh:mm:00 to end hh:mm:59. Rules "no impact observed"
   (7, 19) stay labelled. The 10 Nov file is excluded (first attack 3 min into the file: no attack-free window).
   Timestamps assumed local (GMT+8), as in the paper.

## Label-alignment check allowed before scoring
Plotting/inspecting raw actuator and sensor traces around the documented attack times (not detector scores) to
confirm the time zone. If an offset is found it is corrected and reported; no other label change.

## Power
6, 5 and 17 attacks. With 5-6 attacks the CI in (b) is wide; a non-significant result on A4/A5 or A6 is weak
evidence either way. Stated in advance.

## Secondary / exploratory (no verdict)
- Window rule (H4) and the monolithic detector with the window rule; F1; FA episodes; early detection; 15-min window.
- WaDi A3 (Dec 2023, 105 h, no attacks; first 5 h closed loop, then CrossOver mode): FAR of both detectors under the
  shift rule with A2 models (`Oracles/wadi_a2_none_seed*`, `wadi_a2_mono_none_seed*`), only if >= 95% of the A2
  model features can be mapped by tag name. False alarms only; it cannot test detection.

---
## Results (run 2026-10-09; appended after the run, text above unchanged)

Label-alignment check: Jul 2019 data runs ~105 s ahead of the PDF attack times (second-resolution starts FIT401
-106, P601 -104, MV201 -98, P301 -110 s); all Jul times shifted by -105 s (`CLOCK_OFFSET_S` in
`preprocess_swat_new.py`). Dec 2019 and Nov 2022 agree with the documentation to within a minute; unchanged.
Fault rule replaced AIT503 on Jul 2019 (not a model feature). WaDi A3: 81 of 120 A2 features map by name (67.5%,
FIC controllers and MCV outputs renamed or absent) -> secondary analysis not run, as pre-registered.

`python orvote_deploy.py --release <rel> --calibration shift` -> `orvote_deploy_<rel>_shift*.csv`, `.log`

| release | attacks | PIVOT FAR | mono FAR | (a) seeds | beyond chance PIVOT - mono (95% CI) | verdict |
|---|---|---|---|---|---|---|
| Jul 2019 | 6 | 62.4% | 37.3% | 0/5 | +0.65 (-0.68 .. +1.86) | not supported |
| Dec 2019 | 5 | 10.7% | 9.9% | 2/5 | +3.46 (+1.68 .. +4.66) | not supported (FAR) |
| Nov 2022 | 17 | 61.9% | 39.5% | 0/5 | +0.23 (-0.89 .. +1.43) | not supported |

**H6: NOT SUPPORTED (0 of 3 releases).** Models trained on the 2015 plant do not transfer to the 2019/2022 plant:
with the shift rule both detectors alarm on 10-62% of normal rows. Dec 2019 is the only release with a usable
operating point; there PIVOT catches all 5 attacks (3.5 beyond chance vs 0.1) at a FAR about equal to the
monolithic detector's, failing (a) narrowly. Same pattern as H5: on-site calibration trades FAR for detections.
