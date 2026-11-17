# Label corrections

Three labelling problems were found and fixed. Each can be re-checked from the raw files with
`python validate_data_claims.py` (output: `results/validation/validation_data.txt`).

## HAI 23.05 — test2 attacks lost by a timestamp join

`label-test2.csv` has minute-resolution timestamps while `hai-test2.csv` has seconds, so a join on the timestamp
matches 0% of the test2 rows and labels all of them normal. Joined that way the test set has 14 attack segments
(test1 only); aligned by row it has 52 (test1: 14 segments, 2,981 rows; test2: 38 segments, 8,403 rows).
`preprocess_hai.py` aligns the labels by row and checks the row counts and times.

`hai2305_test_labels_corrected.csv`: one row per test row (test1 then test2), columns `label_old_test1_only`
(timestamp join) and `label_corrected` (row alignment).

## WADI — attacks dropped by corrupted dates

In the A1 release, `attack_description.xlsx` has corrupted date cells, so parsing it yields 8 labelled events
(6,732 rows) instead of 14 (9,985 rows). `preprocess_wadi.py` uses an explicit `ATTACK_WINDOWS` table taken from
the maintainers' attack table. The paper evaluates on the A2 release with the maintainers' table (14 attacks,
9,948 rows; `--release a2 --labels table`). The A2 label file itself places attack 13 at 00:07 instead of 12:07.

`wadi_a1_test_labels_corrected.csv`: one row per A1 test row, columns `y_submitted` (8 events) and `y_corrected`
(14 events).

## SWaT July 2019 — clock offset between data and attack list

In the July 2019 release every documented action appears about 105 s before the time given in the attack PDF
(four second-resolution anchors: -104, -98, -110, -107 s; median -105.5 s). `preprocess_swat_new.py` shifts the
July attack times and the plant start by -105 s (`CLOCK_OFFSET_S`); December 2019 and November 2022 agree with
their documentation to within a minute and are not shifted. No label file is shipped for this release: the labels
are built from the attack windows listed in `preprocess_swat_new.py`.
