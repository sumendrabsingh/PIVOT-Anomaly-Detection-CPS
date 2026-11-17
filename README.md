# PIVOT — code, results and pre-registrations for the evaluation paper

This repository accompanies the paper

> **Deconstructing Stage-Wise Anomaly Detection in Industrial Control Systems: Drift Isolation, Adversarial
> Limits, and Pre-Registered Benchmarks**
> Sumendra B. Singh and Michael Tu, Purdue University Northwest.

**Matching tag:** the submitted manuscript corresponds to the tag **`v1.0`**.

`legacy_preliminary/` is the earlier version of this work that the paper refers to (the "Physics-Informed Voting
Over Subsystems" investigation); its exact original state is the tag `v0.1-preliminary`.

## Layout

| Path | Contents |
|---|---|
| `*.py` (repo root) | Code used in the paper. Scripts use repo-relative paths and are run from the repo root. |
| `Datasets/*.json` | Stage definitions and preprocessing metadata (no sensor data). |
| `results/` | The result files the paper's numbers are computed from; `results/validation/` is the record of the independent re-run. |
| `label_corrections/` | HAI 23.05, WADI and SWaT July 2019 label fixes. |
| `preregistration/` | The hypothesis texts H1–H6 with their outcomes. |
| `reproduce/` | The shell chains used for the independent re-run. |
| `legacy_preliminary/` | The earlier investigation, kept for reference. Not used by the paper. |

## Data

The datasets are not redistributed. Obtain them from their maintainers and place the raw files under
`SWaT_Orig_DataSets/`:

- SWaT (A1 Dec 2015; A4/A5 Jul 2019; A6 Dec 2019; A9 Nov 2022) and WADI (A2, 19 Nov 2019): iTrust, Singapore
  University of Technology and Design.
- HAI 20.07, 21.03, 22.04 and 23.05: <https://github.com/icsdataset/hai>.

## Reproducing the paper

Environment: Python 3.12; install PyTorch and the pinned packages as described at the top of
`requirements-linux.txt`. A GPU is needed in practice.

The independent re-run reported in the paper executed the chains in `reproduce/` in this order. They were written
for one machine (virtual environment in `.venv`, two GPUs selected by index, a `logs/` directory); copy them to the
repo root and adapt those lines.

1. `run_A_data.sh` — rebuild every preprocessed dataset from the raw files.
2. `run_B_swat.sh`, `run_B_wadihai.sh`, `run_D_attacks.sh` — train all models and run every test (H1–H6,
   baselines, attacks).
3. `run_C_final.sh` — leakage analysis. (It also draws the figures of an earlier draft; the paper's figures come
   from `python paper_figures_v4.py`, which writes to `PaperV4/figures/`.)
4. `python validate_data_claims.py`, `python validate_claims.py`, `python compare_results.py` — the checks behind
   `results/validation/`.
5. `python paper_numbers_v4.py` — writes `PaperV4/numbers.tex` (create the `PaperV4/` directory first).

The scripts read and write their result files in the repo root, and `paper_numbers_v4.py` reads the validation
record from `PaperV4/`. To recompute the numbers from the shipped results without retraining:

```bash
mkdir -p PaperV4 && cp results/*.csv . && cp results/validation/* PaperV4/
python paper_numbers_v4.py        # also needs the preprocessed datasets from step 1
```

Retraining on a GPU is not bit-reproducible. `results/validation/VALIDATION.md` and `CLAIMS.md` state which
results the re-run reproduced exactly, which within tolerance, and which differed.

## Pre-registration

`preregistration/` holds each hypothesis as it was written before its data were scored, followed by its result.
See `preregistration/README.md` for where these texts were kept and what their dates do and do not show.

## License and citation

See `LICENSE` and `CITATION.cff`.
