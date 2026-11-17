#!/bin/bash
# Phase A: rebuild every preprocessed dataset from the raw iTrust/HAI files.
set -x
source .venv/bin/activate
B="SWaT_Orig_DataSets/Original Dataset Backup"
python preprocess_swat.py --train "$B/SWaT_Dataset_Normal_v1.csv" --test "$B/SWaT_Dataset_Attack_v0.csv" --out_dir Datasets
python preprocess_wadi.py --release a2 --labels table --out_dir Datasets
python preprocess_hai.py --out_dir Datasets
python preprocess_swat_new.py
echo PHASE_A_DONE
