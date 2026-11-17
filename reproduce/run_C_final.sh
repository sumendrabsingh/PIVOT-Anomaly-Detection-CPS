#!/bin/bash
source .venv/bin/activate
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS=ignore
until grep -q CHAIN_SWAT_DONE logs/STATUS 2>/dev/null && grep -q CHAIN_WADIHAI_DONE logs/STATUS 2>/dev/null; do sleep 60; done
./run_step.sh leakage_effect python leakage_effect.py
mkdir -p PaperV3/figures
./run_step.sh paper_figures_v3 python paper_figures_v3.py
echo ALL_DONE >> logs/STATUS
