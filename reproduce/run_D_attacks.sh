#!/bin/bash
source .venv/bin/activate
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONWARNINGS=ignore
CUDA_VISIBLE_DEVICES=1 ./run_step.sh swat_feasible_attack_retry python feasible_attack_label_free.py
./run_step.sh swat_matched_far_attack_retry python matched_far_attack.py --gpus 0,1
echo ATTACKS_DONE >> logs/STATUS
