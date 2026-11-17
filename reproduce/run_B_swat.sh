#!/bin/bash
source .venv/bin/activate
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS=ignore
until grep -q PHASE_A_DONE logs_A_data.log; do sleep 30; done
S=./run_step.sh
$S swat_decomp_ablation python decomposition_ablation_label_free.py
$S swat_physics_ablation python physics_ablation_label_free.py
$S swat_decomp_mechanism python decomposition_mechanism.py
$S swat_baselines_simple python baselines_simple_label_free.py
$S swat_orvote_check python orvote_check.py --dataset swat
$S swat_physics_or_test python physics_or_test.py --dataset swat
$S swat_calibration_dev python calibration_dev.py --dataset swat
$S swat_commission python deployable_test.py --release swat --calibration commission
for r in swat2019jul swat2019dec swat2022nov; do $S ${r}_shift python orvote_deploy.py --release $r --calibration shift; done
$S swat_matched_far_attack python matched_far_attack.py
$S swat_feasible_attack python feasible_attack_label_free.py
echo CHAIN_SWAT_DONE >> logs/STATUS
