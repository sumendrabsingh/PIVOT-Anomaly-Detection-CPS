#!/bin/bash
source .venv/bin/activate
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTHONWARNINGS=ignore
until grep -q PHASE_A_DONE logs_A_data.log; do sleep 30; done
S=./run_step.sh
$S wadi_decomp_confirm python decomposition_confirm.py --dataset wadi
$S hai_decomp_confirm python decomposition_confirm.py --dataset hai
$S hai2204_deployable python deployable_test.py --release hai2204
$S hai2103_deployable python deployable_test.py --release hai2103
$S hai2007_commission python deployable_test.py --release hai2007 --calibration commission
for r in hai2007 hai2204 hai2103; do
  $S ${r}_orvote_confirm python orvote_confirm.py --release $r
  $S ${r}_orvote_mechanism python orvote_mechanism.py --release $r
  $S ${r}_deploy_window python orvote_deploy.py --release $r
  $S ${r}_deploy_shift python orvote_deploy.py --release $r --calibration shift
done
for d in wadi hai; do
  $S ${d}_orvote_check python orvote_check.py --dataset $d
  $S ${d}_physics_or_test python physics_or_test.py --dataset $d
  $S ${d}_calibration_dev python calibration_dev.py --dataset $d
done
for r in wadi hai2305 hai2204 hai2103; do $S ${r}_commission python deployable_test.py --release $r --calibration commission; done
$S wadihai_baselines_simple python baselines_simple_generalization.py
$S hai_temporal_train python temporal_detector.py --dataset hai --window 60 --train
$S hai_temporal_eval python temporal_detector.py --dataset hai --window 60
echo CHAIN_WADIHAI_DONE >> logs/STATUS
