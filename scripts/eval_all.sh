#!/usr/bin/env bash
# Evaluate everything on validation and test. Run once, after training; do not retune on test results.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/src
OUT=outputs/eval; mkdir -p $OUT
for split in val test; do
  python -m path_following.evaluate --controller baseline --params configs/baseline_params.json \
    --split splits/$split.json --label baseline --out $OUT/baseline_$split.csv
  for s in 0 1 2; do
    python -m path_following.evaluate --controller ppo --model checkpoints/ppo_s$s/best_model.zip \
      --split splits/$split.json --config configs/default.yaml --label ppo_s$s --out $OUT/ppo_s${s}_$split.csv
    python -m path_following.evaluate --controller ppo --model checkpoints/abl_s$s/best_model.zip \
      --split splits/$split.json --config configs/ablation_no_lookahead.yaml --label abl_s$s --out $OUT/abl_s${s}_$split.csv
  done
done
