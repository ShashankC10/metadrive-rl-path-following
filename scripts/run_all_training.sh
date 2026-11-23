#!/usr/bin/env bash
# Train 3 PPO seeds (main) and 3 seeds of the no-lookahead ablation, sequentially. Resumable.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/src
for s in 0 1 2; do
  python -m path_following.train --run ppo_s$s --seed $s --config configs/default.yaml --resume
done
for s in 0 1 2; do
  python -m path_following.train --run abl_s$s --seed $s --config configs/ablation_no_lookahead.yaml --resume
done
