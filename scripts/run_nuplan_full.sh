#!/usr/bin/env bash
# Convert ALL nuPlan mini logs and evaluate frozen PPO + baseline on the valid scenarios.
# Run under caffeinate:  caffeinate -dims scripts/run_nuplan_full.sh
# Resumable: conversion is skipped if the database exists; finished controllers are reused.
set -euo pipefail
cd "$(dirname "$0")/.."
export NUPLAN_DATA_ROOT=$HOME/nuplan/dataset NUPLAN_MAPS_ROOT=$HOME/nuplan/dataset/maps NUPLAN_EXP_ROOT=$HOME/nuplan/exp
export PYTHONPATH=$PWD/src
DB=$HOME/data/nuplan_full_db
NUPLAN_PY=$HOME/data/envs/deep-rl-nuplan/bin/python
CORE_PY=/opt/anaconda3/envs/deep-rl/bin/python

if [ ! -f "$DB/dataset_summary.pkl" ]; then
  free_gb=$(df -g "$HOME" | awk 'NR==2{print $4}')
  echo "free disk: ${free_gb} GB"
  [ "$free_gb" -ge 22 ] || { echo "ABORT: need ~22 GB free for the ~17 GB database"; exit 1; }
  (cd /tmp && $NUPLAN_PY -W ignore -u "$OLDPWD/scripts/convert_nuplan_subset.py" -d "$DB" --logs-per-city 999 --num-workers 4 --overwrite)
fi
echo "database: $(du -sh "$DB" | cut -f1)"
$CORE_PY -W ignore -u scripts/pilot_scenarionet.py --db "$DB" --out outputs/nuplan_full \
  --models checkpoints/ppo_s0/best_model.zip checkpoints/ppo_s1/best_model.zip checkpoints/ppo_s2/best_model.zip
echo "ALL DONE"
