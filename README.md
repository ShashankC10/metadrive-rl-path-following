# MetaDrive RL Path Following

A practice-based course project that trains a reinforcement learning agent to follow a reference route in MetaDrive using continuous steering and acceleration/braking.

## Team

- Shri Gaddad
- Shashank Chandavarkar

## Planned approach

- Train Proximal Policy Optimization (PPO) with Stable-Baselines3 on procedurally generated MetaDrive roads.
- Use numerical observations including speed, lateral tracking error, heading error, and upcoming reference-path points.
- Reward route progress and successful completion while penalizing tracking error, time, and leaving the road.
- Compare with a pure-pursuit steering controller and proportional speed controller.
- Begin with one vehicle and no traffic or camera-based perception.

## Environment setup

The supported development interpreter is CPython 3.11.x; the pinned target is
Python 3.11.9 because MetaDrive 0.4.3 does not support Python 3.12 or newer.

```bash
conda create --name deep-rl python=3.11.9 pip -y
conda activate deep-rl
python -m pip install -r requirements.txt
python -m metadrive.pull_asset
```

The dependency set has been resolved and smoke-tested on macOS ARM64. PPO uses
an MLP, so CPU training is the default; GPU/MPS is not required.

## Evaluation

Measure route completion rate, off-road rate, lateral tracking error, and completion time on successful runs. Use separate training, validation, and test layouts; freeze the policy and observation normalization during evaluation.

Evaluate first on held-out generated roads, then explore reconstructed nuScenes scenarios in MetaDrive. A broader evaluation using nuPlan mini through ScenarioNet is a stretch goal. Dataset availability, conversion, and reference-path validity must be checked before these experiments.

This project studies transfer between scenario sources within simulation. It does not claim real-world deployment or use behavior cloning from human control labels.

## Status

Repository initialized. The `deep-rl` Conda environment and simulator assets
have been validated, while implementation and experiments are pending. The
approach above is the current project plan, not a report of completed results.

## Attribution

The project builds on [MetaDrive](https://github.com/metadriverse/metadrive), [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3), and potentially [ScenarioNet](https://github.com/metadriverse/scenarionet). Borrowed code and ideas must be attributed where used.

Our planned contribution is the task configuration, observation and reward adaptations, baseline controller, and evaluation pipeline.
