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

## Usage

```bash
pip install -e .                                   # or: export PYTHONPATH=$PWD/src
python scripts/make_splits.py                      # immutable layout manifests -> splits/ (committed)
python scripts/tune_baseline.py                    # pure-pursuit tuning on train layouts -> configs/baseline_params.json
scripts/run_all_training.sh                        # 3 PPO seeds + 3 no-lookahead ablation seeds (~10 min each on an M3 Pro)
scripts/eval_all.sh                                # val + test evaluation -> outputs/eval/*.csv
python scripts/make_report.py                      # docs/results.md + docs/figures/
python scripts/plot_trajectories.py --model checkpoints/ppo_s0/best_model.zip --layouts 2000 2001 2002
python -m pytest                                   # geometry + simulator lifecycle tests
python -m path_following.demo --controller ppo --model checkpoints/ppo_s1/best_model.zip --layout 2000   # rendered playback (needs a display)
```

Layout: `src/path_following/` (route geometry, env, baseline, train, evaluate, metrics, demo), `configs/`,
`splits/`, `scripts/`, `tests/`, `docs/`. Checkpoints (`checkpoints/`), logs (`runs/`) and raw results
(`outputs/`) are git-ignored. Observations use fixed scaling constants, so there are no learned
normalization statistics to save or freeze. Optional cluster use: `docs/hprc.md`.

**Rendering on macOS:** stock MetaDrive shows a blank white road on Apple GPUs (the default OpenGL 2.1 context can't compile
its `#version 150` terrain shader, and the core-profile context rejects legacy `texture2D()`). `macos_render.py`
applies a local fix (core profile, `texture2D`->`texture`, lower MSAA) without editing MetaDrive; `demo.py` uses it
automatically. Use it in your own code with `from path_following import macos_render; macos_render.apply()` before creating a rendered env.

## Status

The core experiment is implemented and run: PPO and pure pursuit evaluated on 50 held-out generated roads
(`docs/results.md`). A 10-scene nuScenes-mini transfer pilot (7 valid scenes) is in `docs/nuscenes_pilot.md`;
nuPlan has **not** been attempted.

Test results (50 held-out layouts, one seed per layout): all three PPO seeds and the baseline complete 50/50
roads. PPO is faster (about 9.7 vs 7.8 m/s; the baseline's target speed is 8 m/s while PPO is rewarded up to
10 m/s) and tracks the lane centre more tightly than pure pursuit (mean |lateral| 0.035-0.060 m vs 0.083 m),
but the task is easy and success saturates, so the main differences are speed and tracking error, not
completion. PPO without lookahead points reached 98-100% success with even smaller lateral error here, so
this setup does not show that lookahead is needed. Limitations: single lane-centre reference, no traffic,
one baseline speed setting, three seeds.

## Attribution

The project builds on [MetaDrive](https://github.com/metadriverse/metadrive), [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3), and potentially [ScenarioNet](https://github.com/metadriverse/scenarionet). Borrowed code and ideas must be attributed where used.

Our planned contribution is the task configuration, observation and reward adaptations, baseline controller, and evaluation pipeline.
