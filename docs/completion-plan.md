# Project completion plan

Draft based on repository inspection, 15 November 2026. Scope follows README.md; estimates and acceptance targets below are proposals, not measured results.

## Current state

The project has a README, completion plan, pinned dependency manifest, and Python-version file. There is no task implementation, training configuration, automated test suite, dataset pipeline, or recorded result. The README clearly defines PPO route following, a classical baseline, and held-out evaluation.

The `deep-rl` Conda environment was created on ARM64 macOS with Python 3.11.9. All pinned packages import, `pip check` reports no broken requirements, MetaDrive 0.4.3 assets were installed, and a 10-step headless rollout completed with finite observations. Rendered playback and the full 1,000-step lifecycle gate remain to be checked once the task wrapper exists.

## Completion scope

**Core deliverable:** a reproducible numerical-observation path-following experiment in MetaDrive, with PPO compared fairly against pure pursuit plus proportional speed control on held-out generated roads. One ego vehicle, no traffic, no image perception.

**Extension:** evaluate the frozen policy on a small, validated collection of reconstructed nuScenes scenarios. Require a successful conversion and route-validation pilot before committing time to this extension.

**Stretch:** nuPlan mini via ScenarioNet. Neither dataset extension should prevent completion of the core experiment. This is transfer between simulated scenario sources, not evidence of real-world deployment.

## Requirements

| Area | Required work and decisions |
| --- | --- |
| Runtime | CPython 3.11.x (target 3.11.9) in an isolated environment, using the pinned core stack in `requirements.txt`. MetaDrive 0.4.3 declares Python `<3.12`, so Python 3.12 or newer is outside the supported core setup. |
| Supporting tools | YAML or equivalent configuration, TensorBoard logging, pandas/Matplotlib for results, pytest for geometry and lifecycle checks. Keep optional dataset dependencies separate. |
| Simulator | Download matching assets; verify headless stepping and optional rendered playback; explicitly configure timestep, action repeat, vehicle, road types, traffic disabled, and episode horizon. |
| Compute | Begin with a CPU smoke test and throughput benchmark. Provision training compute only after measuring steps/second. A GPU is optional for a small MLP; simulator throughput may dominate. |
| Storage | Space for assets, checkpoints, normalization statistics, logs, metrics, and demo videos. Estimate actual dataset storage after choosing the subset; keep large artifacts outside Git. |
| Reference route | Ordered centerline/path points, consistent coordinates and units, robust nearest-segment projection, route progress, heading, and lookahead sampling. Handle junctions, endpoints, and nearby parallel segments. |
| Observations | Speed, signed lateral error, wrapped heading error, and a fixed number of upcoming points in ego coordinates. Specify ranges, point spacing, clipping, padding, dtype, and normalization. |
| Actions | Verify simulator steering and acceleration/brake conventions; define bounded continuous action space, steering scaling, speed target/limit, and control frequency. |
| Reward | Progress reward, lateral/heading penalties, time cost, success bonus, and off-road penalty. Log terms separately; prevent reverse motion, repeated crossings, or oscillation from earning artificial progress. |
| Episode rules | Define success using route completion, off-road and crash termination, timeout truncation, stall handling, and end-of-route behavior. Preserve terminated versus truncated semantics for PPO. |
| Baseline | Pure-pursuit steering and proportional speed control; tune lookahead, speed, and gains only on training/validation roads. Use the same route, vehicle, action constraints, and termination rules as PPO. |
| Training | PPO MLP policy; seeded configuration, checkpoint/resume support, validation callback, saved observation statistics, and logged hyperparameters/code revision. |
| Evaluation | Immutable split manifests; frozen policy and normalization; identical episode seeds per controller; raw episode metrics and aggregation with uncertainty. |
| Delivery | Installation instructions, train/evaluate/demo commands, reproducible configs, checkpoint retrieval instructions, results plots/table, report, limitations, and attribution. |

The core dependency set is deliberately conservative and mutually compatible rather than a collection of latest releases: MetaDrive 0.4.3, Stable-Baselines3 2.4.1, Gymnasium 0.29.1, PyTorch 2.5.1, and NumPy 1.26.4. Supporting packages are pinned in `requirements.txt`. MetaDrive requires separate simulator assets and still needs a smoke test on this ARM64 macOS machine. After that test passes, record a complete resolved environment with `python -m pip freeze` alongside the experiment artifacts. ScenarioNet, nuScenes, and nuPlan dependencies are optional and should use a separate requirements file/environment when that phase begins.

## Implementation phases and acceptance gates

### 1. Establish a working environment — estimated 1–2 working days

Create the Python 3.11.9 virtual environment from `.python-version`, install `requirements.txt`, then create the source package, configuration directory, and setup instructions. Pull MetaDrive assets. Run headless reset/step/close and a rendered demonstration when supported. Benchmark a short run and record machine details. Treat `requirements.txt` as the direct dependency baseline and save the fully resolved `pip freeze` output for final experiment reproducibility.

**Gate:** a fresh environment can run at least 1,000 finite-observation steps and repeatedly reset/close without errors. If local graphics or platform compatibility blocks progress, validate a Linux environment before proceeding.

### 2. Implement the task and route geometry — 2–3 days

Build route extraction/projection and the observation/reward wrapper. Start with straight and curved roads before junctions. Add focused tests for lateral-error sign, heading wraparound, progress, endpoint padding, observation shape/range, and termination/truncation. Run the SB3 environment checker and inspect a rollout visually.

**Gate:** errors and progress agree with known vehicle placements, observations remain finite, and completion/off-road/timeout outcomes are correctly distinguished. Route extraction works for every selected road family.

### 3. Build the baseline and evaluator — 2–3 days

Implement pure pursuit and speed control. Create seeded split manifests and episode logging before PPO training. Validate straight-road and curved-road driving; tune on validation layouts only. Fix geometry or control scaling before treating poor baseline behavior as an experimental result.

**Gate:** the baseline reliably completes simple reference routes, and repeated seeded evaluations produce consistent metrics within documented simulator tolerances.

### 4. Train and debug PPO — 3–5 days plus measured compute time

First train on a single easy road, then a small set, then diverse generated layouts. Inspect reward components, failure videos, progress, and success rate. Save the best validation checkpoint and its matching normalization statistics. Verify resumed training and checkpoint loading.

**Gate:** PPO shows learning beyond its initial policy and can solve the simple task. Select checkpoints using validation data only. If learning fails, inspect observations, action scaling, reward magnitudes, and time-limit handling before scaling compute.

### 5. Run final experiments — 2–3 days plus compute time

Proposed initial protocol: 100 training layouts, 20 validation layouts, 50 test layouts; at least three independent PPO training seeds. These counts are adjustable after profiling, before final evaluation. Check layout identity as well as seed separation to avoid accidental duplicate roads. Use matched test initial conditions and report per-training-seed results.

Compare PPO and baseline on completion, off-road rate, mean absolute and RMS lateral error, and completion time on successful episodes. Also report timeout/crash counts, route progress on failures, and speed. Define off-road as an episode event and compute tracking error on every executed step, including failed episodes. Report sample counts and confidence intervals; distinguish uncertainty across layouts from variability across training seeds. Successful-only times are subject to selection bias, so show success counts alongside them.

Include one small ablation if budget permits, such as removing lookahead points. Do not use the test set to retune a failed model.

**Gate:** one evaluation command produces raw results and the report table from saved artifacts. A scientifically complete project need not show PPO outperforming the baseline; failures require honest analysis, not hidden runs.

### 6. Package results and submission — 1–2 days

Write methodology, reward/observation definitions, baseline tuning procedure, learning curves, final comparisons, representative failure cases, limitations, and citations. Provide a short demo and exact reproduction commands. Have the other team member follow setup and evaluation instructions in a fresh environment.

**Gate:** another person can reproduce evaluation from the checkpoint, statistics, configs, and split manifests without undocumented local files.

### Optional transfer experiment — budget only after core completion

Check nuScenes access/license, supported conversion workflow, map compatibility, disk needs, and route availability. Convert a tiny subset first. Validate drivable reference paths, coordinate transforms, spawn states, episode endings, and the same observation/action contract. Disable other actors consistently with the core task. Evaluate frozen weights/statistics and report this source separately. Any fine-tuning is a separate experiment. Apply the same gate to nuPlan mini; ScenarioNet documents a nuPlan conversion command but that alone does not establish suitability for this project's task.

## Suggested layout

```text
configs/                 # environment, PPO, baseline, evaluation settings
splits/                  # versioned train/validation/test manifests
src/path_following/
  env.py                 # task wrapper and lifecycle
  route.py               # projection, errors, lookahead points
  baseline.py            # pursuit and speed controller
  train.py               # PPO training and checkpointing
  evaluate.py            # common controller evaluation
  metrics.py             # episode metrics and aggregation
  demo.py                # rendered policy/baseline playback
scripts/                 # setup checks and report generation
 tests/                  # geometry, environment, evaluator checks
 docs/                   # methodology and results
```

Generated data, checkpoints, and runs remain in already-ignored artifact directories. Document where final artifacts are stored and how to retrieve them.

## Coordination and open decisions

Suggested split for the two-person team: one member owns environment/route geometry/baseline; the other owns training/evaluation/reporting. Agree on the environment and metric interfaces first and review each other's outputs.

Before implementation, settle the submission deadline and grading rubric, available compute/budget, initial road families, intended speed, target performance criteria, and whether dataset transfer is required by the course. None of these are specified in the repository. The estimated core effort above is 11–18 working days, not a calendar guarantee; installation and training time remain uncertain until profiled.

Immediate next step: complete phase 1, then freeze the task interface and implement route geometry. Do not begin expensive runs or dataset downloads before these gates pass.

## Primary references

- MetaDrive installation and assets: https://github.com/metadriverse/metadrive/blob/main/documentation/source/install.rst
- Stable-Baselines3 installation: https://stable-baselines3.readthedocs.io/en/master/guide/install.html
- MetaDrive project: https://github.com/metadriverse/metadrive
- ScenarioNet installation: https://scenarionet.readthedocs.io/en/latest/install.html
- ScenarioNet nuPlan conversion: https://scenarionet.readthedocs.io/en/main/nuplan.html
