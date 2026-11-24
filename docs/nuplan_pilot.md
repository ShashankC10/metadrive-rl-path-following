# nuPlan mini evaluation (stretch goal)

Frozen PPO checkpoints (trained only on generated MetaDrive roads) and the pure-pursuit baseline evaluated on
nuPlan v1.1 mini scenarios converted with ScenarioNet. No training, tuning or fine-tuning on these scenarios.
This is transfer between simulated scenario sources, not a real-world claim.

## Setup
- Data: nuPlan v1.1 mini log databases (64 logs) + nuPlan maps v1.0 (the version v1.1 uses), kept outside the
  repo (`~/nuplan/dataset`). Only maps and logs are used, no sensor data.
- **The whole mini split was converted and evaluated:** all 64 log databases -> 2,647 scenarios (13 GB database).
  An earlier run on a balanced 8-log subset (2 logs per city, 483 scenarios, 370 valid) gave the same picture.
- Conversion (separate Python 3.9 env with the nuplan-devkit source install + ScenarioNet; see "Environment notes"):
  `python scripts/convert_nuplan_subset.py -d ~/data/nuplan_full_db --logs-per-city 999`
  (`--logs-per-city 2` gives the small balanced subset).
- Whole run, resumable, meant to be run under `caffeinate -dims`: `scripts/run_nuplan_full.sh`; evaluation by
  `python scripts/pilot_scenarionet.py --db ~/data/nuplan_full_db --out outputs/nuplan_full --models checkpoints/ppo_s*/best_model.zip`
- Same task contract as the core experiment and the nuScenes pilot: same observation, action space, termination
  rules; no traffic or traffic lights. Reference route = the logged ego trajectory; the vehicle spawns on the logged
  pose, heading and speed.

## Route validation (1,788 of 2,647 scenarios usable)
Same checks as the nuScenes pilot (>= 40 m, >= 95% of route within 2 m of a mapped lane, no heading jump), plus a
duplicate-route filter.

| reason | scenarios |
| --- | --- |
| valid | 1,788 |
| too short / stationary | 840 |
| route leaves mapped lanes | 18 |
| duplicate route | 1 |

By city (valid / total): Las Vegas 1,328/1,987, Singapore One North 188/234, Boston 145/235, Pittsburgh 127/191.
Las Vegas is 48 of the 64 logs, so any pooled number is mostly Las Vegas.

## Results (1,788 scenarios, one deterministic episode each)

| controller | success | off-road / crash / stall / timeout | mean abs lateral (m) | mean speed (m/s) | mean time (s) |
| --- | --- | --- | --- | --- | --- |
| pure pursuit | 1788/1788 | 0 / 0 / 0 / 0 | 0.029 | 7.7 | 15.6 |
| PPO seed 0 | 1788/1788 | 0 / 0 / 0 / 0 | 0.022 | 9.6 | 12.6 |
| PPO seed 1 | 1788/1788 | 0 / 0 / 0 / 0 | 0.041 | 9.6 | 12.5 |
| PPO seed 2 | 1788/1788 | 0 / 0 / 0 / 0 | 0.023 | 9.6 | 12.6 |

Mean absolute lateral error by city (pure pursuit / PPO s0 / s1 / s2), metres:
Las Vegas 0.016 / 0.016 / 0.039 / 0.016; Boston 0.036 / 0.035 / 0.042 / 0.044;
Singapore 0.076 / 0.040 / 0.043 / 0.035; Pittsburgh 0.088 / 0.051 / 0.060 / 0.055.
Per-episode data: `outputs/nuplan_full/episodes_*.csv` (git-ignored).

## Sanity checks
- Spawn pose (12 random valid scenarios from the 8-log run): 0.000 m from the route start, 0 lateral error,
  heading error <= 0.16 rad, start speeds 0.8 to 13.0 m/s.
- Control for the off-road detector: full left steering with throttle 0.5 went off-road in 12 of 12 of those
  scenarios, so a 0% off-road rate is not a blind spot.

## Limitations
- **Not independent.** The 1,788 scenarios are windows from 64 logs, often on the same streets, so the effective
  sample size is much smaller than 1,788. Do not read an interval over scenarios as a per-scenario estimate.
  Success rates are 100% for all controllers, so no interval is reported.
- The reference is a human driver's path; traffic, lights and pedestrians are ignored; start speeds reach 13 m/s
  (training always started from rest).
- 859 of 2,647 scenarios (32%) were excluded, mostly stationary; these are the ones testing waiting and yielding.
- The task saturates (the generated-road test set is also 100%), so there is little room to separate controllers
  on success; tracking error and speed differ, as in the core experiment (PPO faster, baseline's target speed is 8 m/s).
- Not a benchmark result: no nuPlan metrics, planner interface or closed-loop traffic simulation were used.

## Environment notes (what it took)
- Python 3.9 env with `pip<24.1` (hydra-core 1.1.0rc1 metadata), `omegaconf==2.1.0rc1`, `numpy==1.23.4`, and
  `bokeh==2.4.3`, plus about a dozen other packages. The PyPI `nuplan-devkit` wheel lacks the Hydra YAML configs, so
  install the devkit from the GitHub clone (`pip install --no-deps -e nuplan-devkit`).
- **Memory:** MetaDrive's `ScenarioEnv` defaults to `store_map=True, store_data=True`, which caches every loaded
  scenario in RAM; with ~1,800 distinct nuPlan scenarios this exhausted an 18 GB Mac (process killed, swap filled
  the disk). `ScenarioPathFollowingEnv` now sets both to False (RSS stays ~0.8 GB), and the pilot script streams
  scenarios and saves each controller's results as it finishes.
- Set `NUPLAN_DATA_ROOT`, `NUPLAN_MAPS_ROOT`, `NUPLAN_EXP_ROOT`. Run converters from a directory that does not
  contain a folder named `scenarionet`.
