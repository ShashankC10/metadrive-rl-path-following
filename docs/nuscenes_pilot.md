# nuScenes pilot (extension, 10 scenes)

Frozen PPO checkpoints (best-validation, trained only on generated roads) and the pure-pursuit baseline
evaluated on nuScenes v1.0-mini scenes converted with ScenarioNet. No training, tuning or fine-tuning on
these scenes. This is a pilot: transfer between simulated scenario sources, not a real-world claim.

## Setup
- Data: nuScenes v1.0-mini + map expansion v1.3 + CAN bus, kept outside the repo (`~/data/nuscenes`).
- Conversion (separate env, ScenarioNet + nuscenes-devkit):
  `python -m scenarionet.convert_nuscenes -d ~/data/nuscenes_db --split v1.0-mini --dataroot ~/data/nuscenes`
  (run from a directory that does not contain a folder named `scenarionet`; setuptools pinned `<81`).
- Evaluation runs in the core `deep-rl` env: `python scripts/pilot_scenarionet.py --db ~/data/nuscenes_db --out outputs/nuscenes --models checkpoints/ppo_s*/best_model.zip`.
  Code: `src/path_following/scenario_env.py`.
- Same task contract as the core experiment: same observation, action space, reward-independent metrics,
  termination rules, and no traffic or traffic lights (`no_traffic`, `no_light`).
- Reference route = the logged ego trajectory (subsampled to >= 0.5 m, backward pose jitter removed).
  The vehicle spawns on the logged ego pose, heading and speed.

## Route validation (7 of 10 scenes usable)
A scene is kept if its route is >= 40 m, >= 95% of route points lie within 2 m of a mapped lane centerline,
and no heading jump > 0.5 rad between 1 m segments. Raw table: `outputs/nuscenes/validation.csv` (git-ignored).

- Rejected (all three: the logged ego was waiting, so there is no path to follow). Descriptions are the nuScenes
  scene metadata; the observations come from the front-camera keyframes (first/middle/last), not from the
  converted traffic-light data:
  - scene-0553, Boston, 0 m: "Wait at intersection, bicycle, large truck, peds crossing crosswalk, ped with
    stroller". Stopped at a red light behind a crosswalk for the whole scene.
  - scene-0757, Boston, 27 m: "Arrive at busy intersection, bus, wait at intersection, bicycle, peds".
    Approaches an intersection, then waits at a red light while cross traffic passes.
  - scene-1100, Singapore (night), 1 m: "Night, peds in sidewalk, peds cross crosswalk, scooter, PMD, difficult
    lighting". Stationary at a junction with cross traffic passing; the signal in view looks green, so the reason
    for waiting is not clear from the images.
  These scenes exercise waiting at signals and yielding, which this task excludes by design (no traffic, lights
  or pedestrians). Front-camera montage (not committed, nuScenes images are research-only):
  `outputs/nuscenes/rejected_scenes_front_camera.png`.
- First attempt also rejected 0655, 1077, 1094: the end of the logged route jittered backward by decimetres
  while the ego was stopped. The route builder now skips backward steps; those three scenes are valid.

## Results (7 scenes, one deterministic episode each)

| controller | success | off-road | mean abs lateral (m) | mean speed (m/s) | time (s) |
| --- | --- | --- | --- | --- | --- |
| pure pursuit | 7/7 | 0 | 0.064 | 7.7 | 19.5 |
| PPO seed 0 | 7/7 | 0 | 0.038 | 9.7 | 15.5 |
| PPO seed 1 | 7/7 | 0 | 0.051 | 9.8 | 15.5 |
| PPO seed 2 | 7/7 | 0 | 0.053 | 9.7 | 15.5 |

With n = 7 the 95% Wilson interval for 7/7 is [0.65, 1.00], so this shows the policies are not obviously
broken on converted real-map geometry, not that they are equivalent.

## Limitations
- Seven scenes (Boston and Singapore), so wide uncertainty. No dataset-scale claim.
- The reference is a human driver's path, and the scenes' stops, other vehicles, pedestrians and traffic
  lights are ignored; both controllers simply drive the path at their own speeds.
- Start speeds up to 12.8 m/s occur (training always started from rest); speed is clipped by the fixed
  observation scaling.
- The converter's route and lane metadata come from ScenarioNet; they were validated only by the checks above.
- nuPlan mini (stretch goal) has not been attempted.
