"""nuScenes pilot: validate converted scenarios, then evaluate frozen PPO and the baseline on the valid ones.

    python scripts/pilot_nuscenes.py --db ~/data/nuscenes_db --models checkpoints/ppo_s0/best_model.zip ...
Writes outputs/nuscenes/validation.csv and episodes.csv. Nothing is tuned or trained on these scenarios.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from path_following.baseline import PurePursuitParams
from path_following.config import load_config
from path_following.evaluate import baseline_controller, ppo_controller, run_episodes
from path_following.scenario_env import ScenarioPathFollowingEnv, ego_route_points, load_scenarios, validate_route


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="~/data/nuscenes_db")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--params", default="configs/baseline_params.json")
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--out", default="outputs/nuscenes")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(a.config)["env"]

    scns = load_scenarios(a.db)
    rows = []
    for i, s in enumerate(scns):
        ok, st = validate_route(s, ego_route_points(s))
        rows.append({"index": i, "scenario": s["id"], "map": s["metadata"].get("map"), "valid": ok, **st})
    val = pd.DataFrame(rows); val.to_csv(out / "validation.csv", index=False)
    print(val.to_string())
    good = val[val.valid]["index"].tolist()
    if not good:
        return

    env = ScenarioPathFollowingEnv(cfg, db=a.db)
    frames = []
    ctrl, _ = baseline_controller(cfg, PurePursuitParams(**json.loads(Path(a.params).read_text())))
    frames.append(run_episodes(env, ctrl, good, "pure_pursuit"))
    for m in a.models:
        frames.append(run_episodes(env, ppo_controller(m), good, Path(m).parent.name))
    env.close()
    df = pd.concat(frames); df.to_csv(out / "episodes.csv", index=False)
    print(df.groupby("controller")[["success", "off_road", "crash", "stall", "timeout", "progress", "mean_abs_lat"]].mean())


if __name__ == "__main__":
    main()
