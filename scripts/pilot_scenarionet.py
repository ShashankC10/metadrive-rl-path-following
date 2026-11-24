"""Pilot on ScenarioNet-converted scenarios (nuScenes, nuPlan, ...): validate routes, then evaluate frozen PPO and the baseline.

    python scripts/pilot_scenarionet.py --db ~/data/nuscenes_db --out outputs/nuscenes --models checkpoints/ppo_s*/best_model.zip
    python scripts/pilot_scenarionet.py --db ~/data/nuplan_db   --out outputs/nuplan   --models checkpoints/ppo_s*/best_model.zip
Writes <out>/validation.csv and episodes.csv. Nothing is tuned or trained on these scenarios.
Routes that duplicate an earlier route (same rounded geometry) are rejected, so overlapping scenario windows count once.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from path_following.baseline import PurePursuitParams
from path_following.config import load_config
from path_following.evaluate import baseline_controller, ppo_controller, run_episodes
from path_following.scenario_env import ScenarioPathFollowingEnv, ego_route_points, iter_scenarios, validate_route


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

    vfile = out / "validation.csv"
    if vfile.exists():                               # validation is deterministic and slow on big databases: reuse
        print(f"reusing {vfile}", flush=True)
        rows, scns_iter = pd.read_csv(vfile).to_dict("records"), []
    else:
        rows, scns_iter = [], iter_scenarios(a.db)
    seen = set()
    for i, s in enumerate(scns_iter):   # streamed: large databases do not fit in RAM
        if i % 200 == 0:
            print(f"validating {i}...", flush=True)
        pts = ego_route_points(s)
        ok, st = validate_route(s, pts)
        sig = hashlib.md5(np.round(pts - pts[0], 0).tobytes()).hexdigest()[:12] if len(pts) else ""
        if ok and sig in seen:
            ok, st = False, {**st, "reason": "duplicate route"}
        if ok:
            seen.add(sig)
        rows.append({"index": i, "scenario": s["id"], "map": s["metadata"].get("map"), "valid": ok, **st})
    val = pd.DataFrame(rows); val.to_csv(vfile, index=False)
    if len(val) <= 30:
        print(val.to_string())
    print(f"\n{int(val.valid.sum())} of {len(val)} scenarios valid")
    print(val.groupby("map").agg(n=("valid", "size"), valid=("valid", "sum")).to_string())
    print(val[~val.valid].reason.value_counts().to_string())
    good = val[val.valid]["index"].tolist()
    if not good:
        return

    env = ScenarioPathFollowingEnv(cfg, db=a.db)
    frames = []
    def cached(label, make_ctrl):
        """Per-controller results are saved as they finish, so an interrupted run resumes here."""
        f = out / f"episodes_{label}.csv"
        if f.exists():
            print(f"{label}: reusing {f}", flush=True)
            return pd.read_csv(f)
        d = run_episodes(env, make_ctrl(), good, label, progress_every=100)
        d.to_csv(f, index=False)
        return d

    params = PurePursuitParams(**json.loads(Path(a.params).read_text()))
    frames.append(cached("pure_pursuit", lambda: baseline_controller(cfg, params)[0]))
    for m in a.models:
        frames.append(cached(Path(m).parent.name, lambda m=m: ppo_controller(m)))
    env.close()
    df = pd.concat(frames).merge(val[["index", "scenario", "map"]], left_on="layout_seed", right_on="index", how="left")
    df.to_csv(out / "episodes.csv", index=False)
    cols = ["success", "off_road", "crash", "stall", "timeout", "progress", "mean_abs_lat"]
    print(df.groupby("controller")[cols].mean().round(3).to_string())
    print(df.groupby(["map", "controller"])[["success", "off_road", "mean_abs_lat"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
