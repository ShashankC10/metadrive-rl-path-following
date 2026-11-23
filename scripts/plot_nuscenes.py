"""Plot converted nuScenes scenes: mapped lanes, logged ego route, and controller trajectories."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from path_following.baseline import PurePursuitParams
from path_following.config import load_config
from path_following.evaluate import baseline_controller, ppo_controller
from path_following.scenario_env import ScenarioPathFollowingEnv, ego_route_points, load_scenarios


def rollout(env, ctrl, idx):
    obs, _ = env.reset(options={"layout_seed": idx})
    xy, done = [], False
    while not done:
        obs, _, te, tr, info = env.step(ctrl(env, obs)); xy.append(env.state()["pos"]); done = te or tr
    return np.array(xy), info["episode_metrics"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="~/data/nuscenes_db")
    ap.add_argument("--model", required=True)
    ap.add_argument("--scenes", type=int, nargs="+", default=[0, 1, 3, 5, 6, 7, 8])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--params", default="configs/baseline_params.json")
    ap.add_argument("--out", default="docs/figures/nuscenes_scenes.png")
    a = ap.parse_args()
    cfg = load_config(a.config)["env"]
    scns = load_scenarios(a.db)
    env = ScenarioPathFollowingEnv(cfg, db=a.db)
    bl = baseline_controller(cfg, PurePursuitParams(**json.loads(Path(a.params).read_text())))[0]
    ppo = ppo_controller(a.model)
    n = len(a.scenes); cols = 2; rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(7.5 * cols, 5.2 * rows), squeeze=False)
    for ax, i in zip(axes.ravel(), a.scenes):
        s = scns[i]
        shift = ego_route_points(s)[0]  # MetaDrive re-centres each scene so the ego starts at (0, 0)
        for v in s["map_features"].values():
            g = v.get("polyline", v.get("polygon"))
            if g is None: continue
            p = np.asarray(g)[:, :2] - shift
            if v["type"].startswith("LANE_SURFACE"): ax.plot(p[:, 0], p[:, 1], color="0.75", lw=0.8, zorder=1)
            elif v["type"].startswith("ROAD_EDGE"): ax.plot(p[:, 0], p[:, 1], color="0.35", lw=0.8, zorder=1)
            elif v["type"] == "CROSSWALK": ax.plot(p[:, 0], p[:, 1], color="tab:olive", lw=0.8, zorder=1)
        xb, mb = rollout(env, bl, i)
        route = env.route.points
        xp, mp = rollout(env, ppo, i)
        ax.plot(route[:, 0], route[:, 1], "k-", lw=5, alpha=0.2, label="logged ego route", zorder=2)
        ax.plot(xb[:, 0], xb[:, 1], "--", color="tab:orange", lw=1.8, label="pure pursuit", zorder=3)
        ax.plot(xp[:, 0], xp[:, 1], "-", color="tab:blue", lw=1.5, label="PPO", zorder=4)
        ax.plot(*route[0], "go", ms=7, zorder=5); ax.plot(*route[-1], "rs", ms=7, zorder=5)
        lo, hi = route.min(0) - 35, route.max(0) + 35
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
        ax.set_title(f"[{i}] {s['id']} ({s['metadata'].get('map')}), {env.route.length:.0f} m\n"
                     f"pursuit {mb['reason']} |lat|={mb['mean_abs_lat']:.2f} m | PPO {mp['reason']} |lat|={mp['mean_abs_lat']:.2f} m", fontsize=9)
    for ax in axes.ravel()[n:]: ax.axis("off")
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.tight_layout(); Path(a.out).parent.mkdir(parents=True, exist_ok=True); fig.savefig(a.out, dpi=110)
    env.close()


if __name__ == "__main__":
    main()
