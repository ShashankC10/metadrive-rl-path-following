"""Headless plots of baseline vs PPO trajectories (colour = speed) on selected layouts."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from path_following.baseline import PurePursuitParams
from path_following.config import load_config
from path_following.env import PathFollowingEnv
from path_following.evaluate import baseline_controller, ppo_controller


def rollout(env, ctrl, layout):
    obs, _ = env.reset(options={"layout_seed": layout})
    xy, v, done = [], [], False
    while not done:
        obs, _, te, tr, info = env.step(ctrl(env, obs))
        st = env.state(); xy.append(st["pos"]); v.append(st["speed"]); done = te or tr
    return np.array(xy), np.array(v), info["episode_metrics"], env.route.points.copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layouts", type=int, nargs="+", required=True)
    ap.add_argument("--split", default="splits/test.json")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--params", default="configs/baseline_params.json")
    ap.add_argument("--out", default="docs/figures/trajectories.png")
    a = ap.parse_args()
    cfg = load_config(a.config)["env"]
    seeds = json.loads(Path(a.split).read_text())["seeds"]
    env = PathFollowingEnv(cfg, seeds=seeds)
    ctrls = {"pure pursuit": baseline_controller(cfg, PurePursuitParams(**json.loads(Path(a.params).read_text())))[0],
             "PPO": ppo_controller(a.model)}
    fig, axes = plt.subplots(len(a.layouts), 2, figsize=(11, 3.6 * len(a.layouts)), squeeze=False)
    for i, lay in enumerate(a.layouts):
        for j, (name, c) in enumerate(ctrls.items()):
            xy, v, m, route = rollout(env, c, lay)
            ax = axes[i, j]
            ax.plot(route[:, 0], route[:, 1], "k-", lw=5, alpha=0.15, label="reference route")
            sc = ax.scatter(xy[:, 0], xy[:, 1], c=v, s=4, cmap="viridis", vmin=0, vmax=10)
            ax.set_aspect("equal"); ax.set_title(f"layout {lay} - {name}: {m['reason']}, "
                                                  f"|lat|={m['mean_abs_lat']:.2f} m, {m['time_s']:.0f} s", fontsize=9)
    fig.colorbar(sc, ax=axes, label="speed (m/s)", shrink=0.6)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130, bbox_inches="tight")
    env.close()


if __name__ == "__main__":
    main()
