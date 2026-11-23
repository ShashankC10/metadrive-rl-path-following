"""Rendered playback of the baseline or a PPO checkpoint on one layout.

    python -m path_following.demo --controller baseline --layout 2000
    python -m path_following.demo --controller ppo --model checkpoints/ppo_s0/best_model.zip --layout 2000
    python -m path_following.demo --controller baseline --nuscenes-db ~/data/nuscenes_db --layout 0   # nuScenes scene-0061
Needs a display (Panda3D window). On macOS the rendering fix in macos_render.py is applied automatically.
Headless alternatives: scripts/render_topdown.py (GIFs), scripts/plot_trajectories.py, scripts/plot_nuscenes.py.
"""
import argparse
import json
from pathlib import Path

from .baseline import PurePursuitParams
from .config import load_config
from .env import PathFollowingEnv
from .evaluate import baseline_controller, ppo_controller


def main():
    from . import macos_render  # fixes blank-road rendering on macOS; no-op elsewhere
    macos_render.apply()
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", choices=["baseline", "ppo"], required=True)
    ap.add_argument("--model")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--params", default="configs/baseline_params.json")
    ap.add_argument("--layout", type=int, nargs="+", required=True, help="layout seed(s), or scenario indices with --nuscenes-db; several are played in order")
    ap.add_argument("--nuscenes-db", help="converted ScenarioNet database dir (e.g. ~/data/nuscenes_db); layout = scenario index")
    a = ap.parse_args()
    cfg = load_config(a.config)["env"]
    if a.nuscenes_db:
        from .scenario_env import ScenarioPathFollowingEnv
        env = ScenarioPathFollowingEnv(cfg, db=a.nuscenes_db, render=True)
    else:
        env = PathFollowingEnv(cfg, seeds=a.layout, render=True)
    if a.controller == "baseline":
        ctrl, _ = baseline_controller(cfg, PurePursuitParams(**json.loads(Path(a.params).read_text())))
    else:
        ctrl = ppo_controller(a.model)
    import time
    for lay in a.layout:
        obs, _ = env.reset(options={"layout_seed": lay})
        done = False
        while not done:
            obs, _, te, tr, info = env.step(ctrl(env, obs))
            env._md.render(text={"layout": lay, "speed m/s": round(info["speed"], 1), "lateral m": round(info["lateral"], 2)})
            done = te or tr
        m = info["episode_metrics"]
        print(f"layout {lay}: {m['reason']}  progress={m['progress']:.2f}  mean|lat|={m['mean_abs_lat']:.3f} m  time={m['time_s']:.0f} s", flush=True)
        time.sleep(1.5)
    env.close()


if __name__ == "__main__":
    main()
