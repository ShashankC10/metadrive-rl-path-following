"""Record top-down GIFs of a controller driving converted nuScenes scenes (or generated roads).

Uses MetaDrive's 2D top-down renderer, which works headless (no display, no GPU shaders).

    python scripts/render_topdown.py --controller ppo --model checkpoints/ppo_s1/best_model.zip \
        --nuscenes-db ~/data/nuscenes_db --layouts 0 1 3 5 6 7 8 --out-dir docs/videos
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from path_following.baseline import PurePursuitParams
from path_following.config import load_config
from path_following.env import PathFollowingEnv
from path_following.evaluate import baseline_controller, ppo_controller


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", choices=["baseline", "ppo"], required=True)
    ap.add_argument("--model")
    ap.add_argument("--nuscenes-db")
    ap.add_argument("--layouts", type=int, nargs="+", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--params", default="configs/baseline_params.json")
    ap.add_argument("--out-dir", default="docs/videos")
    ap.add_argument("--size", type=int, default=560, help="frame size in pixels")
    ap.add_argument("--scaling", type=float, default=6.0, help="pixels per metre")
    ap.add_argument("--every", type=int, default=2, help="record every n-th control step")
    a = ap.parse_args()
    cfg = load_config(a.config)["env"]
    if a.nuscenes_db:
        from path_following.scenario_env import ScenarioPathFollowingEnv
        env = ScenarioPathFollowingEnv(cfg, db=a.nuscenes_db)
    else:
        env = PathFollowingEnv(cfg, seeds=a.layouts)
    if a.controller == "baseline":
        ctrl, _ = baseline_controller(cfg, PurePursuitParams(**json.loads(Path(a.params).read_text())))
    else:
        ctrl = ppo_controller(a.model)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    tag = "nuscenes" if a.nuscenes_db else "generated"
    for lay in a.layouts:
        obs, _ = env.reset(options={"layout_seed": lay})
        frames, done, t = [], False, 0
        while not done:
            obs, _, te, tr, info = env.step(ctrl(env, obs))
            if t % a.every == 0:
                img = env._md.render(mode="topdown", window=False, screen_size=(a.size, a.size),
                                     film_size=(int(a.size * 2.5), int(a.size * 2.5)), scaling=a.scaling,
                                     target_agent_heading_up=False, draw_target_vehicle_trajectory=True,
                                     text={"layout": lay, "t": f"{t * env.dt:.1f}s", "speed": f"{info['speed']:.1f} m/s"})
                frames.append(Image.fromarray(np.asarray(img).astype("uint8")))
            t += 1
            done = te or tr
        m = info["episode_metrics"]
        path = out / f"{tag}_{a.controller}_{lay}.gif"
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 * env.dt * a.every), loop=0, optimize=True)
        print(f"layout {lay}: {m['reason']}, {len(frames)} frames -> {path}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
