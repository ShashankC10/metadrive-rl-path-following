"""Common evaluation loop for PPO checkpoints and the pure-pursuit baseline.

    python -m path_following.evaluate --controller baseline --split splits/val.json --out outputs/eval/baseline_val.csv
    python -m path_following.evaluate --controller ppo --model checkpoints/ppo_s0/best_model.zip --split splits/test.json --out ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .baseline import PurePursuit, PurePursuitParams
from .config import load_config
from .env import PathFollowingEnv, read_manifest


def run_episodes(env: PathFollowingEnv, controller, layout_seeds, label: str = "", progress_every: int = 0) -> pd.DataFrame:
    """Run one deterministic episode per layout seed; `controller(env, obs, state) -> action`."""
    rows = []
    for n, ls in enumerate(layout_seeds, 1):
        obs, _ = env.reset(options={"layout_seed": ls})
        done = False
        while not done:
            action = controller(env, obs)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        rows.append({"controller": label, **info["episode_metrics"]})
        if progress_every and n % progress_every == 0:
            print(f"  [{label}] {n}/{len(layout_seeds)} episodes", flush=True)
    return pd.DataFrame(rows)


def baseline_controller(env_cfg: dict, params: PurePursuitParams | None = None):
    pp = PurePursuit(env_cfg, params)
    return (lambda env, obs: pp.act(env.state())), pp


def ppo_controller(model_path: str):
    from stable_baselines3 import PPO

    model = PPO.load(model_path, device="cpu")
    return lambda env, obs: model.predict(obs, deterministic=True)[0]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", choices=["baseline", "ppo"], required=True)
    ap.add_argument("--model", help="PPO .zip checkpoint")
    ap.add_argument("--split", required=True, help="split manifest json")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--params", help="json file with PurePursuitParams (baseline)")
    ap.add_argument("--label")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    cfg = load_config(a.config)["env"]
    split = read_manifest(a.split)
    env = PathFollowingEnv(cfg, seeds=split["seeds"])
    if a.controller == "baseline":
        params = PurePursuitParams(**json.loads(Path(a.params).read_text())) if a.params else None
        ctrl, _ = baseline_controller(cfg, params)
    else:
        ctrl = ppo_controller(a.model)
    df = run_episodes(env, ctrl, split["seeds"], a.label or a.controller)
    env.close()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(df[["success", "off_road", "crash", "stall", "timeout", "progress", "mean_abs_lat", "time_s"]].mean(numeric_only=True))


if __name__ == "__main__":
    main()
