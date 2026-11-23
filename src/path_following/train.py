"""PPO training with validation-based checkpoint selection.

    python -m path_following.train --run ppo_s0 --seed 0 --config configs/default.yaml
    python -m path_following.train --run ppo_s0 --seed 0 --resume          # continue from latest.zip
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from .config import load_config
from .env import PathFollowingEnv, read_manifest
from .evaluate import run_episodes


def make_env_fn(cfg, seeds, rank, seed):
    def _init():
        env = PathFollowingEnv(cfg, seeds=seeds)
        env.reset(seed=seed * 1000 + rank)  # seeds the layout-sampling rng
        return env
    return _init


class ValidationCallback(BaseCallback):
    """Every `eval_freq` env steps: evaluate on validation layouts, keep the best checkpoint."""

    def __init__(self, env_cfg, val_seeds, out_dir: Path, eval_freq: int, log):
        super().__init__()
        self.env_cfg, self.val_seeds, self.out, self.freq = env_cfg, val_seeds, out_dir, eval_freq
        self.best, self.next_eval, self.env, self.log = -np.inf, eval_freq, None, log
        self.train_stats = []

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            m = info.get("episode_metrics")
            if m:
                self.train_stats.append((m["success"], m["off_road"], m["progress"], m["ret"]))
        if self.num_timesteps >= self.next_eval:
            self.next_eval += self.freq
            self._evaluate()
        return True

    def _evaluate(self):
        if self.env is None:  # created lazily so the main process doesn't own a simulator before workers start
            self.env = PathFollowingEnv(self.env_cfg, seeds=self.val_seeds)
        model = self.model
        df = run_episodes(self.env, lambda e, o: model.predict(o, deterministic=True)[0], self.val_seeds, "ppo")
        succ, prog, lat = df.success.mean(), df.progress.mean(), df.mean_abs_lat.mean()
        # checkpoint selection uses validation data only: completion first, then progress, then tight tracking
        score = succ + 0.1 * prog - 0.05 * lat
        rec = dict(step=self.num_timesteps, val_success=succ, val_off_road=df.off_road.mean(), val_progress=prog,
                   val_mean_abs_lat=lat, val_score=score)
        if self.train_stats:
            ts = np.array(self.train_stats, dtype=float)
            rec.update(train_success=ts[:, 0].mean(), train_off_road=ts[:, 1].mean(), train_return=ts[:, 3].mean())
            self.train_stats = []
        for k, v in rec.items():
            if k != "step":
                self.logger.record(f"eval/{k}", v)
        self.logger.dump(self.num_timesteps)
        with open(self.out / "val_log.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.model.save(self.out / "latest")
        if score > self.best + 1e-9:
            self.best = score
            self.model.save(self.out / "best_model")
            (self.out / "best_info.json").write_text(json.dumps(rec, indent=1))
        self.log(f"[{self.num_timesteps:>9}] val success={succ:.2f} off_road={df.off_road.mean():.2f} "
                 f"progress={prog:.2f} lat={lat:.3f}  best={self.best:.3f}")

    def _on_training_end(self):
        if self.env is not None:
            self.env.close()


def git_rev() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        d = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
        return r.stdout.strip() + ("+dirty" if d else "")
    except Exception:
        return "unknown"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--train-split", default="splits/train.json")
    ap.add_argument("--val-split", default="splits/val.json")
    ap.add_argument("--timesteps", type=int, help="override config total_timesteps")
    ap.add_argument("--init-from", help="start from this PPO .zip (curriculum)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--log-dir", default="runs")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    env_cfg, ppo = cfg["env"], cfg["ppo"]
    total = a.timesteps or ppo["total_timesteps"]
    out = Path(a.ckpt_dir) / a.run
    out.mkdir(parents=True, exist_ok=True)
    log = lambda s: (print(s, flush=True), open(out / "train.log", "a").write(s + "\n"))
    train_seeds = read_manifest(a.train_split)["seeds"]
    val_seeds = read_manifest(a.val_split)["seeds"]

    (out / "run_info.json").write_text(json.dumps(dict(
        run=a.run, seed=a.seed, git=git_rev(), config=cfg, total_timesteps=total, train_split=a.train_split,
        val_split=a.val_split, torch=torch.__version__, started=time.strftime("%Y-%m-%d %H:%M:%S")), indent=1))

    venv = SubprocVecEnv([make_env_fn(env_cfg, train_seeds, i, a.seed) for i in range(ppo["n_envs"])])
    kwargs = dict(
        n_steps=ppo["n_steps"], batch_size=ppo["batch_size"], n_epochs=ppo["n_epochs"],
        learning_rate=ppo["learning_rate"], gamma=ppo["gamma"], gae_lambda=ppo["gae_lambda"],
        clip_range=ppo["clip_range"], ent_coef=ppo["ent_coef"], seed=a.seed, device="cpu",
        policy_kwargs=dict(net_arch=list(ppo["net_arch"]), log_std_init=ppo["log_std_init"]), tensorboard_log=a.log_dir, verbose=0)
    if a.resume and (out / "latest.zip").exists():
        model = PPO.load(out / "latest.zip", env=venv, device="cpu", tensorboard_log=a.log_dir)
        done = model.num_timesteps
        log(f"resumed at {done} steps")
    else:
        model = PPO("MlpPolicy", venv, **kwargs)
        if a.init_from:
            model.set_parameters(a.init_from, device="cpu")
            log(f"initialised weights from {a.init_from}")
        done = 0
    cb = ValidationCallback(env_cfg, val_seeds, out, ppo["eval_freq"], log)
    cb.next_eval = done + ppo["eval_freq"]
    t0 = time.time()
    model.learn(total_timesteps=max(total - done, 0), callback=cb, tb_log_name=a.run,
                reset_num_timesteps=not (a.resume and done > 0))
    cb._evaluate()
    model.save(out / "final")
    log(f"done in {time.time()-t0:.0f}s; best val score {cb.best:.3f}")
    venv.close()


if __name__ == "__main__":
    main()
