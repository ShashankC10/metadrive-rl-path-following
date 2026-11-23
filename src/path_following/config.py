"""Configuration loading. Defaults live here; YAML files override them."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULT_ENV = {
    # --- simulator ---
    "map": 4,                  # int = number of random blocks, str = fixed block sequence (e.g. "SCSC")
    "road_family": "junctions",  # "curves" (straights+curves) or "junctions" (adds X/T/roundabout)
    "decision_repeat": 5,      # x 0.02 s physics step = 0.1 s control period
    "max_steps_cap": 2000,
    "max_steps_factor": 2.0,   # horizon = factor * route_length / (target_speed*dt) + 50
    # --- observation ---
    "n_lookahead": 10,
    "lookahead_spacing": 3.0,  # metres between lookahead points
    "lookahead_first": 3.0,    # distance of first lookahead point
    "speed_scale": 12.0,       # m/s used to normalise speed
    "lat_scale": 3.0,          # m
    "y_scale": 8.0,            # m, lateral extent of lookahead points in ego frame
    # --- task ---
    "target_speed": 8.0,       # m/s; used for the time limit and by the baseline
    "speed_limit": 10.0,       # m/s; exceeding it is penalised
    "lat_limit": 2.5,          # m; |lateral error| beyond this counts as leaving the route/road
    "end_tol": 3.0,            # m; success when within this of the route end
    "stall_speed": 0.3,        # m/s
    "stall_steps": 60,         # consecutive slow steps (after grace) -> stall
    "stall_grace": 30,
    # --- reward weights (per step unless noted) ---
    "w_progress": 0.1,         # per metre of route progress (clipped)
    "w_lateral": 0.03,         # * min(|lat|, lat_limit)
    "w_heading": 0.02,         # * |heading error| / (pi/2)
    "w_time": 0.005,
    "w_overspeed": 0.05,       # * max(0, v - speed_limit)
    "w_smooth": 0.01,          # * |steer - prev_steer|
    "r_success": 10.0,
    "r_offroad": -10.0,
    "r_stall": -10.0,
}

DEFAULT_PPO = {
    "total_timesteps": 2_000_000,
    "n_envs": 8,
    "n_steps": 512,
    "batch_size": 512,
    "n_epochs": 10,
    "learning_rate": 3.0e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "net_arch": [128, 128],
    "log_std_init": -1.0,      # smaller initial action noise: random steering otherwise leaves the road in ~20 steps
    "eval_freq": 100_000,      # env steps (all envs)
    "eval_episodes": 20,
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path | None = None) -> dict:
    """Return {'env': ..., 'ppo': ...} merged with defaults."""
    raw = {}
    if path:
        raw = yaml.safe_load(Path(path).read_text()) or {}
    return {
        "env": _merge(DEFAULT_ENV, raw.get("env", {})),
        "ppo": _merge(DEFAULT_PPO, raw.get("ppo", {})),
    }
