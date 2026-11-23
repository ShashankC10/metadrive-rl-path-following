"""Aggregation of per-episode metrics with uncertainty estimates."""
from __future__ import annotations

import numpy as np
import pandas as pd


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def bootstrap_ci(values, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI of the mean, resampling episodes (= layouts)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return tuple(np.quantile(means, [alpha / 2, 1 - alpha / 2]))


def summarize(df: pd.DataFrame) -> dict:
    """Summary of one controller on one episode set. `df` has one row per episode."""
    n = len(df)
    out = {"n_episodes": n}
    for flag in ("success", "off_road", "crash", "stall", "timeout"):
        k = int(df[flag].sum())
        lo, hi = wilson_ci(k, n)
        out[f"{flag}_rate"], out[f"{flag}_lo"], out[f"{flag}_hi"] = k / n, lo, hi
    for col in ("mean_abs_lat", "rms_lat", "mean_speed", "progress"):
        lo, hi = bootstrap_ci(df[col])
        out[col], out[f"{col}_lo"], out[f"{col}_hi"] = float(df[col].mean()), lo, hi
    ok = df[df["success"]]
    out["n_success"] = len(ok)
    out["time_success_s"] = float(ok["time_s"].mean()) if len(ok) else np.nan
    lo, hi = bootstrap_ci(ok["time_s"]) if len(ok) else (np.nan, np.nan)
    out["time_success_s_lo"], out["time_success_s_hi"] = lo, hi
    out["progress_failed"] = float(df[~df["success"]]["progress"].mean()) if (~df["success"]).any() else np.nan
    return out
