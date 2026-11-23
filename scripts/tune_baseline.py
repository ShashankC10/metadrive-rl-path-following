"""Grid-search pure-pursuit parameters on TRAIN layouts, then confirm on VALIDATION. Never touches test.

    python scripts/tune_baseline.py --out configs/baseline_params.json
"""
import argparse
import itertools
import json
from dataclasses import asdict

from path_following.baseline import PurePursuitParams
from path_following.config import load_config
from path_following.env import PathFollowingEnv, read_manifest
from path_following.evaluate import baseline_controller, run_episodes

GRID = dict(ld0=[2.0, 3.0, 4.0], k_ld=[0.4, 0.8, 1.2], kp=[0.4, 0.8], a_lat=[1.5, 2.5, 3.5])


def score(df):
    # prefer completion, then fast, then tight tracking
    return df.success.mean() * 100 - df.off_road.mean() * 50 - df.rms_lat.mean() * 5 - df.time_s[df.success].mean() * 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--train-split", default="splits/train.json")
    ap.add_argument("--val-split", default="splits/val.json")
    ap.add_argument("--n-train", type=int, default=30, help="train layouts used for the search")
    ap.add_argument("--out", default="configs/baseline_params.json")
    a = ap.parse_args()
    cfg = load_config(a.config)["env"]
    tr, va = read_manifest(a.train_split)["seeds"], read_manifest(a.val_split)["seeds"]
    env = PathFollowingEnv(cfg, seeds=tr + va)
    results = []
    for vals in itertools.product(*GRID.values()):
        p = PurePursuitParams(**dict(zip(GRID, vals)))
        ctrl, _ = baseline_controller(cfg, p)
        df = run_episodes(env, ctrl, tr[: a.n_train])
        results.append((score(df), p, df.success.mean()))
        print(f"{asdict(p)} train score={results[-1][0]:.2f} success={results[-1][2]:.2f}", flush=True)
    results.sort(key=lambda r: -r[0])
    best = results[0][1]
    ctrl, _ = baseline_controller(cfg, best)
    dv = run_episodes(env, ctrl, va)
    print("BEST", asdict(best), "| validation success", dv.success.mean(), "rms_lat", dv.rms_lat.mean())
    json.dump(asdict(best), open(a.out, "w"), indent=1)
    env.close()


if __name__ == "__main__":
    main()
