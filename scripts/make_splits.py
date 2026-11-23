"""Generate immutable train/val/test layout manifests.

A candidate layout seed is kept only if
  * the reference route is continuous (no jumps > 1.5 m between joined lane pieces),
  * the pure-pursuit baseline does not hit MetaDrive's out_of_road while on the route
    centerline in the first steps (i.e. the start-lane route is valid), and
  * its route polyline hash is not already used by any split (no duplicate roads).
Seed ranges are disjoint per split: train 0.., val 1000.., test 2000..

    python scripts/make_splits.py --family junctions --prefix "" --train 100 --val 20 --test 50
"""
import argparse
import json
from pathlib import Path

import numpy as np

from path_following.baseline import PurePursuit
from path_following.env import PathFollowingEnv

RANGES = {"train": 0, "val": 1000, "test": 2000}


def check_layout(env, seed):
    env.reset(options={"layout_seed": seed})
    pts = env.route.points
    if np.linalg.norm(np.diff(pts, axis=0), axis=1).max() > 1.5:
        return False, "discontinuous"
    pp = PurePursuit(env.cfg)
    for _ in range(10):
        _, _, term, _, info = env.step(pp.act(env.state()))
        if term:
            return False, info["episode_metrics"]["reason"]
    return True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="junctions")
    ap.add_argument("--map", default="4")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--train", type=int, default=100)
    ap.add_argument("--val", type=int, default=20)
    ap.add_argument("--test", type=int, default=50)
    ap.add_argument("--out", default="splits")
    a = ap.parse_args()
    n_blocks = int(a.map) if a.map.isdigit() else a.map
    want = {"train": a.train, "val": a.val, "test": a.test}
    env = PathFollowingEnv({"map": n_blocks, "road_family": a.family}, seeds=range(0, 2000 + 500))
    used, rejected = set(), {}
    for split, n in want.items():
        seeds, sigs, lens, s = [], [], [], RANGES[split]
        while len(seeds) < n:
            ok, why = check_layout(env, s)
            sig = env.layout_signature()
            if ok and sig in used:
                ok, why = False, "duplicate"
            if ok:
                used.add(sig); seeds.append(s); sigs.append(sig); lens.append(round(env.route.length, 1))
            else:
                rejected.setdefault(split, {})[s] = why
            s += 1
            assert s < RANGES[split] + 500, "ran out of candidate seeds"
        manifest = dict(split=split, family=a.family, map=n_blocks, seeds=seeds, signatures=sigs,
                        route_lengths=lens, rejected=rejected.get(split, {}))
        path = Path(a.out) / f"{a.prefix}{split}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(manifest, indent=1))
        print(split, len(seeds), "seeds; rejected:", rejected.get(split, {}))
    env.close()


if __name__ == "__main__":
    main()
