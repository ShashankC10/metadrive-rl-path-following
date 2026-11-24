"""Path-following task on converted ScenarioNet scenarios (e.g. nuScenes), no traffic.

The reference route is the ego vehicle's logged trajectory, subsampled to >=0.5 m steps.
Same observation/action/reward/termination contract as PathFollowingEnv; only the road
source and the route differ. Layout seed = scenario index in the database.
"""
from __future__ import annotations

import os
import pickle

import numpy as np

from .env import PathFollowingEnv
from .route import Route


def load_scenarios(db: str) -> list[dict]:
    """Return scenarios in the same (sorted) order ScenarioEnv indexes them."""
    db = os.path.expanduser(db)
    summary = pickle.load(open(f"{db}/dataset_summary.pkl", "rb"))
    mapping = pickle.load(open(f"{db}/dataset_mapping.pkl", "rb"))
    return [pickle.load(open(os.path.join(db, mapping[f], f), "rb")) for f in summary]


def iter_scenarios(db: str):
    """Yield scenarios one at a time (same order as ScenarioEnv indexes them); use for large databases."""
    db = os.path.expanduser(db)
    summary = pickle.load(open(f"{db}/dataset_summary.pkl", "rb"))
    mapping = pickle.load(open(f"{db}/dataset_mapping.pkl", "rb"))
    for f in summary:
        with open(os.path.join(db, mapping[f], f), "rb") as fh:
            yield pickle.load(fh)


def ego_route_points(scn: dict, min_step: float = 0.5) -> np.ndarray:
    t = scn["tracks"][scn["metadata"]["sdc_id"]]["state"]
    xy = t["position"][:, :2][np.asarray(t["valid"], dtype=bool)]
    keep = [xy[0]]
    for p in xy[1:]:
        if np.linalg.norm(p - keep[-1]) >= min_step:
            # skip backward jitter (pose noise while stopped): a step that reverses the previous direction
            if len(keep) >= 2 and np.dot(keep[-1] - keep[-2], p - keep[-1]) < 0:
                continue
            keep.append(p)
    return np.array(keep)


def validate_route(scn: dict, pts: np.ndarray, min_length=40.0, max_dev=2.0, frac=0.95, max_turn=0.5):
    """Check a logged-ego route is a usable reference. Returns (ok, stats dict)."""
    route = Route(pts, spacing=1.0) if len(pts) >= 2 else None
    stats = {"length": route.length if route else 0.0}
    if route is None or route.length < min_length:
        return False, {**stats, "reason": "too short / stationary"}
    lanes = [np.asarray(v["polyline"])[:, :2] for v in scn["map_features"].values()
             if v["type"].startswith("LANE_SURFACE")]
    if not lanes:
        return False, {**stats, "reason": "no lane map"}
    allpts = np.concatenate([l if len(l) > 1 else l for l in lanes])
    # densify lane polylines so nearest-vertex distance approximates distance to the centerline
    dense = np.concatenate([np.stack([np.interp(np.linspace(0, 1, max(int(np.sum(np.linalg.norm(np.diff(l, axis=0), axis=1))) * 2, 2)),
                                                np.linspace(0, 1, len(l)), l[:, k]) for k in (0, 1)], axis=1)
                            for l in lanes if len(l) > 1])
    d = np.array([np.min(np.linalg.norm(dense - p, axis=1)) for p in route.points])
    turn = np.abs(((np.diff(route.seg_heading) + np.pi) % (2 * np.pi)) - np.pi)
    stats.update(frac_on_lane=float(np.mean(d <= max_dev)), max_dev=float(d.max()), max_turn=float(turn.max()))
    if stats["frac_on_lane"] < frac:
        return False, {**stats, "reason": "route leaves mapped lanes"}
    if stats["max_turn"] > max_turn:
        return False, {**stats, "reason": "heading jump in route"}
    return True, {**stats, "reason": ""}


class ScenarioPathFollowingEnv(PathFollowingEnv):
    def __init__(self, config: dict | None = None, db: str = "~/data/nuscenes_db", seeds=None, render=False):
        self.db = os.path.expanduser(db)
        n = len(pickle.load(open(f"{self.db}/dataset_summary.pkl", "rb")))
        super().__init__(config, seeds=seeds if seeds is not None else list(range(n)), render=render)
        self.n_scenarios = n

    def _make_md(self):
        from metadrive.envs.scenario_env import ScenarioEnv
        from metadrive.policy.env_input_policy import EnvInputPolicy

        c = self.cfg
        self._md = ScenarioEnv(dict(
            use_render=self._render, data_directory=self.db, start_scenario_index=0,
            num_scenarios=self.n_scenarios, sequential_seed=False, agent_policy=EnvInputPolicy,
            no_traffic=True, no_light=True, decision_repeat=c["decision_repeat"], horizon=None,
            crash_vehicle_done=True, crash_object_done=True, log_level=40,
            store_map=False, store_data=False))   # default True caches every scenario in RAM (OOM on large databases)

    def _extract_route(self) -> Route:
        scn = self._md.engine.data_manager.current_scenario
        self.scenario_id = scn["id"]
        return Route(ego_route_points(scn), spacing=1.0)
