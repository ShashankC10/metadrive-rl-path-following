"""Route-following task wrapper around MetaDrive (numerical observations, no traffic)."""
from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import DEFAULT_ENV
from .route import Route, wrap_angle, world_to_ego


# Road families (MetaDrive block-type probabilities). Ramp blocks are excluded: the start lane
# can end/merge inside a ramp block, so a fixed start-lane reference route is not valid there.
ROAD_FAMILIES = {
    "curves": {"Curve": 0.7, "Straight": 0.3},
    "junctions": {"Curve": 0.4, "Straight": 0.1, "StdInterSection": 0.2,
                  "StdTInterSection": 0.2, "Roundabout": 0.1},
}


def _block_config(family: str):
    from metadrive.component.algorithm.blocks_prob_dist import PGBlockDistConfig

    dist = dict(PGBlockDistConfig.BLOCK_TYPE_DISTRIBUTION_V2)
    dist = {k: 0.0 for k in dist}
    dist.update(ROAD_FAMILIES[family])
    return type("FamilyBlockDist", (PGBlockDistConfig,), {"BLOCK_TYPE_DISTRIBUTION_V2": dist})


def read_manifest(path) -> dict:
    return json.loads(Path(path).read_text())


class PathFollowingEnv(gym.Env):
    """One ego vehicle follows the centerline of its start lane to the end of the map.

    Action:  [steer, throttle/brake] in [-1, 1]  (MetaDrive convention; +steer = left).
    Observation (2K+5, float32, roughly in [-1, 1]):
        speed, signed lateral error, heading error, previous steer, previous throttle,
        then K lookahead route points in the ego frame (x, y) -- scaled by fixed constants,
        so no learned normalisation statistics exist (nothing to freeze at eval time).
    Episode ends:  terminated on success / off-road / crash / stall; truncated on timeout.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: dict | None = None, seeds=None, render: bool = False):
        super().__init__()
        self.cfg = {**DEFAULT_ENV, **(config or {})}
        self.seeds = [int(s) for s in (seeds if seeds is not None else [0])]
        self._render = render
        self._md = None
        self._rng = np.random.default_rng()
        k = self.cfg["n_lookahead"]
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(5 + 2 * k,), dtype=np.float32)
        self.dt = None
        self.route: Route | None = None

    # ------------------------------------------------------------------ simulator
    def _make_md(self):
        from metadrive.envs import MetaDriveEnv

        c = self.cfg
        md_cfg = dict(
            use_render=self._render,
            map=c["map"],
            block_dist_config=_block_config(c["road_family"]),
            traffic_density=0.0,
            accident_prob=0.0,
            start_seed=min(self.seeds),
            num_scenarios=max(self.seeds) - min(self.seeds) + 1,
            decision_repeat=c["decision_repeat"],
            horizon=10**6,          # time limit is handled here, not by MetaDrive
            truncate_as_terminate=False,
            out_of_road_done=True,
            crash_vehicle_done=True,
            crash_object_done=True,
            random_spawn_lane_index=False,
            log_level=40,
        )
        self._md = MetaDriveEnv(md_cfg)

    def close(self):
        if self._md is not None:
            self._md.close()
            self._md = None

    # ------------------------------------------------------------------ route
    def _extract_route(self) -> Route:
        md = self._md
        nav = md.agent.navigation
        rn = md.engine.current_map.road_network
        lane_idx = md.agent.lane_index[2]
        pts = []
        for a, b in zip(nav.checkpoints[:-1], nav.checkpoints[1:]):
            lanes = rn.graph[a][b]
            lane = lanes[min(lane_idx, len(lanes) - 1)]
            n = max(int(np.ceil(lane.length / 0.5)), 2)
            pts.append(np.array([lane.position(s, 0.0) for s in np.linspace(0, lane.length, n)]))
        return Route(np.concatenate(pts), spacing=1.0)

    def layout_signature(self) -> str:
        """Hash of the route polyline (for duplicate-layout detection)."""
        import hashlib

        return hashlib.md5(np.round(self.route.points - self.route.points[0], 1).tobytes()).hexdigest()[:12]

    # ------------------------------------------------------------------ state
    def state(self) -> dict:
        """Raw (unscaled) task state, shared by the PPO observation and the baseline."""
        ag = self._md.agent
        pos = np.asarray(ag.position, dtype=np.float64)
        heading = float(ag.heading_theta)
        speed = float(ag.speed_km_h) / 3.6
        s, lat, rh, seg = self.route.project(pos, hint=self._seg)
        self._seg = seg
        c = self.cfg
        look = self.route.lookahead_world(s, c["n_lookahead"], c["lookahead_spacing"], c["lookahead_first"])
        return dict(
            pos=pos, heading=heading, speed=speed, s=s, lateral=lat,
            heading_err=float(wrap_angle(heading - rh)),
            lookahead=world_to_ego(look, pos, heading),
            curvature=self.route.curvature_at(s),
        )

    def _obs(self, st: dict) -> np.ndarray:
        c = self.cfg
        la = st["lookahead"]
        look = np.stack([la[:, 0] / (c["n_lookahead"] * c["lookahead_spacing"] + c["lookahead_first"]),
                         la[:, 1] / c["y_scale"]], axis=1).ravel()
        head = [st["speed"] / c["speed_scale"], st["lateral"] / c["lat_scale"],
                st["heading_err"] / np.pi, self._prev_action[0], self._prev_action[1]]
        return np.clip(np.concatenate([head, look]), -1.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if self._md is None:
            self._make_md()
        layout = (options or {}).get("layout_seed")
        if layout is None:
            layout = int(self._rng.choice(self.seeds))
        self.layout_seed = int(layout)
        self._md.reset(seed=self.layout_seed)
        self.dt = self._md.config["physics_world_step_size"] * self._md.config["decision_repeat"]
        self.route = self._extract_route()
        self._seg = None
        self._prev_action = np.zeros(2, dtype=np.float32)
        st = self.state()
        self._s0, self._s_prev = st["s"], st["s"]
        c = self.cfg
        self.max_steps = min(c["max_steps_cap"],
                             int(c["max_steps_factor"] * self.route.length / (c["target_speed"] * self.dt)) + 50)
        self._t = 0
        self._slow = 0
        self._acc = dict(abs_lat=0.0, sq_lat=0.0, speed=0.0, ret=0.0, terms={})
        return self._obs(st), {"layout_seed": self.layout_seed, "route_length": self.route.length}

    def step(self, action):
        c = self.cfg
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        _, _, md_term, _, md_info = self._md.step(action)
        self._t += 1
        st = self.state()

        ds = float(np.clip(st["s"] - self._s_prev, -0.5, 1.5 * c["speed_limit"] * self.dt))
        self._s_prev = st["s"]
        terms = {
            "progress": c["w_progress"] * ds,
            "lateral": -c["w_lateral"] * min(abs(st["lateral"]), c["lat_limit"]),
            "heading": -c["w_heading"] * abs(st["heading_err"]) / (np.pi / 2),
            "time": -c["w_time"],
            "overspeed": -c["w_overspeed"] * max(0.0, st["speed"] - c["speed_limit"]),
            "smooth": -c["w_smooth"] * abs(float(action[0]) - float(self._prev_action[0])),
        }

        off_road = bool(md_info.get("out_of_road", False)) or abs(st["lateral"]) > c["lat_limit"]
        crash = bool(md_info.get("crash", False)) and not off_road
        success = (st["s"] >= self.route.length - c["end_tol"]) and not off_road and not crash
        if self._t > c["stall_grace"]:
            self._slow = self._slow + 1 if st["speed"] < c["stall_speed"] else 0
        stall = self._slow >= c["stall_steps"]

        terminated = success or off_road or crash or stall
        truncated = (not terminated) and self._t >= self.max_steps
        if success:
            terms["terminal"] = c["r_success"]
        elif off_road or crash:
            terms["terminal"] = c["r_offroad"]
        elif stall:
            terms["terminal"] = c["r_stall"]
        reward = float(sum(terms.values()))

        a = self._acc
        a["abs_lat"] += abs(st["lateral"]); a["sq_lat"] += st["lateral"] ** 2
        a["speed"] += st["speed"]; a["ret"] += reward
        for k, v in terms.items():
            a["terms"][k] = a["terms"].get(k, 0.0) + v
        self._prev_action = action

        info = {"s": st["s"], "lateral": st["lateral"], "speed": st["speed"]}
        if terminated or truncated:
            n = self._t
            reason = ("success" if success else "off_road" if off_road else "crash" if crash
                      else "stall" if stall else "timeout")
            info["episode_metrics"] = dict(
                layout_seed=self.layout_seed, reason=reason, success=success, off_road=off_road,
                crash=crash, stall=stall, timeout=reason == "timeout", steps=n, time_s=n * self.dt,
                route_length=self.route.length,
                progress=float(np.clip((st["s"] - self._s0) / max(self.route.length - self._s0, 1e-6), 0, 1)),
                mean_abs_lat=a["abs_lat"] / n, rms_lat=float(np.sqrt(a["sq_lat"] / n)),
                mean_speed=a["speed"] / n, ret=a["ret"],
                **{f"r_{k}": v for k, v in a["terms"].items()},
            )
        return self._obs(st), reward, terminated, truncated, info
