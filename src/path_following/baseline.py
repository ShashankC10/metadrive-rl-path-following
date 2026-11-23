"""Pure-pursuit steering + proportional speed control.

Uses exactly the information in the PPO observation (ego-frame lookahead points, speed)
and the same action space, vehicle and termination rules.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

WHEELBASE = 2.47       # m (MetaDrive default vehicle: front 1.05 + rear 1.42)
MAX_STEER_RAD = np.deg2rad(40.0)


@dataclass
class PurePursuitParams:
    ld0: float = 3.0       # base lookahead distance (m)
    k_ld: float = 0.8      # lookahead gain on speed (s)
    ld_min: float = 4.0
    ld_max: float = 20.0
    kp: float = 0.6        # proportional speed gain
    a_lat: float = 2.5     # comfortable lateral accel used to cap curve speed (m/s^2)
    v_min: float = 3.0     # lowest curve speed (m/s)


class PurePursuit:
    name = "pure_pursuit"

    def __init__(self, env_cfg: dict, params: PurePursuitParams | None = None):
        self.p = params or PurePursuitParams()
        self.first = env_cfg["lookahead_first"]
        self.spacing = env_cfg["lookahead_spacing"]
        self.v_target = env_cfg["target_speed"]
        self.params = asdict(self.p)

    def _point_at(self, look: np.ndarray, dist: float) -> np.ndarray:
        """Interpolate the ego-frame lookahead polyline at route distance `dist` ahead."""
        d = self.first + self.spacing * np.arange(len(look))
        return np.array([np.interp(dist, d, look[:, 0]), np.interp(dist, d, look[:, 1])])

    def act(self, st: dict) -> np.ndarray:
        p, v, look = self.p, st["speed"], st["lookahead"]
        ld = float(np.clip(p.ld0 + p.k_ld * v, p.ld_min, p.ld_max))
        x, y = self._point_at(look, ld)
        alpha = np.arctan2(y, max(x, 1e-3))
        delta = np.arctan2(2.0 * WHEELBASE * np.sin(alpha), ld)
        steer = float(np.clip(delta / MAX_STEER_RAD, -1, 1))

        # curve speed cap from the curvature implied by two preview points
        kappa = 0.0
        for dist in (ld, 2.0 * ld):
            px, py = self._point_at(look, min(dist, self.first + self.spacing * (len(look) - 1)))
            kappa = max(kappa, 2.0 * abs(py) / max(px * px + py * py, 1e-6))
        v_ref = self.v_target if kappa < 1e-4 else float(np.clip(np.sqrt(p.a_lat / kappa), p.v_min, self.v_target))
        throttle = float(np.clip(p.kp * (v_ref - v), -1, 1))
        return np.array([steer, throttle], dtype=np.float32)
