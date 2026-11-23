"""Reference-route geometry: projection, errors, progress, lookahead.

Pure NumPy, no simulator dependency, so it can be unit-tested in isolation.
Conventions: world frame is x-right/y-up, heading is radians CCW from +x.
Signed lateral error is positive when the vehicle is to the LEFT of the
route direction. Ego frame: +x forward, +y left.
"""
from __future__ import annotations

import numpy as np


def wrap_angle(a):
    """Wrap angle(s) to [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def resample_polyline(points: np.ndarray, spacing: float) -> np.ndarray:
    """Resample an ordered polyline at uniform arc-length spacing (keeps endpoints)."""
    pts = np.asarray(points, dtype=np.float64)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1e-9])  # drop duplicate joints
    pts = pts[keep]
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    n = max(int(np.ceil(total / spacing)), 1)
    s = np.linspace(0.0, total, n + 1)
    return np.stack([np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])], axis=1)


class Route:
    """Ordered centerline with monotone, windowed nearest-segment projection."""

    def __init__(self, points: np.ndarray, spacing: float = 1.0):
        self.points = resample_polyline(points, spacing)
        if len(self.points) < 2:
            raise ValueError("route needs at least two distinct points")
        d = np.diff(self.points, axis=0)
        self.seg_len = np.linalg.norm(d, axis=1)
        self.seg_dir = d / self.seg_len[:, None]
        self.seg_heading = np.arctan2(d[:, 1], d[:, 0])
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.length = float(self.cum[-1])
        self._end_dir = self.seg_dir[-1]

    # ------------------------------------------------------------------ projection
    def project(self, pos, hint: int | None = None, back: int = 5, ahead: int = 60):
        """Project `pos` onto the route.

        If `hint` (segment index from the previous step) is given, only segments in
        [hint-back, hint+ahead] are searched. This avoids jumping to nearby parallel
        or looping segments. Returns (s, lateral, route_heading, seg_index).
        """
        pos = np.asarray(pos, dtype=np.float64)
        n = len(self.seg_len)
        lo, hi = (0, n) if hint is None else (max(hint - back, 0), min(hint + ahead + 1, n))
        a = self.points[lo:hi]
        d = self.seg_dir[lo:hi]
        rel = pos - a
        t = np.clip(np.einsum("ij,ij->i", rel, d), 0.0, self.seg_len[lo:hi])
        closest = a + d * t[:, None]
        dist2 = np.sum((pos - closest) ** 2, axis=1)
        k = int(np.argmin(dist2))
        i = lo + k
        # signed lateral: cross(route_dir, pos - closest); left of direction is positive.
        # For the last segment, allow projection beyond the end (unclipped) so that
        # lateral error stays well-defined past the endpoint.
        tt = t[k]
        if i == n - 1:
            tt = float(np.dot(pos - self.points[i], self.seg_dir[i]))
        diff = pos - (self.points[i] + self.seg_dir[i] * tt)
        lateral = float(self.seg_dir[i][0] * diff[1] - self.seg_dir[i][1] * diff[0])
        s = float(self.cum[i] + tt)
        return s, lateral, float(self.seg_heading[i]), i

    # ------------------------------------------------------------------ sampling
    def point_at(self, s: float) -> np.ndarray:
        """Route point at arc length s. Extrapolates straight past the end / before start."""
        if s >= self.length:
            return self.points[-1] + self._end_dir * (s - self.length)
        if s <= 0.0:
            return self.points[0] + self.seg_dir[0] * s
        i = int(np.searchsorted(self.cum, s, side="right") - 1)
        i = min(i, len(self.seg_len) - 1)
        return self.points[i] + self.seg_dir[i] * (s - self.cum[i])

    def heading_at(self, s: float) -> float:
        if s >= self.length:
            return float(self.seg_heading[-1])
        i = int(np.searchsorted(self.cum, max(s, 0.0), side="right") - 1)
        return float(self.seg_heading[min(i, len(self.seg_len) - 1)])

    def lookahead_world(self, s: float, n: int, spacing: float, first: float | None = None):
        """n points at s+first, s+first+spacing, ... (first defaults to spacing)."""
        first = spacing if first is None else first
        return np.stack([self.point_at(s + first + k * spacing) for k in range(n)])

    def curvature_at(self, s: float, width: float = 6.0) -> float:
        """Approximate |curvature| from heading change over +-width/2 around s."""
        h0 = self.heading_at(s - width / 2)
        h1 = self.heading_at(s + width / 2)
        return float(abs(wrap_angle(h1 - h0)) / width)


def world_to_ego(points_world, pos, heading):
    """Transform world points to the ego frame (+x forward, +y left)."""
    rel = np.asarray(points_world, dtype=np.float64) - np.asarray(pos, dtype=np.float64)
    c, s = np.cos(heading), np.sin(heading)
    x = c * rel[..., 0] + s * rel[..., 1]
    y = -s * rel[..., 0] + c * rel[..., 1]
    return np.stack([x, y], axis=-1)
