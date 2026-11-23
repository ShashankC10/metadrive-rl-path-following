import numpy as np
import pytest

from path_following.route import Route, resample_polyline, world_to_ego, wrap_angle


def straight(length=100.0):
    return Route(np.array([[0.0, 0.0], [length, 0.0]]))


def circle(radius=50.0, deg=90):
    th = np.linspace(0, np.deg2rad(deg), 400)
    return Route(np.stack([radius * np.sin(th), radius * (1 - np.cos(th))], axis=1))


def test_wrap_angle():
    assert wrap_angle(np.pi + 0.1) == pytest.approx(-np.pi + 0.1)
    assert wrap_angle(-3 * np.pi) == pytest.approx(-np.pi)
    assert wrap_angle(0.3) == pytest.approx(0.3)


def test_resample_uniform_and_keeps_endpoints():
    p = resample_polyline(np.array([[0, 0], [0, 0], [10, 0], [10, 10]], float), 1.0)
    assert np.allclose(p[0], [0, 0]) and np.allclose(p[-1], [10, 10])
    assert np.allclose(np.linalg.norm(np.diff(p, axis=0), axis=1), 1.0)


def test_lateral_sign_left_positive():
    r = straight()
    assert r.project([10, 1.5])[1] == pytest.approx(1.5)    # left of +x direction
    assert r.project([10, -2.0])[1] == pytest.approx(-2.0)  # right


def test_progress_is_arclength():
    r = straight()
    assert r.project([37.0, 0.5])[0] == pytest.approx(37.0)
    c = circle()
    # point on the arc at 45 degrees: s = R * pi/4
    th = np.pi / 4
    s, lat, _, _ = c.project([50 * np.sin(th), 50 * (1 - np.cos(th))])
    assert s == pytest.approx(50 * th, abs=0.1) and abs(lat) < 0.05


def test_lateral_on_curve_matches_radius_offset():
    c = circle(50.0)
    th = np.pi / 3
    # moving toward the circle centre (centre is on the left) => positive lateral
    pos = np.array([(50 - 2) * np.sin(th), 50 - (50 - 2) * np.cos(th)])
    assert c.project(pos)[1] == pytest.approx(2.0, abs=0.05)


def test_windowed_projection_ignores_parallel_segment():
    # U-turn: a route that goes out, turns around and comes back 3 m away
    pts = np.array([[0, 0], [100, 0], [100, 3], [0, 3]], float)
    r = Route(pts)
    s1, _, _, seg = r.project([50, 0.2])
    # vehicle on the outbound leg is closer to nothing else, but a global search at
    # (50, 1.4) would pick either leg; with a hint it must stay on the outbound leg
    s2, lat, _, _ = r.project([50, 1.4], hint=seg)
    assert s2 == pytest.approx(50.0, abs=1.0) and lat == pytest.approx(1.4, abs=1e-6)


def test_heading_wraparound_in_ego_frame_and_error():
    r = Route(np.array([[0, 0], [-100, 0]], float))        # route heading = pi
    _, _, rh, _ = r.project([-10, 0])
    veh_heading = -np.pi + 0.05                              # just across the +-pi cut
    assert wrap_angle(veh_heading - rh) == pytest.approx(0.05, abs=1e-6)


def test_lookahead_in_ego_frame():
    r = straight()
    pts = world_to_ego(r.lookahead_world(10.0, 3, 5.0), np.array([10.0, 1.0]), 0.0)
    assert np.allclose(pts, [[5, -1], [10, -1], [15, -1]])
    # rotate the vehicle 90 deg left: route is now on its right-hand side
    pts = world_to_ego(np.array([[10.0, 0.0]]), np.array([0.0, 0.0]), np.pi / 2)
    assert np.allclose(pts, [[0, -10]], atol=1e-9)


def test_endpoint_padding_extrapolates_straight():
    r = straight(100.0)
    look = r.lookahead_world(95.0, 4, 5.0)
    assert np.allclose(look[:, 0], [100, 105, 110, 115]) and np.allclose(look[:, 1], 0)


def test_curvature_estimate():
    c = circle(50.0)
    assert c.curvature_at(20.0) == pytest.approx(1 / 50, rel=0.05)
    assert straight().curvature_at(10.0) == pytest.approx(0.0, abs=1e-9)


def test_projection_past_end_keeps_lateral_defined():
    r = straight(100.0)
    s, lat, _, _ = r.project([105.0, 1.0])
    assert s == pytest.approx(105.0) and lat == pytest.approx(1.0)
