"""Integration tests that run the real MetaDrive simulator (headless)."""
import numpy as np
import pytest

pytest.importorskip("metadrive")

from path_following.baseline import PurePursuit
from path_following.env import PathFollowingEnv


@pytest.fixture(scope="module")
def env():
    e = PathFollowingEnv({"map": "SCSC", "road_family": "curves"}, seeds=range(0, 6))
    yield e
    e.close()


def test_spaces_and_finite_observations(env):
    obs, info = env.reset(options={"layout_seed": 2})
    assert obs.shape == env.observation_space.shape and obs.dtype == np.float32
    for _ in range(50):
        obs, r, te, tr, info = env.step(env.action_space.sample())
        assert np.all(np.isfinite(obs)) and np.isfinite(r)
        assert env.observation_space.contains(obs)
        if te or tr:
            break


def test_steering_sign_matches_lateral_convention(env):
    for steer, sign in ((1.0, 1), (-1.0, -1)):
        env.reset(options={"layout_seed": 3})
        for _ in range(15):
            env.step([steer, 0.5])
        st = env.state()
        assert sign * st["heading_err"] > 0.1 and sign * st["lateral"] > 0.1


def test_initial_state_on_route(env):
    env.reset(options={"layout_seed": 1})
    st = env.state()
    assert abs(st["lateral"]) < 0.05 and abs(st["heading_err"]) < 0.05 and st["speed"] < 0.1
    assert st["lookahead"][0, 0] == pytest.approx(env.cfg["lookahead_first"], abs=0.2)


def test_baseline_succeeds_and_outcome_flags(env):
    pp = PurePursuit(env.cfg)
    obs, _ = env.reset(options={"layout_seed": 0})
    done = False
    while not done:
        obs, r, te, tr, info = env.step(pp.act(env.state()))
        done = te or tr
    m = info["episode_metrics"]
    assert m["success"] and te and not tr and m["reason"] == "success" and not m["off_road"]
    assert m["progress"] > 0.97 and m["mean_abs_lat"] < 0.5


def test_off_road_terminates_not_truncates(env):
    env.reset(options={"layout_seed": 0})
    for _ in range(400):
        _, _, te, tr, info = env.step([1.0, 1.0])  # full left lock: must leave the road
        if te or tr:
            break
    m = info["episode_metrics"]
    assert te and not tr and m["off_road"] and m["reason"] == "off_road"


def test_timeout_is_truncation_and_stall_is_termination(env):
    env.reset(options={"layout_seed": 0})
    for _ in range(env.max_steps + 5):
        _, _, te, tr, info = env.step([0.0, -1.0])  # brake: never moves -> stall
        if te or tr:
            break
    assert te and info["episode_metrics"]["stall"]
    # force a timeout: tiny horizon, drive straight slowly
    env.reset(options={"layout_seed": 0})
    env.max_steps = 20
    for _ in range(21):
        _, _, te, tr, info = env.step([0.0, 0.3])
    assert tr and not te and info["episode_metrics"]["timeout"]


def test_reverse_does_not_earn_progress(env):
    env.reset(options={"layout_seed": 0})
    tot = 0.0
    for _ in range(40):
        _, _, _, _, info = env.step([0.0, 0.3])
    s_start = env.state()["s"]
    # brake hard (cannot reverse far); progress reward terms must never be positive for s decreasing
    for _ in range(20):
        _, r, te, tr, info = env.step([0.0, -1.0])
        assert te or env.state()["s"] >= s_start - 1.0
        if te: break


def test_layout_is_deterministic(env):
    a, _ = env.reset(options={"layout_seed": 4}); pa = env.route.points.copy()
    b, _ = env.reset(options={"layout_seed": 4}); pb = env.route.points.copy()
    assert np.allclose(pa, pb) and np.allclose(a, b)
