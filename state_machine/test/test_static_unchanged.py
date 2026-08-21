"""Case 13: STATIC obstacle behaviour must be bit-for-bit what it was before.

Drives the real _check_free_frenet() against a stub and pins the static branch's
arithmetic to the pre-change formula:

    free_dist = |path_d - obs_d| - obs.size/2 - gb_ego_width_m/2
    blocked  <=> free_dist < lateral_width_m * clip(gap / ref_dist, 0, 1)

i.e. the static branch keeps using `size` (NOT the new Frenet lateral width) and
is NOT bounded by dynamic_prediction_span_m.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from state_machine.state_machine_node import StateMachine  # noqa: E402

TRACK_LENGTH = 40.0


class Obs:
    def __init__(self, oid, s_center, d_center, size, is_static,
                 d_left=None, d_right=None, vs=0.0):
        self.id = oid
        self.s_center = s_center
        self.s_start = s_center - size / 2.0
        self.s_end = s_center + size / 2.0
        self.d_center = d_center
        self.size = size
        self.is_static = is_static
        self.vs = vs
        # Independent of size on purpose, so a test can tell the two apart.
        self.d_left = d_center + size / 2.0 if d_left is None else d_left
        self.d_right = d_center - size / 2.0 if d_right is None else d_right


class Pred:
    def __init__(self, pred_s, pred_d):
        self.pred_s = pred_s
        self.pred_d = pred_d


class WpntStub:
    def __init__(self, s_m, d_m):
        self.s_m = s_m
        self.d_m = d_m


class PathStub:
    """Stand-in for WaypointData: a straight path at constant d."""

    def __init__(self, path_d=0.0, is_ot=False, is_gb=False, lateral_width_m=0.2,
                 max_horizon=10.0, ref_dist=2.0, length_m=12.0):
        n = int(length_m / 0.1) + 1
        s = np.linspace(0.0, length_m, n)
        self.array = np.column_stack([np.zeros(n), np.zeros(n), s])
        self.list = [WpntStub(float(si), path_d) for si in s]
        self.is_init = True
        self.is_closed = False
        self.is_ot_wpnts = is_ot
        self.is_gb_track_wpnts = is_gb
        self.lateral_width_m = lateral_width_m
        self.max_horizon = max_horizon
        self.free_scaling_reference_distance_m = ref_dist
        self.free_dbg = None
        self.closest_target = None
        self.closest_gap = None


class _Logger:
    def info(self, *a, **k):
        pass

    def warn(self, *a, **k):
        pass

    def debug(self, *a, **k):
        pass


class FakeSM:
    def __init__(self, obstacles, predictions=(), pred_id=None, span=3.0):
        self.cur_s = 0.0
        self.cur_vs = 5.0
        self.max_s = TRACK_LENGTH
        self.track_length = TRACK_LENGTH
        self.gb_ego_width_m = 0.29
        self.overtake_min_closing_mps = 2.5
        self.prediction_dt = 0.02
        self.dynamic_prediction_span_m = span
        self.cur_obstacles_in_interest = list(obstacles)
        self.obstacles_prediction = list(predictions)
        self.obstacles_prediction_id = pred_id
        self.pars = {"veh_params": {"length": 0.5}}

    def get_logger(self):
        return _Logger()

    # staticmethod() so binding it onto the stub does not turn it back into
    # an instance method (it is declared @staticmethod on the real class).
    _obs_lateral_half_width = staticmethod(StateMachine._obs_lateral_half_width)
    _prediction_span_end_idx = StateMachine._prediction_span_end_idx
    _check_free_frenet = StateMachine._check_free_frenet


def _static_free_dist(path_d, obs, ego_width):
    """The pre-change static formula, written out independently."""
    return abs(path_d - obs.d_center) - obs.size / 2 - ego_width / 2


def test_static_branch_still_uses_size_not_the_frenet_bounds():
    """An elongated cluster: size 1.2 m, but d bounds say only 0.4 m wide.

    The dynamic branches now prefer the d bounds. The static branch must not:
    changing it would change how far the car gives a static box a berth, which
    is exactly what this task is not allowed to touch.
    """
    obs = Obs(1, s_center=4.0, d_center=0.5, size=1.2, is_static=True,
              d_left=0.7, d_right=0.3)
    path = PathStub(path_d=0.0, is_gb=True)
    sm = FakeSM([obs])
    sm._check_free_frenet(path)

    rec = path.free_dbg["obs"][0]
    assert rec["branch"] == "static/geom"
    assert rec["free_dist"] == pytest.approx(
        round(_static_free_dist(0.0, obs, sm.gb_ego_width_m), 3))
    # size/2 = 0.60 was used, not |d_left - d_right|/2 = 0.20.
    assert rec["free_dist"] == pytest.approx(-0.245, abs=1e-3)
    assert rec["blocked"] is True
    assert path.free_dbg["is_free"] is False


def test_static_verdict_free_when_the_path_gives_it_room():
    obs = Obs(1, s_center=4.0, d_center=0.9, size=0.4, is_static=True)
    path = PathStub(path_d=-0.4, is_ot=True)
    sm = FakeSM([obs])
    assert sm._check_free_frenet(path) is True
    assert path.free_dbg["obs"][0]["blocked"] is False


def test_static_is_not_bounded_by_the_prediction_span():
    """dynamic_prediction_span_m must not reach the static branch at all.

    Same static obstacle, same path, spans 0.1 m and 30 m -- identical verdict
    and identical recorded clearance.
    """
    results = []
    for span in (0.1, 30.0):
        obs = Obs(1, s_center=6.0, d_center=0.4, size=0.5, is_static=True)
        path = PathStub(path_d=0.0, is_ot=True)
        sm = FakeSM([obs], predictions=[Pred(6.0 + i * 0.1, 0.4) for i in range(60)],
                    pred_id=1, span=span)
        results.append((sm._check_free_frenet(path), path.free_dbg["obs"][0]["free_dist"]))
    assert results[0] == results[1]


def test_static_beyond_path_end_still_blocks_on_an_open_path():
    """The `max_gap < gap < max_horizon` guard is untouched."""
    obs = Obs(1, s_center=11.0, d_center=0.0, size=0.4, is_static=True)
    path = PathStub(path_d=0.0, is_ot=True, max_horizon=15.0, length_m=8.0)
    sm = FakeSM([obs])
    assert sm._check_free_frenet(path) is False
    assert path.free_dbg["obs"][0]["branch"] == "static/beyond_path"


def test_static_past_max_horizon_is_ignored():
    obs = Obs(1, s_center=12.0, d_center=0.0, size=0.4, is_static=True)
    path = PathStub(path_d=0.0, is_ot=True, max_horizon=10.0, length_m=14.0)
    sm = FakeSM([obs])
    assert sm._check_free_frenet(path) is True
    assert path.free_dbg["obs"][0]["branch"] == "static/gap>=max_horizon"


def test_dynamic_branch_does_prefer_the_frenet_bounds():
    """The counterpart of the first test: same geometry, is_static False."""
    obs = Obs(1, s_center=4.0, d_center=0.5, size=1.2, is_static=False,
              d_left=0.7, d_right=0.3)
    path = PathStub(path_d=0.0, is_ot=True)
    sm = FakeSM([obs])
    sm._check_free_frenet(path)

    rec = path.free_dbg["obs"][0]
    assert rec["branch"] == "dyn/nopred (id_mismatch or empty)"
    # 0.5 - 0.20 (d-bound half width) - 0.145 = 0.155, not the -0.245 above.
    assert rec["free_dist"] == pytest.approx(0.155, abs=1e-3)
