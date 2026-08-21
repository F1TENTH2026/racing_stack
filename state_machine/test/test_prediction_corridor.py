"""Tests for the limited prediction corridor and the obstacle lateral-width helper."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from state_machine.state_machine_node import StateMachine  # noqa: E402


class Pred:
    """Minimal stand-in for f110_msgs/Prediction."""

    def __init__(self, pred_s, pred_d=0.0):
        self.pred_s = pred_s
        self.pred_d = pred_d


class Obs:
    def __init__(self, d_left, d_right, size):
        self.d_left = d_left
        self.d_right = d_right
        self.size = size


class FakeSM:
    def __init__(self, span=3.0, max_s=20.0):
        self.dynamic_prediction_span_m = span
        self.max_s = max_s

    _prediction_span_end_idx = StateMachine._prediction_span_end_idx


def _prediction(start_s, step, n, max_s=20.0):
    """A predictor-style monotonic track: 200 steps of v*dt each."""
    return [Pred((start_s + i * step) % max_s) for i in range(n)]


def test_case12_corridor_stops_at_the_span():
    """Case 12: prediction runs 6 m, only the first 3 m is used for feasibility.

    0.1 m per step -> poses at 0.0 .. 6.0. Everything at or below 3.0 m from the
    opponent's current pose is kept: indices 0..30, i.e. an end index of 31.
    """
    sm = FakeSM(span=3.0, max_s=60.0)
    preds = _prediction(0.0, 0.1, 61, max_s=60.0)
    end = sm._prediction_span_end_idx(preds)
    assert end == 31
    assert preds[end - 1].pred_s == pytest.approx(3.0)


def test_full_predictor_output_is_cut_to_the_span():
    """The real shape: 200 steps of v*dt (3 m/s, 0.02 s) = 12 m -> 3 m keeps 51."""
    sm = FakeSM(span=3.0, max_s=40.0)
    preds = _prediction(0.0, 3.0 * 0.02, 200, max_s=40.0)
    assert sm._prediction_span_end_idx(preds) == 51


def test_span_zero_disables_the_cap():
    sm = FakeSM(span=0.0, max_s=60.0)
    preds = _prediction(0.0, 0.1, 61, max_s=60.0)
    assert sm._prediction_span_end_idx(preds) == 61


def test_corridor_never_degenerates():
    """Even a span shorter than one step keeps two poses, so there is a start and an end."""
    sm = FakeSM(span=0.001, max_s=60.0)
    preds = _prediction(0.0, 0.1, 61, max_s=60.0)
    assert sm._prediction_span_end_idx(preds) == 2


def test_corridor_across_the_start_finish_seam():
    """A prediction that crosses s=0 is still measured as forward distance."""
    sm = FakeSM(span=3.0, max_s=20.0)
    preds = _prediction(19.0, 0.1, 61, max_s=20.0)   # 19.0 .. 19.9, 0.0 .. 5.1
    end = sm._prediction_span_end_idx(preds)
    assert end == 31
    assert preds[end - 1].pred_s == pytest.approx(2.0)   # 19.0 + 3.0 wrapped


def test_short_prediction_is_returned_whole():
    sm = FakeSM(span=3.0, max_s=20.0)
    assert sm._prediction_span_end_idx([Pred(1.0)]) == 1
    assert sm._prediction_span_end_idx([]) == 0


def test_uninitialised_track_length_disables_the_cap():
    sm = FakeSM(span=3.0, max_s=0.0)
    preds = _prediction(0.0, 0.1, 10, max_s=60.0)
    assert sm._prediction_span_end_idx(preds) == 10


def test_lateral_width_prefers_the_frenet_bounds():
    """A long cluster: bounding circle 1.2 m wide, real lateral extent 0.5 m."""
    assert StateMachine._obs_lateral_half_width(Obs(0.25, -0.25, 1.2)) == pytest.approx(0.25)


def test_lateral_width_falls_back_to_size():
    """Degenerate or unset d bounds -> the old size-based number, unchanged."""
    assert StateMachine._obs_lateral_half_width(Obs(0.0, 0.0, 0.6)) == pytest.approx(0.3)


def test_lateral_width_matches_size_for_perception_obstacles():
    """detect.cpp fills d_left/d_right FROM size, so nothing changes for those."""
    size = 0.5
    obs = Obs(0.1 + size / 2, 0.1 - size / 2, size)
    assert StateMachine._obs_lateral_half_width(obs) == pytest.approx(size / 2)
