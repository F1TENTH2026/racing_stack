"""Tests for the planner-side prediction corridor (_truncate_prediction).

Skipped where the planner's runtime deps (ccma, trajectory_planning_helpers,
grid_filter) are not installed -- i.e. on a dev box; it runs on the car and in a
full workspace build.
"""
import pytest

try:
    from lane_change_planner.change_avoidance_node import ChangeAvoidanceNode
except ImportError as exc:  # pragma: no cover - dev box without the car's deps
    ChangeAvoidanceNode = None
    _IMPORT_ERROR = str(exc)
else:
    _IMPORT_ERROR = None

pytestmark = pytest.mark.skipif(
    ChangeAvoidanceNode is None,
    reason=f"lane_change_planner runtime deps not installed: {_IMPORT_ERROR}",
)


class Obs:
    """Minimal stand-in for f110_msgs/Obstacle."""

    def __init__(self, oid, s_center, d_center=0.0, size=0.5):
        self.id = oid
        self.s_center = s_center
        self.s_start = s_center - size / 2.0
        self.s_end = s_center + size / 2.0
        self.d_center = d_center
        self.d_left = d_center + size / 2.0
        self.d_right = d_center - size / 2.0
        self.size = size
        self.is_static = False


class FakePlanner:
    def __init__(self, span=3.0, max_s=40.0):
        self.prediction_span_m = span
        self.scaled_max_s = max_s

    if ChangeAvoidanceNode is not None:
        _truncate_prediction = ChangeAvoidanceNode._truncate_prediction


def _prediction(start_s, step, n, max_s=40.0):
    return [Obs(i, (start_s + i * step) % max_s) for i in range(n)]


def test_case12_only_the_first_3m_is_planned_for():
    """Case 12: 6 m of prediction, 3 m corridor -> the path spans 3 m, not 6."""
    p = FakePlanner(span=3.0)
    kept = p._truncate_prediction(_prediction(0.0, 0.1, 61))
    assert len(kept) == 31
    assert kept[-1].s_center == pytest.approx(3.0)


def test_full_predictor_output_is_cut():
    """200 steps at 3 m/s * 0.02 s = 12 m -> 51 poses kept."""
    p = FakePlanner(span=3.0)
    assert len(p._truncate_prediction(_prediction(0.0, 0.06, 200))) == 51


def test_span_zero_keeps_everything():
    p = FakePlanner(span=0.0)
    assert len(p._truncate_prediction(_prediction(0.0, 0.1, 61))) == 61


def test_never_degenerate():
    p = FakePlanner(span=0.0001)
    assert len(p._truncate_prediction(_prediction(0.0, 0.1, 61))) == 2


def test_seam_crossing_prediction():
    p = FakePlanner(span=3.0, max_s=20.0)
    kept = p._truncate_prediction(_prediction(19.0, 0.1, 61, max_s=20.0))
    assert len(kept) == 31
    assert kept[-1].s_center == pytest.approx(2.0)


def test_returns_copies_not_aliases():
    """lane_change() unwraps and elongates s in place; the predictor's message
    must not be mutated, or the elongation compounds across loops (the planner
    runs at 20 Hz against a 10 Hz predictor)."""
    p = FakePlanner(span=3.0)
    source = _prediction(0.0, 0.1, 61)
    kept = p._truncate_prediction(source)
    kept[0].s_end += 5.0
    assert source[0].s_end == pytest.approx(0.25)
