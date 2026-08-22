"""Frame-by-frame tests for the learned-trajectory authorization hysteresis.

_update_learned_authorization is pulled off the class and driven against a
lightweight stub, so no ROS node and no GP fit are involved.

    python3 -m pytest src/racing_stack/prediction/gp_traj_predictor/test/test_prediction_hysteresis.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class FakePredictor:
    """Only the attributes _update_learned_authorization reads."""

    def __init__(self, lap_count=1.0, force_trailing=True):
        self.opponent_lap_count = lap_count
        self.min_training_laps = 0.5
        self.learned_deviation_enter_threshold = 0.35
        self.learned_deviation_exit_threshold = 0.55
        self.learned_ready_confirm_frames = 3
        self.learned_reject_confirm_frames = 5
        self.force_trailing = force_trailing
        self._learned_ready_frames = 0
        self._learned_reject_frames = 0

    def feed(self, *deviations):
        """Run one frame per deviation, return force_trailing after each."""
        return [self.update(d) for d in deviations]


def _bind():
    # Imported lazily so a missing rclpy on a dev box fails one test, not collection.
    from opp_prediction import OppTrajPredictor
    FakePredictor.update = OppTrajPredictor._update_learned_authorization


_bind()


def test_case9_three_frames_below_enter_threshold_authorize():
    """Case 9: deviation 0.34 three times -> authorized (force_trailing False)."""
    p = FakePredictor(lap_count=1.0, force_trailing=True)
    assert p.feed(0.34, 0.34, 0.34) == [True, True, False]


def test_two_good_frames_are_not_enough():
    """The confirm count is real: two frames still veto."""
    p = FakePredictor(lap_count=1.0, force_trailing=True)
    assert p.feed(0.34, 0.34) == [True, True]


def test_case10_authorized_does_not_flip_back_inside_the_band():
    """Case 10: once authorized, 0.40/0.48/0.52 must NOT return to force_trailing.

    Every one of those is above the 0.35 enter threshold but below the 0.55 exit
    threshold -- exactly the band the old single 0.25 m test chattered in.
    """
    p = FakePredictor(lap_count=1.0, force_trailing=True)
    p.feed(0.34, 0.34, 0.34)
    assert p.force_trailing is False
    assert p.feed(0.40, 0.48, 0.52) == [False, False, False]


def test_case11_five_frames_above_exit_threshold_veto():
    """Case 11: deviation 0.60 five times -> back to force_trailing."""
    p = FakePredictor(lap_count=1.0, force_trailing=True)
    p.feed(0.34, 0.34, 0.34)
    assert p.force_trailing is False
    assert p.feed(0.60, 0.60, 0.60, 0.60, 0.60) == [False, False, False, False, True]


def test_a_single_bad_frame_does_not_veto():
    """One 0.60 spike while authorized is absorbed."""
    p = FakePredictor(lap_count=1.0, force_trailing=True)
    p.feed(0.34, 0.34, 0.34)
    assert p.feed(0.60, 0.10) == [False, False]


def test_untrained_opponent_is_vetoed_immediately():
    """Below min_training_laps there is no learned trajectory to be near."""
    p = FakePredictor(lap_count=0.2, force_trailing=False)
    assert p.feed(0.01) == [True]


def test_half_a_lap_is_enough_training():
    """min_training_laps is 0.5, not 1 -- half a lap of observation authorizes."""
    p = FakePredictor(lap_count=0.5, force_trailing=True)
    assert p.feed(0.10, 0.10, 0.10) == [True, True, False]


def test_lap_count_none_does_not_raise():
    """opponent_lap_count is None until the first /opponent_trajectory message.

    The old expression `... or self.opponent_lap_count < 1` raised TypeError on
    the first frame whose deviation was small enough to reach it, killing the
    predictor loop.
    """
    p = FakePredictor(lap_count=None, force_trailing=False)
    assert p.feed(0.10) == [True]
