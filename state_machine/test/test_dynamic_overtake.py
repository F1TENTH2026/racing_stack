"""Function-level tests for the DYNAMIC overtake decision.

Deliberately unit-level: the methods under test are pulled off the class and
called against a lightweight stub, so no ROS node, no rclpy.init() and no
network are involved. Run with:

    python3 -m pytest src/racing_stack/state_machine/test/test_dynamic_overtake.py
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from state_machine.state_machine_node import StateMachine  # noqa: E402

TRACK_LENGTH = 20.0


class Obs:
    """Minimal stand-in for f110_msgs/Obstacle."""

    def __init__(self, oid, s_center, vs=0.0, is_static=False, d_center=0.0, size=0.5):
        self.id = oid
        self.s_center = s_center
        self.s_start = s_center - size / 2.0
        self.s_end = s_center + size / 2.0
        self.vs = vs
        self.is_static = is_static
        self.d_center = d_center
        self.d_left = d_center + size / 2.0
        self.d_right = d_center - size / 2.0
        self.size = size


class FakeSM:
    """Only the attributes _check_getting_closer / _nearest_dynamic_opponent_ahead read."""

    def __init__(self, cur_s=0.0, cur_vs=5.0, obstacles=(), track_length=TRACK_LENGTH):
        self.cur_s = cur_s
        self.cur_vs = cur_vs
        self.obstacles_in_interest = list(obstacles)
        self.track_length = track_length
        self.dynamic_overtake_min_rel_speed_mps = -0.5
        self._dyn_ot_target = None

    # Bound under their real names: _check_getting_closer calls
    # self._nearest_dynamic_opponent_ahead().
    _nearest_dynamic_opponent_ahead = StateMachine._nearest_dynamic_opponent_ahead
    getting_closer = StateMachine._check_getting_closer


def test_case1_no_opponent_no_candidate():
    """Case 1: no opponent -> no dynamic overtake candidate (car stays on the raceline)."""
    sm = FakeSM()
    assert sm.getting_closer(10.0) is False
    assert sm._dyn_ot_target is None


def test_case2_opponent_at_15m_is_out_of_range():
    """Case 2: opponent 15 m ahead -> outside the 10 m gate, not yet a candidate.

    This is the regression the gate exists for: the old body ignored threshold_m
    entirely and returned True here.
    """
    sm = FakeSM(cur_s=0.0, cur_vs=5.0, obstacles=[Obs(1, 15.0, vs=3.0)], track_length=60.0)
    assert sm.getting_closer(10.0) is False


def test_case3_opponent_at_8m_closing_is_a_candidate():
    """Case 3: 8 m ahead, ego 5 m/s vs opponent 3 m/s -> candidate."""
    sm = FakeSM(cur_s=0.0, cur_vs=5.0, obstacles=[Obs(3, 8.0, vs=3.0)], track_length=60.0)
    assert sm.getting_closer(10.0) is True
    assert sm._dyn_ot_target["id"] == 3
    assert sm._dyn_ot_target["gap"] == pytest.approx(8.0)
    assert sm._dyn_ot_target["rel_v"] == pytest.approx(2.0)


def test_relative_speed_floor_is_negative_not_zero():
    """racing_stack lets the ego be marginally SLOWER and still qualify (> -0.5 m/s)."""
    sm = FakeSM(cur_s=0.0, cur_vs=5.0, obstacles=[Obs(3, 8.0, vs=5.3)], track_length=60.0)
    assert sm.getting_closer(10.0) is True          # rel = -0.3, above the -0.5 floor
    sm = FakeSM(cur_s=0.0, cur_vs=5.0, obstacles=[Obs(3, 8.0, vs=5.8)], track_length=60.0)
    assert sm.getting_closer(10.0) is False         # rel = -0.8, below it
    assert sm._dyn_ot_target["rel_ok"] is False     # candidate found, speed rejected it


def test_case6_opponent_behind_is_never_the_target():
    """Case 6: a car BEHIND the ego must not become the overtake target."""
    behind = Obs(7, 55.0, vs=3.0)                   # ego at 0 on a 60 m track -> 5 m behind
    sm = FakeSM(cur_s=0.0, cur_vs=5.0, obstacles=[behind], track_length=60.0)
    assert sm.getting_closer(10.0) is False
    assert sm._dyn_ot_target is None


def test_case7_two_ahead_picks_the_nearest():
    """Case 7: two cars ahead -> the nearer one is the target, whatever the list order."""
    far = Obs(2, 9.0, vs=3.0)
    near = Obs(1, 4.0, vs=3.0)
    sm = FakeSM(cur_s=0.0, cur_vs=5.0, obstacles=[far, near], track_length=60.0)
    assert sm.getting_closer(10.0) is True
    assert sm._dyn_ot_target["id"] == 1
    assert sm._dyn_ot_target["gap"] == pytest.approx(4.0)


def test_case8_start_finish_wrap_around():
    """Case 8: ego at track_length - 0.5, opponent at 0.5 -> 1 m AHEAD, not a lap behind."""
    sm = FakeSM(cur_s=TRACK_LENGTH - 0.5, cur_vs=5.0,
                obstacles=[Obs(4, 0.5, vs=3.0)], track_length=TRACK_LENGTH)
    assert sm.getting_closer(10.0) is True
    assert sm._dyn_ot_target["gap"] == pytest.approx(1.0)


def test_static_obstacle_is_not_a_dynamic_candidate():
    """Case 13 support: a STATIC obstacle never enters the dynamic path."""
    sm = FakeSM(cur_s=0.0, cur_vs=5.0,
                obstacles=[Obs(9, 5.0, vs=0.0, is_static=True)], track_length=60.0)
    assert sm.getting_closer(10.0) is False
    assert sm._dyn_ot_target is None


def test_window_capped_at_half_lap_on_a_short_track():
    """On a 17.79 m map a 10 m window would otherwise reach round onto the car's tail."""
    tail = Obs(5, 9.5, vs=0.0)                      # ego at 0 -> 9.5 m ahead == 8.29 m behind
    sm = FakeSM(cur_s=0.0, cur_vs=5.0, obstacles=[tail], track_length=17.79)
    assert sm.getting_closer(10.0) is False         # 9.5 > 17.79/2 = 8.895


def test_zero_track_length_is_not_a_crash():
    """track_length starts at 1.0/0 before /global_waypoints arrives."""
    sm = FakeSM(cur_s=0.0, cur_vs=5.0, obstacles=[Obs(1, 1.0)], track_length=0.0)
    assert sm.getting_closer(10.0) is False
