"""Cases 3, 4, 5: which gate decides OVERTAKE vs TRAILING, and that the existing
racing_stack trailing speed reduction survives the change.

_check_overtaking_mode() is driven directly with each precondition stubbed, so
the test pins the ORDER and the SET of gates, not any one implementation detail.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from state_machine.state_machine_node import StateMachine  # noqa: E402
from state_machine.states_types import StateType  # noqa: E402


class GateSM:
    """Every precondition of _check_overtaking_mode() as a settable flag."""

    def __init__(self, sector=True, in_range=True, force_trailing=False,
                 use_force_trailing=True, path_fresh=True, path_safe=True):
        self._sector = sector
        self._in_range = in_range
        self.force_trailing = force_trailing
        self.use_force_trailing = use_force_trailing
        self._path_fresh = path_fresh
        self._path_safe = path_safe
        self.dynamic_overtake_max_gap_m = 10.0
        self.static_overtaking_mode = True
        self.avoidance_wpnts = object()
        self.cur_avoidance_wpnts = object()
        self.calls = []

    def _check_ot_sector(self):
        self.calls.append("sector")
        return self._sector

    def _check_getting_closer(self, threshold_m):
        self.calls.append("range")
        assert threshold_m == self.dynamic_overtake_max_gap_m
        return self._in_range

    def _check_latest_wpnts(self, _src, _data):
        self.calls.append("fresh")
        return self._path_fresh

    def _check_free_frenet(self, _data):
        self.calls.append("safe")
        return self._path_safe

    _check_overtaking_mode = StateMachine._check_overtaking_mode


def test_case3_all_gates_pass_gives_overtake():
    """Case 3: 8 m ahead, closing, prediction valid, no force_trailing, safe path."""
    sm = GateSM()
    assert sm._check_overtaking_mode() is True
    assert sm.static_overtaking_mode is False


def test_case4_force_trailing_vetoes_entry():
    """Case 4: 8 m ahead but force_trailing -> TRAILING, and the veto is applied
    BEFORE the path checks, so a vetoed frame costs nothing."""
    sm = GateSM(force_trailing=True)
    assert sm._check_overtaking_mode() is False
    assert sm.calls == ["sector", "range"]


def test_force_trailing_can_be_disabled_by_parameter():
    sm = GateSM(force_trailing=True, use_force_trailing=False)
    assert sm._check_overtaking_mode() is True


def test_case5_no_path_gives_trailing():
    """Case 5: opponent close, no avoidance path -> not OVERTAKE."""
    sm = GateSM(path_fresh=False)
    assert sm._check_overtaking_mode() is False


def test_unsafe_path_gives_trailing():
    sm = GateSM(path_safe=False)
    assert sm._check_overtaking_mode() is False


def test_out_of_sector_short_circuits_first():
    sm = GateSM(sector=False)
    assert sm._check_overtaking_mode() is False
    assert sm.calls == ["sector"]


def test_out_of_range_short_circuits_before_the_path_checks():
    sm = GateSM(in_range=False)
    assert sm._check_overtaking_mode() is False
    assert sm.calls == ["sector", "range"]


# --------------------------------------------------------------------------- #
# Case 5, second half: the trailing speed reduction must still be there.
# --------------------------------------------------------------------------- #

class Wpnt:
    def __init__(self, vx_mps):
        self.vx_mps = vx_mps


class Cache:
    def __init__(self, closest_gap):
        self.closest_gap = closest_gap


class GlobalWpnts:
    def __init__(self, v):
        self.wpnts = [Wpnt(v) for _ in range(100)]


class TrailSM:
    def __init__(self, gap, raceline_v=6.0, ramp=True, memory=False):
        self.cur_state = StateType.TRAILING
        self.local_wpnts_src = StateType.GB_TRACK
        self.cur_gb_wpnts = Cache(gap)
        self.cur_recovery_wpnts = Cache(None)
        self._memory = memory
        self._last_dyn_gap_m = gap
        self._last_dyn_seen_sec = 0.0
        self._last_dyn_id = 7
        self._dbg_last_memory_log_sec = 0.0
        self.emergency_break_horizon = 8.0
        self.trailing_speed_scale = 0.35
        self.trailing_min_speed_mps = 1.0
        self.trailing_speed_ramp = ramp
        self.num_glb_wpnts = 100
        self.wpnt_dist = 0.1
        self.cur_s = 0.0
        self.gb_wpnts = GlobalWpnts(raceline_v)
        self.applied_cap = None

    def update_velocity(self, wpnts_msg, safety_factor=1.0, speed_cap=None):
        self.applied_cap = speed_cap

    def _opponent_memory_active(self):
        return self._memory

    def now_sec(self):
        return 0.0

    def get_logger(self):
        class _L:
            def warn(self, *a, **k):
                pass
        return _L()

    def _dbg_log(self, msg):
        pass

    _apply_trailing_speed_cap = StateMachine._apply_trailing_speed_cap


def test_trailing_speed_cap_still_applies_inside_the_horizon():
    """Case 5: no path -> TRAILING, and the cruise cap the stack already had is
    still applied. emergency_break_horizon / trailing_speed_scale /
    trailing_min_speed_mps must keep working exactly as before."""
    sm = TrailSM(gap=4.0, raceline_v=6.0)
    out = sm._apply_trailing_speed_cap([Wpnt(6.0)])
    # ramp: t = 4/8 = 0.5 -> scale = 0.35 + 0.65*0.5 = 0.675 -> cap = 4.05
    assert sm.applied_cap == pytest.approx(4.05)
    assert out is not None


def test_trailing_speed_cap_floor_is_respected():
    sm = TrailSM(gap=0.0, raceline_v=1.0)
    sm._apply_trailing_speed_cap([Wpnt(1.0)])
    assert sm.applied_cap == pytest.approx(1.0)   # trailing_min_speed_mps


def test_trailing_speed_cap_not_applied_beyond_the_horizon():
    sm = TrailSM(gap=9.0)
    sm._apply_trailing_speed_cap([Wpnt(6.0)])
    assert sm.applied_cap is None


def test_trailing_speed_cap_step_mode_unchanged():
    sm = TrailSM(gap=4.0, raceline_v=6.0, ramp=False)
    sm._apply_trailing_speed_cap([Wpnt(6.0)])
    assert sm.applied_cap == pytest.approx(2.1)   # 6.0 * 0.35


def test_trailing_speed_cap_not_applied_outside_trailing_without_memory():
    """No opponent seen recently and not TRAILING -> the profile is untouched."""
    sm = TrailSM(gap=4.0, memory=False)
    sm.cur_state = StateType.OVERTAKE
    wpnts = [Wpnt(6.0)]
    assert sm._apply_trailing_speed_cap(wpnts) is wpnts
    assert sm.applied_cap is None


def test_cap_is_held_after_the_opponent_is_lost():
    """The 2026-08-22 rear-end: obstacle list empties, state leaves TRAILING for
    GB_TRACK, and the car used to go straight back to raceline speed. With the
    memory window active the last known gap still caps the profile."""
    sm = TrailSM(gap=4.0, raceline_v=6.0, memory=True)
    sm.cur_state = StateType.GB_TRACK
    sm._apply_trailing_speed_cap([Wpnt(6.0)])
    # same ramp as while trailing: t = 4/8 -> scale 0.675 -> cap 4.05
    assert sm.applied_cap == pytest.approx(4.05)


def test_memory_cap_uses_the_last_known_gap():
    sm = TrailSM(gap=1.0, raceline_v=6.0, memory=True)
    sm.cur_state = StateType.GB_TRACK
    sm._apply_trailing_speed_cap([Wpnt(6.0)])
    # t = 1/8 = 0.125 -> scale = 0.35 + 0.65*0.125 = 0.43125 -> cap 2.5875
    assert sm.applied_cap == pytest.approx(2.5875)


def test_memory_expiry_releases_the_cap():
    sm = TrailSM(gap=4.0, memory=True)
    sm.cur_state = StateType.GB_TRACK
    sm._memory = False          # window elapsed
    wpnts = [Wpnt(6.0)]
    assert sm._apply_trailing_speed_cap(wpnts) is wpnts
    assert sm.applied_cap is None
