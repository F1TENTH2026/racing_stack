"""When the lost-opponent speed hold may and may not fire.

Driven against the real _opponent_memory_active with a stub, so the exclusions
are pinned rather than re-derived from a log next time.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from state_machine.state_machine_node import StateMachine  # noqa: E402
from state_machine.states_types import StateType  # noqa: E402


class Obs:
    def __init__(self, is_static=False):
        self.is_static = is_static


class MemSM:
    def __init__(self, state=StateType.GB_TRACK, seen_ago=1.0, obstacles=(),
                 overtake_ago=None, window=3.0, grace=2.0):
        self.cur_state = state
        self._now = 1000.0
        self._last_dyn_seen_sec = None if seen_ago is None else self._now - seen_ago
        self._last_overtake_sec = None if overtake_ago is None else self._now - overtake_ago
        self.cur_obstacles_in_interest = list(obstacles)
        self.dynamic_opponent_memory_sec = window
        self.overtake_pass_grace_sec = grace

    def now_sec(self):
        return self._now

    _opponent_memory_active = StateMachine._opponent_memory_active


def test_holds_after_a_real_dropout():
    """The 2026-08-22 rear-end: seen 2.6 s ago, no recent overtake, empty list."""
    sm = MemSM(state=StateType.GB_TRACK, seen_ago=2.6, overtake_ago=8.4)
    assert sm._opponent_memory_active() is True


def test_never_holds_while_overtaking():
    """84 of 136 holds in the 09:19 run capped the car mid-pass."""
    sm = MemSM(state=StateType.OVERTAKE, seen_ago=0.2)
    assert sm._opponent_memory_active() is False


def test_never_holds_just_after_a_pass():
    """A car that vanishes right after we passed it is behind us, not lost."""
    sm = MemSM(state=StateType.GB_TRACK, seen_ago=0.5, overtake_ago=0.4)
    assert sm._opponent_memory_active() is False


def test_holds_again_once_the_pass_grace_expires():
    sm = MemSM(state=StateType.GB_TRACK, seen_ago=1.0, overtake_ago=2.5)
    assert sm._opponent_memory_active() is True


def test_window_expiry_releases():
    sm = MemSM(state=StateType.GB_TRACK, seen_ago=3.5, overtake_ago=None)
    assert sm._opponent_memory_active() is False


def test_no_hold_while_a_dynamic_obstacle_is_visible():
    sm = MemSM(state=StateType.GB_TRACK, seen_ago=0.5, obstacles=[Obs(is_static=False)])
    assert sm._opponent_memory_active() is False


def test_static_obstacles_do_not_block_the_hold():
    sm = MemSM(state=StateType.GB_TRACK, seen_ago=1.0, obstacles=[Obs(is_static=True)])
    assert sm._opponent_memory_active() is True


def test_never_seen_an_opponent():
    sm = MemSM(state=StateType.GB_TRACK, seen_ago=None)
    assert sm._opponent_memory_active() is False


def test_window_zero_disables_the_feature():
    sm = MemSM(state=StateType.GB_TRACK, seen_ago=0.5, window=0.0)
    assert sm._opponent_memory_active() is False


def test_grace_zero_disables_only_the_pass_suppression():
    sm = MemSM(state=StateType.GB_TRACK, seen_ago=0.5, overtake_ago=0.1, grace=0.0)
    assert sm._opponent_memory_active() is True
