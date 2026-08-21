from types import SimpleNamespace

import numpy as np

from state_machine.state_machine_node import StateMachine
from state_machine.states_types import StateType


def obstacle(obs_id, s, is_static):
    return SimpleNamespace(id=obs_id, s_center=s, is_static=is_static)


def safety_fixture(collision_last=1, include_other_blocker=False, target_s=13.0, static_s=16.0):
    node = object.__new__(StateMachine)
    target = obstacle(1, target_s, False)
    static = obstacle(100, static_s, True)
    node.cur_s = 10.0
    node.max_s = 100.0
    node.local_wpnts_src = StateType.GB_TRACK
    node._static_trailing_target_id = None
    node.cur_gb_wpnts = SimpleNamespace(closest_target=target)
    node.cur_recovery_wpnts = SimpleNamespace(closest_target=None)
    node.cur_obstacles_in_interest = [target, static]
    node._static_safety_dbg = None
    node._check_free_frenet = lambda _: False

    blockers = [{
        "id": 1, "blocked": True, "collision_path_idx": 0,
        "collision_path_last_idx": collision_last,
    }]
    if include_other_blocker:
        blockers.append({
            "id": 2, "blocked": True, "collision_path_idx": 0,
            "collision_path_last_idx": 0,
        })
    path = SimpleNamespace(
        is_init=True,
        lateral_width_m=0.1,
        max_horizon=10.0,
        # Divergence starts at index 2: |0.11 - 0.0| > lateral_width_m.
        array=np.asarray([
            [0.0, 0.0, 10.0, 0.00],
            [0.0, 0.0, 11.0, 0.05],
            [0.0, 0.0, 12.0, 0.11],
            [0.0, 0.0, 13.0, 0.30],
        ]),
        free_dbg={"obs": blockers},
    )
    return node, path


def test_current_target_shared_prefix_is_preparation_not_hard_block():
    node, path = safety_fixture(collision_last=1)
    result = node._check_static_path_safety_detailed(path)
    assert result["safe"] is False
    assert result["prefix_only"] is True
    assert result["target_id"] == 1
    assert result["entry_idx"] == 2
    assert result["reason"] == "TRAILING_TARGET_SHARED_PREFIX"


def test_current_target_post_entry_conflict_remains_blocked():
    node, path = safety_fixture(collision_last=2)
    result = node._check_static_path_safety_detailed(path)
    assert result["prefix_only"] is False
    assert result["reason"] == "POST_ENTRY_CONFLICT"


def test_non_target_dynamic_obstacle_remains_blocker():
    node, path = safety_fixture(collision_last=1, include_other_blocker=True)
    result = node._check_static_path_safety_detailed(path)
    assert result["prefix_only"] is False
    assert result["reason"] == "NON_TARGET_BLOCKER"


def test_target_must_be_between_ego_and_static():
    node, path = safety_fixture(collision_last=1, target_s=18.0, static_s=16.0)
    result = node._check_static_path_safety_detailed(path)
    assert result["prefix_only"] is False
    assert result["reason"] == "TARGET_NOT_BETWEEN_EGO_AND_STATIC"
