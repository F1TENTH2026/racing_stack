import time
from types import SimpleNamespace

import numpy as np

from controller.controller_manager import ControllerManager


def manager_stub():
    node = object.__new__(ControllerManager)
    node.controller = object()
    node.last_behavior_received_time = time.monotonic()
    node.last_odom_received_time = time.monotonic()
    node.behavior_timeout_sec = 0.25
    node.odom_timeout_sec = 0.15
    node.minimum_trajectory_points = 20
    node.position_in_map = np.array([[0.0, 0.0, 0.0]])
    node.position_in_map_frenet = np.zeros(4)
    node.waypoint_array_in_map = np.zeros((20, 9))
    node.max_steering_angle = 0.53
    node.last_finite_steering = 0.2
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(to_msg=lambda: SimpleNamespace()))
    return node


def test_stale_behavior_rejected_before_control():
    node = manager_stub()
    node.last_behavior_received_time -= 1.0
    assert node._trajectory_invalid_reason(time.monotonic()) == 'behavior stale'


def test_nonfinite_and_short_trajectory_rejected():
    node = manager_stub()
    node.waypoint_array_in_map[3, 2] = np.nan
    assert node._trajectory_invalid_reason(time.monotonic()) == 'trajectory non-finite'
    node.waypoint_array_in_map = np.zeros((2, 9))
    assert node._trajectory_invalid_reason(time.monotonic()) == 'trajectory missing/short'


def test_actuator_boundary_stops_nan_speed_and_holds_finite_steer():
    node = manager_stub()
    msg = node.create_ack_msg(np.nan, np.inf, np.nan, np.inf)
    assert msg.drive.speed == 0.0
    assert msg.drive.steering_angle == 0.2
    assert msg.drive.acceleration == 0.0
    assert msg.drive.jerk == 0.0


def test_actuator_boundary_clamps_steering():
    node = manager_stub()
    msg = node.create_ack_msg(3.0, 0.0, 0.0, 2.0)
    assert msg.drive.speed == 3.0
    assert msg.drive.steering_angle == 0.53
