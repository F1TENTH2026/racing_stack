#!/usr/bin/env python3
import time
from typing import List, Any, Tuple
import copy

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import (
    FloatingPointRange,
    IntegerRange,
    ParameterDescriptor,
    ParameterType,
    SetParametersResult,
)

import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32, Bool
from visualization_msgs.msg import Marker, MarkerArray
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from scipy.interpolate import BPoly
from scipy.signal import argrelextrema
from f110_msgs.msg import Obstacle, ObstacleArray, OTWpntArray, Wpnt, WpntArray, BehaviorStrategy
from frenet_conversion.frenet_converter import FrenetConverter
from transforms3d.euler import quat2euler
from grid_filter.grid_filter import GridFilter
import trajectory_planning_helpers as tph


class ObstacleSpliner(Node):
    """
    This class implements a ROS node that splines a start trajectory through manually clicked poses.

    It subscribes to the following topics:
        - `/behavior_strategy`: Subscribes to the behavior strategy (overtaking targets).
        - `/car_state/odom_frenet`: Subscribes to the car state in Frenet coordinates.
        - `/car_state/odom`: Subscribes to the car state in cartesian coordinates.
        - `/global_waypoints`: Subscribes to global waypoints.
        - `/global_waypoints_scaled`: Subscribes to the scaled global waypoints.
        - `/move_base_simple/goal`: Subscribes to clicked goal poses.
        - `/save_start_traj`: Resets the accumulated clicked poses.

    The node publishes the following topics:
        - `/planner/start_wpnts/markers`: Publishes spline markers.
        - `/planner/start_wpnts`: Publishes splined waypoints.
    """

    def __init__(self):
        """
        Initialize the node, subscribe to topics, and create publishers and service proxies.
        """
        # Initialize the node
        self.name = "start_spliner_node"
        super().__init__('start_spline_node_v2')

        # initialize the instance variable
        self.obs_in_interest = None
        self.gb_wpnts = None
        self.gb_vmax = None
        self.gb_max_idx = None
        self.gb_max_s = None
        self.cur_s = None
        self.cur_d = None
        self.cur_vs = None
        self.cur_x = None
        self.cur_y = None
        self.cur_yaw = None
        self.start_target_m = 1.0
        self.gb_scaled_wpnts = None
        self.waypoints = None
        self.lookahead = 10  # in meters [m]
        self.last_switch_time = self.get_clock().now().to_msg()
        self.last_ot_side = ""
        self.points_without_pose = []
        self.tangents_without_pose = []

        # Static parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('from_bag', False),
                ('measure', False),
            ])
        self.from_bag = self.get_parameter('from_bag').get_parameter_value().bool_value
        self.measuring = self.get_parameter('measure').get_parameter_value().bool_value

        self.sampling_dist = 20.0
        self.spline_scale = 0.8
        self.post_min_dist = 1.5
        self.post_max_dist = 5.0
        self.kernel_size = 4

        self.map_filter = GridFilter(node=self, map_topic="/map", debug=False)
        self.map_filter.set_erosion_kernel_size(self.kernel_size)

        # dyn params default
        self.evasion_dist = 0.65
        self.obs_traj_tresh = 0.3
        self.spline_bound_mindist = 0.2
        self.n_loc_wpnts = 80
        self.width_car = 0.30

        # start_target_m was read from /dyn_statemachine/parameter_updates in ROS1.
        # Folded into a declared parameter with a param callback.
        self.declare_parameter(
            'start_target_m', 1.0,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description="Target distance ahead of the car for the start spline [m]",
                floating_point_range=[FloatingPointRange(from_value=0.0, to_value=20.0, step=0.001)],
            ))
        self.start_target_m = self.get_parameter('start_target_m').get_parameter_value().double_value
        self.add_on_set_parameters_callback(self.dyn_param_cb)

        # Subscribe to the topics
        self.create_subscription(BehaviorStrategy, "/behavior_strategy", self.behavior_cb, 10)
        self.create_subscription(Odometry, "/car_state/odom_frenet", self.state_frenet_cb, 10)
        self.create_subscription(Odometry, "/car_state/odom", self.state_cb, 10)
        self.create_subscription(WpntArray, "/global_waypoints", self.gb_cb, 10)
        self.create_subscription(WpntArray, "/global_waypoints_scaled", self.gb_scaled_cb, 10)
        self.create_subscription(PoseStamped, "/move_base_simple/goal", self.pose_cb, 10)
        self.create_subscription(Bool, "/save_start_traj", self.save_start_traj_cb, 10)

        self.mrks_pub = self.create_publisher(MarkerArray, "/planner/start_wpnts/markers", 10)
        self.evasion_pub = self.create_publisher(OTWpntArray, "/planner/start_wpnts", 10)
        if self.measuring:
            self.latency_pub = self.create_publisher(Float32, "/planner/avoidance/latency", 10)

        # Wait for critical messages
        self.wait_for_messages()

        self.converter = self.initialize_converter()

        # Set the rate at which the loop runs
        self.create_timer(1.0 / 20.0, self.loop)

    #############
    # CALLBACKS #
    #############
    def save_start_traj_cb(self, pose):
        self.points_without_pose = []
        self.tangents_without_pose = []

    def pose_cb(self, data):
        quat = data.pose.orientation
        # transforms3d uses (w, x, y, z) quaternion ordering
        euler = quat2euler([quat.w, quat.x, quat.y, quat.z])

        self.points_without_pose.append([data.pose.position.x, data.pose.position.y])
        self.tangents_without_pose.append([np.cos(euler[2]), np.sin(euler[2])])

    def behavior_cb(self, data: BehaviorStrategy):
        if len(data.overtaking_targets) != 0:
            self.obs_in_interest = data.overtaking_targets[0]
        else:
            self.obs_in_interest = None

    def state_frenet_cb(self, data: Odometry):
        self.cur_s = data.pose.pose.position.x
        self.cur_d = data.pose.pose.position.y
        self.cur_vs = data.twist.twist.linear.x

    def state_cb(self, data: Odometry):
        self.cur_x = data.pose.pose.position.x
        self.cur_y = data.pose.pose.position.y
        quat = data.pose.pose.orientation
        # transforms3d uses (w, x, y, z) quaternion ordering
        euler = quat2euler([quat.w, quat.x, quat.y, quat.z])
        self.cur_yaw = euler[2]

    # Callback for global waypoint topic
    def gb_cb(self, data: WpntArray):
        self.waypoints = np.array([[wpnt.x_m, wpnt.y_m] for wpnt in data.wpnts])
        self.gb_wpnts = data
        if self.gb_vmax is None:
            self.gb_vmax = np.max(np.array([wpnt.vx_mps for wpnt in data.wpnts]))
            self.gb_max_idx = data.wpnts[-1].id
            self.gb_max_s = data.wpnts[-1].s_m

    # Callback for scaled global waypoint topic
    def gb_scaled_cb(self, data: WpntArray):
        self.gb_scaled_wpnts = data

    # Callback triggered by dynamic reconf (folded statemachine tunable)
    def dyn_param_cb(self, params: List[Parameter]):
        """
        Notices the change in the parameters and changes spline params.
        """
        for param in params:
            if param.name == 'start_target_m':
                self.start_target_m = param.value
        return SetParametersResult(successful=True)

    #############
    # MAIN LOOP #
    #############
    def loop(self):
        if self.measuring:
            start = time.perf_counter()
        # Sample data
        gb_scaled_wpnts = self.gb_scaled_wpnts.wpnts
        wpnts = OTWpntArray()
        mrks = MarkerArray()

        wpnts, mrks = self.do_spline(obs=copy.deepcopy(self.obs_in_interest), gb_wpnts=gb_scaled_wpnts)

        # Publish wpnts and markers
        if self.measuring:
            end = time.perf_counter()
            self.latency_pub.publish(Float32(data=float(end - start)))
        self.evasion_pub.publish(wpnts)
        self.mrks_pub.publish(mrks)

    #########
    # UTILS #
    #########
    def wait_for_messages(self):
        self.get_logger().info(f"[{self.name}] Waiting for messages and services...")
        waitlist = [self.cur_s, self.cur_x, self.gb_wpnts, self.gb_scaled_wpnts]
        while None in waitlist:
            rclpy.spin_once(self)
            waitlist = [self.cur_s, self.cur_x, self.gb_wpnts, self.gb_scaled_wpnts]
        self.get_logger().info(f"[{self.name}] Ready!")

    def initialize_converter(self) -> FrenetConverter:
        """
        Initialize the FrenetConverter object"""
        waypoint_array = self.gb_wpnts.wpnts
        waypoints_x = np.array([wpnt.x_m for wpnt in waypoint_array])
        waypoints_y = np.array([wpnt.y_m for wpnt in waypoint_array])
        waypoints_psi = np.array([wpnt.psi_rad for wpnt in waypoint_array])
        converter = FrenetConverter(waypoints_x, waypoints_y, waypoints_psi)
        self.get_logger().info(f"[{self.name}] initialized FrenetConverter object")
        return converter

    def _more_space(self, obstacle: Obstacle, gb_wpnts: List[Any], obs_s_idx: int) -> Tuple[str, float]:
        left_gap = abs(gb_wpnts[obs_s_idx].d_left - obstacle.d_left)
        right_gap = abs(gb_wpnts[obs_s_idx].d_right + obstacle.d_right)
        min_space = self.evasion_dist + self.spline_bound_mindist

        if right_gap > min_space and left_gap < min_space:
            # Compute apex distance to the right of the opponent
            d_apex_right = obstacle.d_right - self.evasion_dist
            # If we overtake to the right of the opponent BUT the apex is to the left of the raceline, then we set the apex to 0
            if d_apex_right > 0:
                d_apex_right = 0
            return "right", d_apex_right

        elif left_gap > min_space and right_gap < min_space:
            # Compute apex distance to the left of the opponent
            d_apex_left = obstacle.d_left + self.evasion_dist
            # If we overtake to the left of the opponent BUT the apex is to the right of the raceline, then we set the apex to 0
            if d_apex_left < 0:
                d_apex_left = 0
            return "left", d_apex_left
        else:
            candidate_d_apex_left = obstacle.d_left + self.evasion_dist
            candidate_d_apex_right = obstacle.d_right - self.evasion_dist

            if abs(candidate_d_apex_left) <= abs(candidate_d_apex_right):
                # If we overtake to the left of the opponent BUT the apex is to the right of the raceline, then we set the apex to 0
                if candidate_d_apex_left < 0:
                    candidate_d_apex_left = 0
                return "left", candidate_d_apex_left
            else:
                # If we overtake to the right of the opponent BUT the apex is to the left of the raceline, then we set the apex to 0
                if candidate_d_apex_right > 0:
                    candidate_d_apex_right = 0
                return "right", candidate_d_apex_right

    def do_spline(self, obs: Obstacle, gb_wpnts: WpntArray) -> Tuple[WpntArray, MarkerArray]:
        """
        Creates a start trajectory by splining through the manually clicked poses.

        Returns:
        - wpnts (WpntArray): An array of waypoints that describe the start trajectory.
        - mrks (MarkerArray): An array of markers that represent the waypoints in a visualization format.
        """
        # Return wpnts and markers
        mrks = MarkerArray()
        wpnts = OTWpntArray()
        wpnts.header.stamp = self.get_clock().now().to_msg()
        wpnts.header.frame_id = "map"
        # Get spacing between wpnts for rough approximations
        wpnt_dist = gb_wpnts[1].s_m - gb_wpnts[0].s_m

        if len(self.points_without_pose) > 0:
            s_list = [self.cur_s + self.start_target_m]
            d_list = [0]

            s_array = np.array(s_list)
            d_array = np.array(d_list)

            s_array = s_array % self.gb_max_s

            s_idx = np.round((s_array / wpnt_dist)).astype(int) % self.gb_max_idx

            danger_flag = False
            resp = self.converter.get_cartesian(s_array, d_array)

            points = [[self.cur_x, self.cur_y]]
            tangents = [[np.cos(self.cur_yaw), np.sin(self.cur_yaw)]]

            points.extend(self.points_without_pose[:-1])  # exclude last item
            tangents.extend(self.tangents_without_pose[:-1])  # exclude last item

            # Use the gb_wpnts point closest to the last clicked pose for the final point
            last_point = self.points_without_pose[-1]
            last_tangent = self.tangents_without_pose[-1]
            s_, d_ = self.converter.get_frenet([last_point[0]], [last_point[1]])

            last_s_idx = int((s_[0] / wpnt_dist) % self.gb_max_idx)

            last_wpnt = gb_wpnts[last_s_idx]
            points.append([last_wpnt.x_m, last_wpnt.y_m])
            tangents.append([np.cos(last_wpnt.psi_rad), np.sin(last_wpnt.psi_rad)])

            tangents = np.dot(tangents, self.spline_scale * np.eye(2))
            points = np.asarray(points)
            nPoints, dim = points.shape

            # Parametrization parameter s.
            dp = np.diff(points, axis=0)                 # difference between points
            dp = np.linalg.norm(dp, axis=1)              # distance between points
            d = np.cumsum(dp)                            # cumsum along the segments
            d = np.hstack([[0], d])                      # add distance from first point
            l = d[-1]                                    # length of point sequence
            nSamples = int(l / wpnt_dist)                # number of samples
            s, r = np.linspace(0, l, nSamples, retstep=True)  # sample parameter and step

            # Bring points and (optional) tangent information into correct format.
            assert(len(points) == len(tangents))
            spline_result = np.empty([nPoints, dim], dtype=object)
            for i, ref in enumerate(points):
                t = tangents[i]
                # Either tangent is None or has the same
                # number of dimensions as the point ref.
                assert(t is None or len(t) == dim)
                fuse = list(zip(ref, t) if t is not None else zip(ref,))
                spline_result[i, :] = fuse

            # Compute splines per dimension separately.
            samples = np.zeros([nSamples, dim])
            for i in range(dim):
                poly = BPoly.from_derivatives(d, spline_result[:, i])
                samples[:, i] = poly(s)

            n_additional = 40
            xy_additional = np.array([
                (
                    gb_wpnts[(last_s_idx + i + 1) % self.gb_max_idx].x_m,
                    gb_wpnts[(last_s_idx + i + 1) % self.gb_max_idx].y_m
                )
                for i in range(n_additional)
            ])

            samples = np.vstack([samples, xy_additional])

            s_, d_ = self.converter.get_frenet(samples[:, 0], samples[:, 1])

            psi_, kappa_ = tph.calc_head_curv_num.\
                calc_head_curv_num(
                    path=samples,
                    el_lengths=0.1 * np.ones(len(samples) - 1),
                    is_closed=False
                )

            danger_flag = False
            for i in range(samples.shape[0]):
                gb_wpnt_i = int((s_[i] / wpnt_dist) % self.gb_max_idx)

                inside = self.map_filter.is_point_inside(samples[i, 0], samples[i, 1])
                if not inside:
                    self.get_logger().info(
                        f"[{self.name}]: Evasion trajectory too close to TRACKBOUNDS, aborting evasion",
                        throttle_duration_sec=2,
                    )
                    danger_flag = True
                    break
                outside = True
                # Get V from gb wpnts and go slower if we are going through the inside
                vi = gb_wpnts[gb_wpnt_i].vx_mps if outside else gb_wpnts[gb_wpnt_i].vx_mps * 0.9  # TODO make speed scaling ros param

                wpnts.wpnts.append(
                    self.xyv_to_wpnts(x=samples[i, 0], y=samples[i, 1], s=s_[i], d=d_[i], v=2, psi=psi_[i] + np.pi / 2, kappa=kappa_[i], wpnts=wpnts)
                )
                mrks.markers.append(self.xyv_to_markers(x=samples[i, 0], y=samples[i, 1], v=vi, mrks=mrks))

            # Fill the rest of OTWpnts
            if danger_flag:
                wpnts.wpnts = []
                mrks.markers = []
        return wpnts, mrks

    ######################
    # VIZ + MSG WRAPPING #
    ######################
    def xyv_to_markers(self, x: float, y: float, v: float, mrks: MarkerArray) -> Marker:
        mrk = Marker()
        mrk.header.frame_id = "map"
        mrk.header.stamp = self.get_clock().now().to_msg()
        mrk.type = mrk.CYLINDER
        mrk.scale.x = 0.1
        mrk.scale.y = 0.1
        mrk.scale.z = float(v / self.gb_vmax)
        mrk.color.a = 1.0
        mrk.color.b = 0.75
        mrk.color.r = 0.75
        if self.from_bag:
            mrk.color.g = 0.75

        mrk.id = len(mrks.markers)
        mrk.pose.position.x = float(x)
        mrk.pose.position.y = float(y)
        mrk.pose.position.z = float(v / self.gb_vmax / 2)
        mrk.pose.orientation.w = 1.0

        return mrk

    def xy_to_point(self, x: float, y: float, opponent=True) -> Marker:
        mrk = Marker()
        mrk.header.frame_id = "map"
        mrk.header.stamp = self.get_clock().now().to_msg()
        mrk.type = mrk.SPHERE
        mrk.scale.x = 0.5
        mrk.scale.y = 0.5
        mrk.scale.z = 0.5
        mrk.color.a = 0.8
        mrk.color.b = 0.65
        mrk.color.r = 1.0 if opponent else 0.0
        mrk.color.g = 0.65

        mrk.pose.position.x = float(x)
        mrk.pose.position.y = float(y)
        mrk.pose.position.z = 0.01
        mrk.pose.orientation.w = 1.0

        return mrk

    def xyv_to_wpnts(self, s: float, d: float, x: float, y: float, v: float, psi: float, kappa: float, wpnts: WpntArray) -> Wpnt:
        wpnt = Wpnt()
        wpnt.id = len(wpnts.wpnts)
        wpnt.x_m = float(x)
        wpnt.y_m = float(y)
        wpnt.s_m = float(s)
        wpnt.d_m = float(d)
        wpnt.vx_mps = float(v)
        wpnt.psi_rad = float(psi)
        wpnt.kappa_radpm = float(kappa)

        return wpnt


def main(args=None):
    rclpy.init(args=args)
    spliner = ObstacleSpliner()
    try:
        rclpy.spin(spliner)
    except KeyboardInterrupt:
        pass
    spliner.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
