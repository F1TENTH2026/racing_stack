#!/usr/bin/env python3
import time
from typing import List, Any, Tuple

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
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker, MarkerArray
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

from f110_msgs.msg import Obstacle, ObstacleArray, OTWpntArray, Wpnt, WpntArray
from frenet_conversion.frenet_converter import FrenetConverter


class ObstacleSpliner(Node):
    """
    This class implements a ROS node that performs splining around obstacles.

    It subscribes to the following topics:
        - `/tracking/obstacles`: Subscribes to the obstacle array.
        - `/car_state/odom_frenet`: Subscribes to the car state in Frenet coordinates.
        - `/global_waypoints`: Subscribes to global waypoints.
        - `/global_waypoints_scaled`: Subscribes to the scaled global waypoints.

    The node publishes the following topics:
        - `/planner/avoidance/markers`: Publishes spline markers.
        - `/planner/avoidance/otwpnts`: Publishes splined waypoints.
        - `/planner/avoidance/considered_OBS`: Publishes markers for the closest obstacle.
        - `/planner/avoidance/propagated_obs`: Publishes markers for the propagated obstacle.
        - `/planner/avoidance/latency`: Publishes the latency of the spliner node. (only if measuring is enabled)
    """

    def __init__(self):
        """
        Initialize the node, subscribe to topics, and create publishers and service proxies.
        """
        # Initialize the node
        self.name = "obs_spliner_node"
        super().__init__('spliner_node')

        # initialize the instance variable
        self.obs = ObstacleArray()
        self.gb_wpnts = None
        self.gb_vmax = None
        self.gb_max_idx = None
        self.gb_max_s = None
        self.cur_s = None
        self.cur_d = None
        self.cur_vs = None
        self.gb_scaled_wpnts = None
        self.waypoints = None
        self.lookahead = 10  # in meters [m]
        self.last_switch_time = self.get_clock().now().to_msg()
        self.last_ot_side = ""

        # Static parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('from_bag', False),
                ('measure', False),
            ])
        self.from_bag = self.get_parameter('from_bag').get_parameter_value().bool_value
        self.measuring = self.get_parameter('measure').get_parameter_value().bool_value

        # Subscribe to the topics
        self.create_subscription(ObstacleArray, "/tracking/obstacles", self.obs_cb, 10)
        self.create_subscription(Odometry, "/car_state/odom_frenet", self.state_cb, 10)
        self.create_subscription(WpntArray, "/global_waypoints", self.gb_cb, 10)
        self.create_subscription(WpntArray, "/global_waypoints_scaled", self.gb_scaled_cb, 10)

        # dyn params defaults
        self.save_params = False
        self.kernel_size = 8
        self.post_sampling_dist = 5.0
        self.post_min_dist = 1.5
        self.post_max_dist = 5.0
        self.spline_scale = 0.8
        self.pre_apex_0 = -4.0
        self.pre_apex_1 = -3.0
        self.pre_apex_2 = -1.5
        self.post_apex_0 = 2.0
        self.post_apex_1 = 3.0
        self.post_apex_2 = 4.0
        self.evasion_dist = 0.65
        self.obs_traj_tresh = 0.3
        self.spline_bound_mindist = 0.2
        self.kd_obs_pred = 1.0
        self.fixed_pred_time = 0.15

        self.declare_all_parameters()
        # Apply loaded params to working members at startup (callback only fires on later set).
        self.dyn_param_cb(self.get_parameters([
            'save_params', 'kernel_size', 'post_sampling_dist', 'post_min_dist',
            'post_max_dist', 'spline_scale', 'evasion_dist', 'obs_traj_tresh',
            'spline_bound_mindist', 'pre_apex_dist0', 'pre_apex_dist1', 'pre_apex_dist2',
            'post_apex_dist0', 'post_apex_dist1', 'post_apex_dist2', 'kd_obs_pred',
            'fixed_pred_time',
        ]))
        self.add_on_set_parameters_callback(self.dyn_param_cb)

        self.mrks_pub = self.create_publisher(MarkerArray, "/planner/avoidance/markers", 10)
        self.evasion_pub = self.create_publisher(OTWpntArray, "/planner/avoidance/otwpnts", 10)
        self.closest_obs_pub = self.create_publisher(Marker, "/planner/avoidance/considered_OBS", 10)
        self.pub_propagated = self.create_publisher(Marker, "/planner/avoidance/propagated_obs", 10)
        if self.measuring:
            self.latency_pub = self.create_publisher(Float32, "/planner/avoidance/latency", 10)

        # Wait for critical messages
        self.wait_for_messages()

        self.converter = self.initialize_converter()

        # Set the rate at which the loop runs
        self.create_timer(1.0 / 20.0, self.loop)

    #####################
    # DYNAMIC PARAMETERS #
    #####################
    def declare_all_parameters(self):
        """
        Declare the dynamic-reconfigure tunables (from cfg/dyn_spliner_tuner.cfg) as ROS2
        parameters with proper descriptor ranges.
        """
        def dbl(min_v, max_v, desc=""):
            return ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description=desc,
                floating_point_range=[FloatingPointRange(from_value=float(min_v),
                                                         to_value=float(max_v),
                                                         step=0.001)],
            )

        def intd(min_v, max_v, desc=""):
            return ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description=desc,
                integer_range=[IntegerRange(from_value=int(min_v),
                                            to_value=int(max_v),
                                            step=1)],
            )

        # name, default, descriptor  (defaults/min/max from cfg/dyn_spliner_tuner.cfg)
        self.declare_parameter('save_params', False,
                               ParameterDescriptor(type=ParameterType.PARAMETER_BOOL,
                                                   description="Save params"))
        self.declare_parameter('kernel_size', 8, intd(1, 20))
        self.declare_parameter('post_sampling_dist', 5.0, dbl(0.5, 20.0))
        self.declare_parameter('post_min_dist', 1.5, dbl(0.5, 3.0))
        self.declare_parameter('post_max_dist', 5.0, dbl(3.0, 20.0))
        self.declare_parameter('spline_scale', 0.8, dbl(0.5, 2.0))
        self.declare_parameter('evasion_dist', 0.6, dbl(0.25, 1.25))
        self.declare_parameter('obs_traj_tresh', 1.0, dbl(0.1, 1.5))
        self.declare_parameter('spline_bound_mindist', 0.30, dbl(0.05, 1.0))
        self.declare_parameter('pre_apex_dist0', 4.0, dbl(0.5, 8.0))
        self.declare_parameter('pre_apex_dist1', 3.0, dbl(0.5, 8.0))
        self.declare_parameter('pre_apex_dist2', 2.0, dbl(0.5, 8.0))
        self.declare_parameter('post_apex_dist0', 4.5, dbl(0.5, 12.0))
        self.declare_parameter('post_apex_dist1', 5.0, dbl(0.5, 12.0))
        self.declare_parameter('post_apex_dist2', 5.5, dbl(0.5, 12.0))
        self.declare_parameter('kd_obs_pred', 1.0, dbl(0.1, 10.0))
        self.declare_parameter('fixed_pred_time', 0.15, dbl(0.0, 1.0))

    # Callback triggered by dynamic spline reconf
    def dyn_param_cb(self, params: List[Parameter]):
        """
        Notices the change in the parameters and changes spline params.

        Mirrors the rounding/ordering clamps that the ROS1 dynamic_spline_server applied.
        """
        # snapshot of current raw (pre-clamp) tunable values
        vals = {
            'save_params': self.save_params,
            'kernel_size': self.kernel_size,
            'post_sampling_dist': self.post_sampling_dist,
            'post_min_dist': self.post_min_dist,
            'post_max_dist': self.post_max_dist,
            'spline_scale': self.spline_scale,
            'evasion_dist': self.evasion_dist,
            'obs_traj_tresh': self.obs_traj_tresh,
            'spline_bound_mindist': self.spline_bound_mindist,
            'pre_apex_dist0': abs(self.pre_apex_0),
            'pre_apex_dist1': abs(self.pre_apex_1),
            'pre_apex_dist2': abs(self.pre_apex_2 - 0.1),
            'post_apex_dist0': self.post_apex_0,
            'post_apex_dist1': self.post_apex_1,
            'post_apex_dist2': self.post_apex_2,
            'kd_obs_pred': self.kd_obs_pred,
            'fixed_pred_time': self.fixed_pred_time,
        }
        for param in params:
            if param.name in vals:
                vals[param.name] = param.value

        # Ensuring nice rounding by either 0.05 or 0.5 (from dynamic_spline_server)
        evasion_dist = round(vals['evasion_dist'] * 20) / 20
        obs_traj_tresh = round(vals['obs_traj_tresh'] * 20) / 20
        spline_bound_mindist = round(vals['spline_bound_mindist'] * 20) / 20

        pre_apex_dist0 = round(vals['pre_apex_dist0'] * 2) / 2
        # Ensure pre_apex_dist1 always >= pre_apex_dist0, pre_apex_dist2 >= pre_apex_dist1
        pre_apex_dist1 = round(min(pre_apex_dist0 + 0.5, vals['pre_apex_dist1']) * 2) / 2
        pre_apex_dist2 = round(min(pre_apex_dist1 + 0.5, vals['pre_apex_dist2']) * 2) / 2
        post_apex_dist0 = round(vals['post_apex_dist0'] * 2) / 2
        post_apex_dist1 = round(max(post_apex_dist0 + 0.5, vals['post_apex_dist1']) * 2) / 2
        post_apex_dist2 = round(max(post_apex_dist1 + 0.5, vals['post_apex_dist2']) * 2) / 2
        kd_obs_pred = round(vals['kd_obs_pred'] * 20) / 20
        fixed_pred_time = round(vals['fixed_pred_time'] * 100) / 100

        # Store, preserving the UNICORN spliner_node sign/offset convention
        self.save_params = vals['save_params']
        self.kernel_size = vals['kernel_size']
        self.post_sampling_dist = vals['post_sampling_dist']
        self.post_min_dist = vals['post_min_dist']
        self.post_max_dist = vals['post_max_dist']
        self.spline_scale = vals['spline_scale']

        self.pre_apex_0 = -1 * pre_apex_dist0
        self.pre_apex_1 = -1 * pre_apex_dist1
        self.pre_apex_2 = -1 * pre_apex_dist2 + 0.1
        self.post_apex_0 = post_apex_dist0
        self.post_apex_1 = post_apex_dist1
        self.post_apex_2 = post_apex_dist2

        self.evasion_dist = evasion_dist
        self.obs_traj_tresh = obs_traj_tresh
        self.spline_bound_mindist = spline_bound_mindist
        self.kd_obs_pred = kd_obs_pred
        self.fixed_pred_time = fixed_pred_time

        spline_params = [
            self.pre_apex_0,
            self.pre_apex_1,
            self.pre_apex_2,
            0,
            self.post_apex_0,
            self.post_apex_1,
            self.post_apex_2,
        ]
        self.get_logger().info(
            f"[{self.name}] Dynamic reconf triggered new spline params: {spline_params} [m],\n"
            f" evasion apex distance: {self.evasion_dist} [m],\n"
            f" obstacle trajectory treshold: {self.obs_traj_tresh} [m]\n"
            f" obstacle prediciton k_d: {self.kd_obs_pred},    obstacle prediciton constant time: {self.fixed_pred_time} [s] "
        )
        return SetParametersResult(successful=True)

    #############
    # CALLBACKS #
    #############
    # Callback for obstacle topic
    def obs_cb(self, data: ObstacleArray):
        self.obs = data

    def state_cb(self, data: Odometry):
        self.cur_s = data.pose.pose.position.x
        self.cur_d = data.pose.pose.position.y
        self.cur_vs = data.twist.twist.linear.x

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

    #############
    # MAIN LOOP #
    #############
    def loop(self):
        if self.measuring:
            start = time.perf_counter()
        # Sample data
        obs = self.obs
        gb_scaled_wpnts = self.gb_scaled_wpnts.wpnts
        wpnts = OTWpntArray()
        mrks = MarkerArray()

        # If obs then do splining around it
        if len(obs.obstacles) > 0:
            wpnts, mrks = self.do_spline(obstacles=obs, gb_wpnts=gb_scaled_wpnts)
        # Else delete spline markers
        else:
            del_mrk = Marker()
            del_mrk.header.stamp = self.get_clock().now().to_msg()
            del_mrk.action = Marker.DELETEALL
            mrks.markers.append(del_mrk)

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
        waitlist = [self.cur_s, self.gb_wpnts, self.gb_scaled_wpnts]
        while None in waitlist:
            rclpy.spin_once(self)
            waitlist = [self.cur_s, self.gb_wpnts, self.gb_scaled_wpnts]
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

    def _more_space(self, obstacle: Obstacle, gb_wpnts: List[Any], gb_idxs: List[int]) -> Tuple[str, float]:
        left_gap = abs(gb_wpnts[gb_idxs[0]].d_left - obstacle.d_left)
        right_gap = abs(gb_wpnts[gb_idxs[0]].d_right + obstacle.d_right)
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

    def do_spline(self, obstacles: ObstacleArray, gb_wpnts: WpntArray) -> Tuple[WpntArray, MarkerArray]:
        """
        Creates an evasion trajectory for the closest obstacle by splining between pre- and post-apex points.

        This function takes as input the obstacles to be evaded, and a list of global waypoints that describe a reference raceline.
        It only considers obstacles that are within a threshold of the raceline and generates an evasion trajectory for each of these obstacles.
        The evasion trajectory consists of a spline between pre- and post-apex points that are spaced apart from the obstacle.
        The spatial and velocity components of the spline are calculated using the `Spline` class, and the resulting waypoints and markers are returned.

        Args:
        - obstacles (ObstacleArray): An array of obstacle objects to be evaded.
        - gb_wpnts (WpntArray): A list of global waypoints that describe a reference raceline.
        - state (Odometry): The current state of the car.

        Returns:
        - wpnts (WpntArray): An array of waypoints that describe the evasion trajectory to the closest obstacle.
        - mrks (MarkerArray): An array of markers that represent the waypoints in a visualization format.

        """
        # Return wpnts and markers
        mrks = MarkerArray()
        wpnts = OTWpntArray()

        # Get spacing between wpnts for rough approximations
        wpnt_dist = gb_wpnts[1].s_m - gb_wpnts[0].s_m

        # Only use obstacles that are within a threshold of the raceline, else we don't care about them
        close_obs = self._obs_filtering(obstacles=obstacles)

        # If there are obstacles within the lookahead distance, then we need to generate an evasion trajectory considering the closest one
        if len(close_obs) > 0:
            # Get the closest obstacle handling wraparound
            closest_obs = min(
                close_obs, key=lambda obs: (obs.s_center - self.cur_s) % self.gb_max_s
            )

            # Get Apex for evasion that is further away from the trackbounds
            if closest_obs.s_end < closest_obs.s_start:
                s_apex = (closest_obs.s_end + self.gb_max_s + closest_obs.s_start) / 2
            else:
                s_apex = (closest_obs.s_end + closest_obs.s_start) / 2
            # Approximate next 20 indexes of global wpnts with wrapping => 2m and compute which side is the outside of the raceline
            gb_idxs = [int(s_apex / wpnt_dist + i) % self.gb_max_idx for i in range(20)]
            kappas = np.array([gb_wpnts[gb_idx].kappa_radpm for gb_idx in gb_idxs])
            outside = "left" if np.sum(kappas) < 0 else "right"
            # Choose the correct side and compute the distance to the apex based on left of right of the obstacle
            more_space, d_apex = self._more_space(closest_obs, gb_wpnts, gb_idxs)

            # Publish the point around which we are splining
            mrk = self.xy_to_point(x=gb_wpnts[gb_idxs[0]].x_m, y=gb_wpnts[gb_idxs[0]].y_m, opponent=False)
            self.closest_obs_pub.publish(mrk)

            # Choose wpnts from global trajectory for splining with velocity
            evasion_points = []
            spline_params = [
                self.pre_apex_0,
                self.pre_apex_1,
                self.pre_apex_2,
                0,
                self.post_apex_0,
                self.post_apex_1,
                self.post_apex_2,
            ]
            for i, dst in enumerate(spline_params):
                # scale dst linearly between 1 and 1.5 depending on the speed normalised to the max speed
                dst = dst * np.clip(1.0 + self.cur_vs / self.gb_vmax, 1, 1.5)
                # If we overtake on the outside, we smoothen the spline
                if outside == more_space:
                    si = s_apex + dst * 1.75  # TODO make parameter
                else:
                    si = s_apex + dst
                di = d_apex if dst == 0 else 0
                evasion_points.append([si, di])
            # Convert to nump
            evasion_points = np.array(evasion_points)

            # Spline spatialy for d with s as base
            spline_resolution = 0.1  # TODO read from ros params to make consistent in case it changes
            spatial_spline = Spline(x=evasion_points[:, 0], y=evasion_points[:, 1])
            evasion_s = np.arange(evasion_points[0, 0], evasion_points[-1, 0], spline_resolution)
            # Clipe the d to the apex distance
            if d_apex < 0:
                evasion_d = np.clip(spatial_spline(evasion_s), d_apex, 0)
            else:
                evasion_d = np.clip(spatial_spline(evasion_s), 0, d_apex)

            # Handle Wrapping of s
            evasion_s = evasion_s % self.gb_max_s

            # Do frenet conversion via conversion service for spline and create markers and wpnts
            danger_flag = False
            resp = self.converter.get_cartesian(evasion_s, evasion_d)

            # Check if a side switch is possible
            if not self._check_ot_side_possible(more_space):
                danger_flag = True

            for i in range(evasion_s.shape[0]):
                gb_wpnt_i = int((evasion_s[i] / wpnt_dist) % self.gb_max_idx)
                # Check if wpnt is too close to the trackbounds but only if spline is actually off the raceline
                if abs(evasion_d[i]) > spline_resolution:
                    tb_dist = gb_wpnts[gb_wpnt_i].d_left if more_space == "left" else gb_wpnts[gb_wpnt_i].d_right
                    if abs(evasion_d[i]) > abs(tb_dist) - self.spline_bound_mindist:
                        self.get_logger().info(
                            f"[{self.name}]: Evasion trajectory too close to TRACKBOUNDS, aborting evasion",
                            throttle_duration_sec=2,
                        )
                        danger_flag = True
                        break
                # Get V from gb wpnts and go slower if we are going through the inside
                vi = gb_wpnts[gb_wpnt_i].vx_mps if outside == more_space else gb_wpnts[gb_wpnt_i].vx_mps * 0.9  # TODO make speed scaling ros param
                wpnts.wpnts.append(
                    self.xyv_to_wpnts(x=resp[0, i], y=resp[1, i], s=evasion_s[i], d=evasion_d[i], v=vi, wpnts=wpnts)
                )
                mrks.markers.append(self.xyv_to_markers(x=resp[0, i], y=resp[1, i], v=vi, mrks=mrks))

            # Fill the rest of OTWpnts
            wpnts.header.stamp = self.get_clock().now().to_msg()
            wpnts.header.frame_id = "map"
            if not danger_flag:
                wpnts.ot_side = more_space
                wpnts.ot_line = outside
                wpnts.side_switch = True if self.last_ot_side != more_space else False
                wpnts.last_switch_time = self.last_switch_time

                # Update the last switch time and the last side
                if self.last_ot_side != more_space:
                    self.last_switch_time = self.get_clock().now().to_msg()
                self.last_ot_side = more_space
            else:
                wpnts.wpnts = []
                mrks.markers = []
                # This fools the statemachine to cool down
                wpnts.side_switch = True
                self.last_switch_time = self.get_clock().now().to_msg()
        return wpnts, mrks

    def _obs_filtering(self, obstacles: ObstacleArray) -> List[Obstacle]:
        # Only use obstacles that are within a threshold of the raceline, else we don't care about them
        obs_on_traj = [obs for obs in obstacles.obstacles if abs(obs.d_center) < self.obs_traj_tresh]

        # Only use obstacles that within self.lookahead in front of the car
        close_obs = []
        for obs in obs_on_traj:
            obs = self._predict_obs_movement(obs)
            # Handle wraparound
            dist_in_front = (obs.s_center - self.cur_s) % self.gb_max_s
            # dist_in_back = abs(dist_in_front % (-self.gb_max_s)) # distance from ego to obstacle in the back
            if dist_in_front < self.lookahead:
                close_obs.append(obs)
                # Not within lookahead
            else:
                pass
        return close_obs

    def _predict_obs_movement(self, obs: Obstacle, mode: str = "constant") -> Obstacle:
        """
        Predicts the movement of an obstacle based on the current state and mode.

        TODO: opponent prediction should be completely isolated for added modularity

        Args:
            obs (Obstacle): The obstacle to predict the movement for.
            mode (str, optional): The mode for predicting the movement. Defaults to "constant".

        Returns:
            Obstacle: The updated obstacle with the predicted movement.
        """
        # propagate opponent by time dependent on distance
        if (obs.s_center - self.cur_s) % self.gb_max_s < 10:  # TODO make param
            if mode == "adaptive":
                # distance in s coordinate
                cur_s = self.cur_s
                ot_distance = (obs.s_center - cur_s) % self.gb_max_s
                rel_speed = np.clip(self.gb_scaled_wpnts.wpnts[int(cur_s * 10)].vx_mps - obs.vs, 0.1, 10)
                ot_time_distance = np.clip(ot_distance / rel_speed, 0, 5) * 0.5

                delta_s = ot_time_distance * obs.vs
                delta_d = ot_time_distance * obs.vd
                delta_d = -(obs.d_center + delta_d) * np.exp(-np.abs(self.kd_obs_pred * obs.d_center))

            elif mode == "adaptive_velheuristic":
                opponent_scaler = 0.7
                cur_s = self.cur_s
                ego_speed = self.gb_scaled_wpnts.wpnts[int(cur_s * 10)].vx_mps

                # distance in s coordinate
                ot_distance = (obs.s_center - cur_s) % self.gb_max_s
                rel_speed = (1 - opponent_scaler) * ego_speed
                ot_time_distance = np.clip(ot_distance / rel_speed, 0, 5)

                delta_s = ot_time_distance * opponent_scaler * ego_speed
                delta_d = -(obs.d_center) * np.exp(-np.abs(self.kd_obs_pred * obs.d_center))

            # propagate opponent by constant time
            elif mode == "constant":
                delta_s = self.fixed_pred_time * obs.vs
                delta_d = self.fixed_pred_time * obs.vd
                # delta_d = -(obs.d_center+delta_d)*np.exp(-np.abs(self.kd_obs_pred*obs.d_center))

            elif mode == "heuristic":
                # distance in s coordinate
                ot_distance = (obs.s_center - self.cur_s) % self.gb_max_s
                rel_speed = 3
                ot_time_distance = ot_distance / rel_speed

                delta_d = ot_time_distance * obs.vd
                delta_d = -(obs.d_center + delta_d) * np.exp(-np.abs(self.kd_obs_pred * obs.d_center))

            # update
            obs.s_start += delta_s
            obs.s_center += delta_s
            obs.s_end += delta_s
            obs.s_start %= self.gb_max_s
            obs.s_center %= self.gb_max_s
            obs.s_end %= self.gb_max_s

            obs.d_left += delta_d
            obs.d_center += delta_d
            obs.d_right += delta_d

            # NOTE: pass scalars (not 1-element lists) so get_cartesian returns
            # 0-d arrays; float() on a size-1 1-d array raises under numpy 2.x.
            resp = self.converter.get_cartesian(obs.s_center, obs.d_center)

            marker = self.xy_to_point(resp[0], resp[1], opponent=True)
            self.pub_propagated.publish(marker)

        return obs

    def _check_ot_side_possible(self, more_space) -> bool:
        if abs(self.cur_d) > 0.25 and more_space != self.last_ot_side:  # TODO make rosparam for cur_d threshold
            self.get_logger().info(f"[{self.name}]: Can't switch sides, because we are not on the raceline")
            return False
        return True

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

    def xyv_to_wpnts(self, s: float, d: float, x: float, y: float, v: float, wpnts: OTWpntArray) -> Wpnt:
        wpnt = Wpnt()
        wpnt.id = len(wpnts.wpnts)
        wpnt.x_m = float(x)
        wpnt.y_m = float(y)
        wpnt.s_m = float(s)
        wpnt.d_m = float(d)
        wpnt.vx_mps = float(v)
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
