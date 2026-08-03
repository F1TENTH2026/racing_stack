#!/usr/bin/env python3

import time
from typing import List

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rcl_interfaces.msg import ParameterDescriptor, ParameterType, SetParametersResult
from rclpy.parameter import Parameter

from nav_msgs.msg import Odometry
from f110_msgs.msg import WpntArray
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import String, Header


class UpdateWaypoints(Node):
    def __init__(self):
        # Initialize the node
        super().__init__('waypoint_updater')

        # Init and params
        # Adaptive rate would be nice
        self.loop_rate = 1  # Hz

        # Parameters
        self.state = "GB_TRACK"
        self.hysteresis_time = 2.0  # s
        self.gb_track_start_time = None
        self.max_speed_scaled = 10.0  # m/s
        self.speed_offset = 0.0  # m/s

        # Callback data
        self.wpnts_scaled_msg = WpntArray()
        self.wpnts_updated_msg = WpntArray()
        self.s_points_array = np.array([])
        self.update_waypoints = True

        # Wait helpers
        self.wpnts_scaled_received = None
        self.global_wpnts_msg = None

        # Tunable params (dynamic_reconfigure -> rclpy parameters)
        self.declare_parameter(
            'update_waypoints', True,
            ParameterDescriptor(type=ParameterType.PARAMETER_BOOL,
                                description="Toggle updating the waypoints with measured speed"))
        self.declare_parameter(
            'speed_offset', 0.0,
            ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE,
                                description="Speed offset added to measured ego speed [m/s]"))
        self.update_waypoints = self.get_parameter(
            'update_waypoints').get_parameter_value().bool_value
        self.speed_offset = self.get_parameter(
            'speed_offset').get_parameter_value().double_value

        # Subscriber
        self.create_subscription(Odometry, "/car_state/odom_frenet", self.odom_cb, QoSProfile(depth=10))
        self.create_subscription(String, "/state_machine", self.state_machine_cb, QoSProfile(depth=10))
        self.create_subscription(WpntArray, "/global_waypoints_scaled", self.wpnts_scaled_cb, QoSProfile(depth=10))
        self.create_subscription(WpntArray, "/global_waypoints", self.global_wpnts_cb, QoSProfile(depth=10))

        # Waypoint publisher
        self.wpnts_updated_pub = self.create_publisher(WpntArray, "/global_waypoints_updated", QoSProfile(depth=10))
        self.marker_pub = self.create_publisher(MarkerArray, "/updated_waypoints_marker", QoSProfile(depth=10))

        self.add_on_set_parameters_callback(self.dyn_param_cb)

        # Wait for critical messages and set up internal state
        self.wait_for_messages()
        self.wpnts_updated_msg = self.wpnts_scaled_msg
        self.max_speed_scaled = max([self.global_wpnts_msg.wpnts[i].vx_mps for i in range(len(self.global_wpnts_msg.wpnts))])
        self.s_points_array = np.array([wpnt.s_m for wpnt in self.wpnts_updated_msg.wpnts])
        self.get_logger().info("[Update Wptns] Update Wpnts ready!")

        # Main loop timer
        self.create_timer(1.0 / self.loop_rate, self.loop)

    ### Callbacks ###
    def state_machine_cb(self, data: String):
        self.state = data.data
        if self.state == "GB_TRACK" and self.gb_track_start_time is None:
            self.gb_track_start_time = time.time()
        elif self.state != "GB_TRACK":
            self.gb_track_start_time = None

    def wpnts_scaled_cb(self, data: WpntArray):
        if self.wpnts_scaled_received is None:
            self.wpnts_scaled_msg = data
            self.wpnts_scaled_received = True

    def global_wpnts_cb(self, data: WpntArray):
        if self.global_wpnts_msg is None:
            self.global_wpnts_msg = data

    def odom_cb(self, data: Odometry):
        car_odom = data
        if self.update_waypoints == True:
            if self.s_points_array.any():
                ego_position = car_odom.pose.pose.position.x
                ego_speed = car_odom.twist.twist.linear.x
                ego_approx_indx = np.abs(self.s_points_array - ego_position).argmin()
                # Hysteresis added to prevent the waypoints from being updated to soon afer switching to GB_TRACK
                if self.state == "GB_TRACK" and self.gb_track_start_time is not None and time.time() - self.gb_track_start_time >= self.hysteresis_time:
                    self.wpnts_updated_msg.wpnts[ego_approx_indx].vx_mps = ego_speed + self.speed_offset
                    if ego_approx_indx == 0 or ego_approx_indx == (len(self.wpnts_updated_msg.wpnts) - 1):  # First and last waypoint are the same
                        self.wpnts_updated_msg.wpnts[0].vx_mps
                        self.wpnts_updated_msg.wpnts[-1].vx_mps
        else:
            pass

    def dyn_param_cb(self, params: List[Parameter]):
        for param in params:
            if param.name == 'update_waypoints':
                self.update_waypoints = param.value
            elif param.name == 'speed_offset':
                self.speed_offset = param.value

        self.get_logger().info(
            f"[Opp. Pred.] Toggled update waypoints"
            f"[Opp. Pred.] Speed offset: {self.speed_offset}"
        )
        return SetParametersResult(successful=True)

    ### Helper functions ###
    def wait_for_messages(self):
        self.get_logger().info("[Update Wpnts] Update Wpnts wating...")
        while self.wpnts_scaled_received is None or self.global_wpnts_msg is None:
            rclpy.spin_once(self)

    def visualize_waypoints(self):
        marker_array = MarkerArray()
        for i in range(len(self.wpnts_updated_msg.wpnts)):
            marker = Marker(header=Header(frame_id="map"), id=i, type=Marker.CYLINDER)
            marker.pose.position.x = self.wpnts_updated_msg.wpnts[i].x_m
            marker.pose.position.y = self.wpnts_updated_msg.wpnts[i].y_m

            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = self.wpnts_updated_msg.wpnts[i].vx_mps / self.max_speed_scaled
            marker.pose.position.z = marker.scale.z / 2

            marker.color.a = 1.0
            marker.color.g = 1.0

            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)

    ### Main loop ###
    def loop(self):
        self.wpnts_updated_msg.header.stamp = self.get_clock().now().to_msg()
        self.wpnts_updated_pub.publish(self.wpnts_updated_msg)
        self.visualize_waypoints()


def main(args=None):
    rclpy.init(args=args)
    node = UpdateWaypoints()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
