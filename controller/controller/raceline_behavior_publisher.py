#!/usr/bin/env python3
"""Minimal obstacle-free race adapter: scaled raceline -> BehaviorStrategy."""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from f110_msgs.msg import BehaviorStrategy, WpntArray


class RacelineBehaviorPublisher(Node):
    def __init__(self):
        super().__init__('raceline_behavior_publisher')
        self.declare_parameter('publish_rate_hz', 40.0)
        self._waypoints = None
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=1,
            reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(
            WpntArray, '/global_waypoints_scaled', self._waypoints_cb, qos)
        self._publisher = self.create_publisher(BehaviorStrategy, '/behavior_strategy', qos)
        rate = float(self.get_parameter('publish_rate_hz').value)
        self.create_timer(1.0 / max(rate, 1.0), self._publish)

    def _waypoints_cb(self, msg):
        self._waypoints = msg

    def _publish(self):
        if self._waypoints is None or not self._waypoints.wpnts:
            return
        msg = BehaviorStrategy()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._waypoints.header.frame_id
        msg.local_wpnts = self._waypoints.wpnts
        msg.need_vel_planner = False
        msg.state = 'GB_TRACK'
        self._publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = RacelineBehaviorPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
