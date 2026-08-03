#!/usr/bin/env python3
"""Convert geometry_msgs/PoseStamped to nav_msgs/Odometry for robot_localization."""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry


class PoseToOdomNode(Node):
    def __init__(self):
        super().__init__('pose_to_odom_node')
        self.declare_parameter('child_frame_id', 'base_link')
        self.child_frame_id = self.get_parameter('child_frame_id').value

        self.sub = self.create_subscription(PoseStamped, 'pose_in', self.cb, 10)
        self.pub = self.create_publisher(Odometry, 'odom_out', 10)

    def cb(self, msg: PoseStamped):
        odom = Odometry()
        odom.header = msg.header
        odom.child_frame_id = self.child_frame_id
        odom.pose.pose = msg.pose
        # Diagonal pose covariance: [x, y, z, roll, pitch, yaw]
        # Zero covariance → singular matrix → EKF NaN
        odom.pose.covariance[0]  = 0.01   # x
        odom.pose.covariance[7]  = 0.01   # y
        odom.pose.covariance[14] = 1e6    # z  (unused in 2D)
        odom.pose.covariance[21] = 1e6    # roll  (unused in 2D)
        odom.pose.covariance[28] = 1e6    # pitch (unused in 2D)
        odom.pose.covariance[35] = 0.01   # yaw
        self.pub.publish(odom)


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(PoseToOdomNode())


if __name__ == '__main__':
    main()
