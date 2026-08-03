#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import PoseWithCovarianceStamped
from cartographer_ros_msgs.srv import FinishTrajectory, StartTrajectory
import time


class SetInitialPose(Node):
    def __init__(self):
        super().__init__('set_initial_pose_node')

        # Declare and get parameters
        self.declare_parameter('config_dir', '')
        self.declare_parameter('config_base', 'localization_2d.lua')

        self.config_dir = self.get_parameter('config_dir').value
        self.config_base = self.get_parameter('config_base').value

        self.get_logger().info(f'Config directory: {self.config_dir}')
        self.get_logger().info(f'Config basename: {self.config_base}')

        # Create service clients for Cartographer
        self.finish_trajectory_client = self.create_client(
            FinishTrajectory, '/finish_trajectory')
        self.start_trajectory_client = self.create_client(
            StartTrajectory, '/start_trajectory')

        # Wait for services to be available
        self.get_logger().info('Waiting for Cartographer services...')
        self.finish_trajectory_client.wait_for_service(timeout_sec=5.0)
        self.start_trajectory_client.wait_for_service(timeout_sec=5.0)
        self.get_logger().info('Cartographer services available!')

        # Subscribe to initial pose from RViz. RViz's "2D Pose Estimate" tool
        # publishes /initialpose as BEST_EFFORT; a RELIABLE subscriber (the default
        # you get by passing depth=10) is QoS-incompatible and receives NOTHING
        # ("incompatible QoS ... No messages will be received ... RELIABILITY"), so
        # clicking the tool never fires the callback. A BEST_EFFORT subscriber
        # accepts a best-effort OR a reliable publisher, so it works either way.
        initialpose_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10)
        self.subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            '/initialpose',
            self.initial_pose_callback,
            initialpose_qos)

        # 0 is the one we saved during mapping, so in localization we start trajectory 1
        self.trajectory_num = 1

        self.get_logger().info("Click the '2D Pose Estimate' button in RViz to set the robot's pose...")

    def initial_pose_callback(self, msg):
        self.get_logger().info(f'Initial position: ({msg.pose.pose.position.x}, {msg.pose.pose.position.y}, {msg.pose.pose.position.z})')
        self.get_logger().info(f'Initial orientation: ({msg.pose.pose.orientation.x}, {msg.pose.pose.orientation.y}, {msg.pose.pose.orientation.z}, {msg.pose.pose.orientation.w})')

        # Store the pose message for later use
        self.current_pose_msg = msg

        # Finish current trajectory
        self.get_logger().info(f'Finishing trajectory {self.trajectory_num}')
        finish_req = FinishTrajectory.Request()
        finish_req.trajectory_id = self.trajectory_num

        finish_future = self.finish_trajectory_client.call_async(finish_req)
        finish_future.add_done_callback(self.finish_trajectory_callback)

    def finish_trajectory_callback(self, future):
        """Callback when finish trajectory is done"""
        try:
            result = future.result()
            if result is not None:
                self.get_logger().info('Trajectory finished successfully')
                # Schedule start_trajectory to run after a small delay (one-shot timer)
                self._start_timer = self.create_timer(0.5, self._start_trajectory_once)
            else:
                self.get_logger().warn('Failed to finish trajectory')
        except Exception as e:
            self.get_logger().error(f'Error finishing trajectory: {str(e)}')

    def _start_trajectory_once(self):
        """One-shot timer callback to start new trajectory"""
        self._start_timer.cancel()
        self.destroy_timer(self._start_timer)
        self.start_new_trajectory()

    def start_new_trajectory(self):
        """Start new trajectory with initial pose"""
        self.get_logger().info('Starting new trajectory with initial pose')
        start_req = StartTrajectory.Request()
        start_req.configuration_directory = self.config_dir
        start_req.configuration_basename = self.config_base
        start_req.use_initial_pose = True
        start_req.initial_pose.position.x = self.current_pose_msg.pose.pose.position.x
        start_req.initial_pose.position.y = self.current_pose_msg.pose.pose.position.y
        start_req.initial_pose.position.z = self.current_pose_msg.pose.pose.position.z
        start_req.initial_pose.orientation.x = self.current_pose_msg.pose.pose.orientation.x
        start_req.initial_pose.orientation.y = self.current_pose_msg.pose.pose.orientation.y
        start_req.initial_pose.orientation.z = self.current_pose_msg.pose.pose.orientation.z
        start_req.initial_pose.orientation.w = self.current_pose_msg.pose.pose.orientation.w
        start_req.relative_to_trajectory_id = 0

        start_future = self.start_trajectory_client.call_async(start_req)
        start_future.add_done_callback(self.start_trajectory_callback)

    def start_trajectory_callback(self, future):
        """Callback when start trajectory is done"""
        try:
            result = future.result()
            if result is not None:
                self.get_logger().info(f'New trajectory started with ID: {result.trajectory_id}')
                self.trajectory_num += 1
            else:
                self.get_logger().error('Failed to start new trajectory')
        except Exception as e:
            self.get_logger().error(f'Error starting new trajectory: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    node = SetInitialPose()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Shutting down set_initial_pose_node...')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
