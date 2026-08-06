#!/usr/bin/env python3
"""Measure the VESC IMU's gyro bias (zero-rate offset) while the car sits still.

Place the car STILL. The node averages angular_velocity over `duration` seconds
and reports the bias per axis, plus the heading error that bias alone would
accumulate over example mapping run lengths. Prints a paste-ready YAML block for
stack_master/config/CAR/vehicle_config.yaml — does not write any file.

    ros2 run stack_master imu_gyro_bias_check.py
    ros2 run stack_master imu_gyro_bias_check.py --ros-args -p duration:=15.0
"""
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

EXAMPLE_DURATIONS_S = (60.0, 120.0, 300.0)


class ImuGyroBiasCheck(Node):
    def __init__(self):
        super().__init__('imu_gyro_bias_check')
        self.declare_parameter('imu_topic', '/vesc/sensors/imu')
        self.declare_parameter('duration', 10.0)
        self.duration = float(self.get_parameter('duration').value)
        self.samples = []
        self.t_first = None
        topic = self.get_parameter('imu_topic').value
        self.sub = self.create_subscription(Imu, topic, self.cb, qos_profile_sensor_data)
        self.get_logger().info(
            f'keep the car COMPLETELY STILL — collecting {self.duration:.1f}s of {topic} ...')

    def cb(self, msg: Imu):
        now = self.get_clock().now()
        if self.t_first is None:
            self.t_first = now
        w = msg.angular_velocity
        self.samples.append((w.x, w.y, w.z))
        if (now - self.t_first).nanoseconds * 1e-9 >= self.duration:
            self.finish()

    def finish(self):
        self.destroy_subscription(self.sub)
        w = np.array(self.samples)
        mean = w.mean(axis=0)
        std = w.std(axis=0)
        n = len(self.samples)
        self.get_logger().info(
            f'{n} samples   mean=[{mean[0]:+.5f} {mean[1]:+.5f} {mean[2]:+.5f}] rad/s   '
            f'std=[{std[0]:.5f} {std[1]:.5f} {std[2]:.5f}] rad/s')

        ok = True
        if n < 50:
            self.get_logger().warn(f'only {n} samples — is the IMU publishing?')
            ok = False
        if float(std.max()) > 0.02:
            self.get_logger().warn(
                f'per-axis std {std.round(4).tolist()} rad/s is high for a still car '
                '— vibration, engine running, or car not actually still?')
            ok = False

        bias_z_deg = math.degrees(mean[2])
        self.get_logger().info(f'gyro-z bias: {mean[2]:+.5f} rad/s ({bias_z_deg:+.4f} deg/s)')
        if abs(mean[2]) < 1e-4:
            self.get_logger().info(
                'gyro-z bias negligible (<1e-4 rad/s) — unlikely to explain a visible map '
                'skew; look elsewhere (loop closure, mount yaw, scan matching).')
        else:
            self.get_logger().info('projected heading drift from this bias ALONE:')
            for d in EXAMPLE_DURATIONS_S:
                drift_deg = math.degrees(mean[2] * d)
                self.get_logger().info(f'    after {d:6.0f}s driving: {drift_deg:+7.2f} deg')

        print('\n# paste into stack_master/config/CAR/vehicle_config.yaml'
              + ('' if ok else '   # WARNING: see log above, result may be unreliable'))
        print('imu_bias_corrector:')
        print('  ros__parameters:')
        print(f'    gyro_bias: [{mean[0]:.7f}, {mean[1]:.7f}, {mean[2]:.7f}]  '
              f'# rad/s, measured stationary\n')
        raise SystemExit(0)


def main(args=None):
    rclpy.init(args=args)
    node = ImuGyroBiasCheck()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()