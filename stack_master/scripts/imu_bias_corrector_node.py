#!/usr/bin/env python3
"""Subtract a fixed gyro bias from /vesc/sensors/imu and republish it.

The bias comes from imu_gyro_bias_check.py (run with the car stationary) and is
stored in vehicle_config.yaml under `imu_bias_corrector.gyro_bias`. Cartographer
(mapping_2d.lua / localization_2d.lua) should subscribe to the OUTPUT topic
instead of the raw one, so its heading estimate no longer integrates the raw
sensor's zero-rate offset.

This node also stamps the message's COVARIANCES, which vesc_driver leaves at
zero. That is not a cosmetic gap: robot_localization replaces a zero variance
with 1e-9 (ekf.cpp, "no measurement can be completely without error, so add a
small amount") -- a standard deviation of 3e-5 rad/s. That is a numerical guard
against the Kalman gain blowing up, NOT a safe default, and its effect is a gain
of ~1: the filter snaps its state onto whatever the gyro just said, spike
included. Declaring the real noise here is what lets the EKF weigh the gyro
against its own prediction, and what makes ekf_*.yaml's
imu0_twist_rejection_threshold mean anything (the Mahalanobis gate is scaled by
this covariance).

This node is the right place for it rather than vesc_driver: the residual noise
after bias removal is what consumers actually see, every consumer in this stack
reads the corrected topic, and vesc/ is a vendored upstream checkout we would
rather not diverge further.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu


class ImuBiasCorrectorNode(Node):
    def __init__(self):
        super().__init__('imu_bias_corrector')
        self.declare_parameter('input_topic', '/vesc/sensors/imu')
        self.declare_parameter('output_topic', '/vesc/sensors/imu_corrected')
        self.declare_parameter('gyro_bias', [0.0, 0.0, 0.0])  # rad/s [x, y, z]
        # [(rad/s)^2] Variance of the gyro rate AFTER the bias above is removed.
        # Per axis; only z is fused in this 2D stack. Measure it with
        # imu_gyro_bias_check.py, which reports the variance alongside the bias.
        self.declare_parameter('gyro_variance', [4.0e-4, 4.0e-4, 4.0e-4])
        # [(m/s^2)^2] Accelerometer variance. Nothing fuses it today (the EKFs
        # disable accel, cartographer's ImuData carries no covariance at all),
        # so this is declared for honesty rather than effect.
        self.declare_parameter('accel_variance', [9.0e-2, 9.0e-2, 9.0e-2])

        self.bias = list(self.get_parameter('gyro_bias').value)
        self.gyro_var = list(self.get_parameter('gyro_variance').value)
        self.accel_var = list(self.get_parameter('accel_variance').value)
        self.add_on_set_parameters_callback(self._on_set_params)

        self.pub = self.create_publisher(Imu, self.get_parameter('output_topic').value, 10)
        self.sub = self.create_subscription(
            Imu, self.get_parameter('input_topic').value, self._cb, 50)

    def _on_set_params(self, params):
        from rcl_interfaces.msg import SetParametersResult
        for p in params:
            if p.name == 'gyro_bias':
                self.bias = list(p.value)
            elif p.name == 'gyro_variance':
                self.gyro_var = list(p.value)
            elif p.name == 'accel_variance':
                self.accel_var = list(p.value)
        return SetParametersResult(successful=True)

    def _cb(self, msg: Imu):
        msg.angular_velocity.x -= self.bias[0]
        msg.angular_velocity.y -= self.bias[1]
        msg.angular_velocity.z -= self.bias[2]

        # Diagonal only: the VESC gives no cross-axis information, and claiming
        # a correlation we have not measured would be worse than claiming none.
        for i in range(3):
            msg.angular_velocity_covariance[i * 3 + i] = self.gyro_var[i]
            msg.linear_acceleration_covariance[i * 3 + i] = self.accel_var[i]

        # -1 in the first slot is the sensor_msgs/Imu spec's way of saying "this
        # message carries no orientation". The field holds the VESC's own AHRS
        # quaternion, which free-runs and drifts, and nothing in this stack is
        # meant to consume it (ekf_*.yaml fuse yaw rate only; cartographer's
        # ImuData is {time, linear_acceleration, angular_velocity} and never
        # reads it). robot_localization honours the -1 explicitly
        # (ros_filter.cpp: "Ignoring orientation..."), so this turns a comment
        # in the EKF configs into something the filter enforces -- a future
        # config edit cannot accidentally start fusing a drifting heading.
        msg.orientation_covariance[0] = -1.0

        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ImuBiasCorrectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()