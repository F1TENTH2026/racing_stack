#!/usr/bin/env python3
"""Subtract a fixed gyro bias from /vesc/sensors/imu and republish it.

The bias comes from imu_gyro_bias_check.py (run with the car stationary) and is
stored in vehicle_config.yaml under `imu_bias_corrector.gyro_bias`. Cartographer
(mapping_2d.lua / localization_2d.lua) should subscribe to the OUTPUT topic
instead of the raw one, so its heading estimate no longer integrates the raw
sensor's zero-rate offset.
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
        # (rad/s)^2 written onto angular_velocity_covariance's diagonal. The VESC
        # driver leaves every covariance in sensor_msgs/Imu at zero, and
        # robot_localization substitutes a near-zero variance (with a warning)
        # for any variable it is told to fuse that arrives with none -- which
        # makes it treat one gyro sample as exact. ekf_pf.yaml fuses vyaw from
        # this topic, so the honest number has to come from somewhere: default
        # 1e-4 == sigma 0.01 rad/s (~0.6 deg/s), which covers sample noise plus
        # the thermal drift left over after gyro_bias is subtracted. Measure
        # your own with imu_gyro_bias_check.py and override in vehicle_config.
        self.declare_parameter('gyro_variance', 1e-4)

        self.bias = list(self.get_parameter('gyro_bias').value)
        self.gyro_variance = float(self.get_parameter('gyro_variance').value)
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
                self.gyro_variance = float(p.value)
        return SetParametersResult(successful=True)

    def _cb(self, msg: Imu):
        msg.angular_velocity.x -= self.bias[0]
        msg.angular_velocity.y -= self.bias[1]
        msg.angular_velocity.z -= self.bias[2]
        v = self.gyro_variance
        msg.angular_velocity_covariance = [v, 0.0, 0.0,
                                           0.0, v, 0.0,
                                           0.0, 0.0, v]
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