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

        self.bias = list(self.get_parameter('gyro_bias').value)
        self.add_on_set_parameters_callback(self._on_set_params)

        self.pub = self.create_publisher(Imu, self.get_parameter('output_topic').value, 10)
        self.sub = self.create_subscription(
            Imu, self.get_parameter('input_topic').value, self._cb, 50)

    def _on_set_params(self, params):
        from rcl_interfaces.msg import SetParametersResult
        for p in params:
            if p.name == 'gyro_bias':
                self.bias = list(p.value)
        return SetParametersResult(successful=True)

    def _cb(self, msg: Imu):
        msg.angular_velocity.x -= self.bias[0]
        msg.angular_velocity.y -= self.bias[1]
        msg.angular_velocity.z -= self.bias[2]
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