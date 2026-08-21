#!/usr/bin/env python3
"""Forward the freshest /scan to /scan_viz at a low, visualization-only rate."""

import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


class ScanThrottle(Node):
    def __init__(self):
        super().__init__('scan_throttle')
        self.declare_parameter('input_topic', '/scan')
        self.declare_parameter('output_topic', '/scan_viz')
        self.declare_parameter('rate_hz', 5.0)
        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self._pub = self.create_publisher(LaserScan, output_topic, qos_profile_sensor_data)
        self._sub = self.create_subscription(
            LaserScan, input_topic, self._on_scan, qos_profile_sensor_data)
        self._last_stamp = None
        self._set_rate(float(self.get_parameter('rate_hz').value))
        self.add_on_set_parameters_callback(self._on_parameters)

    def _set_rate(self, rate_hz):
        self._period = 1.0 / rate_hz if rate_hz > 0.0 else None
        self._last_stamp = None

    def _on_parameters(self, parameters):
        for parameter in parameters:
            if parameter.name == 'rate_hz':
                self._set_rate(float(parameter.value))
        return SetParametersResult(successful=True)

    def _on_scan(self, msg):
        if self._period is None or self._pub.get_subscription_count() == 0:
            return
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        # Also accept a clock reset (bag replay or /clock restart).
        if self._last_stamp is not None and 0.0 <= stamp - self._last_stamp < self._period:
            return
        self._last_stamp = stamp
        self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ScanThrottle()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
