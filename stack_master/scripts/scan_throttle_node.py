#!/usr/bin/env python3
"""scan_throttle_node.py — a low-rate /scan mirror for RViz, so watching the
LiDAR does not cost the car its control loop.

WHY THIS EXISTS
---------------
/scan is consumed on TWO paths and only one of them is expensive:

  A. localisation/perception — particle_filter, detect and the controller's FTG
     all subscribe on the car itself. This is loopback: a memory copy, ~free,
     and it runs at the full 40 Hz whether or not anyone is watching.
  B. visualisation — RViz on the pitwall laptop subscribes across WiFi. At 40 Hz
     a UST-10LX scan is 4.3 KB (8.5 KB with intensities on), i.e. ~175-350 KB/s
     of wireless traffic plus the serialisation that produces it.

Path B was enough on its own to starve the control loop: with pitwall connected
the car lagged, stopped mid-track and hit a wall, while a 15-minute run with
pitwall closed was clean.

Path B does not need 40 Hz. It answers "do the returns line up with the map
walls", which is a question about geometry, not about dynamics -- 5 Hz shows it
just as well and costs an eighth as much. So this node mirrors /scan onto
/scan_viz at rate_hz, and RViz watches the mirror.

Path A is untouched. Localisation still gets every one of its 40 scans.

TWO GATES, both of which make this free when nobody is looking:
  * the publish is skipped entirely while /scan_viz has no subscriber, so with
    pitwall closed this node costs one near-empty callback per scan (measured:
    ~9.6 us to deserialise a full LaserScan, i.e. ~0.04% of a core at 40 Hz);
  * when there IS a subscriber, output is capped at rate_hz.

Forwarding happens in the subscriber callback rather than on a timer on
purpose: a timer would republish whatever scan happened to be buffered, so the
picture could be up to a period stale. This way every forwarded scan is the
freshest one, just fewer of them.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


class ScanThrottleNode(Node):
    def __init__(self):
        super().__init__('scan_throttle_node')

        self.declare_parameter('input_topic', '/scan')
        self.declare_parameter('output_topic', '/scan_viz')
        # [Hz] 0 disables forwarding outright (the subscription stays, the
        # publish never happens) -- same effect as closing the display.
        self.declare_parameter('rate_hz', 5.0)

        self.rate_hz = float(self.get_parameter('rate_hz').value)
        in_topic = self.get_parameter('input_topic').value
        out_topic = self.get_parameter('output_topic').value

        self._min_period = (1.0 / self.rate_hz) if self.rate_hz > 0.0 else None
        self._last_sec = None

        # Sensor QoS on both sides: RViz's LaserScan display defaults to
        # BEST_EFFORT, and a RELIABLE writer would not match it.
        self._pub = self.create_publisher(LaserScan, out_topic, qos_profile_sensor_data)
        self._sub = self.create_subscription(
            LaserScan, in_topic, self._cb, qos_profile_sensor_data)

        # Live-tunable, so the rate can be dropped (or the mirror killed with 0)
        # from the pitwall mid-session without relaunching the stack:
        #   ros2 param set /scan_throttle rate_hz 2.0
        self.add_on_set_parameters_callback(self._on_set_params)

        self.get_logger().info(
            f'{in_topic} -> {out_topic} at {self.rate_hz:.1f} Hz '
            f'(only while {out_topic} has a subscriber)')

    def _on_set_params(self, params):
        from rcl_interfaces.msg import SetParametersResult
        for p in params:
            if p.name == 'rate_hz':
                self.rate_hz = float(p.value)
                self._min_period = (1.0 / self.rate_hz) if self.rate_hz > 0.0 else None
                self._last_sec = None
                self.get_logger().info(f'scan_viz rate -> {self.rate_hz:.1f} Hz')
        return SetParametersResult(successful=True)

    def _cb(self, msg: LaserScan):
        if self._min_period is None:
            return
        if self._pub.get_subscription_count() == 0:
            return
        # Scan header stamps, not wall clock: this throttles the DATA stream, and
        # the stamps are what RViz orders the display by.
        now = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self._last_sec is not None and (now - self._last_sec) < self._min_period:
            return
        self._last_sec = now
        self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ScanThrottleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
