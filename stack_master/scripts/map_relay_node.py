#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from nav_msgs.msg import OccupancyGrid


class MapRelayNode(Node):
    """
    Subscribes to /map (TRANSIENT_LOCAL) locally and republishes it
    periodically so that remote subscribers can reliably receive it.

    CycloneDDS TRANSIENT_LOCAL cross-machine delivery can be unreliable
    when the publisher sends only once (nav2 map_server behaviour).
    This relay re-publishes every `period` seconds from the same machine,
    ensuring remote nodes that join late will eventually receive the map.
    """

    def __init__(self):
        super().__init__('map_relay_node')

        self.declare_parameter('period', 3.0)
        period = self.get_parameter('period').value

        transient_local_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._map: OccupancyGrid | None = None

        self._sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self._map_cb,
            transient_local_qos,
        )

        self._pub = self.create_publisher(
            OccupancyGrid,
            '/map_relay',
            transient_local_qos,
        )

        self._timer = self.create_timer(period, self._publish)
        self.get_logger().info(f'Map relay node started (period={period}s)')

    def _map_cb(self, msg: OccupancyGrid):
        self._map = msg
        self.get_logger().info('Map received, relaying...', once=True)
        self._publish()

    def _publish(self):
        if self._map is not None:
            self._pub.publish(self._map)


def main(args=None):
    rclpy.init(args=args)
    node = MapRelayNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
