#!/usr/bin/env python3
"""static_obstacle_manager — click-to-place static obstacles as f110_msgs.

RViz "Publish Point" (/clicked_point) appends a static obstacle; published as an
f110_msgs/ObstacleArray on /sim/static_obstacles -> consumed by scan_overlay
(lidar overlay) and tracking_merger (object concat). Clear via /sim/clear_obstacles.
"""
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Empty
from visualization_msgs.msg import Marker, MarkerArray
from f110_msgs.msg import Obstacle, ObstacleArray


class StaticObstacleManager(Node):
    def __init__(self):
        super().__init__('static_obstacle_manager')
        self.declare_parameter('size', 0.3)
        self.declare_parameter('topic', '/sim/static_obstacles')
        self.size = float(self.get_parameter('size').value)
        self.obstacles = []   # list of (x, y)
        self._prev_n = 0      # markers published last tick (for per-id DELETE)

        self.create_subscription(PointStamped, '/clicked_point', self._click_cb, 10)
        self.create_subscription(Empty, '/sim/clear_obstacles', self._clear_cb, 10)
        self.pub = self.create_publisher(ObstacleArray, self.get_parameter('topic').value, 10)
        # share the opponent's marker topic so a single RViz MarkerArray display
        # shows BOTH the opponent box and the static obstacles.
        self.viz = self.create_publisher(MarkerArray, '/vil/opponents_viz', 1)
        self.create_timer(0.1, self._publish)
        self.get_logger().info(
            '[static_obstacle_manager] RViz "Publish Point" to add; /sim/clear_obstacles to clear')

    def _click_cb(self, msg):
        self.obstacles.append((msg.point.x, msg.point.y))
        self.get_logger().info(
            f'[static_obstacle_manager] added @ ({msg.point.x:.2f}, {msg.point.y:.2f}) '
            f'(total {len(self.obstacles)})')

    def _clear_cb(self, _msg):
        n = len(self.obstacles)
        self.obstacles = []
        self.get_logger().info(f'[static_obstacle_manager] cleared {n} obstacles')

    def _publish(self):
        stamp = self.get_clock().now().to_msg()
        arr = ObstacleArray()
        arr.header.stamp = stamp
        arr.header.frame_id = 'map'
        viz = MarkerArray()
        for i, (x, y) in enumerate(self.obstacles):
            o = Obstacle()
            o.id = 100 + i
            o.x_m = x
            o.y_m = y
            o.theta = 0.0
            o.size = self.size
            o.is_static = True
            o.is_visible = True
            arr.obstacles.append(o)

            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = stamp
            m.ns = 'static_obs'
            m.id = i
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.lifetime = Duration(seconds=0.3).to_msg()  # auto-expire (~3 ticks @ 10 Hz) so a cleared/moved obstacle vanishes even if the per-id DELETE is dropped
            m.pose.position.x = x
            m.pose.position.y = y
            m.pose.position.z = 0.1
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = self.size
            m.scale.z = 0.2
            m.color.b = 1.0
            m.color.a = 0.8
            viz.markers.append(m)
        # delete markers for obstacles removed since last tick (per-id DELETE, NOT
        # DELETEALL — this marker topic is shared with the opponent box marker).
        for i in range(len(self.obstacles), self._prev_n):
            dm = Marker()
            dm.header.frame_id = 'map'
            dm.ns = 'static_obs'
            dm.id = i
            dm.action = Marker.DELETE
            viz.markers.append(dm)
        self._prev_n = len(self.obstacles)
        self.pub.publish(arr)
        self.viz.publish(viz)


def main(args=None):
    rclpy.init(args=args)
    node = StaticObstacleManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
