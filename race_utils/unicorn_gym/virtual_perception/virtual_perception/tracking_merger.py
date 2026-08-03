#!/usr/bin/env python3
"""tracking_merger — object-level VIL seam.

Intercepts the real tracking/detection output and merges the virtual obstacles
into it, so downstream planners see real + virtual as ONE f110_msgs/ObstacleArray
(same man-in-the-middle pattern as scan_overlay, but at the object level):

  /tracking/obstacles_raw (real)   ─┐
  /sim/dynamic_obstacles (virtual) ─┼─► /tracking/obstacles (merged)
  /sim/static_obstacles  (virtual) ─┘

Remap your tracking node to publish /tracking/obstacles_raw; this node owns
/tracking/obstacles. In pure sim (no real tracking) it just emits the virtual
obstacles, so the object-level path is testable without perception.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String
from f110_msgs.msg import ObstacleArray, WpntArray

try:
    from frenet_conversion.frenet_converter import FrenetConverter
except Exception:
    FrenetConverter = None


class TrackingMerger(Node):
    def __init__(self):
        super().__init__('tracking_merger')
        self.declare_parameter('in_topic', '/tracking/obstacles_raw')
        self.declare_parameter('out_topic', '/tracking/obstacles')
        self.declare_parameter('virtual_topics',
                               ['/sim/dynamic_obstacles', '/sim/static_obstacles'])
        self.declare_parameter('rate_hz', 25.0)
        self.declare_parameter('watchdog_sec', 0.5)
        # Reject obstacles whose lateral Frenet offset exceeds this (they are
        # off the racing corridor — i.e. walls picked up by the simple detector
        # in the overlay seam). 0 disables the filter.
        self.declare_parameter('max_obstacle_d', 1.0)
        # mutually-exclusive injection seam selector: this (object/tracking) seam is
        # active only when /vp/inject_mode == 'merge'. The other value, 'overlay',
        # routes injection through scan_overlay instead (never both at once).
        self.declare_parameter('inject_mode', 'overlay')

        self.watchdog = float(self.get_parameter('watchdog_sec').value)
        self.max_obstacle_d = float(self.get_parameter('max_obstacle_d').value)
        self.merge_enabled = (str(self.get_parameter('inject_mode').value).strip().lower() == 'merge')
        self.real = None             # (obstacles, time)
        self.virt = {}               # topic -> (obstacles, time)

        for t in self.get_parameter('virtual_topics').value:
            self.create_subscription(ObstacleArray, t,
                                     lambda m, tt=t: self._virt_cb(tt, m), 10)
        self.create_subscription(ObstacleArray, self.get_parameter('in_topic').value,
                                 self._real_cb, 10)
        self.create_subscription(String, '/vp/inject_mode', self._mode_cb, 10)
        # Global raceline -> Frenet converter, so the merged obstacles carry
        # Frenet fields (s/d) that the planners & state machine consume. The
        # virtual sources only fill Cartesian (x_m,y_m,theta); real tracking
        # output already has Frenet and is left untouched.
        self.converter = None
        self.create_subscription(WpntArray, '/global_waypoints', self._gb_cb, 10)
        self.pub = self.create_publisher(ObstacleArray, self.get_parameter('out_topic').value, 10)
        self.create_timer(1.0 / float(self.get_parameter('rate_hz').value), self._tick)
        self.get_logger().info(
            f"[tracking_merger] {self.get_parameter('in_topic').value} + virtual "
            f"-> {self.get_parameter('out_topic').value}")

    def _real_cb(self, msg):
        self.real = (list(msg.obstacles), self.get_clock().now())

    def _virt_cb(self, topic, msg):
        self.virt[topic] = (list(msg.obstacles), self.get_clock().now())

    def _mode_cb(self, msg):
        self.merge_enabled = (msg.data.strip().lower() == 'merge')

    def _gb_cb(self, msg):
        if FrenetConverter is None or len(msg.wpnts) < 3:
            return
        x = np.array([w.x_m for w in msg.wpnts])
        y = np.array([w.y_m for w in msg.wpnts])
        psi = np.array([w.psi_rad for w in msg.wpnts])
        try:
            self.converter = FrenetConverter(x, y, psi)
        except Exception as e:
            self.get_logger().warn(f'[tracking_merger] FrenetConverter init failed: {e}')

    def _fill_frenet(self, obstacles):
        """Populate Frenet fields (s/d) from Cartesian (x_m,y_m), and drop
        obstacles that fall outside the racing corridor (walls). Returns the
        kept list."""
        if self.converter is None:
            return obstacles
        kept = []
        for o in obstacles:
            already = (o.s_center != 0.0 or o.s_start != 0.0 or o.d_center != 0.0)
            if not already:
                try:
                    fr = self.converter.get_frenet(np.array([o.x_m]), np.array([o.y_m]))
                    s = float(fr[0, 0]); d = float(fr[1, 0])
                except Exception:
                    continue
                half = max(o.size, 0.05) * 0.5
                o.s_center = s
                o.d_center = d
                o.s_start = s - half
                o.s_end = s + half
                o.d_left = d + half
                o.d_right = d - half
            if self.max_obstacle_d > 0.0 and abs(o.d_center) > self.max_obstacle_d:
                continue                       # off-corridor -> wall, drop it
            kept.append(o)
        return kept

    def _fresh(self, entry):
        if entry is None:
            return []
        obs, t = entry
        if (self.get_clock().now() - t).nanoseconds * 1e-9 < self.watchdog:
            return obs
        return []

    def _tick(self):
        out = ObstacleArray()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = 'map'
        out.obstacles = list(self._fresh(self.real))
        if self.merge_enabled:
            for entry in self.virt.values():
                out.obstacles.extend(self._fresh(entry))
        out.obstacles = self._fill_frenet(out.obstacles)
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = TrackingMerger()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
