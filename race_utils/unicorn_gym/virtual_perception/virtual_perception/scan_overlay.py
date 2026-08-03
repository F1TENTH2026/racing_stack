#!/usr/bin/env python3
"""scan_overlay — VIL injection seam (f110_msgs/ObstacleArray based).

Republishes the base lidar scan UNCHANGED (passthrough) until virtual obstacles
(opponent + static) are published as f110_msgs/ObstacleArray, then overlays each
as a small axis-aligned square of side `size` centred at (x_m, y_m) onto the
scan (min(real_range, box_range)). The opponent is intentionally a small box at
its base_link (rear axle), mirroring a real detection box rather than the whole
car.

Obstacles are carried as f110_msgs so the SAME source feeds two consumers:
  (a) this overlay (sensor-level VIL), and
  (b) a future concat with detection/tracking output (object-level VIL).

Ego pose comes from odometry (stable; no per-scan TF lookup). If no fresh
obstacles or no ego pose, it falls back to byte-for-byte passthrough.
"""
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from f110_msgs.msg import ObstacleArray
import tf2_ros
from rclpy.time import Time
from rclpy.duration import Duration


def _yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _overlay_box(scan, cx, cy, theta, half, px, py, dx, dy):
    """min(scan, distance to a `2*half` square centred at (cx,cy), rotated by
    theta) per beam (ray vs 4 edges)."""
    c, s = math.cos(theta), math.sin(theta)
    loc = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    V = loc @ np.array([[c, s], [-s, c]]) + np.array([cx, cy])
    E = np.roll(V, -1, 0) - V
    for a, e in zip(V, E):
        det = e[0] * dy - e[1] * dx
        safe = np.abs(det) > 1e-12
        den = np.where(safe, det, 1.0)
        wx, wy = a[0] - px, a[1] - py
        t = (e[0] * wy - e[1] * wx) / den
        u = (dx * wy - dy * wx) / den
        ok = safe & (t > 0) & (u >= 0) & (u <= 1) & (t < scan)
        scan = np.where(ok, t, scan)
    return scan


class ScanOverlay(Node):
    def __init__(self):
        super().__init__('scan_overlay')
        self.declare_parameter('in_topic', '/scan_raw')
        self.declare_parameter('out_topic', '/scan')
        self.declare_parameter('obstacle_topics',
                               ['/sim/dynamic_obstacles', '/sim/static_obstacles'])
        self.declare_parameter('ego_odom_topic', '/car_state/odom')
        self.declare_parameter('scan_distance_to_base_link', 0.275)
        self.declare_parameter('watchdog_sec', 0.5)
        # mutually-exclusive injection seam selector: this (lidar overlay) seam is
        # active only when /vp/inject_mode == 'overlay'. The other value, 'merge',
        # routes injection through tracking_merger instead (never both at once).
        self.declare_parameter('inject_mode', 'overlay')

        in_topic = self.get_parameter('in_topic').value
        out_topic = self.get_parameter('out_topic').value
        self.scan_dist = float(self.get_parameter('scan_distance_to_base_link').value)
        self.watchdog = float(self.get_parameter('watchdog_sec').value)

        self.enabled = (str(self.get_parameter('inject_mode').value).strip().lower() == 'overlay')
        self.ego = None                  # (x, y, yaw) base_link in map
        self.obs = {}                    # topic -> (obstacles list, recv_time)

        for topic in self.get_parameter('obstacle_topics').value:
            self.create_subscription(
                ObstacleArray, topic,
                lambda msg, t=topic: self._obs_cb(t, msg), 10)
        self.create_subscription(
            Odometry, self.get_parameter('ego_odom_topic').value, self._ego_cb, 10)
        # panel/launch picks overlay XOR merge via /vp/inject_mode (String)
        self.create_subscription(String, '/vp/inject_mode', self._mode_cb, 10)

        # TF so the laser origin can be sampled AT THE SCAN'S TIMESTAMP (see
        # _laser_pose). The listener's /tf subscriptions use a ReentrantCallbackGroup,
        # so under the MultiThreadedExecutor in main() the /tf buffer keeps updating
        # in another thread while _scan_cb is blocked inside lookup_transform(...).
        # Under a single-threaded executor that lookup could never see a newer /tf,
        # so it timed out and fell back to the latest odom -> the motion-dependent
        # overlay lag. The multi-threaded executor is what makes the trailing work.
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.sub = self.create_subscription(
            LaserScan, in_topic, self._scan_cb, qos_profile_sensor_data)
        self.pub = self.create_publisher(LaserScan, out_topic, qos_profile_sensor_data)
        self.get_logger().info(
            f"[scan_overlay] {in_topic} -> {out_topic} (f110_msgs obstacles; "
            "passthrough until obstacles arrive)")

    def _obs_cb(self, topic, msg):
        self.obs[topic] = (msg.obstacles, self.get_clock().now())

    def _ego_cb(self, msg):
        p = msg.pose.pose
        self.ego = (p.position.x, p.position.y, _yaw(p.orientation))

    def _mode_cb(self, msg):
        self.enabled = (msg.data.strip().lower() == 'overlay')

    def _fresh_obstacles(self):
        now = self.get_clock().now()
        out = []
        for obstacles, t in self.obs.values():
            if (now - t).nanoseconds * 1e-9 < self.watchdog:
                out.extend(obstacles)
        return out

    def _laser_pose(self, stamp, frame_id):
        """Laser origin (x, y, yaw) in map AT THE SCAN'S TIMESTAMP. Sampling the TF
        at the scan stamp — the SAME transform the consumer/RViz use to project this
        scan back to map — keeps the overlaid boxes pinned to the objects' true
        positions. If instead we used the latest odom (a different, newer time), the
        boxes would slide along the direction of motion as the ego/opponent move
        (that is the 'overlay lags a few steps behind' artifact). Falls back to the
        latest odom (base_link + scan_dist) only if the TF isn't available yet."""
        try:
            tf = self.tf_buffer.lookup_transform(
                'map', frame_id, Time.from_msg(stamp), timeout=Duration(seconds=0.005))
            return (tf.transform.translation.x, tf.transform.translation.y,
                    _yaw(tf.transform.rotation))
        except Exception:
            if self.ego is None:
                return None
            ex, ey, eyaw = self.ego
            return (ex + self.scan_dist * math.cos(eyaw),
                    ey + self.scan_dist * math.sin(eyaw), eyaw)

    def _scan_cb(self, msg):
        obstacles = self._fresh_obstacles() if self.enabled else []
        pose = self._laser_pose(msg.header.stamp, msg.header.frame_id) if obstacles else None
        if pose is None:
            self.pub.publish(msg)                     # passthrough
            return

        px, py, yaw = pose                            # laser origin in map @ scan stamp
        n = len(msg.ranges)
        ang = yaw + msg.angle_min + np.arange(n) * msg.angle_increment
        dx, dy = np.cos(ang), np.sin(ang)

        orig = np.asarray(msg.ranges, dtype=np.float64)
        valid = np.isfinite(orig) & (orig > 0.0)
        eff = np.where(valid, orig, msg.range_max)    # no-return -> open for overlay
        work = eff.copy()
        for o in obstacles:
            half = max(float(o.size), 0.01) / 2.0
            work = _overlay_box(work, float(o.x_m), float(o.y_m), float(o.theta),
                                half, px, py, dx, dy)

        hit = work < eff
        out_ranges = np.where(hit, work, orig)        # preserve original elsewhere

        out = LaserScan()
        out.header = msg.header
        out.angle_min = msg.angle_min
        out.angle_max = msg.angle_max
        out.angle_increment = msg.angle_increment
        out.time_increment = msg.time_increment
        out.scan_time = msg.scan_time
        out.range_min = msg.range_min
        out.range_max = msg.range_max
        out.ranges = out_ranges.astype(np.float32).tolist()
        out.intensities = msg.intensities
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ScanOverlay()
    # MultiThreadedExecutor: lets the TF listener's /tf callbacks (ReentrantCallbackGroup)
    # keep filling the buffer in another thread while _scan_cb blocks in
    # lookup_transform() at the scan stamp -> the timestamp-synced laser pose
    # actually resolves instead of falling back to the latest odom.
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
