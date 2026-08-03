#!/usr/bin/env python3
"""
Opponent controller for the f1tenth_gym_ros sim.

Drives the spawned physical opponent (gym agent[1]) by publishing
AckermannDriveStamped on the opponent drive topic. Two autonomous modes,
selectable at runtime from the RViz Sim Control panel via /sim/opp_mode:

  - "ftg"  : Follow-The-Gap reactive driving from the opponent's own lidar
             (ported from unicorn-racing-stack-ros1/controller/ftg/ftg.py).
  - "path" : pure-pursuit along the global racing line (/global_waypoints).
  - "manual": controller stays silent; the opponent obeys the panel speed +/-
              (handled directly in gym_bridge). This is the default.

Panel speed +/- (/sim/opp_speed_delta) raises/lowers this controller's speed
cap too, so it works in every mode.
"""

import math
import os
import csv
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Float32, Bool
from f110_msgs.msg import WpntArray


# --------------------------------------------------------------------------- #
# Follow-The-Gap (ported from the ROS1 ftg.py, numpy-only, angle-from-metadata)
# --------------------------------------------------------------------------- #
class FollowTheGap:
    PREPROCESS_CONV_SIZE = 3
    STRAIGHTS_STEERING_ANGLE = np.pi / 18   # 10 deg
    MILD_CURVE_ANGLE = np.pi / 6            # 30 deg
    ULTRASTRAIGHTS_ANGLE = np.pi / 60       # 3 deg
    MAX_STEER = 0.4

    def __init__(self, safety_radius=30, max_lidar_dist=20.0,
                 range_offset=0, track_width=2.0):
        self.SAFETY_RADIUS = int(safety_radius)
        self.MAX_LIDAR_DIST = float(max_lidar_dist)
        self.range_offset = int(range_offset)
        self.track_width = float(track_width)
        self.velocity = 0.0

    def set_vel(self, v):
        self.velocity = float(v)

    def _preprocess(self, ranges):
        if self.range_offset > 0:
            ranges = ranges[self.range_offset:len(ranges) - self.range_offset]
        proc = np.asarray(ranges, dtype=np.float64)
        # no-return (NaN/inf) means nothing within range -> treat as open/far
        proc = np.nan_to_num(proc, nan=self.MAX_LIDAR_DIST, posinf=self.MAX_LIDAR_DIST)
        kernel = np.ones(self.PREPROCESS_CONV_SIZE) / self.PREPROCESS_CONV_SIZE
        proc = np.convolve(proc, kernel, 'same')
        return np.clip(proc, 0.0, self.MAX_LIDAR_DIST)

    def _safety_border(self, ranges):
        """Extend a safety bubble where the range jumps, so the car keeps clear
        of obstacle edges (same idea as the ROS1 reference)."""
        filtered = ranges.copy()
        n = len(ranges)
        i = 0
        while i < n - 1:
            if ranges[i + 1] - ranges[i] > 0.5:
                for j in range(self.SAFETY_RADIUS):
                    if i + j < n:
                        filtered[i + j] = min(filtered[i + j], ranges[i])
                i += max(self.SAFETY_RADIUS - 2, 1)
            i += 1
        i = n - 1
        while i > 0:
            if ranges[i - 1] - ranges[i] > 0.5:
                for j in range(self.SAFETY_RADIUS):
                    if i - j >= 0:
                        filtered[i - j] = min(filtered[i - j], ranges[i])
                i -= max(self.SAFETY_RADIUS - 2, 1)
            i -= 1
        return filtered

    def _radius(self, max_speed):
        return min(5.0, self.track_width / 2.0 + 2.0 * (self.velocity / max(max_speed, 1e-3)))

    @staticmethod
    def _find_largest_gap(ranges, radius):
        bin_ranges = np.where(ranges >= radius, 1, 0)
        if bin_ranges.sum() == 0:
            return None  # nowhere open enough -> caller goes straight & slow
        bin_diffs = np.abs(np.diff(bin_ranges))
        bin_diffs[0] = 1
        bin_diffs[-1] = 1
        diff_idxs = bin_diffs.nonzero()[0]
        high_gaps = []
        for i in range(len(diff_idxs) - 1):
            lo, hi = diff_idxs[i], diff_idxs[i + 1]
            high_gaps.append(np.mean(bin_ranges[lo:hi]) > 0.5)
        widths = np.array(high_gaps) * np.diff(diff_idxs)
        if widths.max() <= 0:
            return None
        gap_left = diff_idxs[int(np.argmax(widths))]
        gap_right = gap_left + int(widths.max())
        return gap_left, gap_right

    def process(self, ranges, angle_min, angle_increment, max_speed):
        """Return (speed, steering_angle) from one lidar scan."""
        proc = self._safety_border(self._preprocess(ranges))
        gap = self._find_largest_gap(proc, self._radius(max_speed))
        if gap is None:
            return 0.3 * max_speed, 0.0
        gap_left, gap_right = gap
        gap_mid = (gap_left + gap_right) // 2
        beam_idx = gap_mid + self.range_offset
        theta = angle_min + beam_idx * angle_increment
        steer = float(np.clip(theta, -self.MAX_STEER, self.MAX_STEER))

        a = abs(steer)
        if a > self.MILD_CURVE_ANGLE:
            speed = 0.30 * max_speed
        elif a > self.STRAIGHTS_STEERING_ANGLE:
            speed = 0.45 * max_speed
        elif a > self.ULTRASTRAIGHTS_ANGLE:
            speed = 0.80 * max_speed
        else:
            speed = max_speed
        return speed, steer


# --------------------------------------------------------------------------- #
class OpponentController(Node):
    def __init__(self):
        super().__init__('opponent_controller')

        self.declare_parameter('opp_scan_topic', '/opp_scan')
        self.declare_parameter('opp_odom_topic', '/opp_racecar/odom')
        self.declare_parameter('opp_drive_topic', '/opp_drive')
        self.declare_parameter('global_wpnts_topic', '/global_waypoints')
        self.declare_parameter('map_path', '')             # <map>.yaml; centerline.csv sits next to it
        self.declare_parameter('centerline_csv', 'centerline.csv')
        self.declare_parameter('mode', 'manual')           # manual | path | ftg
        self.declare_parameter('max_speed', 2.0)           # speed cap (m/s), tuned by +/-
        self.declare_parameter('wheelbase', 0.33)
        self.declare_parameter('lookahead', 1.2)
        self.declare_parameter('track_width', 2.0)
        self.declare_parameter('teleop_steer_gain', 1.0)   # Twist.angular.z -> steering
        self.declare_parameter('ftg_safety_radius', 30)
        self.declare_parameter('ftg_max_lidar_dist', 10.0)
        self.declare_parameter('ftg_range_offset', 0)

        self.mode = self.get_parameter('mode').value
        self.max_speed = float(self.get_parameter('max_speed').value)
        self.wheelbase = float(self.get_parameter('wheelbase').value)
        self.lookahead = float(self.get_parameter('lookahead').value)

        # path-following source: prefer live /global_waypoints (has speed),
        # else fall back to a file in the map dir: centerline.csv (constant
        # cruise) or global_waypoints.json (IQP racing line, with speeds).
        self.path_pts = None        # (N, 2) map-frame x,y
        self.path_speeds = None     # (N,) m/s, or None -> use max_speed
        self._load_centerline()

        self.ftg = FollowTheGap(
            safety_radius=self.get_parameter('ftg_safety_radius').value,
            max_lidar_dist=self.get_parameter('ftg_max_lidar_dist').value,
            range_offset=self.get_parameter('ftg_range_offset').value,
            track_width=self.get_parameter('track_width').value)

        self.scan = None
        self.odom = None
        self.teleop = None
        self.steer_gain = float(self.get_parameter('teleop_steer_gain').value)

        # match the lidar publisher's SENSOR_DATA (best-effort) QoS, else no scans arrive
        self.create_subscription(
            LaserScan, self.get_parameter('opp_scan_topic').value, self._scan_cb,
            qos_profile_sensor_data)
        self.create_subscription(
            Odometry, self.get_parameter('opp_odom_topic').value, self._odom_cb, 10)
        self.create_subscription(
            WpntArray, self.get_parameter('global_wpnts_topic').value, self._wp_cb, 10)
        self.create_subscription(String, '/sim/opp_mode', self._mode_cb, 10)
        self.create_subscription(Float32, '/sim/opp_speed_delta', self._speed_cb, 10)
        self.create_subscription(Twist, '/opp_cmd_vel', self._teleop_cb, 10)

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, self.get_parameter('opp_drive_topic').value, 10)
        # so selecting FTG can auto-enable the opponent lidar it depends on.
        # Volatile to match the sim_control_panel's publisher on this same topic
        # (a transient_local sub here would be QoS-incompatible with the panel).
        self.opp_lidar_pub = self.create_publisher(Bool, '/sim/opp_lidar_enable', 10)
        if self.mode == 'ftg':
            self.opp_lidar_pub.publish(Bool(data=True))

        self.timer = self.create_timer(0.025, self._loop)   # 40 Hz
        self.get_logger().info(
            f"[OpponentController] up (mode={self.mode}, max_speed={self.max_speed:.1f}). "
            "Pick Path/FTG from the RViz Sim Control panel.")

    # --- callbacks ---
    def _scan_cb(self, msg):
        self.scan = msg

    def _odom_cb(self, msg):
        self.odom = msg

    def _wp_cb(self, msg):
        if not msg.wpnts:
            return
        self.path_pts = np.array([[w.x_m, w.y_m] for w in msg.wpnts])
        self.path_speeds = np.array([w.vx_mps for w in msg.wpnts])

    def _load_centerline(self):
        map_path = self.get_parameter('map_path').value
        if not map_path:
            self.get_logger().warn('[OpponentController] no map_path -> path mode needs /global_waypoints')
            return
        map_dir = os.path.dirname(os.path.abspath(map_path))

        # 1) centerline.csv (x_m,y_m; no speed -> cruise at max_speed)
        csv_path = os.path.join(map_dir, self.get_parameter('centerline_csv').value)
        if os.path.exists(csv_path):
            xs, ys = [], []
            with open(csv_path, 'r') as f:
                for row in csv.DictReader(f):
                    xs.append(float(row['x_m']))
                    ys.append(float(row['y_m']))
            if xs:
                self.path_pts = np.column_stack([xs, ys])
                self.path_speeds = None
                self.get_logger().info(
                    f'[OpponentController] loaded centerline ({len(xs)} pts) from {csv_path}')
                return

        # 2) global_waypoints.json -> the SAME IQP racing line /global_waypoints
        #    serves (x_m,y_m,vx_mps). Reuse gb_optimizer's canonical reader so the
        #    JSON parsing lives in one place; lazy import keeps virtual_perception
        #    usable even where gb_optimizer isn't present.
        if os.path.exists(os.path.join(map_dir, 'global_waypoints.json')):
            map_name = os.path.splitext(os.path.basename(map_path))[0]   # f.yaml -> f
            try:
                from gb_optimizer.readwrite_global_waypoints import read_global_waypoints
                # tuple order per read_global_waypoints(): index 5 = global_traj_wpnts_iqp,
                # the WpntArray that /global_waypoints publishes.
                (_, _, _, _, _, gtw_iqp, _, _, _) = read_global_waypoints(map_name)
                wpnts = gtw_iqp.wpnts
                if wpnts:
                    self.path_pts = np.array([[w.x_m, w.y_m] for w in wpnts])
                    spd = np.array([w.vx_mps for w in wpnts])
                    self.path_speeds = spd if np.any(spd > 0.0) else None
                    self.get_logger().info(
                        f'[OpponentController] loaded racing line ({len(wpnts)} pts) from '
                        f'{map_name}/global_waypoints.json via gb_optimizer reader')
                    return
            except Exception as e:
                self.get_logger().warn(
                    f'[OpponentController] could not read global_waypoints.json ({e})')

        self.get_logger().warn(
            f'[OpponentController] no centerline.csv / global_waypoints.json in {map_dir} '
            '-> path mode needs /global_waypoints')

    def _teleop_cb(self, msg):
        self.teleop = msg

    def _mode_cb(self, msg):
        m = msg.data.strip().lower()
        if m in ('manual', 'path', 'ftg'):
            self.mode = m
            self.get_logger().info(f"[OpponentController] mode -> {m}")
            if m == 'ftg':
                # FTG drives off the opponent's lidar -> make sure it's on
                self.opp_lidar_pub.publish(Bool(data=True))

    def _speed_cb(self, msg):
        self.max_speed = float(np.clip(self.max_speed + msg.data, 0.0, 8.0))

    # --- helpers ---
    @staticmethod
    def _yaw(q):
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                          1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def _pure_pursuit(self):
        p = self.odom.pose.pose
        x, y, yaw = p.position.x, p.position.y, self._yaw(p.orientation)
        pts = self.path_pts
        n = len(pts)
        nearest = int(np.argmin(np.hypot(pts[:, 0] - x, pts[:, 1] - y)))
        idx = nearest
        for k in range(n):
            j = (nearest + k) % n
            if math.hypot(pts[j, 0] - x, pts[j, 1] - y) >= self.lookahead:
                idx = j
                break
        dx, dy = pts[idx, 0] - x, pts[idx, 1] - y
        lx = math.cos(-yaw) * dx - math.sin(-yaw) * dy
        ly = math.sin(-yaw) * dx + math.cos(-yaw) * dy
        l2 = lx * lx + ly * ly
        if l2 < 1e-6 or lx <= 0.0:
            return 0.5, 0.0          # target behind/at car -> creep straight
        steer = float(np.clip(math.atan(self.wheelbase * (2.0 * ly / l2)), -0.4, 0.4))
        cruise = float(self.path_speeds[idx]) if self.path_speeds is not None else self.max_speed
        speed = min(cruise, self.max_speed)
        return speed, steer

    # --- main loop ---
    def _loop(self):
        speed = steer = 0.0
        if self.mode == 'manual':
            # keyboard teleop: run teleop_twist_keyboard remapped to /opp_cmd_vel
            if self.teleop is None:
                return
            speed = float(np.clip(self.teleop.linear.x, -self.max_speed, self.max_speed))
            steer = float(np.clip(self.teleop.angular.z * self.steer_gain, -0.4, 0.4))
        elif self.mode == 'ftg':
            if self.scan is None:
                return
            self.ftg.set_vel(
                math.hypot(self.odom.twist.twist.linear.x,
                           self.odom.twist.twist.linear.y) if self.odom else 0.0)
            speed, steer = self.ftg.process(
                self.scan.ranges, self.scan.angle_min, self.scan.angle_increment,
                self.max_speed)
        elif self.mode == 'path':
            if self.odom is None or self.path_pts is None:
                return
            speed, steer = self._pure_pursuit()
        else:
            return

        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steer)
        self.drive_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = OpponentController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
