#!/usr/bin/env python3
"""ROS-level smoke test for kiss_icp_localization.

Builds a synthetic room map (ASCII PCD), starts localization_node on it,
then simulates a robot driving an arc while publishing PointCloud2 scans
(sampled from the map, sensor frame) and IMU at 100 Hz. Passes if the
node's odometry tracks the ground-truth trajectory.

Run (with workspace sourced):
    python3 smoke_test.py
"""
import math
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

DOMAIN_ID = "77"
IMU_RATE = 100.0
SCAN_EVERY = 10  # imu ticks per scan -> 10 Hz
STATIC_SECS = 1.5
MOTION_SECS = 4.0
VX = 0.4          # m/s forward
WZ = 0.25         # rad/s yaw


def make_room():
    pts = []
    step = 0.1
    xmin, xmax, ymin, ymax, zmin, zmax = -5, 5, -4, 4, 0.0, 3.0
    xs = np.arange(xmin, xmax + 1e-6, step)
    ys = np.arange(ymin, ymax + 1e-6, step)
    zs = np.arange(zmin, zmax + 1e-6, step)
    for x in xs:
        for y in ys:
            pts.append((x, y, zmin))
            pts.append((x, y, zmax))
    for x in xs:
        for z in zs:
            pts.append((x, ymin, z))
            pts.append((x, ymax, z))
    for y in ys:
        for z in zs:
            pts.append((xmin, y, z))
            pts.append((xmax, y, z))
    for a in np.arange(0, 2 * math.pi, 0.05):
        for z in zs:
            pts.append((2.0 + 0.3 * math.cos(a), 1.0 + 0.3 * math.sin(a), z))
    return np.array(pts, dtype=np.float64)


def write_pcd(path, pts):
    with open(path, "w") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n")
        f.write(f"WIDTH {len(pts)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {len(pts)}\nDATA ascii\n")
        for p in pts:
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")


def sample_surfaces(rng, n):
    """Random continuous points on the room surfaces (independent of the map
    grid — reusing map points verbatim makes ICP lock onto the lattice)."""
    pts = np.empty((n, 3))
    s = rng.uniform(0, 1, n)
    x = rng.uniform(-5, 5, n)
    y = rng.uniform(-4, 4, n)
    z = rng.uniform(0, 3, n)
    half = rng.uniform(0, 1, n) < 0.5
    # floor/ceiling
    m = s < 0.30
    pts[m] = np.column_stack([x[m], y[m], np.where(half[m], 0.0, 3.0)])
    # y walls
    m = (s >= 0.30) & (s < 0.55)
    pts[m] = np.column_stack([x[m], np.where(half[m], -4.0, 4.0), z[m]])
    # x walls
    m = (s >= 0.55) & (s < 0.80)
    pts[m] = np.column_stack([np.where(half[m], -5.0, 5.0), y[m], z[m]])
    # pillar
    m = s >= 0.80
    a = rng.uniform(0, 2 * math.pi, int(m.sum()))
    pts[m] = np.column_stack(
        [2.0 + 0.3 * np.cos(a), 1.0 + 0.3 * np.sin(a), z[m]])
    return pts


def yaw_rot(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class Sim(Node):
    def __init__(self, room):
        super().__init__("kiss_loc_smoke_sim")
        self.room = room
        self.lidar_pub = self.create_publisher(PointCloud2, "/livox/lidar", 10)
        self.imu_pub = self.create_publisher(Imu, "/livox/imu", 100)
        self.odom_msgs = []
        self.create_subscription(Odometry, "/kiss_loc/odometry", self.on_odom, 100)
        self.rng = np.random.default_rng(7)

    def on_odom(self, msg):
        self.odom_msgs.append(msg)

    def publish_imu(self, t, wz):
        msg = Imu()
        msg.header = Header()
        msg.header.stamp = self.to_stamp(t)
        msg.header.frame_id = "livox"
        msg.angular_velocity.z = wz + float(self.rng.normal(0, 1e-3))
        msg.angular_velocity.x = float(self.rng.normal(0, 1e-3))
        msg.angular_velocity.y = float(self.rng.normal(0, 1e-3))
        msg.linear_acceleration.z = 9.81
        self.imu_pub.publish(msg)

    def publish_scan(self, t, pos, yaw):
        pts_map = sample_surfaces(self.rng, 4000)
        R = yaw_rot(yaw)
        pts_sensor = (pts_map - pos) @ R  # R^T (p - t), row-vector form
        pts_sensor += self.rng.normal(0, 0.005, pts_sensor.shape)
        header = Header()
        header.stamp = self.to_stamp(t)
        header.frame_id = "livox"
        msg = point_cloud2.create_cloud_xyz32(header, pts_sensor.astype(np.float32))
        self.lidar_pub.publish(msg)

    @staticmethod
    def to_stamp(t):
        from builtin_interfaces.msg import Time

        s = Time()
        s.sec = int(t)
        s.nanosec = int((t - int(t)) * 1e9)
        return s


def main():
    os.environ["ROS_DOMAIN_ID"] = DOMAIN_ID

    room = make_room()
    tmpdir = tempfile.mkdtemp(prefix="kiss_loc_smoke_")
    pcd_path = os.path.join(tmpdir, "room.pcd")
    write_pcd(pcd_path, room)
    print(f"[smoke] map: {len(room)} pts -> {pcd_path}")

    # exec the binary directly — `ros2 run` is a wrapper, so terminating it
    # leaves the actual node process running
    node_bin = os.path.expanduser(
        "~/ros2_ws/install/kiss_icp_localization/lib/kiss_icp_localization/"
        "localization_node")
    node_log = open(os.path.join(tmpdir, "node.log"), "w")
    node_proc = subprocess.Popen(
        [
            node_bin,
            "--ros-args",
            "-p", f"map_pcd_path:={pcd_path}",
            "-p", "use_custom_msg:=false",
            "-p", "imu_init_samples:=50",
            "-p", "min_range:=0.0",
            "-p", "publish_tf:=false",
            "-p", "publish_aligned_scan:=false",
        ],
        env=dict(os.environ),
        stdout=node_log,
        stderr=subprocess.STDOUT,
    )

    sim = None
    ok = False
    try:
        rclpy.init()
        sim = Sim(room)
        time.sleep(2.0)  # let the node load the map and subscribe
        if node_proc.poll() is not None:
            print("[smoke] FAIL: node exited early")
            return 1

        dt = 1.0 / IMU_RATE
        t = time.time()
        pos = np.zeros(3)
        yaw = 0.0
        n_ticks = int((STATIC_SECS + MOTION_SECS) * IMU_RATE)
        gt_at_scan = []
        for i in range(n_ticks):
            moving = i * dt > STATIC_SECS
            wz = WZ if moving else 0.0
            vx = VX if moving else 0.0
            sim.publish_imu(t, wz)
            if i % SCAN_EVERY == 0:
                sim.publish_scan(t, pos, yaw)
                gt_at_scan.append((t, pos.copy(), yaw))
            # integrate GT forward
            yaw += wz * dt
            pos += yaw_rot(yaw)[:, 0] * vx * dt
            t += dt
            rclpy.spin_once(sim, timeout_sec=0.0)
            time.sleep(dt * 0.8)

        # drain remaining callbacks
        end = time.time() + 1.0
        while time.time() < end:
            rclpy.spin_once(sim, timeout_sec=0.05)

        n = len(sim.odom_msgs)
        print(f"[smoke] received {n} odometry msgs")
        if n < 50:
            print("[smoke] FAIL: too few odometry messages")
            return 1

        for m in sim.odom_msgs:
            vals = [
                m.pose.pose.position.x, m.pose.pose.position.y,
                m.pose.pose.position.z, m.pose.pose.orientation.w,
            ]
            if any(math.isnan(v) or math.isinf(v) for v in vals):
                print("[smoke] FAIL: NaN/inf in odometry")
                return 1

        # compare final odom pose with GT at its stamp
        last = sim.odom_msgs[-1]
        t_last = last.header.stamp.sec + last.header.stamp.nanosec * 1e-9
        gt_t, gt_pos, gt_yaw = min(gt_at_scan, key=lambda g: abs(g[0] - t_last))
        est = np.array([
            last.pose.pose.position.x, last.pose.pose.position.y,
            last.pose.pose.position.z,
        ])
        q = last.pose.pose.orientation
        est_yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                             1 - 2 * (q.y * q.y + q.z * q.z))
        pos_err = float(np.linalg.norm(est - gt_pos))
        yaw_err = abs((est_yaw - gt_yaw + math.pi) % (2 * math.pi) - math.pi)
        print(f"[smoke] final pose err: {pos_err:.3f} m, yaw err: "
              f"{math.degrees(yaw_err):.2f} deg (gt dt {abs(gt_t - t_last):.3f}s)")
        if pos_err > 0.2 or yaw_err > math.radians(5.0):
            print("[smoke] FAIL: tracking error too large")
            return 1

        ok = True
        print("[smoke] PASS")
        return 0
    finally:
        node_proc.terminate()
        try:
            node_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            node_proc.kill()
        node_log.close()
        if sim is not None:
            sim.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if not ok:
            print(f"[smoke] artifacts kept in {tmpdir} for debugging")


if __name__ == "__main__":
    sys.exit(main())
