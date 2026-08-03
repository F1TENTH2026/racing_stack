#!/usr/bin/env python3
"""Bag-replay validation for kiss_icp_localization.

Plays a real MID360 bag against a prior map and checks that the node
keeps map-lock: finite odometry, no pose jumps, plausible speeds, and a
low scan-to-map residual (mean NN distance of the aligned scan).

Usage (workspace sourced):
    python3 bag_validation.py [bag_dir] [map_pcd]
"""
import math
import os
import re
import struct
import subprocess
import sys
import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scipy.spatial import cKDTree
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

DEFAULT_BAG = "/Users/mini/bag/rosbag2_2026_06_10-12_13_30"
DEFAULT_MAP = ("/Users/mini/ros2_ws/src/IFAC2026/src/system/stack_master/maps/"
               "hall_0609/cloudGlobal.pcd")
DOMAIN_ID = "77"

# pass criteria
MIN_SCAN_POSES = 300        # bag has ~470 lidar frames
MAX_JUMP = 0.5              # m between consecutive scan-rate poses
MAX_SPEED = 8.0             # m/s
MAX_MEAN_RESIDUAL = 0.15    # m, average over locked frames
MIN_LOCK_FRACTION = 0.9     # fraction of frames with median residual < 0.2


def load_pcd_xyz(path):
    """Minimal reader for PCL binary (uncompressed) PCD; returns Nx3 float32."""
    with open(path, "rb") as f:
        header = b""
        while True:
            line = f.readline()
            header += line
            if line.startswith(b"DATA"):
                break
        h = header.decode("ascii", errors="replace")
        fields = re.search(r"FIELDS (.+)", h).group(1).split()
        sizes = [int(s) for s in re.search(r"SIZE (.+)", h).group(1).split()]
        counts = [int(c) for c in re.search(r"COUNT (.+)", h).group(1).split()]
        npts = int(re.search(r"POINTS (\d+)", h).group(1))
        if "binary_compressed" in h:
            raise RuntimeError("compressed PCD not supported by this reader")
        if "ascii" in h.split("DATA")[1]:
            data = np.loadtxt(f, dtype=np.float32)
            idx = [fields.index(a) for a in ("x", "y", "z")]
            return data[:, idx]
        point_step = sum(s * c for s, c in zip(sizes, counts))
        buf = f.read(npts * point_step)
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(npts, point_step)
        out = np.empty((npts, 3), dtype=np.float32)
        for k, name in enumerate(("x", "y", "z")):
            i = fields.index(name)
            off = sum(s * c for s, c in zip(sizes[:i], counts[:i]))
            out[:, k] = arr[:, off:off + 4].copy().view(np.float32)[:, 0]
        return out


class Recorder(Node):
    def __init__(self):
        super().__init__("kiss_loc_bag_validator")
        self.odom = []        # (t, xyz, speed_from_twist)
        self.scan_samples = []  # (t, Nx3 subsampled aligned points)
        self.create_subscription(Odometry, "/kiss_loc/odometry", self.on_odom, 200)
        self.create_subscription(
            PointCloud2, "/kiss_loc/scan_aligned", self.on_scan, 20)
        self._scan_count = 0

    def on_odom(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p = msg.pose.pose.position
        v = msg.twist.twist.linear
        self.odom.append((t, p.x, p.y, p.z,
                          math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2), abs(v.x)))

    def on_scan(self, msg):
        self._scan_count += 1
        if self._scan_count % 5 != 0:
            return  # keep every 5th frame for residual stats
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        pts = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"))
        if len(pts) > 500:
            pts = pts[:: len(pts) // 500]
        self.scan_samples.append((t, np.asarray(pts, dtype=np.float32)))


def parse_stat_lines(log_path):
    rows = []
    with open(log_path) as f:
        for line in f:
            if "STAT " not in line:
                continue
            kv = dict(p.split("=") for p in line.split("STAT ")[1].split())
            rows.append({k: float(v) for k, v in kv.items()})
    if not rows:
        return None
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def read_wheel_speed(bag):
    from rclpy.serialization import deserialize_message
    from nav_msgs.msg import Odometry as OdomMsg
    from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions

    r = SequentialReader()
    r.open(StorageOptions(uri=bag, storage_id="mcap"), ConverterOptions("cdr", "cdr"))
    t, v = [], []
    while r.has_next():
        topic, data, _ = r.read_next()
        if topic != "/odom":
            continue
        m = deserialize_message(data, OdomMsg)
        t.append(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)
        v.append(abs(m.twist.twist.linear.x))
    return np.array(t), np.array(v)


def main():
    bag = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BAG
    map_pcd = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MAP
    os.environ["ROS_DOMAIN_ID"] = DOMAIN_ID

    print(f"[bag-val] map: {map_pcd}")
    map_pts = load_pcd_xyz(map_pcd)
    print(f"[bag-val] map loaded: {len(map_pts)} pts; building KDTree...")
    tree = cKDTree(map_pts)

    # exec the binary directly — `ros2 run` is a wrapper, so terminating it
    # leaves the actual node process running
    node_bin = os.path.expanduser(
        "~/ros2_ws/install/kiss_icp_localization/lib/kiss_icp_localization/"
        "localization_node")
    node_log_path = "/tmp/kiss_loc_bag_val_node.log"
    node_log = open(node_log_path, "w")
    node_proc = subprocess.Popen(
        [
            node_bin,
            "--ros-args",
            "-p", f"map_pcd_path:={map_pcd}",
            "-p", "use_custom_msg:=false",
            "-p", "publish_tf:=false",
            "-p", "print_stats:=true",
        ] + [a for kv in sys.argv[3:] for a in ("-p", kv)],
        env=dict(os.environ), stdout=node_log, stderr=subprocess.STDOUT,
    )

    bag_proc = None
    rec = None
    ok = False
    try:
        time.sleep(8.0)  # node: load 900k-pt map + build voxel hash
        if node_proc.poll() is not None:
            print(f"[bag-val] FAIL: node exited early — see {node_log_path}")
            return 1

        rclpy.init()
        rec = Recorder()

        print("[bag-val] playing bag (real time, ~47 s)...")
        bag_proc = subprocess.Popen(
            ["ros2", "bag", "play", bag, "--topics", "/livox/lidar", "/livox/imu"],
            env=dict(os.environ),
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
        )
        t_end = time.time() + 75
        while bag_proc.poll() is None and time.time() < t_end:
            rclpy.spin_once(rec, timeout_sec=0.05)
        drain = time.time() + 2.0
        while time.time() < drain:
            rclpy.spin_once(rec, timeout_sec=0.05)

        # ---------------- checks ----------------
        n = len(rec.odom)
        odom = np.array(rec.odom)
        print(f"[bag-val] odometry msgs: {n}, aligned-scan samples: "
              f"{len(rec.scan_samples)}")
        if n < MIN_SCAN_POSES:
            print("[bag-val] FAIL: too few odometry messages")
            return 1
        if not np.isfinite(odom).all():
            print("[bag-val] FAIL: NaN/inf in odometry")
            return 1

        d = np.linalg.norm(np.diff(odom[:, 1:4], axis=0), axis=1)
        path_len = float(d.sum())
        # skip the first second: snapping from the initial-pose parameter to
        # the first registration fix is acquisition, not a tracking jump
        tracking = odom[:-1, 0] > odom[0, 0] + 1.0
        max_jump = float(d[tracking].max())
        max_speed = float(odom[:, 4].max())
        print(f"[bag-val] path length {path_len:.1f} m, max inter-msg jump "
              f"{max_jump:.3f} m, max |v| {max_speed:.2f} m/s")

        med_res, mean_res = [], []
        for _, pts in rec.scan_samples:
            dist, _ = tree.query(pts, k=1)
            med_res.append(float(np.median(dist)))
            mean_res.append(float(np.mean(dist)))
        med_res = np.array(med_res)
        lock_frac = float((med_res < 0.2).mean()) if len(med_res) else 0.0
        overall_mean = float(np.mean(mean_res)) if mean_res else 1e9
        print(f"[bag-val] scan-to-map residual: mean {overall_mean:.3f} m, "
              f"median-of-medians {np.median(med_res):.3f} m, "
              f"lock fraction {lock_frac:.2%} ({len(med_res)} frames)")

        fails = []
        if max_jump > MAX_JUMP:
            fails.append(f"jump {max_jump:.2f} > {MAX_JUMP}")
        if max_speed > MAX_SPEED:
            fails.append(f"speed {max_speed:.2f} > {MAX_SPEED}")
        if overall_mean > MAX_MEAN_RESIDUAL:
            fails.append(f"mean residual {overall_mean:.3f} > {MAX_MEAN_RESIDUAL}")
        if lock_frac < MIN_LOCK_FRACTION:
            fails.append(f"lock fraction {lock_frac:.2%} < {MIN_LOCK_FRACTION:.0%}")

        np.savetxt("/tmp/kiss_loc_bag_val_traj.csv", odom,
                   header="t x y z speed", comments="")
        print("[bag-val] trajectory saved to /tmp/kiss_loc_bag_val_traj.csv")

        # ---- node STAT lines: ICP timing / convergence health ----
        node_log.flush()
        stats = parse_stat_lines(node_log_path)
        if stats:
            for key, fmt in (("icp_ms", "%.1f"), ("prep_ms", "%.1f"),
                             ("iters", "%.0f"), ("corr", "%.0f"),
                             ("th", "%.3f"), ("dev_t", "%.3f"),
                             ("lat_ms", "%.0f"), ("ds", "%.0f")):
                v = stats[key]
                print(f"[bag-val] {key}: p50 {fmt % np.percentile(v, 50)}, "
                      f"p95 {fmt % np.percentile(v, 95)}, "
                      f"max {fmt % v.max()}")
            print(f"[bag-val] scans processed: {len(stats['t'])}, "
                  f"converged: {stats['conv'].mean():.1%}")

        # ---- speed vs wheel odometry (longitudinal reference) ----
        wheel_t, wheel_v = read_wheel_speed(bag)
        if len(wheel_t):
            est = odom[np.abs(odom[:, 4]) > 1e-9]  # msgs carrying velocity
            wv = np.interp(est[:, 0], wheel_t, wheel_v)
            err = est[:, 5] - wv  # |vx| vs wheel longitudinal speed
            print(f"[bag-val] |vx| vs wheel: rmse {np.sqrt((err**2).mean()):.2f} "
                  f"m/s, bias {err.mean():+.2f}, p95 |err| "
                  f"{np.percentile(np.abs(err), 95):.2f}")

        if fails:
            print("[bag-val] FAIL: " + "; ".join(fails))
            return 1
        ok = True
        print("[bag-val] PASS")
        return 0
    finally:
        for p in (bag_proc, node_proc):
            if p is not None:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
        node_log.close()
        if rec is not None:
            rec.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if not ok:
            print(f"[bag-val] node log: {node_log_path}")


if __name__ == "__main__":
    sys.exit(main())
