#!/usr/bin/env python3
"""
Accuracy comparison: every backend's lidar scan published on its OWN color-coded
topic, from the SAME pose, so you can SEE where they differ in RViz.

  green  /scan_rm       rm  (exact DT march, no angle quantization) = REFERENCE
  red    /scan_glt112   glt theta_disc=112  (what the demo used; coarse)
  yellow /scan_glt720   glt theta_disc=720  (finer LUT)
  blue   /scan_cddt112  cddt theta_disc=112
  white  /scan_bl       bl  (Bresenham, exact)

Where colors diverge from green = that backend's quantization error.
Edit CONFIGS below to add/remove backends or theta_disc values.

Run: examples/run_compare.sh   (static pose; set -p ego_speed:=20 to drive)
"""
import os, sys, math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, qos_profile_sensor_data
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from raycaster import RaycastEngine
CAC = os.environ.get("CAC_DIR", "/home/js/unicorn_racing_stack/src/creating_autonomous_car")

# (label, backend, theta_disc) -- color is bound to the topic in compare_demo.rviz
CONFIGS = [
    ("rm",      "rm",   112),   # exact reference (green)
    ("glt112",  "glt",  112),   # red
    ("glt720",  "glt",  720),   # yellow
    ("cddt112", "cddt", 112),   # blue
    ("bl",      "bl",   112),   # white (Bresenham, exact)
]


class CompareDemo(Node):
    def __init__(self):
        super().__init__("raycast_compare")
        mapname = self.declare_parameter("map", "f").value
        self.nb = self.declare_parameter("num_beams", 1080).value
        self.fov = self.declare_parameter("fov", 4.7).value
        self.mr = self.declare_parameter("max_range", 10.0).value
        self.speed = float(self.declare_parameter("ego_speed", 0.0).value)   # 0 = static
        ydir = f"{CAC}/stack_master/maps/{mapname}"
        occ, self.res, self.origin = RaycastEngine.load_map_yaml(f"{ydir}/{mapname}.yaml")
        self.occ = occ; self.H, self.W = occ.shape
        self.cl = np.loadtxt(f"{ydir}/centerline.csv", delimiter=",", skiprows=1)[:, :2]
        self.n = len(self.cl); self.ego_i = float(self.declare_parameter("ego_start", 0).value)
        self.engines = []
        for label, be, td in CONFIGS:
            try:
                e = RaycastEngine(be, max_range_m=self.mr, theta_disc=td).set_map(occ, self.res, self.origin)
                pub = self.create_publisher(LaserScan, f"/scan_{label}", qos_profile_sensor_data)
                self.engines.append((label, e, pub))
                self.get_logger().info(f"built {label:8s} ({be} td={td}) -> /scan_{label}")
            except Exception as ex:
                self.get_logger().warn(f"skip {label}: {ex}")
        latched = QoSProfile(depth=1); latched.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.map_pub = self.create_publisher(OccupancyGrid, "/map", latched)
        self.tfb = TransformBroadcaster(self)
        self.map_pub.publish(self._grid())
        # one-shot MAE-vs-rm print so you also get numbers
        self._report()
        self.create_timer(1 / 20, self.tick)

    def _grid(self):
        g = OccupancyGrid(); g.header.frame_id = "map"; g.info.resolution = float(self.res)
        g.info.width = self.W; g.info.height = self.H
        g.info.origin.position.x = float(self.origin[0]); g.info.origin.position.y = float(self.origin[1])
        g.info.origin.orientation.w = 1.0
        g.data = np.where(self.occ, 100, 0).astype(np.int8).flatten().tolist(); return g

    def _pose(self, i):
        i0 = int(i) % self.n; p = self.cl[i0]; nx = self.cl[(i0 + 3) % self.n]
        return np.array([p[0], p[1], math.atan2(nx[1] - p[1], nx[0] - p[0])])

    def _report(self):
        ego = self._pose(self.ego_i)
        ref = dict((l, e.scan(ego, self.nb, self.fov)) for l, e, _ in self.engines).get("rm")
        if ref is None:
            return
        self.get_logger().info("MAE vs rm (cm) at the current pose:")
        for l, e, _ in self.engines:
            d = np.abs(e.scan(ego, self.nb, self.fov) - ref)
            m = (ref < self.mr - 0.05)
            self.get_logger().info(f"  {l:8s}: {d[m].mean()*100:5.2f} cm")

    def tick(self):
        self.ego_i = (self.ego_i + self.speed / 20) % self.n
        ego = self._pose(self.ego_i)
        now = self.get_clock().now().to_msg()
        t = TransformStamped(); t.header.stamp = now; t.header.frame_id = "map"; t.child_frame_id = "laser"
        t.transform.translation.x = float(ego[0]); t.transform.translation.y = float(ego[1])
        t.transform.rotation.z = math.sin(ego[2] / 2); t.transform.rotation.w = math.cos(ego[2] / 2)
        self.tfb.sendTransform(t)
        for label, e, pub in self.engines:
            scan = e.scan(ego, self.nb, self.fov, miss=np.nan)   # un-returned beams -> NaN (no false max-range wall)
            s = LaserScan(); s.header.stamp = now; s.header.frame_id = "laser"
            s.angle_min = -self.fov / 2; s.angle_max = self.fov / 2; s.angle_increment = self.fov / (self.nb - 1)
            s.range_min = 0.0; s.range_max = float(self.mr); s.ranges = scan.astype(np.float32).tolist()
            pub.publish(s)


def main():
    rclpy.init(); rclpy.spin(CompareDemo())


if __name__ == "__main__":
    main()
