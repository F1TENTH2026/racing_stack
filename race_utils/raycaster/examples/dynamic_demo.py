#!/usr/bin/env python3
"""
Dynamic-overlay RViz demo: PRECOMPUTE static (glt/lut, built once) + overlay
dynamics (NO map rebuild).
  - ego drives the centerline; opponent car drives ahead (green box)
  - RViz 'Publish Point' adds/removes a circular obstacle (red) -> appears in /scan
  - lidar 40 Hz, dynamics (ego + opponent) 120 Hz

Run: examples/run_dynamic.sh glt f
"""
import os, sys, math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, qos_profile_sensor_data
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TransformStamped, PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformBroadcaster

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from raycaster import RaycastEngine
CAC = os.environ.get("CAC_DIR", "/home/js/unicorn_racing_stack/src/creating_autonomous_car")


class DynamicDemo(Node):
    def __init__(self):
        super().__init__("raycast_dynamic_demo")
        self.backend = self.declare_parameter("backend", "rm").value   # rm = exact for sim; glt/lut for PF batch
        mapname = self.declare_parameter("map", "f").value
        self.nb = self.declare_parameter("num_beams", 1080).value
        self.fov = self.declare_parameter("fov", 4.7).value
        self.mr = self.declare_parameter("max_range", 10.0).value
        ydir = f"{CAC}/stack_master/maps/{mapname}"
        occ, self.res, self.origin = RaycastEngine.load_map_yaml(f"{ydir}/{mapname}.yaml")
        self.occ = occ; self.H, self.W = occ.shape
        td = 720 if self.backend == "lut" else 112
        self.eng = RaycastEngine(self.backend, max_range_m=self.mr, theta_disc=td).set_map(occ, self.res, self.origin)
        self.get_logger().info(f"[{self.backend}] static table built ONCE; dynamics via overlay (no rebuild)")
        self.cl = np.loadtxt(f"{ydir}/centerline.csv", delimiter=",", skiprows=1)[:, :2]
        self.n = len(self.cl); self.opp_gap = 35; self.obstacles = []
        self.ego_speed = float(self.declare_parameter("ego_speed", 25.0).value)   # idx/s (0 = static)
        self.ego_i = float(self.declare_parameter("ego_start", 0).value)

        latched = QoSProfile(depth=1); latched.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.map_pub = self.create_publisher(OccupancyGrid, "/map", latched)
        self.scan_pub = self.create_publisher(LaserScan, "/scan", qos_profile_sensor_data)
        self.opp_pub = self.create_publisher(Marker, "/opponent", 10)
        self.obs_pub = self.create_publisher(MarkerArray, "/obstacles", 10)
        self.tfb = TransformBroadcaster(self)
        self.create_subscription(PointStamped, "/clicked_point", self.click_cb, 10)
        self.map_pub.publish(self._grid())
        self._opp = self._pose_at(self.opp_gap)
        self.create_timer(1/120, self.dyn_tick)    # dynamics 120 Hz
        self.create_timer(1/40, self.lidar_tick)   # lidar 40 Hz
        self.create_timer(0.15, self.marker_tick)

    def _pose_at(self, i):
        i0 = int(i) % self.n; p = self.cl[i0]; nxt = self.cl[(i0 + 3) % self.n]
        return np.array([p[0], p[1], math.atan2(nxt[1] - p[1], nxt[0] - p[0])])

    def _grid(self):
        g = OccupancyGrid(); g.header.frame_id = "map"; g.info.resolution = float(self.res)
        g.info.width = self.W; g.info.height = self.H
        g.info.origin.position.x = float(self.origin[0]); g.info.origin.position.y = float(self.origin[1])
        g.info.origin.orientation.w = 1.0
        g.data = np.where(self.occ, 100, 0).astype(np.int8).flatten().tolist(); return g

    def dyn_tick(self):
        self.ego_i = (self.ego_i + self.ego_speed / 120) % self.n   # advance @ 120 Hz
        self._opp = self._pose_at(self.ego_i + self.opp_gap)

    def lidar_tick(self):
        ego = self._pose_at(self.ego_i)
        scan = self.eng.scan_with_dynamics(ego, self.nb, self.fov,
                                           opp_poses=[self._opp], obstacles=self.obstacles)
        now = self.get_clock().now().to_msg()
        t = TransformStamped(); t.header.stamp = now; t.header.frame_id = "map"; t.child_frame_id = "laser"
        t.transform.translation.x = float(ego[0]); t.transform.translation.y = float(ego[1])
        t.transform.rotation.z = math.sin(ego[2] / 2); t.transform.rotation.w = math.cos(ego[2] / 2)
        self.tfb.sendTransform(t)
        s = LaserScan(); s.header.stamp = now; s.header.frame_id = "laser"
        s.angle_min = -self.fov / 2; s.angle_max = self.fov / 2; s.angle_increment = self.fov / (self.nb - 1)
        s.range_min = 0.0; s.range_max = float(self.mr); s.ranges = scan.astype(np.float32).tolist()
        self.scan_pub.publish(s)

    def marker_tick(self):
        m = Marker(); m.header.frame_id = "map"; m.ns = "opp"; m.id = 0; m.type = Marker.CUBE; m.action = Marker.ADD
        m.pose.position.x = float(self._opp[0]); m.pose.position.y = float(self._opp[1]); m.pose.position.z = 0.1
        m.pose.orientation.z = math.sin(self._opp[2] / 2); m.pose.orientation.w = math.cos(self._opp[2] / 2)
        m.scale.x = 0.58; m.scale.y = 0.31; m.scale.z = 0.2; m.color.g = 1.0; m.color.a = 0.9
        self.opp_pub.publish(m)
        ma = MarkerArray()
        d = Marker(); d.header.frame_id = "map"; d.ns = "obs"; d.action = Marker.DELETEALL; ma.markers.append(d)
        for k, (x, y, r) in enumerate(self.obstacles):
            c = Marker(); c.header.frame_id = "map"; c.ns = "obs"; c.id = k; c.type = Marker.CYLINDER; c.action = Marker.ADD
            c.pose.position.x = float(x); c.pose.position.y = float(y); c.pose.position.z = 0.15; c.pose.orientation.w = 1.0
            c.scale.x = c.scale.y = float(2 * r); c.scale.z = 0.3; c.color.r = 1.0; c.color.b = 0.2; c.color.a = 0.85
            ma.markers.append(c)
        self.obs_pub.publish(ma)

    def click_cb(self, msg):
        x, y = msg.point.x, msg.point.y
        for k, (ox, oy, r) in enumerate(self.obstacles):
            if math.hypot(ox - x, oy - y) < 0.5:
                del self.obstacles[k]; self.get_logger().info("obstacle removed"); return
        self.obstacles.append((x, y, 0.3)); self.get_logger().info(f"obstacle added at ({x:.1f},{y:.1f}); total {len(self.obstacles)}")


def main():
    rclpy.init(); rclpy.spin(DynamicDemo())


if __name__ == "__main__":
    main()
