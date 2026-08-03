#!/usr/bin/env python3
"""
RViz demo: drive a pose around an F1TENTH map and publish the RaycastEngine scan.

Publishes:  /map (OccupancyGrid, latched)  +  /scan (LaserScan)  +  TF map->laser
View:       fixed frame = map, add Map + LaserScan (or use examples/raycast_demo.rviz)

Run (ROS 2 Jazzy sourced; range_libc built for the system python):
  RC=.../tools/raycaster
  PYTHONPATH="$RC:$RC/range_libc/pywrapper:$PYTHONPATH" \
    python3 examples/rviz_demo.py --ros-args -p backend:=rm -p map:=f
"""
import os, sys, math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from raycaster import RaycastEngine

CAC = os.environ.get("CAC_DIR", "/home/js/unicorn_racing_stack/src/creating_autonomous_car")


class RaycastDemo(Node):
    def __init__(self):
        super().__init__("raycast_demo")
        self.backend = self.declare_parameter("backend", "rm").value
        mapname = self.declare_parameter("map", "f").value
        self.num_beams = self.declare_parameter("num_beams", 1080).value
        self.fov = self.declare_parameter("fov", 4.7).value
        self.max_range = self.declare_parameter("max_range", 10.0).value

        ydir = f"{CAC}/stack_master/maps/{mapname}"
        occ, self.res, self.origin = RaycastEngine.load_map_yaml(f"{ydir}/{mapname}.yaml")
        self.occ = occ; self.H, self.W = occ.shape
        td = 720 if self.backend == "lut" else 112
        self.eng = RaycastEngine(self.backend, max_range_m=self.max_range, theta_disc=td)
        self.eng.set_map(occ, self.res, self.origin)
        self.get_logger().info(f"RaycastEngine backend='{self.backend}', map='{mapname}' {occ.shape}")

        # path to drive: centerline if present, else free cells
        cl = f"{ydir}/centerline.csv"
        if os.path.exists(cl):
            self.path = np.loadtxt(cl, delimiter=",", skiprows=1)[:, :2]
        else:
            ys, xs = np.where(~occ)
            self.path = np.stack([self.origin[0] + xs[::50] * self.res,
                                  self.origin[1] + ys[::50] * self.res], 1)
        self.idx = 0

        latched = QoSProfile(depth=1); latched.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.map_pub = self.create_publisher(OccupancyGrid, "/map", latched)
        self.scan_pub = self.create_publisher(LaserScan, "/scan", 10)
        self.tfb = TransformBroadcaster(self)
        self.map_pub.publish(self._grid())
        self.create_timer(0.05, self.tick)        # 20 Hz

    def _grid(self):
        g = OccupancyGrid()
        g.header.frame_id = "map"
        g.info.resolution = float(self.res)
        g.info.width = self.W; g.info.height = self.H
        g.info.origin.position.x = float(self.origin[0])
        g.info.origin.position.y = float(self.origin[1])
        g.info.origin.orientation.w = 1.0
        g.data = np.where(self.occ, 100, 0).astype(np.int8).flatten().tolist()
        return g

    def tick(self):
        p = self.path[self.idx % len(self.path)]
        nxt = self.path[(self.idx + 3) % len(self.path)]
        yaw = math.atan2(nxt[1] - p[1], nxt[0] - p[0])
        self.idx += 1
        now = self.get_clock().now().to_msg()

        t = TransformStamped()
        t.header.stamp = now; t.header.frame_id = "map"; t.child_frame_id = "laser"
        t.transform.translation.x = float(p[0]); t.transform.translation.y = float(p[1])
        t.transform.rotation.z = math.sin(yaw / 2); t.transform.rotation.w = math.cos(yaw / 2)
        self.tfb.sendTransform(t)

        ranges = self.eng.scan(np.array([p[0], p[1], yaw]), self.num_beams, self.fov)
        s = LaserScan()
        s.header.stamp = now; s.header.frame_id = "laser"
        s.angle_min = -self.fov / 2; s.angle_max = self.fov / 2
        s.angle_increment = self.fov / (self.num_beams - 1)
        s.range_min = 0.0; s.range_max = float(self.max_range)
        s.ranges = ranges.astype(np.float32).tolist()
        self.scan_pub.publish(s)


def main():
    rclpy.init()
    rclpy.spin(RaycastDemo())


if __name__ == "__main__":
    main()
