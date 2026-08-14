#!/usr/bin/env python3
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import (
    FloatingPointRange,
    IntegerRange,
    ParameterDescriptor,
    ParameterType,
    SetParametersResult,
)

import numpy as np
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from scipy.interpolate import BPoly
from f110_msgs.msg import Obstacle, OTWpntArray, Wpnt, WpntArray, BehaviorStrategy
from frenet_conversion.frenet_converter import FrenetConverter
from transforms3d.euler import quat2euler
from grid_filter.grid_filter import GridFilter
import trajectory_planning_helpers as tph

# --- Fixed geometry. These are NOT tunables: they define the collision invariant. ---
CAR_WIDTH_M = 0.30                                          # measured vehicle width
SAFETY_MARGIN_M = 0.025                                     # per side (0.05 m total across the car)
MIN_EVASION_DIST_M = CAR_WIDTH_M / 2 + SAFETY_MARGIN_M      # 0.175 m from the obstacle edge to the
                                                            # path. The relaxation ladder never goes
                                                            # below this, and _clearance_ok enforces it
                                                            # on the finished path.

RATE_HZ = 40.0
MIN_APEX_LEAD_M = 0.5     # the spline needs this much length ahead of the car to be well posed
GB_TAIL_POINTS = 100      # global-line tail appended so the state machine can measure gaps ahead
MIN_SAMPLES = 3           # below this the curvature/velocity math is degenerate

# Relaxation ladder: (evasion scale, post-distance scale, side).
# Tried in order, first path that passes the wall + obstacle checks wins. The effective
# evasion distance is max(scale * evasion_dist, MIN_EVASION_DIST_M), so scale 0.0 means
# "fall back to the hard floor" -- no rung can ever plan a path closer than the floor.
RELAXATION_LADDER = (
    (1.0, 1.0, "preferred"),   # normal case: latched/wider side at the configured clearance
    (1.0, 1.0, "opposite"),    # other side, same clearance
    (0.7, 1.5, "wider"),       # tighter clearance, longer and therefore flatter return
    (0.0, 1.5, "wider"),       # last resort: hard floor clearance
)


def _opposite(side: str) -> str:
    return "right" if side == "left" else "left"


class ObstacleSpliner(Node):
    """
    Plans an evasion path around a single static obstacle.

    Subscribes:
        - `/behavior_strategy`         : overtaking target picked by the state machine
        - `/car_state/odom_frenet`     : ego state in Frenet coordinates
        - `/car_state/odom`            : ego state in cartesian coordinates
        - `/global_waypoints`          : global waypoints
        - `/global_waypoints_scaled`   : scaled global waypoints

    Publishes:
        - `/planner/avoidance/otwpnts` : the evasion trajectory (remapped to static_otwpnts)
        - `/planner/avoidance/markers` : trajectory visualization
    """

    def __init__(self):
        self.name = "static_avoidance_planner"
        super().__init__('static_avoidance_planner')

        self.obs_in_interest = None
        self.gb_wpnts = None
        self.gb_vmax = None
        self.gb_max_idx = None
        self.gb_max_s = None
        self.cur_s = None
        self.cur_d = None
        self.cur_vs = None
        self.cur_x = None
        self.cur_y = None
        self.cur_yaw = None
        self.gb_scaled_wpnts = None
        self.last_ot_side = ""
        self.last_ot_obstacle_id = None

        # Tunables (defaults, overwritten by the declared parameters below)
        self.kernel_size = 8
        self.post_min_dist = 1.5
        self.post_max_dist = 5.0
        self.spline_scale = 0.8
        self.evasion_dist = 0.3
        self.spline_bound_mindist = 0.2

        self.map_filter = GridFilter(node=self, map_topic="/map", debug=False)
        self.map_filter.set_erosion_kernel_size(self.kernel_size)

        self.declare_all_parameters()
        # Apply loaded params at startup; the callback only fires on later set() calls.
        self.dyn_param_cb(self.get_parameters([
            'kernel_size', 'post_min_dist', 'post_max_dist',
            'spline_scale', 'evasion_dist', 'spline_bound_mindist',
        ]))
        self.add_on_set_parameters_callback(self.dyn_param_cb)

        self.create_subscription(BehaviorStrategy, "/behavior_strategy", self.behavior_cb, 10)
        self.create_subscription(Odometry, "/car_state/odom_frenet", self.state_frenet_cb, 10)
        self.create_subscription(Odometry, "/car_state/odom", self.state_cb, 10)
        self.create_subscription(WpntArray, "/global_waypoints", self.gb_cb, 10)
        self.create_subscription(WpntArray, "/global_waypoints_scaled", self.gb_scaled_cb, 10)

        self.mrks_pub = self.create_publisher(MarkerArray, "/planner/avoidance/markers", 10)
        self.evasion_pub = self.create_publisher(OTWpntArray, "/planner/avoidance/otwpnts", 10)

        self.wait_for_messages()
        self.converter = self.initialize_converter()
        self.create_timer(1.0 / RATE_HZ, self.loop)

    ######################
    # DYNAMIC PARAMETERS #
    ######################
    def declare_all_parameters(self):
        def dbl(min_v, max_v, desc=""):
            return ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description=desc,
                floating_point_range=[FloatingPointRange(from_value=float(min_v),
                                                         to_value=float(max_v),
                                                         step=0.001)],
            )

        def intd(min_v, max_v, desc=""):
            return ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description=desc,
                integer_range=[IntegerRange(from_value=int(min_v),
                                            to_value=int(max_v),
                                            step=1)],
            )

        self.declare_parameter('kernel_size', 8, intd(1, 20, "map erosion kernel"))
        self.declare_parameter('post_min_dist', 1.5, dbl(0.5, 3.0, "min return distance after apex"))
        self.declare_parameter('post_max_dist', 5.0, dbl(3.0, 20.0, "max return distance after apex"))
        self.declare_parameter('spline_scale', 0.8, dbl(0.5, 2.0, "tangent scale, higher = wider swing"))
        self.declare_parameter('evasion_dist', 0.3,
                               dbl(MIN_EVASION_DIST_M, 1.25, "nominal clearance from the obstacle edge"))
        self.declare_parameter('spline_bound_mindist', 0.2, dbl(0.05, 1.0, "wall margin for side selection"))

    def dyn_param_cb(self, params: List[Parameter]):
        for param in params:
            if param.name == 'evasion_dist':
                self.evasion_dist = max(round(param.value * 20) / 20, MIN_EVASION_DIST_M)
            elif param.name == 'spline_bound_mindist':
                self.spline_bound_mindist = round(param.value * 20) / 20
            elif param.name == 'spline_scale':
                self.spline_scale = param.value
            elif param.name == 'post_min_dist':
                self.post_min_dist = param.value
            elif param.name == 'post_max_dist':
                self.post_max_dist = param.value
            elif param.name == 'kernel_size':
                self.kernel_size = param.value
                self.map_filter.set_erosion_kernel_size(self.kernel_size)

        self.get_logger().info(
            f"[{self.name}] evasion_dist: {self.evasion_dist:.2f} m (floor {MIN_EVASION_DIST_M:.3f} m), "
            f"wall margin: {self.spline_bound_mindist:.2f} m, "
            f"return dist: {self.post_min_dist:.1f}-{self.post_max_dist:.1f} m, "
            f"spline scale: {self.spline_scale:.2f}, erosion: {self.kernel_size}"
        )
        return SetParametersResult(successful=True)

    #############
    # CALLBACKS #
    #############
    def behavior_cb(self, data: BehaviorStrategy):
        self.obs_in_interest = data.overtaking_targets[0] if len(data.overtaking_targets) else None

    def state_frenet_cb(self, data: Odometry):
        self.cur_s = data.pose.pose.position.x
        self.cur_d = data.pose.pose.position.y
        self.cur_vs = data.twist.twist.linear.x

    def state_cb(self, data: Odometry):
        self.cur_x = data.pose.pose.position.x
        self.cur_y = data.pose.pose.position.y
        quat = data.pose.pose.orientation
        # transforms3d uses (w, x, y, z) quaternion ordering
        self.cur_yaw = quat2euler([quat.w, quat.x, quat.y, quat.z])[2]

    def gb_cb(self, data: WpntArray):
        self.gb_wpnts = data
        if self.gb_vmax is None:
            self.gb_vmax = np.max(np.array([wpnt.vx_mps for wpnt in data.wpnts]))
            self.gb_max_idx = data.wpnts[-1].id
            self.gb_max_s = data.wpnts[-1].s_m

    def gb_scaled_cb(self, data: WpntArray):
        self.gb_scaled_wpnts = data

    #############
    # MAIN LOOP #
    #############
    def loop(self):
        wpnts = OTWpntArray()
        mrks = MarkerArray()

        if self.obs_in_interest is not None:
            wpnts, mrks = self.do_spline(obs=self.obs_in_interest, gb_wpnts=self.gb_scaled_wpnts.wpnts)
        else:
            del_mrk = Marker()
            del_mrk.header.stamp = self.get_clock().now().to_msg()
            del_mrk.action = Marker.DELETEALL
            mrks.markers.append(del_mrk)

        self.evasion_pub.publish(wpnts)
        self.mrks_pub.publish(mrks)

    #########
    # UTILS #
    #########
    def wait_for_messages(self):
        self.get_logger().info(f"[{self.name}] Waiting for messages and services...")
        waitlist = [self.cur_s, self.cur_x, self.gb_wpnts, self.gb_scaled_wpnts]
        while None in waitlist:
            rclpy.spin_once(self)
            waitlist = [self.cur_s, self.cur_x, self.gb_wpnts, self.gb_scaled_wpnts]
        self.get_logger().info(f"[{self.name}] Ready!")

    def initialize_converter(self) -> FrenetConverter:
        waypoint_array = self.gb_wpnts.wpnts
        converter = FrenetConverter(
            np.array([wpnt.x_m for wpnt in waypoint_array]),
            np.array([wpnt.y_m for wpnt in waypoint_array]),
            np.array([wpnt.psi_rad for wpnt in waypoint_array]),
        )
        self.get_logger().info(f"[{self.name}] initialized FrenetConverter object")
        return converter

    ############
    # PLANNING #
    ############
    def _side_gaps(
        self,
        obstacle: Obstacle,
        gb_wpnts: List[Wpnt],
        apex_idx: int,
        post_dist: float,
        wpnt_dist: float,
    ) -> Tuple[float, float]:
        """
        Free width left and right of the obstacle, over the whole corridor the path must fit
        through. f1tenth Frenet: +d is LEFT, -d is RIGHT. gb_wp.d_left / d_right are POSITIVE
        wall distances, so the walls sit at signed +d_left and -d_right.

        Looking only at the obstacle waypoint would pick a side that becomes narrow right after
        the obstacle, so the whole return corridor is scanned.
        """
        horizon_steps = max(1, int(np.ceil(post_dist / wpnt_dist)))
        corridor = [gb_wpnts[(apex_idx + i) % len(gb_wpnts)] for i in range(horizon_steps + 1)]
        min_left_width = min(wp.d_left for wp in corridor)
        min_right_width = min(wp.d_right for wp in corridor)

        obs_radius = obstacle.size / 2
        left_gap = min_left_width - (obstacle.d_center + obs_radius)
        right_gap = (obstacle.d_center - obs_radius) + min_right_width
        return left_gap, right_gap

    def _preferred_side(self, obstacle: Obstacle, left_gap: float, right_gap: float) -> str:
        """
        Side to try first. Keep one side for the lifetime of a tracked obstacle: on a nearly
        symmetric straight, centimetre-level d noise otherwise flips the path left/right every
        planner tick. The latch is overridden only when it became infeasible and the other side
        did not.
        """
        min_space = self.evasion_dist + self.spline_bound_mindist
        left_ok = left_gap >= min_space
        right_ok = right_gap >= min_space

        if self.last_ot_obstacle_id == obstacle.id and self.last_ot_side in ("left", "right"):
            side = self.last_ot_side
            if side == "left" and not left_ok and right_ok:
                side = "right"
            elif side == "right" and not right_ok and left_ok:
                side = "left"
            return side

        if left_ok and not right_ok:
            return "left"
        if right_ok and not left_ok:
            return "right"
        return "left" if left_gap >= right_gap else "right"

    def _apex_d(self, obstacle: Obstacle, gb_wp: Wpnt, side: str, evasion_dist: float) -> float:
        """Lateral apex offset for a given side. Clamped to the track walls."""
        obs_radius = obstacle.size / 2
        if side == "left":
            d_apex = (obstacle.d_center + obs_radius) + evasion_dist
            d_apex = max(d_apex, 0.0)              # never flip across the raceline to the wrong side
            return min(d_apex, gb_wp.d_left)       # clamp to the +d wall
        d_apex = (obstacle.d_center - obs_radius) - evasion_dist
        d_apex = min(d_apex, 0.0)
        return max(d_apex, -gb_wp.d_right)         # clamp to the -d wall

    def _clearance_ok(self, samples: np.ndarray, obs: Obstacle) -> bool:
        """
        Hard collision invariant, checked on the FINISHED path.

        The apex offset alone is not a guarantee: _apex_d clamps to the track walls, and a
        relaxed ladder rung plans tighter on purpose. This is the only place that actually
        verifies the car stays clear of the obstacle.

        The car's current pose is the first sample and is a fact, not a decision -- no path can
        undo it. So when the car is already inside the minimum clearance (it is passing the
        obstacle), the requirement becomes "do not get any closer than we already are".
        """
        obs_xy = np.asarray(self.converter.get_cartesian(
            np.array([obs.s_center]), np.array([obs.d_center]))).reshape(2)
        required = obs.size / 2 + MIN_EVASION_DIST_M
        clearances = np.linalg.norm(samples - obs_xy, axis=1)
        floor = min(required, float(clearances[0]))
        # 1 mm of numerical slack: the last ladder rung plans exactly at the floor, and sample
        # discretisation would otherwise reject it on rounding alone. Negligible against the
        # 25 mm safety margin already baked into MIN_EVASION_DIST_M.
        return float(np.min(clearances)) >= floor - 1e-3

    def _build_path(
        self,
        obs: Obstacle,
        gb_wpnts: List[Wpnt],
        apex_s: float,
        side: str,
        evasion_dist: float,
        post_dist: float,
        wpnt_dist: float,
    ) -> Optional[Tuple[np.ndarray, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Build one candidate evasion path. Returns None if it is degenerate or fails either
        safety check, so the caller can move down the relaxation ladder.
        """
        apex_idx = int(apex_s / wpnt_dist) % self.gb_max_idx
        d_apex = self._apex_d(obs, gb_wpnts[apex_idx], side, evasion_dist)

        # Apex, then a single point back on the global line: BPoly with the global tangents
        # already shapes the return, extra intermediate points only add noise.
        s_array = np.array([apex_s, apex_s + post_dist]) % self.gb_max_s
        d_array = np.array([d_apex, 0.0])
        s_idx = np.round(s_array / wpnt_dist).astype(int) % self.gb_max_idx

        resp = self.converter.get_cartesian(s_array, d_array)

        points = [[self.cur_x, self.cur_y]]
        tangents = [[np.cos(self.cur_yaw), np.sin(self.cur_yaw)]]
        for i in range(len(s_idx)):
            points.append(resp[:, i])
            tangents.append(np.array([np.cos(gb_wpnts[s_idx[i]].psi_rad),
                                      np.sin(gb_wpnts[s_idx[i]].psi_rad)]))

        tangents = np.dot(tangents, self.spline_scale * np.eye(2))
        points = np.asarray(points)
        nPoints, dim = points.shape

        # Parametrize by cumulative chord length.
        dp = np.linalg.norm(np.diff(points, axis=0), axis=1)
        d = np.hstack([[0], np.cumsum(dp)])
        length = d[-1]
        nSamples = int(length / wpnt_dist)
        if nSamples < MIN_SAMPLES:
            return None
        s = np.linspace(0, length, nSamples)

        spline_input = np.empty([nPoints, dim], dtype=object)
        for i, ref in enumerate(points):
            spline_input[i, :] = list(zip(ref, tangents[i]))

        samples = np.zeros([nSamples, dim])
        for i in range(dim):
            samples[:, i] = BPoly.from_derivatives(d, spline_input[:, i])(s)

        # The BPoly is already C1 and built from explicit tangents, so it needs no smoothing.
        # It is published exactly as checked below -- nothing may move the apex after this point.
        generated_count = len(samples)

        # Append the global line ahead so the state machine can measure gaps past the manoeuvre.
        xy_tail = np.array([
            (gb_wpnts[(s_idx[-1] + i + 1) % self.gb_max_idx].x_m,
             gb_wpnts[(s_idx[-1] + i + 1) % self.gb_max_idx].y_m)
            for i in range(GB_TAIL_POINTS)
        ])
        samples = np.vstack([samples, xy_tail])

        # Validate only what this planner generated. The appended tail is the already accepted
        # global path; rejecting the whole manoeuvre over one eroded map pixel 10 m downstream
        # made static avoidance fail at random.
        for i in range(generated_count):
            if not self.map_filter.is_point_inside(samples[i, 0], samples[i, 1]):
                return None
        if not self._clearance_ok(samples[:generated_count], obs):
            return None

        s_, d_ = self.converter.get_frenet(samples[:, 0], samples[:, 1])
        psi_, kappa_ = tph.calc_head_curv_num.calc_head_curv_num(
            path=samples,
            el_lengths=wpnt_dist * np.ones(len(samples) - 1),
            is_closed=False,
        )
        return samples, generated_count, s_, d_, psi_, kappa_

    def do_spline(self, obs: Obstacle, gb_wpnts: List[Wpnt]) -> Tuple[OTWpntArray, MarkerArray]:
        """
        Creates an evasion trajectory for a static obstacle, splining from the current pose
        around the obstacle and back onto the global line.
        """
        mrks = MarkerArray()
        wpnts = OTWpntArray()
        wpnts.header.stamp = self.get_clock().now().to_msg()
        wpnts.header.frame_id = "map"

        if not obs.is_static:
            return wpnts, mrks

        wpnt_dist = gb_wpnts[1].s_m - gb_wpnts[0].s_m
        pre_dist = (obs.s_center - self.cur_s) % self.gb_max_s
        if pre_dist > self.gb_max_s / 2:
            return wpnts, mrks  # obstacle is behind us

        # Keep planning all the way past the obstacle. The apex is pushed ahead of the car once
        # we are alongside the obstacle instead of dropping the path: publishing an empty array
        # here makes the state machine leave OVERTAKE mid-manoeuvre and snap the car back onto
        # the raceline -- straight into the obstacle it was avoiding.
        apex_s = obs.s_center if pre_dist >= MIN_APEX_LEAD_M else self.cur_s + MIN_APEX_LEAD_M
        apex_s %= self.gb_max_s
        lead = max(pre_dist, MIN_APEX_LEAD_M)

        apex_idx = int(apex_s / wpnt_dist) % self.gb_max_idx
        base_post_dist = min(min(max(lead, self.post_min_dist), self.post_max_dist),
                             self.gb_max_s / 2)

        left_gap, right_gap = self._side_gaps(obs, gb_wpnts, apex_idx, base_post_dist, wpnt_dist)
        preferred = self._preferred_side(obs, left_gap, right_gap)
        wider = "left" if left_gap >= right_gap else "right"
        sides = {"preferred": preferred, "opposite": _opposite(preferred), "wider": wider}

        tried = set()
        for rung, (evasion_scale, post_scale, side_key) in enumerate(RELAXATION_LADDER):
            side = sides[side_key]
            evasion_dist = max(evasion_scale * self.evasion_dist, MIN_EVASION_DIST_M)
            post_dist = min(base_post_dist * post_scale, self.gb_max_s / 2)

            key = (side, round(evasion_dist, 3), round(post_dist, 3))
            if key in tried:
                continue
            tried.add(key)

            built = self._build_path(obs, gb_wpnts, apex_s, side, evasion_dist,
                                     post_dist, wpnt_dist)
            if built is None:
                continue

            samples, generated_count, s_, d_, psi_, kappa_ = built
            if rung > 0:
                self.get_logger().info(
                    f"[{self.name}]: evasion found on ladder rung {rung} "
                    f"(side={side}, clearance={evasion_dist:.2f} m, return={post_dist:.1f} m)",
                    throttle_duration_sec=2,
                )

            # Latch only a direction that produced a valid path, so a failed direction cannot
            # trap the same obstacle in repeated retries.
            self.last_ot_obstacle_id = obs.id
            self.last_ot_side = side

            for i in range(samples.shape[0]):
                gb_wpnt_i = int((s_[i] / wpnt_dist) % self.gb_max_idx)
                wpnts.wpnts.append(self.xyv_to_wpnts(
                    x=samples[i, 0], y=samples[i, 1], s=s_[i], d=d_[i],
                    v=2, psi=psi_[i] + np.pi / 2, kappa=kappa_[i], wpnts=wpnts))
                mrks.markers.append(self.xyv_to_markers(
                    x=samples[i, 0], y=samples[i, 1],
                    v=gb_wpnts[gb_wpnt_i].vx_mps, mrks=mrks))
            return wpnts, mrks

        # Every rung failed: the gap is not passable at the minimum safe clearance. Publishing
        # nothing is deliberate -- the controller falls back to crawling towards the obstacle,
        # which changes the approach pose and lets the next tick try again.
        self.get_logger().warning(
            f"[{self.name}]: no evasion path passes the safety checks (obstacle {obs.id}, "
            f"gaps L/R {left_gap:.2f}/{right_gap:.2f} m)",
            throttle_duration_sec=2,
        )
        return wpnts, mrks

    ######################
    # VIZ + MSG WRAPPING #
    ######################
    def xyv_to_markers(self, x: float, y: float, v: float, mrks: MarkerArray) -> Marker:
        mrk = Marker()
        mrk.header.frame_id = "map"
        mrk.header.stamp = self.get_clock().now().to_msg()
        mrk.type = mrk.CYLINDER
        mrk.scale.x = 0.1
        mrk.scale.y = 0.1
        mrk.scale.z = float(v / self.gb_vmax)
        mrk.color.a = 1.0
        mrk.color.b = 0.75
        mrk.color.r = 0.75

        mrk.id = len(mrks.markers)
        mrk.pose.position.x = float(x)
        mrk.pose.position.y = float(y)
        mrk.pose.position.z = float(v / self.gb_vmax / 2)
        mrk.pose.orientation.w = 1.0

        return mrk

    def xyv_to_wpnts(self, s: float, d: float, x: float, y: float, v: float, psi: float,
                     kappa: float, wpnts: WpntArray) -> Wpnt:
        wpnt = Wpnt()
        wpnt.id = len(wpnts.wpnts)
        wpnt.x_m = float(x)
        wpnt.y_m = float(y)
        wpnt.s_m = float(s)
        wpnt.d_m = float(d)
        wpnt.vx_mps = float(v)
        wpnt.psi_rad = float(psi)
        wpnt.kappa_radpm = float(kappa)

        return wpnt


def main(args=None):
    rclpy.init(args=args)
    spliner = ObstacleSpliner()
    try:
        rclpy.spin(spliner)
    except KeyboardInterrupt:
        pass
    spliner.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
