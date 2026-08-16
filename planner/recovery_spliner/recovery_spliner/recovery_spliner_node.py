#!/usr/bin/env python3
import datetime
import os
import time
from pathlib import Path
from typing import List, Any, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rcl_interfaces.msg import (
    FloatingPointRange,
    IntegerRange,
    ParameterDescriptor,
    SetParametersResult,
    ParameterType,
)
from rclpy.parameter import Parameter

import numpy as np
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker, MarkerArray
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from scipy.interpolate import BPoly
from scipy.signal import argrelextrema
from f110_msgs.msg import Obstacle, ObstacleArray, OTWpntArray, Wpnt, WpntArray
from frenet_conversion.frenet_converter import FrenetConverter
from transforms3d.euler import quat2euler
from grid_filter.grid_filter import GridFilter
import trajectory_planning_helpers as tph


def _default_debug_log_dir() -> Path:
    """<repo>/logfile if this source file lives inside a git checkout, so debug logs
    land somewhere `git add logfile/` already picks up instead of needing a manual
    copy out of the home directory after every run. Falls back to a fixed
    home-directory path otherwise. Kept identical to the helpers of the same name in
    state_machine_node.py and static_avoidance_node.py -- separate packages, no
    shared import between them.
    """
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / ".git").exists():
            return parent / "logfile"
    return Path.home() / "racing_logs"


class ObstacleSpliner(Node):
    """
    This class implements a ROS node that performs splining around obstacles.

    It subscribes to the following topics:
        - `/tracking/obstacles`: Subscribes to the obstacle array.
        - `/car_state/odom_frenet`: Subscribes to the car state in Frenet coordinates.
        - `/global_waypoints`: Subscribes to global waypoints.
        - `/global_waypoints_scaled`: Subscribes to the scaled global waypoints.

    The node publishes the following topics:
        - `/planner/avoidance/markers`: Publishes spline markers.
        - `/planner/avoidance/otwpnts`: Publishes splined waypoints.
        - `/planner/avoidance/considered_OBS`: Publishes markers for the closest obstacle.
        - `/planner/avoidance/propagated_obs`: Publishes markers for the propagated obstacle.
        - `/planner/avoidance/latency`: Publishes the latency of the spliner node. (only if measuring is enabled)
    """

    def __init__(self):
        """
        Initialize the node, subscribe to topics, and create publishers and service proxies.
        """
        # Initialize the node
        self.name = "recovery_spliner_node"
        super().__init__(self.name)
        self._setup_debug_log_file()
        # 1 Hz perf summary accumulators (see _flush_perf). Kept as plain counters so
        # the 40 Hz loop only does integer adds; the formatting happens once a second.
        self._perf_window_start = time.perf_counter()
        self._perf_cpu_ticks = self._read_self_cpu_ticks()
        self._perf_loops = 0
        self._perf_splines = 0
        self._perf_blended = 0
        self._perf_samples = 0
        self._perf_danger = 0
        self._perf_marker_pubs = 0
        self._perf_loop_s_sum = 0.0
        self._perf_loop_s_max = 0.0

        # initialize the instance variable
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
        self.waypoints = None

        self.gb_scaled_wpnts = None
        self.inflection_points = None
        self.ot_wpnts = None
        self.ot_wpnts_stamp = None

        # static params
        self.declare_parameters(
            namespace='',
            parameters=[
                ('from_bag', False),
                ('measure', False),
                ('n_loc_wpnts', 80),
                ('blend_start_dist_m', 1.0),
                ('blend_hold_max_sec', 0.5),
            ])
        self.from_bag = self.get_parameter(
            'from_bag').get_parameter_value().bool_value
        self.measuring = self.get_parameter(
            'measure').get_parameter_value().bool_value
        self.n_loc_wpnts = self.get_parameter(
            'n_loc_wpnts').get_parameter_value().integer_value
        self.blend_start_dist_m = self.get_parameter(
            'blend_start_dist_m').get_parameter_value().double_value
        self.blend_hold_max_sec = self.get_parameter(
            'blend_hold_max_sec').get_parameter_value().double_value

        # dynamic (tunable) params - folded from dynamic_reconfigure cfg
        self.save_params = False
        self.min_candidates_lookahead_n = 20
        self.num_kappas = 20
        self.spline_scale = 0.8
        self.kernel_size = 5
        self.smooth_len = 1.0
        # Per-sample bounds-check viz on /planner/recovery/spline_samples. One Marker
        # per spline sample, rebuilt and published on every do_spline() call -- at
        # 40 Hz, and twice per loop whenever the OT-blended path also splines, that is
        # thousands of Marker objects a second built and serialised in Python for a
        # topic only RViz reads. Off by default; flip it live while debugging.
        self.publish_spline_samples = False
        # Rate for the RViz marker arrays only. The planner still replans and
        # publishes /planner/recovery/wpnts -- the data the state machine actually
        # consumes -- at the full loop rate; this decimates the visualisation, which
        # profiling showed to be the single largest cost in the loop (one CYLINDER
        # Marker per waypoint, ~90 of them, rebuilt 40x a second for a topic no node
        # subscribes to). 0 disables the marker arrays entirely.
        self.marker_rate_hz = 5.0
        # -inf so the very first loop always builds markers.
        self._last_marker_build = float("-inf")

        int_lookahead_pd = ParameterDescriptor(
            type=ParameterType.PARAMETER_INTEGER,
            integer_range=[IntegerRange(from_value=10, to_value=100, step=1)]
        )
        int_num_kappas_pd = ParameterDescriptor(
            type=ParameterType.PARAMETER_INTEGER,
            integer_range=[IntegerRange(from_value=10, to_value=50, step=1)]
        )
        int_kernel_pd = ParameterDescriptor(
            type=ParameterType.PARAMETER_INTEGER,
            integer_range=[IntegerRange(from_value=1, to_value=20, step=1)]
        )
        double_spline_scale_pd = ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE,
            floating_point_range=[FloatingPointRange(from_value=0.5, to_value=2.0, step=0.001)]
        )
        double_smooth_len_pd = ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE,
            floating_point_range=[FloatingPointRange(from_value=0.0, to_value=3.0, step=0.001)]
        )
        double_marker_rate_pd = ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE,
            floating_point_range=[FloatingPointRange(from_value=0.0, to_value=40.0, step=0.5)]
        )
        bool_pd = ParameterDescriptor(type=ParameterType.PARAMETER_BOOL)

        param_dicts = [
            {'name': 'save_params', 'default': self.save_params, 'descriptor': bool_pd},
            {'name': 'min_candidates_lookahead_n', 'default': self.min_candidates_lookahead_n, 'descriptor': int_lookahead_pd},
            {'name': 'num_kappas', 'default': self.num_kappas, 'descriptor': int_num_kappas_pd},
            {'name': 'spline_scale', 'default': self.spline_scale, 'descriptor': double_spline_scale_pd},
            {'name': 'kernel_size', 'default': self.kernel_size, 'descriptor': int_kernel_pd},
            {'name': 'smooth_len', 'default': self.smooth_len, 'descriptor': double_smooth_len_pd},
            {'name': 'publish_spline_samples', 'default': self.publish_spline_samples, 'descriptor': bool_pd},
            {'name': 'marker_rate_hz', 'default': self.marker_rate_hz, 'descriptor': double_marker_rate_pd},
        ]
        self.declare_all_parameters(param_dicts=param_dicts)

        # Subscribe to the topics
        self.create_subscription(
            Odometry, "/car_state/odom_frenet", self.state_frenet_cb, QoSProfile(depth=10))
        self.create_subscription(
            Odometry, "/car_state/odom", self.state_cb, QoSProfile(depth=10))
        self.create_subscription(
            WpntArray, "/global_waypoints", self.gb_cb, QoSProfile(depth=10))
        self.create_subscription(
            WpntArray, "/global_waypoints_scaled", self.gb_scaled_cb, QoSProfile(depth=10))
        # OT trajectory, used to seed the blended-recovery path (first ~1 m of OT).
        self.create_subscription(
            OTWpntArray, "/planner/avoidance/otwpnts", self.ot_wpnts_cb, QoSProfile(depth=10))

        self.mrks_pub = self.create_publisher(
            MarkerArray, "/planner/recovery/markers", QoSProfile(depth=10))
        self.recovery_wpnts_pub = self.create_publisher(
            WpntArray, "/planner/recovery/wpnts", QoSProfile(depth=10))
        # OT-blended recovery: OT heading kept for the first ~1 m, then splined to GB.
        self.ot_blended_wpnts_pub = self.create_publisher(
            WpntArray, "/planner/ot_blended_recovery/wpnts", QoSProfile(depth=10))
        self.ot_blended_mrks_pub = self.create_publisher(
            MarkerArray, "/planner/ot_blended_recovery/markers", QoSProfile(depth=10))
        self.recovery_lookahead_pub = self.create_publisher(
            Marker, "/planner/recovery/lookahead_point", QoSProfile(depth=10))
        # Debug: per-sample spline bounds-check viz (green=pass, red=fail, blue=unchecked tail)
        self.spline_samples_pub = self.create_publisher(
            MarkerArray, "/planner/recovery/spline_samples", QoSProfile(depth=10))

        if self.measuring:
            self.latency_pub = self.create_publisher(
                Float32, "/planner/recovery/latency", QoSProfile(depth=10))
            self.checkpoints_pub = self.create_publisher(
                MarkerArray, "/planner/recovery/checkpoints", QoSProfile(depth=10))

        # Register dynamic params callback after publishers exist (checkpoints may publish in gb_cb)
        self.add_on_set_parameters_callback(self.dyn_param_cb)

        # Wait for critical messages
        self.wait_for_messages()

        self.converter = self.initialize_converter()
        self.map_filter = GridFilter(node=self, map_topic="/map", debug=False)
        self.map_filter.set_erosion_kernel_size(self.kernel_size)

        # Main loop at 40 Hz
        self.create_timer(1.0 / 40.0, self.loop)
        self.get_logger().info(f"[{self.name}] Ready!")

    #################### DYNAMIC PARAMS ####################
    def declare_all_parameters(self, param_dicts: List[dict]):
        params = []
        for param_dict in param_dicts:
            param = self.declare_parameter(
                param_dict['name'], param_dict['default'], param_dict['descriptor'])
            params.append(param)
        return params

    #############
    # CALLBACKS #
    #############
    def state_frenet_cb(self, data: Odometry):
        self.cur_s = data.pose.pose.position.x
        self.cur_d = data.pose.pose.position.y
        self.cur_vs = data.twist.twist.linear.x

    def state_cb(self, data: Odometry):
        self.cur_x = data.pose.pose.position.x
        self.cur_y = data.pose.pose.position.y
        quat = data.pose.pose.orientation
        # transforms3d expects (w, x, y, z)
        euler = quat2euler([quat.w, quat.x, quat.y, quat.z])
        self.cur_yaw = euler[2]

    def ot_wpnts_cb(self, data: OTWpntArray):
        # Keep the last VALID OT trajectory. Empty publishes (planner produced nothing
        # this frame) must not wipe it: the blended path keeps re-anchoring off this
        # stored OT while the planner is silent, instead of collapsing to empty.
        if len(data.wpnts) >= 2:
            self.ot_wpnts = data
            self.ot_wpnts_stamp = self.get_clock().now().nanoseconds * 1e-9

    def _ot_blend_seed(self):
        """Blend seed: the anchor pose ~blend_start_dist_m ahead of the car along the
        current OT trajectory, PLUS the raw OT segment from the car's nearest point up
        to the anchor. The head segment is prepended so the blended path starts under
        the car (near_idx), not 1 m ahead at the anchor -- otherwise on_spline's
        min_dist check rejects it. Returns (seed, head_xy) or None (-> empty blended)."""
        ot = self.ot_wpnts
        if ot is None or len(ot.wpnts) < 2:
            return None
        # Drop a too-old OT: re-anchoring off a stale trajectory would steer around
        # obstacles that have since moved. Past the hold window, fall back to plain
        # recovery (empty blended -> state machine uses recovery/GB).
        if self.ot_wpnts_stamp is not None and (
            self.get_clock().now().nanoseconds * 1e-9 - self.ot_wpnts_stamp
        ) > self.blend_hold_max_sec:
            return None
        xy = np.array([(w.x_m, w.y_m) for w in ot.wpnts])
        diff = np.linalg.norm(xy - np.array([self.cur_x, self.cur_y]), axis=1)
        near = int(np.argmin(diff))
        # Walk forward from the nearest OT point until blend_start_dist_m of arclength.
        acc = 0.0
        i = near
        while i + 1 < len(ot.wpnts) and acc < self.blend_start_dist_m:
            acc += float(np.linalg.norm(xy[i + 1] - xy[i]))
            i += 1
        # Car reached the end of the held OT path before a full 1 m of anchor lookahead
        # remained: no usable anchor -> emit empty (state machine falls back).
        if acc < self.blend_start_dist_m:
            return None
        w = ot.wpnts[i]
        wp = ot.wpnts[i - 1] if i > 0 else ot.wpnts[i]
        seed_yaw = float(np.arctan2(w.y_m - wp.y_m, w.x_m - wp.x_m))
        seed = (float(w.x_m), float(w.y_m), seed_yaw, float(w.s_m), float(w.d_m))
        head_xy = xy[near:i]  # OT points from under the car up to (not incl.) the anchor
        return seed, head_xy

    # Callback for global waypoint topic
    def gb_cb(self, data: WpntArray):

        self.waypoints = np.array([[wpnt.x_m, wpnt.y_m] for wpnt in data.wpnts])
        self.gb_wpnts = data
        if self.gb_vmax is None:
            self.gb_vmax = np.max(np.array([wpnt.vx_mps for wpnt in data.wpnts]))
            self.gb_max_idx = data.wpnts[-1].id
            self.gb_max_s = data.wpnts[-1].s_m

        # psi_array = np.array([wpnt.psi_rad for wpnt in data.wpnts])
        kappas = np.array([wpnt.kappa_radpm for wpnt in data.wpnts])

        sign_changes = np.sign(kappas)

        self.inflection_points = np.where(np.diff(sign_changes) != 0)[0]

        # max_indices = argrelextrema(psi_array, np.greater)[0]
        # min_indices = argrelextrema(psi_array, np.less)[0]

        if self.measuring:
            mrks = MarkerArray()
            for idx in self.inflection_points:
                # print(idx)
                mrk = Marker()
                mrk.header.frame_id = "map"
                mrk.header.stamp = self.get_clock().now().to_msg()
                mrk.type = mrk.CYLINDER
                mrk.scale.x = 0.3
                mrk.scale.y = 0.3
                mrk.scale.z = 0.3
                mrk.color.a = 1.0
                mrk.color.b = 0.75
                mrk.color.r = 0.75
                mrk.id = int(idx)
                mrk.pose.position.x = data.wpnts[idx].x_m
                mrk.pose.position.y = data.wpnts[idx].y_m
                mrk.pose.position.z = 0.0
                mrk.pose.orientation.w = 1.0
                mrks.markers.append(mrk)
            self.checkpoints_pub.publish(mrks)

    # Callback for scaled global waypoint topic
    def gb_scaled_cb(self, data: WpntArray):
        self.gb_scaled_wpnts = data

    # Callback triggered by dynamic spline reconf
    def dyn_param_cb(self, params: List[Parameter]):
        """
        Notices the change in the parameters and changes spline params
        """
        for param in params:
            param_name = param.name
            if param_name == 'save_params':
                self.save_params = param.value
            elif param_name == 'min_candidates_lookahead_n':
                self.min_candidates_lookahead_n = param.value
            elif param_name == 'num_kappas':
                self.num_kappas = param.value
            elif param_name == 'spline_scale':
                self.spline_scale = param.value
            elif param_name == 'kernel_size':
                self.kernel_size = param.value
            elif param_name == 'smooth_len':
                self.smooth_len = param.value
            elif param_name == 'publish_spline_samples':
                self.publish_spline_samples = param.value
            elif param_name == 'marker_rate_hz':
                self.marker_rate_hz = param.value

        if hasattr(self, 'map_filter'):
            self.map_filter.set_erosion_kernel_size(self.kernel_size)

        self.get_logger().info(
            f"[{self.name}] Dynamic reconf triggered new params: "
            f"min_candidates_lookahead_n: {self.min_candidates_lookahead_n}, "
            f"num_kappas: {self.num_kappas}, spline_scale: {self.spline_scale}, "
            f"kernel_size: {self.kernel_size}, smooth_len: {self.smooth_len}, "
            f"publish_spline_samples: {self.publish_spline_samples}"
        )
        self._dbg_log(f"[PARAM] publish_spline_samples={self.publish_spline_samples} "
                      f"marker_rate_hz={self.marker_rate_hz} "
                      f"kernel_size={self.kernel_size} spline_scale={self.spline_scale}")
        return SetParametersResult(successful=True)

    #############
    # MAIN LOOP #
    #############
    def loop(self):
        loop_start = time.perf_counter()
        if self.measuring:
            start = loop_start
        # Sample data
        gb_scaled_wpnts = self.gb_scaled_wpnts.wpnts

        # RViz marker arrays run at marker_rate_hz, the waypoints at the full loop
        # rate. Nothing subscribes to the marker topics in code -- only pitwall.rviz --
        # so decimating them costs nothing but visual refresh rate, while the path the
        # state machine follows keeps updating every tick.
        marker_period = 1.0 / self.marker_rate_hz if self.marker_rate_hz > 0 else float("inf")
        build_markers = (loop_start - self._last_marker_build) >= marker_period
        if build_markers:
            self._last_marker_build = loop_start
            self._perf_marker_pubs += 1

        wpnts, mrks = self.do_spline(gb_wpnts=gb_scaled_wpnts, build_markers=build_markers)

        # Publish wpnts and markers
        if self.measuring:
            end = time.perf_counter()
            latency_msg = Float32()
            latency_msg.data = float(1 / (end - start))
            self.latency_pub.publish(latency_msg)
        self.recovery_wpnts_pub.publish(wpnts)

        # Prepend DELETEALL so stale markers from the previous frame are cleared, then
        # publish once (also clears them when do_spline returns no markers). Skipped
        # entirely on decimated ticks rather than publishing an empty array, which
        # would blank the display between refreshes instead of holding the last frame.
        if build_markers:
            del_mrk = Marker()
            del_mrk.header.stamp = self.get_clock().now().to_msg()
            del_mrk.action = Marker.DELETEALL
            mrks.markers.insert(0, del_mrk)
            self.mrks_pub.publish(mrks)

        # OT-blended recovery: keep the first ~1 m of the OT path, then spline to GB.
        # Published every loop; empty when no OT path exists so the state machine
        # transparently falls back to the plain recovery path.
        blended = WpntArray()
        blended_mrks = MarkerArray()
        seed_info = self._ot_blend_seed()
        if seed_info is not None:
            seed, head_xy = seed_info
            self._perf_blended += 1
            blended, blended_mrks = self.do_spline(
                gb_wpnts=gb_scaled_wpnts, seed=seed, head_xy=head_xy,
                build_markers=build_markers)
        blended.header.stamp = self.get_clock().now().to_msg()
        blended.header.frame_id = "map"
        self.ot_blended_wpnts_pub.publish(blended)

        if build_markers:
            blended_del = Marker()
            blended_del.header.stamp = self.get_clock().now().to_msg()
            blended_del.action = Marker.DELETEALL
            blended_mrks.markers.insert(0, blended_del)
            self.ot_blended_mrks_pub.publish(blended_mrks)

        elapsed = time.perf_counter() - loop_start
        self._perf_loops += 1
        self._perf_loop_s_sum += elapsed
        self._perf_loop_s_max = max(self._perf_loop_s_max, elapsed)
        self._flush_perf()

    ########################
    # DEBUG LOG + PERF     #
    ########################
    def _setup_debug_log_file(self):
        """Per-run plain-text perf log for this node -- how long the 40 Hz loop
        actually takes, how many splines it ran, and this process's own CPU share.

        Written unconditionally so the numbers exist without anyone remembering to
        start a separate measurement tool: a profiler you have to launch by hand is a
        profiler that does not get launched. Same convention as state_machine's and
        static_avoidance_node's helpers of the same name -- one file per run plus a
        "latest" symlink, directory overridable via RACE_DEBUG_LOG_DIR. Never allowed
        to crash the node: a read-only home just disables file logging.
        """
        self._dbg_fh = None
        try:
            log_dir = Path(os.environ.get("RACE_DEBUG_LOG_DIR", str(_default_debug_log_dir()))).expanduser()
            log_dir.mkdir(parents=True, exist_ok=True)
            run_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = log_dir / f"recovery_spliner_{run_stamp}.log"
            self._dbg_fh = open(log_path, "a", buffering=1)
            latest = log_dir / "latest_recovery_spliner.log"
            if latest.is_symlink() or latest.exists():
                latest.unlink()
            latest.symlink_to(log_path.name)
            self._dbg_fh.write(f"# recovery_spliner perf log started {run_stamp}\n")
            self._dbg_fh.write("# [PERF] fields: loops=timer ticks in the window, "
                               "hz=achieved loop rate, loop_ms avg/max=wall time inside loop(), "
                               "splines=do_spline calls, blended=loops that also splined the "
                               "OT-blended path, samples=mean spline samples bounds-checked, "
                               "danger=splines aborted on the bounds check, mrk_hz=achieved "
                               "RViz marker rate (see marker_rate_hz; the wpnt topics stay at "
                               "the full loop rate), cpu=this process's own CPU over the window "
                               "(100% = one core), viz=spline-sample marker publishing\n")
            self.get_logger().info(f"[{self.name}] perf log: {log_path} (latest: {latest})")
        except OSError as e:
            self.get_logger().warn(f"[{self.name}] could not open debug log file: {e}")

    def _dbg_log(self, msg: str) -> None:
        if getattr(self, "_dbg_fh", None) is None:
            return
        try:
            now = self.get_clock().now().nanoseconds * 1e-9
            self._dbg_fh.write(f"{now:.3f} {msg}\n")
        except OSError:
            pass

    @staticmethod
    def _read_self_cpu_ticks():
        """utime+stime for this process in clock ticks, or None if unreadable.

        Read straight from /proc rather than shelling out, so the 1 Hz summary can
        carry the node's own CPU without a sysstat dependency or a second process.
        """
        try:
            with open("/proc/self/stat") as fh:
                raw = fh.read()
            # Drop "pid (comm) " first: comm may itself contain spaces or parens,
            # which would otherwise shift every field index after it.
            raw = raw[raw.rfind(") ") + 2:].split()
            return int(raw[11]) + int(raw[12])       # utime + stime
        except (OSError, ValueError, IndexError):
            return None

    def _flush_perf(self, period_s: float = 1.0) -> None:
        """Emit one [PERF] line per period_s. Aggregating rather than logging every
        tick keeps this from becoming its own 40 Hz load -- the thing being measured.
        """
        now = time.perf_counter()
        window = now - self._perf_window_start
        if window < period_s:
            return

        cpu = "n/a"
        ticks = self._read_self_cpu_ticks()
        if ticks is not None and self._perf_cpu_ticks is not None:
            clk = os.sysconf("SC_CLK_TCK") or 100
            cpu = f"{(ticks - self._perf_cpu_ticks) / clk / window * 100:.1f}%"
        self._perf_cpu_ticks = ticks

        loops = self._perf_loops
        self._dbg_log(
            f"[PERF] loops={loops} hz={loops / window:.1f} "
            f"loop_ms avg={self._perf_loop_s_sum / max(loops, 1) * 1e3:.2f} "
            f"max={self._perf_loop_s_max * 1e3:.2f} "
            f"splines={self._perf_splines} blended={self._perf_blended} "
            f"samples={self._perf_samples / max(self._perf_splines, 1):.1f} "
            f"danger={self._perf_danger} mrk_hz={self._perf_marker_pubs / window:.1f} "
            f"cpu={cpu} viz={'on' if self.publish_spline_samples else 'off'}"
        )

        self._perf_window_start = now
        self._perf_loops = 0
        self._perf_splines = 0
        self._perf_blended = 0
        self._perf_samples = 0
        self._perf_danger = 0
        self._perf_marker_pubs = 0
        self._perf_loop_s_sum = 0.0
        self._perf_loop_s_max = 0.0

    #########
    # UTILS #
    #########
    def wait_for_messages(self):
        self.get_logger().info(f"[{self.name}] Waiting for messages and services...")
        waitlist = [self.gb_wpnts, self.gb_scaled_wpnts, self.cur_x, self.cur_s]
        while None in waitlist:
            rclpy.spin_once(self)
            waitlist = [self.gb_wpnts, self.gb_scaled_wpnts, self.cur_x, self.cur_s]
        self.get_logger().info(f"[{self.name}] All required messages received.")

    def initialize_converter(self) -> FrenetConverter:
        """
        Initialize the FrenetConverter object"""
        # Initialize the FrenetConverter object
        converter = FrenetConverter(self.waypoints[:, 0], self.waypoints[:, 1])
        self.get_logger().info(f"[{self.name}] initialized FrenetConverter object")

        return converter

    def find_tangent_idx(self, xy_m, psi_rads, start_x=None, start_y=None, start_yaw=None):
        # Start from the given seed pose (blended path) or the car pose (recovery).
        cur_x = self.cur_x if start_x is None else start_x
        cur_y = self.cur_y if start_y is None else start_y
        cur_yaw = self.cur_yaw if start_yaw is None else start_yaw
        smooth = np.cos(cur_yaw), np.sin(cur_yaw) * self.smooth_len

        # Compute direction vectors from the current position to waypoints
        dx = xy_m[:, 0] - (cur_x + smooth[0])
        dy = xy_m[:, 1] - (cur_y + smooth[1])

        # Normalize vectors
        norm = np.sqrt(dx**2 + dy**2)  # Vector magnitude
        unit_vectors = np.vstack((dx / norm, dy / norm)).T  # (N, 2) unit vectors

        # Convert waypoint heading angles to unit vectors
        psi_unit_vectors = np.vstack((np.cos(psi_rads), np.sin(psi_rads))).T  # (N, 2)

        # Compute cosine similarity between vectors
        cos_theta = np.sum(unit_vectors * psi_unit_vectors, axis=1)  # Dot product
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Prevent numerical errors

        # Compute angular difference in radians
        angles = np.arccos(cos_theta)  # Range: 0 ~ π

        # Find the index with the smallest angle
        tangent_idx = np.argmin(angles)

        return tangent_idx

    def do_spline(self, gb_wpnts: WpntArray, seed=None, head_xy=None,
                  build_markers: bool = True) -> Tuple[WpntArray, MarkerArray]:
        """
        Creates an evasion trajectory for the closest obstacle by splining between pre- and post-apex points.

        seed: optional (x, y, yaw, s, d) tuple to start the spline from instead of the
        car pose. Used to build the "OT-blended recovery" path: the first ~1 m of the
        OT trajectory is kept and the rest is splined back onto the GB line, so the
        controller keeps the overtake heading while trailing instead of snapping to GB.
        When None the path starts from the current car pose (plain recovery).

        This function takes as input the obstacles to be evaded, and a list of global waypoints that describe a reference raceline.
        It only considers obstacles that are within a threshold of the raceline and generates an evasion trajectory for each of these obstacles.
        The evasion trajectory consists of a spline between pre- and post-apex points that are spaced apart from the obstacle.
        The spatial and velocity components of the spline are calculated using the `Spline` class, and the resulting waypoints and markers are returned.

        Args:
        - obstacles (ObstacleArray): An array of obstacle objects to be evaded.
        - gb_wpnts (WpntArray): A list of global waypoints that describe a reference raceline.
        - state (Odometry): The current state of the car.

        Returns:
        - wpnts (WpntArray): An array of waypoints that describe the evasion trajectory to the closest obstacle.
        - mrks (MarkerArray): An array of markers that represent the waypoints in a visualization format.

        """
        # Return wpnts and markers
        mrks = MarkerArray()
        wpnts = WpntArray()

        # Get spacing between wpnts for rough approximations
        wpnt_dist = gb_wpnts[1].s_m - gb_wpnts[0].s_m

        # Spline seed: car pose (recovery) or OT-forward point (blended).
        if seed is None:
            seed_x, seed_y, seed_yaw, seed_s, seed_d = \
                self.cur_x, self.cur_y, self.cur_yaw, self.cur_s, self.cur_d
        else:
            seed_x, seed_y, seed_yaw, seed_s, seed_d = seed

        cur_s = seed_s
        cur_d = seed_d
        cur_s_idx = int(cur_s / wpnt_dist)

        if len(self.inflection_points) != 0:
            infl_sector_idx = np.searchsorted(self.inflection_points, cur_s_idx)

            next_infl_sector_idx = self.inflection_points[(infl_sector_idx) % len(self.inflection_points)]

            candidate_len = next_infl_sector_idx - cur_s_idx + self.gb_max_idx if infl_sector_idx == len(self.inflection_points) \
                else next_infl_sector_idx - cur_s_idx
        else:
            candidate_len = int(self.gb_max_idx / 2)

        candidate_len = max(candidate_len, self.min_candidates_lookahead_n)

        gb_idxs = [(cur_s_idx + i) % self.gb_max_idx for i in range(candidate_len)]

        num_kappas_ = min(self.num_kappas, self.min_candidates_lookahead_n)
        kappas = np.array([gb_wpnts[gb_idx].kappa_radpm for gb_idx in gb_idxs[:num_kappas_]])

        outside = True if np.sum(kappas) * cur_d < 0 else False

        tangent_idx = 20
        # if outside: # TODO : need to consider outside and inside
        if True:
            xy_m = np.array([(gb_wpnts[gb_idx].x_m, gb_wpnts[gb_idx].y_m) for gb_idx in gb_idxs])
            psi_rads = np.array([gb_wpnts[gb_idx].psi_rad for gb_idx in gb_idxs])

            tangent_idx = self.find_tangent_idx(
                xy_m, psi_rads, start_x=seed_x, start_y=seed_y, start_yaw=seed_yaw)

            if self.measuring:
                mrk = Marker()
                mrk.header.frame_id = "map"
                mrk.header.stamp = self.get_clock().now().to_msg()
                mrk.type = mrk.SPHERE
                mrk.scale.x = 0.5
                mrk.scale.y = 0.5
                mrk.scale.z = 0.5
                mrk.color.a = 1.0
                mrk.color.b = 1.0
                mrk.color.r = 0.0
                mrk.color.g = 0.65

                mrk.pose.position.x = float(xy_m[tangent_idx, 0])
                mrk.pose.position.y = float(xy_m[tangent_idx, 1])
                mrk.pose.position.z = 0.01
                mrk.pose.orientation.w = 1.0
                self.recovery_lookahead_pub.publish(mrk)

            if tangent_idx != 0:
                target_s = tangent_idx * wpnt_dist

        points = []
        tangents = []

        points.append([seed_x, seed_y])
        points.append([xy_m[tangent_idx, 0], xy_m[tangent_idx, 1]])

        tangents.append(np.array([np.cos(seed_yaw), np.sin(seed_yaw)]))
        tangents.append(np.array([np.cos(psi_rads[tangent_idx]), np.sin(psi_rads[tangent_idx])]))

        tangents = np.dot(tangents, self.spline_scale * np.eye(2))
        points = np.asarray(points)
        nPoints, dim = points.shape

        # Parametrization parameter s.
        dp = np.diff(points, axis=0)                 # difference between points
        dp = np.linalg.norm(dp, axis=1)              # distance between points
        d = np.cumsum(dp)                            # cumsum along the segments
        d = np.hstack([[0], d])                      # add distance from first point
        l = d[-1]                                    # length of point sequence
        nSamples = int(l / wpnt_dist)                # number of samples
        s, r = np.linspace(0, l, nSamples, retstep=True)  # sample parameter and step

        # Bring points and (optional) tangent information into correct format.
        assert(len(points) == len(tangents))
        spline_result = np.empty([nPoints, dim], dtype=object)
        for i, ref in enumerate(points):
            t = tangents[i]
            # Either tangent is None or has the same
            # number of dimensions as the point ref.
            assert(t is None or len(t) == dim)
            fuse = list(zip(ref, t) if t is not None else zip(ref,))
            spline_result[i, :] = fuse

        # Compute splines per dimension separately.
        samples = np.zeros([nSamples, dim])
        for i in range(dim):
            poly = BPoly.from_derivatives(d, spline_result[:, i])
            samples[:, i] = poly(s)

        # if samples.shape[0] < self.n_loc_wpnts:
        n_additional = 80
        xy_additional = np.array([
            (
                gb_wpnts[(tangent_idx + cur_s_idx + i + 1) % self.gb_max_idx].x_m,
                gb_wpnts[(tangent_idx + cur_s_idx + i + 1) % self.gb_max_idx].y_m
            )
            for i in range(n_additional)
        ])

        # Prepend the raw OT head (car -> anchor) so the blended path starts under the
        # car, then the anchor->GB spline, then the GB tail.
        if head_xy is not None and len(head_xy) > 0:
            samples = np.vstack([head_xy, samples, xy_additional])
        else:
            samples = np.vstack([samples, xy_additional])

        s_, d_ = self.converter.get_frenet(samples[:, 0], samples[:, 1])

        psi_, kappa_ = tph.calc_head_curv_num.\
            calc_head_curv_num(
                path=samples,
                el_lengths=0.1 * np.ones(len(samples) - 1),
                is_closed=False
            )

        # Bounds-check every sample in one vectorised pass instead of a Python call
        # per sample. are_points_inside() is element-for-element identical to
        # is_point_inside(), so the accepted/rejected split below is unchanged; only
        # the cost moves from ~N interpreter round-trips to one numpy pass.
        n_samples = samples.shape[0]
        inside_all = self.map_filter.are_points_inside(samples[:, 0], samples[:, 1])
        failed = np.flatnonzero(~inside_all)
        # First sample that fails, or n_samples if all pass. The loop below stops
        # there, exactly where the previous per-sample `break` did.
        first_fail = int(failed[0]) if failed.size else n_samples
        danger_flag = first_fail < n_samples

        for i in range(first_fail):
            gb_wpnt_i = int((s_[i] / wpnt_dist) % self.gb_max_idx)
            # Get V from gb wpnts and go slower if we are going through the inside
            vi = gb_wpnts[gb_wpnt_i].vx_mps if outside else gb_wpnts[gb_wpnt_i].vx_mps * 0.9  # TODO make speed scaling ros param
            wpnts.wpnts.append(
                self.xyv_to_wpnts(x=samples[i, 0], y=samples[i, 1], s=s_[i], d=d_[i], v=vi, psi=psi_[i] + np.pi / 2, kappa=kappa_[i], wpnts=wpnts)
            )
            # Skipping the append, not just the publish: the cost being avoided is
            # Marker.__init__ itself, which profiling put at ~43% of the loop.
            if build_markers:
                mrk_rgb = (0.0, 0.0, 1.0) if seed is not None else None  # blue for OT-blended
                mrks.markers.append(self.xyv_to_markers(x=samples[i, 0], y=samples[i, 1], v=vi, mrks=mrks, rgb=mrk_rgb))

        if danger_flag:
            self.get_logger().info(
                f"[{self.name}]: Evasion trajectory too close to TRACKBOUNDS, aborting evasion",
                throttle_duration_sec=2,
            )

        # Debug viz, off by default -- see publish_spline_samples. The old per-sample
        # loop only recorded results up to and including the failing sample and left
        # the remainder as the "unchecked tail"; slicing here reproduces that.
        # Also tied to build_markers, so turning the debug viz on for a look does not
        # reintroduce a full-rate marker flood.
        if self.publish_spline_samples and build_markers:
            checked = inside_all[:first_fail + 1] if danger_flag else inside_all
            self._publish_spline_samples_markers(samples, checked.tolist())

        # Per-loop perf accounting; flushed as a 1 Hz summary by loop().
        self._perf_splines += 1
        self._perf_samples += n_samples
        if danger_flag:
            self._perf_danger += 1

        # Fill the rest of OTWpnts
        wpnts.header.stamp = self.get_clock().now().to_msg()
        wpnts.header.frame_id = "map"

        if danger_flag:
            wpnts.wpnts = []
            mrks.markers = []

        return wpnts, mrks

    def _publish_spline_samples_markers(self, samples: np.ndarray, bounds_check_results: List[bool]):
        """Debug viz: each spline sample as a sphere. green=passed bounds check,
        red=failed (the point that aborted evasion), blue=unchecked tail point."""
        markers = MarkerArray()
        del_mrk = Marker()
        del_mrk.header.frame_id = "map"
        del_mrk.action = Marker.DELETEALL
        markers.markers.append(del_mrk)
        for i in range(samples.shape[0]):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "spline_samples"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(samples[i, 0])
            marker.pose.position.y = float(samples[i, 1])
            marker.pose.position.z = 0.1
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            if i < len(bounds_check_results):
                if bounds_check_results[i]:
                    marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.0, 1.0, 0.0, 0.8  # green: passed
                else:
                    marker.color.r, marker.color.g, marker.color.b, marker.color.a = 1.0, 0.0, 0.0, 1.0  # red: failed
            else:
                marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.0, 0.0, 1.0, 0.5  # blue: unchecked tail
            markers.markers.append(marker)
        self.spline_samples_pub.publish(markers)

    ######################
    # VIZ + MSG WRAPPING #
    ######################
    def xyv_to_markers(self, x: float, y: float, v: float, mrks: MarkerArray, rgb=None) -> Marker:
        mrk = Marker()
        mrk.header.frame_id = "map"
        mrk.header.stamp = self.get_clock().now().to_msg()
        mrk.type = mrk.CYLINDER
        mrk.scale.x = 0.1
        mrk.scale.y = 0.1
        mrk.scale.z = float(v / self.gb_vmax)
        mrk.color.a = 1.0
        if rgb is not None:
            mrk.color.r, mrk.color.g, mrk.color.b = rgb
        else:
            mrk.color.b = 0.75
            mrk.color.r = 0.75
            if self.from_bag:
                mrk.color.g = 0.75

        mrk.id = len(mrks.markers)
        mrk.pose.position.x = float(x)
        mrk.pose.position.y = float(y)
        mrk.pose.position.z = float(v / self.gb_vmax / 2)
        mrk.pose.orientation.w = 1.0

        return mrk

    def xy_to_point(self, x: float, y: float, opponent=True) -> Marker:
        mrk = Marker()
        mrk.header.frame_id = "map"
        mrk.header.stamp = self.get_clock().now().to_msg()
        mrk.type = mrk.SPHERE
        mrk.scale.x = 0.5
        mrk.scale.y = 0.5
        mrk.scale.z = 0.5
        mrk.color.a = 0.8
        mrk.color.b = 0.65
        mrk.color.r = 1.0 if opponent else 0.0
        mrk.color.g = 0.65

        mrk.pose.position.x = float(x)
        mrk.pose.position.y = float(y)
        mrk.pose.position.z = 0.01
        mrk.pose.orientation.w = 1.0

        return mrk

    def xyv_to_wpnts(self, s: float, d: float, x: float, y: float, v: float, psi: float, kappa: float, wpnts: WpntArray) -> Wpnt:
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
