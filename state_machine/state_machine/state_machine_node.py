#!/usr/bin/env python3
"""
UNICORN racing state machine - ROS2 (Jazzy / rclpy) port.

Ported from the ROS1 (catkin/rospy) `state_machine` package. This is the racing
"brain": it subscribes to perception / planning / localization topics, computes a
set of boolean conditions, runs the state-transition graph and publishes the chosen
driving behaviour (local waypoints + BehaviorStrategy).

The full UNICORN feature set is preserved (RECOVERY / START / multi-planner
sustainability / prediction-aware free checks / velocity replanning / BehaviorStrategy
trailing & overtaking targets). The race_stack ROS2 template was used only for the
ament/rclpy structural idioms.
"""
import os
import time
import json
import copy
import datetime
import configparser
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile

import transforms3d
from ament_index_python.packages import get_package_share_directory

from scipy.interpolate import InterpolatedUnivariateSpline as Spline

from std_msgs.msg import String, Float32, Float32MultiArray, Bool
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from f110_msgs.msg import (
    ObstacleArray,
    OTWpntArray,
    WpntArray,
    BehaviorStrategy,
    PredictionArray,
)

import trajectory_planning_helpers as tph

from vel_planner.vel_planner import calc_vel_profile
from state_machine.states_types import StateType
from state_machine import states
from state_machine import state_transitions
from state_machine.state_machine_params import StateMachineParams

try:
    # if we are in the car, vesc msgs are built and we read them
    from vesc_msgs.msg import VescStateStamped
except Exception:
    pass


class WaypointData:
    """Holds the latest waypoints of a given planner together with its (dynamic)
    parameters. In ROS1 these parameters were served by a per-planner
    `dynamic_reconfigure` server (dyn_planner_tuner.cfg). In ROS2 they are declared on
    the state-machine node as nested parameters `<planner_name>.<param>` (loaded from
    the planner yaml in this package's config/planners directory).
    """

    def __init__(self, node: "StateMachine", planner_name: str, is_closed: bool):
        self.node = node
        self.name = planner_name
        self.list = []
        self.array = None
        self.stamp = None
        self.is_init = False
        # Debug: wall-clock of the last initialize_traj (cache replacement) and
        # how many times it has happened. None/0 until the first real update.
        self.last_init_sec = None
        self.init_count = 0
        self.is_gb_track_wpnts = False
        self.is_ot_wpnts = False
        self.closest_target = None
        self.closest_gap = None
        # Debug: last _check_free_frenet decision detail (per-obstacle branch/free_dist).
        self.free_dbg = None
        self.is_closed = is_closed
        self.vel_planner_safety_factor = 1.0
        # When frozen, the cache is NOT re-initialized from fresh planner output; the
        # path captured on entry is kept and only sliced (tail trimmed) as the car
        # advances. Used to hold one blended-recovery path while trailing on it, so the
        # controller target stops jumping every frame (see _hold_recovery_freeze).
        self.frozen = False
        self.update_param()

    def update_param(self):
        get = self.node.get_planner_param
        self.min_horizon = get(self.name, "min_horizon")
        self.max_horizon = get(self.name, "max_horizon")
        self.lateral_width_m = get(self.name, "lateral_width_m")
        self.free_scaling_reference_distance_m = get(self.name, "free_scaling_reference_distance_m")
        self.latest_threshold = get(self.name, "latest_threshold")
        self.on_spline_front_horizon_thres_m = get(self.name, "on_spline_front_horizon_thres_m")
        self.on_spline_min_dist_thres_m = get(self.name, "on_spline_min_dist_thres_m")
        self.hyst_timer_sec = get(self.name, "hyst_timer_sec")
        self.killing_timer_sec = get(self.name, "killing_timer_sec")

    def initialize_traj(self, wpnt):
        if len(wpnt.wpnts) != 0:
            self.stamp = wpnt.header.stamp
            self.list = wpnt.wpnts
            self.array = np.array([[w.x_m, w.y_m, w.s_m, w.d_m] for w in wpnt.wpnts])
            self.is_init = True
            # Debug: when this cache was last replaced with fresh planner output.
            # Lets the loop snapshot report cache staleness (wall-clock since the
            # last real re-init) independently of the message header stamp.
            self.last_init_sec = self.node.now_sec()
            self.init_count += 1


def time_to_float(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def _default_debug_log_dir() -> Path:
    """<repo>/logfile if this source file lives inside a git checkout (true for a
    symlink-install, which is how this workspace is normally built -- __file__
    resolves through the symlink to the tracked source), so debug logs land
    somewhere `git add logfile/` already picks up instead of needing a manual copy
    out of the home directory after every run. Falls back to a fixed home-directory
    path if the source isn't inside a repo (e.g. a non-symlink install, where
    __file__ points into install/ instead of src/).
    """
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / ".git").exists():
            return parent / "logfile"
    return Path("~/roboracer_debug_logs").expanduser()


class StateMachine(Node):
    """
    This state machine subscribes to topics and calculates flags/conditions.
    State transitions and state behaviors are described in `transitions.py` and `states.py`
    """

    def __init__(self) -> None:
        super().__init__(
            "state_machine",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )
        self.name = "state_machine"

        self.main_loop = None  # set later, referenced by params callback

        # Load planner configs (planner_name -> {param: value}) before declaring params
        self._planner_param_cache = {}
        self._load_planner_configs()

        # PARAMETER DECLARATION (replaces rospy.get_param + dyn_reconfigure)
        self.params = StateMachineParams(self)
        self.add_on_set_parameters_callback(self.params.parameters_callback)

        # Convenience aliases (kept as attributes for parity with the ROS1 code which
        # read these directly off `self`). They mirror self.params.* values.
        self.rate_hz = self.params.rate_hz
        self.n_loc_wpnts = self.params.n_loc_wpnts
        self.timetrials_only = self.params.timetrials_only
        self.racecar_version = self.params.racecar_version
        self.ot_planner = self.params.ot_planner
        self.track_length = self.params.track_length
        self.volt_threshold = self.params.volt_threshold

        self.local_wpnts = WpntArray()
        self.waypoints_dist = 0.1  # [m]
        self.measuring = self.params.measuring

        # sectors: read the map yamls at startup, live-update from the sector tuner nodes
        # (ROS1: /map_params + /ot_map_params and the dyn_sector_* servers)
        self.map_name = self._get_str_param("map", "")
        self.sectors_params = {}
        self.ot_sectors_params = {}
        self.only_ftg_zones = []
        self.ftg_counter = 0

        # Per-run debug log file (logfile/state_machine_*.log): off by default so a
        # normal race doesn't pay disk-write overhead for output nobody reads; pass
        # debug:=true on race.launch.xml (-> debug_log_enabled:=true here) to get it.
        # has_parameter guard: automatically_declare_parameters_from_overrides=True
        # above already auto-declares it if the launch passed an override.
        if not self.has_parameter("debug_log_enabled"):
            self.declare_parameter("debug_log_enabled", False)
        self.debug_log_enabled = bool(self.get_parameter("debug_log_enabled").value)
        self._setup_debug_log_file()

        self.cur_s = 0.0
        self.cur_d = 0.0
        self.cur_vs = 0.0

        # Velocity Planning - load racecar config from stack_master
        self._load_vehicle_dynamics()

        # overtaking variables
        self.n_ot_sectors = 0
        self.overtake_wpnts = None
        self.overtake_zones = []
        self.ot_begin_margin = 0.5
        # read the map sector yamls, then build only_ftg_zones / overtake_zones
        self._load_sector_yamls()
        self._load_sector_params()
        self.cur_volt = 11.69  # default value for sim
        self.static_overtaking_mode = False
        # Per-loop slice diagnostics, set by get_splini_wpts / get_recovery_wpts,
        # reset at the top of loop() so a snapshot only shows the source actually used.
        self._splini_dbg = None
        self._recovery_dbg = None
        # Diagnostics for the static OVERTAKE entry gate (_static_path_available /
        # _check_static_overtaking_mode): last availability detail, and the wall-clock
        # of when the path first became valid+safe (for [STATIC_OT_TRANSITION] latency).
        self._static_path_dbg = None
        self._static_first_valid_sec = None
        # Previous loop's source cache, for rule 2 (drop the cache on a real src change).
        self._prev_src_cache = None

        # waypoint variables
        self.cur_id_ot = 1
        self.max_speed = -1
        self.max_s = 0
        self.current_position = None
        self.gb_wpnts = None
        self.recovery_wpnts = None
        self._recovery_plain = None
        self._recovery_blended = None
        self.gb_max_idx = None
        self.wpnt_dist = self.waypoints_dist
        self.num_glb_wpnts = 0
        self.num_ot_points = 0
        self.previous_index = 0

        # dynamic-parameter-backed attributes (aliases onto params)
        self.gb_ego_width_m = self.params.gb_ego_width_m
        self.lateral_width_gb_m = self.params.lateral_width_gb_m
        self.gb_horizon_m = self.params.gb_horizon_m
        self.interest_horizon_m = self.params.interest_horizon_m
        self.overtake_min_closing_mps = self.params.overtake_min_closing_mps

        self.last_recovery_update_time = None
        self.cur_gb_wpnts = WaypointData(self, "global_tracking", True)
        self.cur_recovery_wpnts = WaypointData(self, "recovery_planner", False)
        self.cur_avoidance_wpnts = WaypointData(self, "dynamic_avoidance_planner", False)
        self.cur_static_avoidance_wpnts = WaypointData(self, "static_avoidance_planner", False)
        self.cur_start_wpnts = WaypointData(self, "start_planner", False)

        self.cur_avoidance_wpnts.is_ot_wpnts = True
        self.cur_static_avoidance_wpnts.is_ot_wpnts = True
        self.cur_gb_wpnts.is_gb_track_wpnts = True
        self.cur_recovery_wpnts.vel_planner_safety_factor = 0.5

        self.gb_closest_target = None
        self.gb_closest_gap = None
        self.recovery_closest_target = None
        self.recovery_closest_gap = None
        self.ot_closest_target = None
        self.ot_closest_gap = None

        self.behavior_strategy = BehaviorStrategy()

        # mincurv spline
        self.mincurv_spline_x = None
        self.mincurv_spline_y = None
        # ot spline
        self.ot_spline_x = None
        self.ot_spline_y = None
        self.ot_spline_d = None
        self.recompute_ot_spline = True
        # live sector retune from the sector tuner nodes (after recompute_ot_spline exists)
        self._setup_sector_live_update()

        # obstacle avoidance variables
        self.obstacles = []
        self.obstacles_in_interest = []
        self.cur_obstacles_in_interest = []
        self.obstacles_perception = []
        self.obstacles_prediction_id = None
        self.obstacles_prediction = []
        self.prediction_dt = 0.02  # updated from PredictionArray.dt; matches predictor
        self.ego_prediction = []
        self.obstacle_was_here = True
        self.side_by_side_threshold = 0.6
        self.merger = None
        # force_trailing: published by the GP predictor, True while the opponent
        # prediction is only a constant-velocity fallback. Used as an OVERTAKE
        # *ENTRY* veto (_check_overtaking_mode) and deliberately NOT consulted by
        # _check_overtaking_mode_sustainability -- see the comment there.
        self.force_trailing = False
        # Was `not self.params.use_force_trailing`, while the live-update callback
        # in state_machine_params.py assigned the value straight through: the flag
        # meant the opposite of itself depending on how it was set. It never
        # showed, because self.force_trailing was written by the callback and then
        # read by nothing at all.
        self.use_force_trailing = bool(self.params.use_force_trailing)
        # Dynamic-overtake gating (see _check_getting_closer / _check_overtaking_mode).
        self.dynamic_overtake_max_gap_m = self.params.dynamic_overtake_max_gap_m
        self.dynamic_overtake_min_rel_speed_mps = self.params.dynamic_overtake_min_rel_speed_mps
        self.dynamic_prediction_span_m = self.params.dynamic_prediction_span_m
        # Last dynamic-overtake candidate, filled by _check_getting_closer for the
        # [DYNAMIC_OT] decision log. None means "no candidate in range this loop".
        self._dyn_ot_target = None
        # Latest /planner/avoidance/dynamic_diag payload, and when the last
        # non-empty prediction arrived. Both feed the [DYNAMIC_OT] line only.
        self._planner_diag = None
        self._prediction_stamp_sec = None
        self._dbg_last_dynamic_log_sec = 0.0
        self._dbg_last_memory_log_sec = 0.0
        self._dbg_last_memstate_log_sec = 0.0
        # Mirrors what _check_ot_sector() publishes on /ot_section_check, so the
        # decision log can report it without re-running the sector scan. Reset to
        # None at the top of every loop: None means "not evaluated this loop".
        self.ot_section_check = None
        # Short-term memory of a dynamic opponent that was ahead and has since
        # dropped out of /tracking/obstacles. See _update_opponent_memory.
        self._last_dyn_seen_sec = None
        self._last_dyn_gap_m = None
        self._last_dyn_id = None
        self._last_overtake_sec = None
        self.dynamic_opponent_memory_sec = self.params.dynamic_opponent_memory_sec
        self.overtake_pass_grace_sec = self.params.overtake_pass_grace_sec
        self.overtake_speed_scale = self.params.overtake_speed_scale

        # spliner variables
        self.splini_ttl = self.params.splini_ttl
        self.splini_ttl_counter = int(self.splini_ttl * self.rate_hz)
        self.avoidance_wpnts = None
        self.static_avoidance_wpnts = None
        self.start_wpnts = None
        self.start_wpnts_array = None
        self.last_valid_avoidance_wpnts = None
        self.last_valid_avoidance_array = None
        self.last_valid_static_avoidance_wpnts = None

        self.overtaking_horizon_m = self.params.overtaking_horizon_m
        self.lateral_width_ot_m = self.params.lateral_width_ot_m
        self.splini_hyst_timer_sec = self.params.splini_hyst_timer_sec
        self.emergency_break_horizon = self.params.emergency_break_horizon
        self.emergency_break_d = 0.12  # [m]
        self.trailing_speed_scale = self.params.trailing_speed_scale
        self.trailing_min_speed_mps = self.params.trailing_min_speed_mps
        self.trailing_speed_ramp = bool(getattr(self.params, 'trailing_speed_ramp', True))

        # Graph based variables
        self.graph_based_wpts = None
        self.gb_wpnts_arr = None
        # Frenet variables
        self.frenet_wpnts = WpntArray()

        # FTG params
        self.ftg_speed_mps = self.params.ftg_speed_mps
        self.ftg_timer_sec = self.params.ftg_timer_sec
        self.ftg_disabled = not self.params.ftg_active

        # Force GBTRACK state
        self.force_gbtrack_state = self.params.force_GBTRACK

        self.overtaking_ttl_count = 0
        # Grace window (in loops) during which the OT-blended recovery path is allowed
        # as the recovery source. The blended path (OT heading -> GB) only makes sense
        # when leaving OVERTAKE; outside that window plain recovery is used, so a car
        # that never overtook (OT sector off) never trails on the blended OT line.
        # Set to a positive count while in OVERTAKE and decremented each loop after.
        self.blended_recovery_grace_loops = int(0.5 * self.rate_hz)
        self._blended_grace_count = 0

        self.save_start_traj = False
        self.cur_start_wpnts_candidate = OTWpntArray()
        self.need_start_traj = False
        # visualization variables
        self.first_visualization = True
        self.x_viz = 0
        self.y_viz = 0

        # STATES
        self.cur_state = StateType.GB_TRACK
        self.local_wpnts_src = StateType.GB_TRACK
        self.static_avoid = False
        self.fail_trailing = False

        self.states = {
            StateType.GB_TRACK: states.GlobalTracking,
            StateType.OVERTAKE: states.Overtaking,
            StateType.FTGONLY: states.FTGOnly,
            StateType.RECOVERY: states.RECOVERY,
            StateType.START: states.START,
        }
        self.state_transitions = {
            StateType.GB_TRACK: state_transitions.GlobalTrackingTransition,
            StateType.RECOVERY: state_transitions.RecoveryTransition,
            StateType.TRAILING: state_transitions.TrailingTransition,
            StateType.ATTACK: state_transitions.TrailingTransition,
            StateType.OVERTAKE: state_transitions.OvertakingTransition,
            StateType.FTGONLY: state_transitions.FTGOnlyTransition,
            StateType.START: state_transitions.StartTransition,
        }

        self.opponent = ObstacleArray()

        qos = QoSProfile(depth=10)

        # SUBSCRIPTIONS
        self.create_subscription(Odometry, "/car_state/odom", self.odom_cb, qos)
        self._wait_for_attr("current_position", "/car_state/odom")

        self.create_subscription(WpntArray, "/global_waypoints_scaled", self.glb_wpnts_cb, qos)
        self.create_subscription(WpntArray, "/planner/recovery/wpnts", self.recovery_wpnts_cb, qos)
        self.create_subscription(
            WpntArray, "/planner/ot_blended_recovery/wpnts", self.ot_blended_recovery_cb, qos)
        self.create_subscription(WpntArray, "/global_waypoints/overtaking", self.overtake_cb, qos)
        self._wait_for_attr("gb_wpnts", "/global_waypoints_scaled")
        self._wait_for_attr("overtake_wpnts", "/global_waypoints/overtaking")

        self.create_subscription(Odometry, "/car_state/odom_frenet", self.frenet_pose_cb, qos)
        self.create_subscription(WpntArray, "/global_waypoints", self.glb_wpnts_og_cb, qos)

        self.create_subscription(ObstacleArray, "/tracking/obstacles", self.obstacle_perception_cb, qos)
        self.create_subscription(
            PredictionArray, "/opponent_prediction/obstacles_pred", self.obstacle_prediction_cb, qos
        )
        self.create_subscription(PredictionArray, "/mpc_controller/ego_prediction", self.ego_prediction_cb, qos)

        if self.ot_planner == "spliner" or self.ot_planner == "sqp" or self.ot_planner == "lane_change":
            self.create_subscription(OTWpntArray, "/planner/avoidance/otwpnts", self.avoidance_cb, qos)
            if self.ot_planner == "sqp" or self.ot_planner == "lane_change":
                self.create_subscription(
                    OTWpntArray, "/planner/avoidance/static_otwpnts", self.static_avoidance_cb, qos
                )
        if self.ot_planner == "sqp" or self.ot_planner == "lane_change":
            self.create_subscription(Float32MultiArray, "/planner/avoidance/merger", self.merger_cb, qos)
            self.create_subscription(Bool, "/opponent_prediction/force_trailing", self.force_trailing_cb, qos)
            self.create_subscription(
                String, "/planner/avoidance/dynamic_diag", self.dynamic_diag_cb,
                QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
            self.create_subscription(Bool, "planner/avoidance/fail_trailing", self.fail_trailing_cb, qos)

        if not self.params.sim:
            self.create_subscription(VescStateStamped, "/vesc/sensors/core", self.vesc_state_cb, qos)

        self.create_subscription(OTWpntArray, "/planner/start_wpnts", self.start_wpnts_cb, qos)
        self.create_subscription(Bool, "/save_start_traj", self.save_start_traj_cb, qos)

        # PUBLICATIONS
        self.behavior_strategy_pub = self.create_publisher(BehaviorStrategy, "behavior_strategy", 1)
        self.trailing_marker_pub = self.create_publisher(Marker, "/state_machine/trailing_target", 10)
        self.overtaking_marker_pub = self.create_publisher(Marker, "/state_machine/overtaking_target", 10)
        self.loc_wpnt_pub = self.create_publisher(WpntArray, "local_waypoints", 1)
        self.vis_loc_wpnt_pub = self.create_publisher(MarkerArray, "local_waypoints/markers", 10)
        # [Hz] How often the local-waypoint markers are DRAWN, independent of the
        # decision rate. Nothing the car does depends on this: /local_waypoints,
        # /behavior_strategy and the state string all go out at `rate` regardless.
        #
        # The subscriber gate in _pub_local_wpnts makes drawing free while pitwall
        # is closed, but the moment RViz connects the gate opens and every tick
        # draws again -- which is exactly when the car has the least CPU to spare.
        # This caps that cost: at 10 Hz a connected pitwall costs a fifth of what
        # the decision rate would, and a marker cloud is not worth watching faster.
        # 0 disables drawing outright.
        if not self.has_parameter("viz_rate_hz"):
            self.declare_parameter("viz_rate_hz", 10.0)
        self.viz_rate_hz = float(self.get_parameter("viz_rate_hz").value)
        self._last_viz_sec = 0.0
        self.state_pub = self.create_publisher(String, "state_machine", 1)
        # Per-loop diagnostic snapshot (JSON) for offline/live debugging of the
        # local_wpnts source selection and stale-cache leaks.
        self.debug_pub = self.create_publisher(String, "/state_machine/debug", 10)
        self.state_mrk = self.create_publisher(Marker, "/state_marker", 10)
        self.emergency_pub = self.create_publisher(Marker, "/emergency_marker", 5)
        self.ot_section_check_pub = self.create_publisher(Bool, "/ot_section_check", 1)
        # ROS1 published this from dynamic_statemachine_server when the save_start_traj
        # rqt button was pressed; re-homed here as a momentary param (see loop()).
        self.save_start_traj_pub = self.create_publisher(Bool, "/save_start_traj", 1)
        self._save_start_traj_requested = False
        self._save_params_requested = False
        if self.measuring:
            self.latency_pub = self.create_publisher(Float32, "/state_machine/latency", 10)

        # MAIN LOOP at fixed rate
        self.main_loop = self.create_timer(1.0 / self.rate_hz, self.loop)

    # ---------------------------------------------------------------------- #
    # SETUP HELPERS                                                           #
    # ---------------------------------------------------------------------- #
    def _setup_debug_log_file(self):
        """Per-run plain-text debug log: obstacle recognition, static OVERTAKE entry
        decisions (why TRAILING/blocked, with actual clearance in metres), and every
        state change -- independent of rclpy's own per-process log (which rcutils
        names by PID under ~/.ros/log, making it hard to find after the fact on the
        car or in sim). One file per run, plus a "latest" symlink so `tail -f` works
        without knowing the timestamp. Directory overridable via RACE_DEBUG_LOG_DIR.
        Never allowed to crash the node: a read-only home just disables file logging.
        Gated on debug_log_enabled (race.launch.xml debug:=true) -- disabled, this
        only sets up the tracking attrs so _dbg_log()'s `_dbg_fh is None` check
        no-ops the rest of the class out without touching the disk.
        """
        self._dbg_fh = None
        self._dbg_last_state_value = None
        self._dbg_last_obs_log_sec = 0.0
        self._dbg_last_static_log_sec = 0.0
        if not self.debug_log_enabled:
            return
        try:
            log_dir = Path(os.environ.get("RACE_DEBUG_LOG_DIR", str(_default_debug_log_dir()))).expanduser()
            log_dir.mkdir(parents=True, exist_ok=True)
            run_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = log_dir / f"state_machine_{run_stamp}.log"
            self._dbg_fh = open(log_path, "a", buffering=1)
            latest = log_dir / "latest_state_machine.log"
            if latest.is_symlink() or latest.exists():
                latest.unlink()
            latest.symlink_to(log_path.name)
            self._dbg_fh.write(f"# state_machine debug log started {run_stamp}  map={self.map_name!r}\n")
            self.get_logger().info(f"[{self.name}] debug log: {log_path} (latest: {latest})")
        except OSError as e:
            self.get_logger().warn(f"[{self.name}] could not open debug log file: {e}")

    def _dbg_log(self, msg: str) -> None:
        if self._dbg_fh is None:
            return
        try:
            self._dbg_fh.write(f"{self.now_sec():.3f} {msg}\n")
        except OSError:
            pass

    def _wait_for_attr(self, attr, topic):
        """rclpy equivalent of rospy.wait_for_message."""
        while rclpy.ok() and getattr(self, attr, None) is None:
            self.get_logger().info(f"Waiting for message on {topic}", throttle_duration_sec=1.0)
            rclpy.spin_once(self, timeout_sec=0.1)

    def _load_planner_configs(self):
        """Load the per-planner yaml files shipped in this package's config/planners dir
        and declare them as nested ROS2 parameters (<planner>.<key>)."""
        import yaml

        try:
            share = get_package_share_directory("state_machine")
        except Exception:
            share = None

        planner_names = [
            "global_tracking",
            "recovery_planner",
            "dynamic_avoidance_planner",
            "static_avoidance_planner",
            "start_planner",
        ]
        for pname in planner_names:
            data = {}
            if share is not None:
                cfg = os.path.join(share, "config", "planners", pname + ".yaml")
                if os.path.exists(cfg):
                    with open(cfg, "r") as f:
                        data = yaml.safe_load(f) or {}
            self._planner_param_cache[pname] = data
            for key, val in data.items():
                pname_param = f"{pname}.{key}"
                try:
                    self.declare_parameter(pname_param, val)
                except Exception:
                    pass

    def _get_str_param(self, name, default=""):
        try:
            if not self.has_parameter(name):
                self.declare_parameter(name, default)
            v = self.get_parameter(name).value
            return v if v is not None else default
        except Exception:
            return default

    def _load_sector_yamls(self):
        # read the map sector yamls into sectors_params / ot_sectors_params (ROS1 /map_params, /ot_map_params)
        import yaml
        try:
            maps_dir = os.path.join(get_package_share_directory("stack_master"), "maps", self.map_name)
        except Exception:
            self.get_logger().warn(f"[{self.name}] could not locate stack_master maps dir; no sectors loaded")
            return
        sp = os.path.join(maps_dir, "speed_scaling.yaml")
        if os.path.exists(sp):
            with open(sp, "r") as f:
                d = yaml.safe_load(f) or {}
            self.sectors_params = (d.get("speed_sector_tuner", {}) or {}).get("ros__parameters", {}) or {}
        else:
            self.get_logger().warn(f"[{self.name}] {sp} not found; no FTG-only zones")
        op = os.path.join(maps_dir, "ot_sectors.yaml")
        if os.path.exists(op):
            with open(op, "r") as f:
                d = yaml.safe_load(f) or {}
            self.ot_sectors_params = (d.get("ot_sector_tuner", {}) or {}).get("ros__parameters", {}) or {}
            self.ot_begin_margin = float(self.ot_sectors_params.get("ot_sector_begin", self.ot_begin_margin))
        else:
            self.get_logger().warn(f"[{self.name}] {op} not found; no overtake zones")

    def _load_sector_params(self):
        # build zones from the sector dicts (ROS1 sector_dyn_param_cb / ot_dyn_param_cb)
        self.only_ftg_zones = []
        self.n_sectors = int(self.sectors_params.get("n_sectors", 0))
        for i in range(self.n_sectors):
            sec = self.sectors_params.get(f"Sector{i}", {}) or {}
            if sec.get("only_FTG", False):
                # end+1 == next sector's start: close the 1-index gap so adjacent FTG
                # sectors don't briefly drop to GB_TRACK (ROS1 used [start, end]).
                self.only_ftg_zones.append([sec.get("start", 0), sec.get("end", 0) + 1])

        self.overtake_zones = []
        self.n_ot_sectors = int(self.ot_sectors_params.get("n_sectors", 0))
        for i in range(self.n_ot_sectors):
            sec = self.ot_sectors_params.get(f"Overtaking_sector{i}", {}) or {}
            if sec.get("ot_flag", False):
                self.overtake_zones.append([sec.get("start", 0), sec.get("end", 0) + 1])

    def _setup_sector_live_update(self):
        # ROS2 replacement of ROS1 /dyn_sector_speed & /dyn_sector_overtake subscriptions
        from rclpy.parameter_event_handler import ParameterEventHandler
        self._sector_evt_handler = ParameterEventHandler(self)
        self._sector_evt_cb_handle = self._sector_evt_handler.add_parameter_event_callback(
            self._sector_param_event_cb)

    @staticmethod
    def _param_msg_value(p):
        # rcl_interfaces/Parameter -> python value (bool/int/double only needed here)
        t = p.value.type
        if t == 1:
            return p.value.bool_value
        if t == 2:
            return p.value.integer_value
        if t == 3:
            return p.value.double_value
        return None

    def _sector_param_event_cb(self, event):
        node = event.node.lstrip("/")
        if node == "speed_sector_tuner":
            for p in list(event.new_parameters) + list(event.changed_parameters):
                if p.name.startswith("Sector") and p.name.endswith(".only_FTG"):
                    key = p.name.split(".")[0]
                    self.sectors_params.setdefault(key, {})["only_FTG"] = bool(self._param_msg_value(p))
            self._load_sector_params()
        elif node == "ot_sector_tuner":
            for p in list(event.new_parameters) + list(event.changed_parameters):
                if p.name.startswith("Overtaking_sector") and p.name.endswith(".ot_flag"):
                    key = p.name.split(".")[0]
                    self.ot_sectors_params.setdefault(key, {})["ot_flag"] = bool(self._param_msg_value(p))
                elif p.name == "ot_sector_begin":
                    self.ot_begin_margin = float(self._param_msg_value(p))
                    self.recompute_ot_spline = True
            self._load_sector_params()

    def get_planner_param(self, planner_name, key):
        """Read a planner parameter; falls back to cached yaml value."""
        full = f"{planner_name}.{key}"
        if self.has_parameter(full):
            return self.get_parameter(full).value
        return self._planner_param_cache.get(planner_name, {}).get(key)

    def _load_vehicle_dynamics(self):
        """Load veh params + ggv / ax_max machine info from stack_master config."""
        self.pars = {}
        try:
            stack_master_path = get_package_share_directory("stack_master")
        except Exception:
            stack_master_path = None

        parser = configparser.ConfigParser()
        ini_ok = False
        if stack_master_path is not None:
            ini_path = os.path.join(
                stack_master_path, "config", self.params.racecar_version, "racecar_f110.ini"
            )
            ini_ok = bool(parser.read(ini_path))

        if not ini_ok:
            # Sim / missing config fallback: provide sane defaults so the node still runs.
            self.get_logger().warn(
                "racecar_f110.ini not found; using default vehicle params (velocity replanning degraded)"
            )
            self.pars["veh_params"] = {
                "v_max": 7.0, "length": 0.535, "width": 0.3,
                "mass": 3.5, "dragcoeff": 0.0136, "g": 9.81,
            }
            self.pars["vel_calc_opts"] = {"dyn_model_exp": 1.0, "vel_profile_conv_filt_window": None}
            self.ggv = None
            self.ax_max_machines = None
            self.b_ax_max_machines = None
            return

        self.pars["veh_params"] = json.loads(parser.get("GENERAL_OPTIONS", "veh_params"))
        self.pars["vel_calc_opts"] = json.loads(parser.get("GENERAL_OPTIONS", "vel_calc_opts"))
        vdyn = os.path.join(stack_master_path, "config", self.params.racecar_version, "veh_dyn_info")
        ggv_path = os.path.join(vdyn, "ggv.csv")
        ax_max_path = os.path.join(vdyn, "ax_max_machines.csv")
        b_ax_max_path = os.path.join(vdyn, "b_ax_max_machines.csv")
        self.ggv, self.ax_max_machines = tph.import_veh_dyn_info.import_veh_dyn_info(
            ggv_import_path=ggv_path, ax_max_machines_import_path=ax_max_path
        )
        _, self.b_ax_max_machines = tph.import_veh_dyn_info.import_veh_dyn_info(
            ggv_import_path=ggv_path, ax_max_machines_import_path=b_ax_max_path
        )

    def now_sec(self) -> float:
        return time_to_float(self.get_clock().now().to_msg())

    #############
    # CALLBACKS #
    #############
    def save_start_traj_cb(self, msg):
        if len(self.cur_start_wpnts_candidate.wpnts) != 0:
            self.update_velocity(self.cur_start_wpnts_candidate, self.cur_start_wpnts.vel_planner_safety_factor)
            self.cur_start_wpnts.initialize_traj(self.cur_start_wpnts_candidate)
            self.cur_state = StateType.START

    def vesc_state_cb(self, data):
        self.cur_volt = data.state.voltage_input

    def frenet_planner_cb(self, data: WpntArray):
        self.frenet_wpnts = data

    def recovery_wpnts_cb(self, data: WpntArray):
        if len(data.wpnts) != 0:
            self.update_velocity(data, self.cur_recovery_wpnts.vel_planner_safety_factor)
        self._recovery_plain = data
        self._select_recovery_source()

    def ot_blended_recovery_cb(self, data: WpntArray):
        # OT-blended recovery: OT heading for the first ~1 m then splined to GB.
        # Published every loop by recovery_spliner (empty when no OT path). When it
        # carries a valid path we prefer it over plain recovery so the RECOVERY src
        # keeps the overtake line instead of snapping straight to GB.
        if len(data.wpnts) != 0:
            self.update_velocity(data, self.cur_recovery_wpnts.vel_planner_safety_factor)
        self._recovery_blended = data
        self._select_recovery_source()

    def _select_recovery_source(self):
        # recovery_wpnts feeds _check_latest_wpnts (freshness + on-spline). Prefer the
        # blended path when it is fresh and non-empty, else fall back to plain recovery.
        # While the recovery cache is frozen (held path in use), don't swap the source
        # from under it -- the freeze in _check_latest_wpnts keeps the captured cache.
        if self.cur_recovery_wpnts.frozen:
            return
        # The blended path is only meaningful when leaving OVERTAKE: it keeps the OT
        # heading for ~1 m then splines back to GB. Outside the post-OVERTAKE grace
        # window it must NOT stand in as the recovery source, otherwise a car that is
        # merely trailing (OT sector off, OVERTAKE never entered) would follow the OT
        # line whenever it drifts off the raceline. Fall back to plain recovery then.
        allow_blended = self.cur_state == StateType.OVERTAKE or self._blended_grace_count > 0
        blended = self._recovery_blended
        if allow_blended and blended is not None and len(blended.wpnts) != 0 and (
            self.now_sec() - time_to_float(blended.header.stamp)
        ) <= self.cur_recovery_wpnts.latest_threshold:
            self.recovery_wpnts = blended
        else:
            self.recovery_wpnts = self._recovery_plain

    def _hold_recovery_freeze(self):
        # Called once per loop after the src is decided. Freeze the recovery cache while
        # RECOVERY is the active source; release it the moment the src leaves RECOVERY so
        # the next entry captures a fresh path.
        if self.local_wpnts_src == StateType.RECOVERY:
            # On entry the cache was just re-inited with fresh output (frozen was False
            # during this loop's transition); now latch it so later loops hold it.
            self.cur_recovery_wpnts.frozen = True
        else:
            self.cur_recovery_wpnts.frozen = False

    def avoidance_cb(self, data: OTWpntArray):
        if len(data.wpnts) != 0:
            self.update_velocity(data, self.cur_avoidance_wpnts.vel_planner_safety_factor,
                                 speed_scale=self.overtake_speed_scale)
        self.avoidance_wpnts = data

    def static_avoidance_cb(self, data: OTWpntArray):
        if len(data.wpnts) != 0:
            self.update_velocity(data, self.cur_static_avoidance_wpnts.vel_planner_safety_factor,
                                 speed_scale=self.overtake_speed_scale)
        self.static_avoidance_wpnts = data

    def start_wpnts_cb(self, data: OTWpntArray):
        if len(data.wpnts) != 0:
            self.cur_start_wpnts_candidate = data

    def overtake_cb(self, data):
        self.overtake_wpnts = data.wpnts
        self.num_ot_points = len(self.overtake_wpnts)
        if self.recompute_ot_spline and self.num_ot_points != 0:
            self.ot_splinification()
            self.recompute_ot_spline = False

    def glb_wpnts_cb(self, data: WpntArray):
        # last point's s == loop length (ROS1 read this from /global_republisher/track_length)
        track_len = data.wpnts[-1].s_m
        data.wpnts = data.wpnts[:-1]  # exclude last point (== first)
        self.gb_wpnts = data
        self.num_glb_wpnts = len(data.wpnts)
        self.n_loc_wpnts = min(self.n_loc_wpnts, int(self.num_glb_wpnts / 2))
        self.max_s = data.wpnts[-1].s_m
        if track_len > 1.0:
            self.track_length = track_len
        self.wpnt_dist = data.wpnts[1].s_m - data.wpnts[0].s_m
        self.gb_max_idx = data.wpnts[-1].id
        if self.ot_planner == "graph_based":
            self.gb_wpnts_arr = np.array([
                [w.s_m, w.d_m, w.x_m, w.y_m, w.d_right, w.d_left, w.psi_rad,
                 w.kappa_radpm, w.vx_mps, w.ax_mps2] for w in data.wpnts
            ])

    def glb_wpnts_og_cb(self, data):
        if self.max_speed == -1:
            self.max_speed = max([wpnt.vx_mps for wpnt in data.wpnts])

    def graphbased_wpts_cb(self, data):
        arr = np.asarray(data.data)
        self.graph_based_wpts = arr.reshape(data.layout.dim[0].size, data.layout.dim[1].size)
        self.graph_based_action = data.layout.dim[0].label

    def obstacle_perception_cb(self, data):
        if not self.timetrials_only:
            self.obstacles_perception = data.obstacles[:]
            self.obstacles = data.obstacles
            obstacles_in_interest = []
            for obs in data.obstacles:
                gap = (obs.s_start - self.cur_s) % self.track_length
                if gap < self.interest_horizon_m:
                    obstacles_in_interest.append(obs)
            self.obstacles_in_interest = obstacles_in_interest

    def ego_prediction_cb(self, data):
        self.ego_prediction = data.predictions if len(data.predictions) != 0 else []

    def obstacle_prediction_cb(self, data):
        if len(data.predictions) != 0:
            self.obstacles_prediction_id = data.id
            self.obstacles_prediction = data.predictions
            self._prediction_stamp_sec = time_to_float(data.header.stamp)
            # Time step between consecutive predictions, carried on the message so the
            # ttc->prediction-index conversion in _check_free_frenet stays in sync with
            # the predictor's dt (falls back to the last known dt if a msg omits it).
            if data.dt > 0.0:
                self.prediction_dt = data.dt
        else:
            self.obstacles_prediction = []

    def frenet_pose_cb(self, data: Odometry):
        self.cur_s = data.pose.pose.position.x
        self.cur_d = data.pose.pose.position.y
        self.cur_vs = data.twist.twist.linear.x
        if self.num_ot_points != 0:
            self.cur_id_ot = int(self._find_nearest_ot_s())

    def odom_cb(self, data):
        x = data.pose.pose.position.x
        y = data.pose.pose.position.y
        q = data.pose.pose.orientation
        # transforms3d uses [w, x, y, z]
        _, _, theta = transforms3d.euler.quat2euler([q.w, q.x, q.y, q.z])
        self.current_position = [x, y, theta]

    def merger_cb(self, data):
        self.merger = data.data

    def force_trailing_cb(self, data):
        self.force_trailing = bool(data.data)

    def dynamic_diag_cb(self, data):
        """Side-availability / no-path reason from the lane_change planner.

        The planner republishes only on change, so this is a handful of messages
        per race, not a 20 Hz stream. Parsed here rather than in the log line so
        a malformed payload cannot take down the main loop.
        """
        try:
            self._planner_diag = json.loads(data.data)
        except (ValueError, TypeError):
            self._planner_diag = None

    def fail_trailing_cb(self, data):
        self.fail_trailing = data.data

    ######################################
    # ATTRIBUTES/CONDITIONS CALCULATIONS #
    ######################################
    def _check_only_ftg_zone(self) -> bool:
        ftg_only = False
        if len(self.only_ftg_zones) != 0:
            for sector in self.only_ftg_zones:
                if sector[0] <= self.cur_s / self.waypoints_dist <= sector[1]:
                    ftg_only = True
                    break
        return ftg_only

    def _check_close_to_raceline(self, threshold_m=None) -> bool:
        if threshold_m is None:
            return np.abs(self.cur_d) < self.gb_ego_width_m
        else:
            return np.abs(self.cur_d) < threshold_m

    def _check_close_to_raceline_heading(self, threshold_deg=None) -> bool:
        cloest_wpnt_idx = int(self.cur_s / self.waypoints_dist) % self.num_glb_wpnts
        cloest_wpnt_psi = self.cur_gb_wpnts.list[cloest_wpnt_idx].psi_rad
        if threshold_deg is None:
            return np.abs(self.current_position[2] - cloest_wpnt_psi) < np.deg2rad(20)
        else:
            return np.abs(self.cur_d) < np.deg2rad(threshold_deg)

    def _check_ot_sector(self) -> bool:
        # ROS1: no overtake zone matching cur_s -> not in an OT sector (return False).
        # (An empty overtake_zones means overtaking is suppressed, as in ROS1.)
        for sector in self.overtake_zones:
            if sector[0] <= self.cur_s / self.waypoints_dist <= sector[1]:
                self.ot_section_check = True
                self.ot_section_check_pub.publish(Bool(data=True))
                return True
        self.ot_section_check = False
        self.ot_section_check_pub.publish(Bool(data=False))
        return False

    @staticmethod
    def _obs_lateral_half_width(obs) -> float:
        """Half the obstacle's LATERAL extent, in Frenet d.

        Prefers |d_left - d_right| / 2 over size / 2, so an elongated cluster
        whose bounding circle is much wider than the car is not treated as if it
        blocked that whole width sideways. Falls back to size / 2 when the d
        bounds are absent or degenerate.

        NOTE for this stack specifically: perception fills d_left = d + size/2 and
        d_right = d - size/2 (detect.cpp publishObstaclesMessage, likewise
        multi_tracking.py), so for perception obstacles the two are identical and
        this changes nothing today. It becomes load-bearing for any producer that
        reports a real lateral extent -- e.g. opp_prediction, which writes
        d_left/d_right at +/-0.25 m (a car width) while copying `size` straight
        from the bounding circle.
        """
        width = abs(float(obs.d_left) - float(obs.d_right))
        if np.isfinite(width) and width > 1e-3:
            return 0.5 * width
        return 0.5 * float(obs.size)

    def _nearest_dynamic_opponent_ahead(self, threshold_m):
        """The closest NON-static obstacle ahead of the ego, within `threshold_m`.

        Returns ``(obstacle, forward_gap_m)`` or ``(None, None)``.

        This replaces the old ``obstacles_in_interest[0]`` pick, which was
        whatever order perception happened to publish in -- an opponent already
        BEHIND the car, or the further of two cars ahead, could become the
        overtake target.

        `forward_gap` is measured on s_center modulo track_length, so the
        start/finish seam is handled: ego at ``track_length - 0.5`` and an
        opponent at ``0.5`` is 1 m AHEAD, not a lap behind. The search window is
        additionally capped at half the lap, because past that a "gap" of nearly
        one track_length is really a car behind us -- on the short maps in this
        workspace (0804test is 17.79 m) an uncapped 10 m window would reach
        round to the car's own tail.
        """
        if self.track_length <= 0.0:
            return None, None
        window = min(float(threshold_m), 0.5 * self.track_length)
        best_obs = None
        best_gap = None
        for obs in self.obstacles_in_interest:
            if obs.is_static:
                continue
            gap = (obs.s_center - self.cur_s) % self.track_length
            if gap > window:
                continue
            if best_gap is None or gap < best_gap:
                best_obs = obs
                best_gap = gap
        return best_obs, best_gap

    def _check_getting_closer(self, threshold_m=3.0) -> bool:
        """Is there a dynamic overtake candidate inside `threshold_m` right now?

        Two independent conditions, deliberately kept apart:

        * **range** -- ``0 <= forward_gap <= threshold_m``. `threshold_m` used to
          be an argument the body never read, so every caller got the same
          "any obstacle in the 20 m interest horizon" answer regardless of the
          number it passed. It is a real gate now.
        * **relative speed** -- ``ego_vs - opp_vs > dynamic_overtake_min_rel_speed_mps``
          (default -0.5), i.e. the ego may be marginally SLOWER than the opponent
          and still qualify. This is racing_stack behaviour and is kept as-is; it
          is explicitly *not* the same thing as "the gap is actually shrinking",
          which is why it no longer shares a name with it.
        """
        obs, gap = self._nearest_dynamic_opponent_ahead(threshold_m)
        if obs is None:
            self._dyn_ot_target = None
            return False

        relative_speed = self.cur_vs - obs.vs
        relative_speed_ok = relative_speed > self.dynamic_overtake_min_rel_speed_mps
        self._dyn_ot_target = {
            "id": int(obs.id),
            "gap": float(gap),
            "opp_vs": float(obs.vs),
            "rel_v": float(relative_speed),
            "rel_ok": bool(relative_speed_ok),
        }
        return bool(relative_speed_ok)

    def _check_enemy_in_front(self) -> bool:
        horizon = self.gb_horizon_m
        for obs in self.obstacles:
            gap = (obs.s_start - self.cur_s) % self.track_length
            if gap < horizon:
                return True
        return False

    def _check_latest_wpnts(self, src_wpnts, wpnts_data: WaypointData):
        # Frozen cache: keep the captured path, do NOT replace it with fresh output.
        # Stay "available" as long as we are still on the held path (on_spline); once
        # the car runs off its tail the freeze naturally lapses to unavailable.
        if wpnts_data.frozen:
            return bool(wpnts_data.is_init and self._check_on_spline(wpnts_data))
        if src_wpnts is None or len(src_wpnts.wpnts) == 0:
            return False
        elif (self.now_sec() - time_to_float(src_wpnts.header.stamp)) > wpnts_data.latest_threshold:
            return False
        else:
            wpnts_data.initialize_traj(src_wpnts)
            return bool(self._check_on_spline(wpnts_data))

    def _check_ftg(self) -> bool:
        threshold = self.ftg_timer_sec * self.rate_hz
        if self.ftg_disabled:
            return False
        else:
            if (self.cur_state == StateType.TRAILING or self.cur_state == StateType.ATTACK) and \
                    self.cur_vs < self.ftg_speed_mps:
                self.ftg_counter += 1
                self.get_logger().warn(
                    f"[{self.name}] FTG counter: {self.ftg_counter}/{threshold}",
                    throttle_duration_sec=0.5,
                )
            else:
                self.ftg_counter = 0
            return self.ftg_counter > threshold

    def _check_on_spline(self, wpnt_data) -> bool:
        if wpnt_data.is_init:
            gap = (wpnt_data.list[-1].s_m - self.cur_s) % self.track_length
            min_dist = np.min(np.linalg.norm(wpnt_data.array[:, 0:2] - self.current_position[:2], axis=1))
            if gap > wpnt_data.on_spline_front_horizon_thres_m and min_dist < wpnt_data.on_spline_min_dist_thres_m:
                return True
        return False

    def _prediction_span_end_idx(self, obstacle_predictions) -> int:
        """Index one past the last prediction inside dynamic_prediction_span_m.

        Measured forward in Frenet s from the FIRST predicted pose (the opponent
        where it is now), wrap-around handled. `dynamic_prediction_span_m <= 0`
        disables the cap.

        The predictions are monotonically increasing in s (the predictor
        integrates the opponent forward), so this is a bisect, not a scan --
        O(log n) per obstacle per loop on the Jetson instead of O(n).
        """
        n = len(obstacle_predictions)
        span = float(self.dynamic_prediction_span_m)
        if span <= 0.0 or n < 2 or self.max_s <= 0.0:
            return n
        origin = float(obstacle_predictions[0].pred_s)
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if (float(obstacle_predictions[mid].pred_s) - origin) % self.max_s <= span:
                lo = mid + 1
            else:
                hi = mid
        return max(lo, 2)

    def _check_free_frenet(self, wpnts_data) -> bool:
        is_free = True
        closest_obs = None
        min_gap = 2.0
        max_horizon = wpnts_data.max_horizon
        is_gb_track_wpnts = wpnts_data.is_gb_track_wpnts
        is_ot_wpnts = wpnts_data.is_ot_wpnts
        free_scaling_reference_distance_m = wpnts_data.free_scaling_reference_distance_m
        lateral_width_m = wpnts_data.lateral_width_m

        obstacles = self.cur_obstacles_in_interest
        obstacle_predictions = self.obstacles_prediction

        # Debug: per-obstacle record of which branch decided free/blocked, so a
        # "GB judged free while an obstacle is right ahead" can be explained
        # (empty obstacle list vs prediction branch vs static/dynamic geom).
        dbg = {"is_init": bool(wpnts_data.is_init), "n_obs": len(obstacles), "obs": []}

        if wpnts_data.is_init:
            max_gap = (wpnts_data.array[-1, 2] - self.cur_s) % self.max_s
            for obs in obstacles:
                obs_s = obs.s_center
                gap = (obs_s - self.cur_s) % self.max_s
                relative_vs = self.cur_vs - obs.vs
                clip_vs = max(relative_vs, self.overtake_min_closing_mps)
                ttc = (gap - self.pars["veh_params"]["length"]) / clip_vs
                tt0 = (gap + 0.3 * self.pars["veh_params"]["length"]) / clip_vs

                rec = {"id": int(obs.id), "static": bool(obs.is_static),
                       "gap": round(float(gap), 2), "d": round(float(obs.d_center), 3),
                       "branch": None, "free_dist": None, "blocked": False}

                if obs.is_static:
                    # `gap < max_horizon` bounds this to obstacles the planner is
                    # actually responsible for. Obstacles come from
                    # cur_obstacles_in_interest (interest_horizon_m, 20 m) while each
                    # planner declares its own horizon (10 m for static avoidance).
                    # Without the bound, a static obstacle 15 m away is "beyond the
                    # path" for every avoidance path that could ever be published --
                    # no achievable path length clears it -- so static avoidance was
                    # unreachable whenever a second obstacle sat between the two
                    # horizons. Inside max_horizon the check is unchanged: a path
                    # that does not reach an obstacle it is meant to cover still
                    # counts as blocked.
                    if not wpnts_data.is_closed and max_gap < gap < max_horizon:
                        rec["branch"] = "static/beyond_path"
                        is_free = False
                        rec["blocked"] = True
                        if closest_obs is None or min_gap > gap:
                            closest_obs = obs
                            min_gap = gap
                    elif gap < max_horizon:
                        obs_d = obs.d_center
                        ot_d = 0
                        if not is_gb_track_wpnts:
                            avoid_wpnt_idx = np.argmin(abs(wpnts_data.array[:, 2] - obs_s))
                            ot_d = wpnts_data.list[avoid_wpnt_idx].d_m
                        min_dist = abs(ot_d - obs_d)
                        free_dist = min_dist - obs.size / 2 - self.gb_ego_width_m / 2
                        scaling_factor = np.clip(gap / free_scaling_reference_distance_m, 0.0, 1.0)
                        rec["branch"] = "static/geom"
                        rec["free_dist"] = round(float(free_dist), 3)
                        if free_dist < lateral_width_m * scaling_factor:
                            is_free = False
                            rec["blocked"] = True
                            self.get_logger().info(
                                "[State Machine] FREE False, obs dist to ot lane: {} m".format(free_dist),
                                throttle_duration_sec=1.0,
                            )
                            if closest_obs is None or min_gap > gap:
                                closest_obs = obs
                                min_gap = gap
                    else:
                        rec["branch"] = "static/gap>=max_horizon"
                else:
                    if len(obstacle_predictions) != 0 and self.obstacles_prediction_id == obs.id:
                        rec["branch"] = "dyn/pred"
                        obs_half_width = self._obs_lateral_half_width(obs)
                        start_idx = 0
                        end_idx = len(obstacle_predictions)
                        if is_ot_wpnts:
                            if ttc > 0:
                                start_idx = min(int(ttc / self.prediction_dt), len(obstacle_predictions))
                            if tt0 > 0:
                                end_idx = min(int(tt0 / self.prediction_dt), len(obstacle_predictions))
                            # Bound the window by DISTANCE as well as by time. The
                            # predictor emits n_time_steps * dt = 4 s of future; on
                            # a 20-40 m track that is most of a lap, and demanding
                            # the candidate overtake path clear all of it is what
                            # made the manoeuvre unreachable. The car re-decides at
                            # 50 Hz and the planner replans at 20 Hz, so only the
                            # stretch covered before the next decision has to hold.
                            #
                            # ONLY for the OT path (is_ot_wpnts). The blocked/free
                            # verdict on the raceline and on recovery still sees the
                            # whole prediction: shortening THAT would delay noticing
                            # an opponent, which is the opposite of the point.
                            end_idx = min(end_idx, self._prediction_span_end_idx(obstacle_predictions))
                            if end_idx - start_idx < 2:
                                end_idx = min(start_idx + 2, len(obstacle_predictions))
                        worst_fd = None
                        for obs_pred in obstacle_predictions[start_idx:end_idx]:
                            wpnt_idx = np.argmin(abs(wpnts_data.array[:, 2] - obs_pred.pred_s))
                            wpnt_d = wpnts_data.list[wpnt_idx].d_m
                            min_dist = abs(wpnt_d - obs_pred.pred_d)
                            free_dist = min_dist - obs_half_width - self.gb_ego_width_m / 2
                            scaling_factor = np.clip(gap / free_scaling_reference_distance_m, 0.0, 1.0)
                            if worst_fd is None or free_dist < worst_fd:
                                worst_fd = free_dist
                            if is_ot_wpnts:
                                self.get_logger().debug(
                                    f"free_dist: {free_dist}, lateral_width_m: {lateral_width_m}, "
                                    f"scaling_factor: {scaling_factor}, obs_half_width: {obs_half_width}, "
                                    f"wpnt_d:{wpnt_d}, obs_pred.pred_d: {obs_pred.pred_d} ",
                                    throttle_duration_sec=0.5,
                                )
                            if free_dist < lateral_width_m * scaling_factor:
                                is_free = False
                                rec["blocked"] = True
                                if closest_obs is None or min_gap > gap:
                                    closest_obs = obs
                                    min_gap = gap
                        rec["free_dist"] = None if worst_fd is None else round(float(worst_fd), 3)
                        rec["pred_n"] = int(end_idx - start_idx)
                    else:
                        rec["branch"] = "dyn/nopred (id_mismatch or empty)"
                        rec["pred_id"] = int(self.obstacles_prediction_id) if self.obstacles_prediction_id is not None else None
                        rec["pred_len"] = len(obstacle_predictions)
                        if not wpnts_data.is_closed and gap > max_gap:
                            rec["branch"] = "dyn/nopred/beyond_path"
                            is_free = False
                            rec["blocked"] = True
                            if closest_obs is None or min_gap > gap:
                                closest_obs = obs
                                min_gap = gap
                        elif gap < max_horizon:
                            ot_d = 0
                            if not is_gb_track_wpnts:
                                avoid_wpnt_idx = np.argmin(abs(wpnts_data.array[:, 2] - obs.s_center))
                                ot_d = wpnts_data.list[avoid_wpnt_idx].d_m
                            min_dist = abs(ot_d - obs.d_center)
                            free_dist = min_dist - self._obs_lateral_half_width(obs) - self.gb_ego_width_m / 2
                            scaling_factor = np.clip(gap / free_scaling_reference_distance_m, 0.0, 1.0)
                            rec["free_dist"] = round(float(free_dist), 3)
                            if free_dist < lateral_width_m * scaling_factor:
                                is_free = False
                                rec["blocked"] = True
                                if closest_obs is None or min_gap > gap:
                                    closest_obs = obs
                                    min_gap = gap
                        else:
                            rec["branch"] = "dyn/nopred/gap>=max_horizon"
                dbg["obs"].append(rec)
        else:
            # An OT/recovery cache with no valid path (is_init False, e.g. expired by
            # _expire_stale_cache) must NOT read as "free": treating a missing avoidance
            # path as clear keeps OVERTAKE alive on an empty cache, which then emits an
            # empty local_wpnts. Report blocked so the source is dropped instead.
            is_free = not (wpnts_data.is_ot_wpnts and not wpnts_data.is_init)

        dbg["is_free"] = bool(is_free)
        wpnts_data.free_dbg = dbg
        wpnts_data.closest_target = closest_obs
        wpnts_data.closest_gap = min_gap
        return is_free

    def _check_free_cartesian(self, wpnts_data) -> bool:
        is_free = True
        closest_obs = None
        min_gap = None
        min_horizon = wpnts_data.min_horizon
        max_horizon = wpnts_data.max_horizon
        free_scaling_reference_distance_m = wpnts_data.free_scaling_reference_distance_m
        lateral_width_m = wpnts_data.lateral_width_m

        obstacles = self.cur_obstacles_in_interest
        if wpnts_data.is_init:
            for obs in obstacles:
                obs_s = obs.s_center
                gap = (obs_s - self.cur_s) % self.max_s
                if gap < max_horizon or min_horizon < (gap - self.max_s):
                    dists = np.linalg.norm(wpnts_data.array[:, 0:2] - np.array([obs.x_m, obs.y_m]), axis=1)
                    min_dist = np.min(dists)
                    free_dist = min_dist - obs.size / 2 - self.gb_ego_width_m / 2
                    scaling_factor = np.clip(gap / free_scaling_reference_distance_m, 0.0, 1.0)
                    if free_dist < lateral_width_m * scaling_factor:
                        is_free = False
                        if closest_obs is None or min_gap > gap:
                            closest_obs = obs
                            min_gap = gap
                        self.get_logger().info(
                            f"[{self.name}] RECOVERY_FREE False, obs dist to recovery lane: {min_dist} m",
                            throttle_duration_sec=1.0,
                        )
        else:
            is_free = True
        wpnts_data.closest_target = closest_obs
        wpnts_data.closest_gap = min_gap
        return is_free

    def _src_cache(self, src):
        # The planner-output cache a given local_wpnts_src slices from (None if the
        # source is not backed by an OT/recovery cache, e.g. GB_TRACK).
        if src == StateType.OVERTAKE:
            # 초기 코드 : self.cur_S_wpnts 
            return self.cur_static_avoidance_wpnts if self.static_overtaking_mode else self.cur_avoidance_wpnts
        if src == StateType.RECOVERY:
            return self.cur_recovery_wpnts
        return None

    def _expire_stale_cache(self, wpnts_data, ttl_sec):
        # Drop a stale planner-output cache: one that is NOT the current source and
        # whose planner stopped emitting for > ttl_sec (a ghost / old frozen path).
        # The cache actively driven as local_wpnts_src is exempt -- kept alive by the
        # on_spline/hyst/killing hysteresis in _check_availability so the car keeps
        # following it through a few skipped solver frames. A cache that stays alive
        # (planner keeps publishing) but is not the current source is left intact so
        # it can be re-selected instantly with fresh data.
        if not wpnts_data.is_init:
            return
        if wpnts_data is self._src_cache(self.local_wpnts_src):
            return
        if wpnts_data.last_init_sec is None:
            return
        if self.now_sec() - wpnts_data.last_init_sec > ttl_sec:
            wpnts_data.is_init = False
            wpnts_data.closest_target = None

    def _check_availability(self, wpnts, wpnts_data) -> bool:
        if (self.now_sec() - time_to_float(wpnts_data.stamp)) > wpnts_data.killing_timer_sec:
            wpnts_data.is_init = False
            return bool(self._check_latest_wpnts(wpnts, wpnts_data))

        if (self.now_sec() - time_to_float(wpnts_data.stamp)) > wpnts_data.hyst_timer_sec:
            if self._check_latest_wpnts(wpnts, wpnts_data):
                return True

        if not self._check_on_spline(wpnts_data):
            return bool(self._check_latest_wpnts(wpnts, wpnts_data))

        return True

    def _check_sustainability(self, src_wpnts, wpnts_data) -> bool:
        if self._check_availability(src_wpnts, wpnts_data) and self._check_free_frenet(wpnts_data):
            return True
        return False

    @property
    def overtaking_ttl_sec(self) -> float:
        """[s] How long OVERTAKE is held after the obstacle stops being in front.

        Read live from the parameter, not cached. It used to be snapshotted in
        __init__ together with overtaking_ttl_count_threshold, so
        `ros2 param set /state_machine overtaking_ttl_sec X` updated the parameter
        the save-back writes out but NOT the number OvertakingTransition compares
        against -- the knob looked live and was not.
        """
        return float(self.params.overtaking_ttl_sec)

    @property
    def overtaking_ttl_count_threshold(self) -> int:
        """overtaking_ttl_sec expressed in main-loop iterations."""
        return int(self.overtaking_ttl_sec * self.rate_hz)

    def _check_overtaking_mode(self) -> bool:
        """DYNAMIC OVERTAKE *entry* gate.

            OT sector
            AND a dynamic opponent ahead within dynamic_overtake_max_gap_m
            AND the relative-speed condition
            AND the avoidance path is fresh
            AND the avoidance path is safe
            AND NOT force_trailing
            -> OVERTAKE

        force_trailing is an entry veto only. Once the car is in OVERTAKE,
        _check_overtaking_mode_sustainability decides whether to stay, and it
        does not look at force_trailing: a single frame in which the predictor
        drops back to its constant-velocity fallback must not abort a manoeuvre
        already underway and pull a car that is side by side back in behind the
        opponent. Leaving OVERTAKE stays governed by the existing path
        availability / free-frenet / overtaking_ttl logic.
        """
        if (
            self._check_ot_sector()
            and self._check_getting_closer(threshold_m=self.dynamic_overtake_max_gap_m)
            and not (self.use_force_trailing and self.force_trailing)
            and self._check_latest_wpnts(self.avoidance_wpnts, self.cur_avoidance_wpnts)
            and self._check_free_frenet(self.cur_avoidance_wpnts)
        ):
            self.static_overtaking_mode = False
            return True
        else:
            return False

    def _static_path_available(self) -> bool:
        """Static-only replacement for _check_latest_wpnts(), used as the OVERTAKE
        *entry* gate for static obstacles. Same freshness/init semantics (fresh,
        non-empty topic -> initialize_traj), but deliberately does NOT require
        _check_on_spline(): the static planner plans a path starting ahead of the
        ego (see 0e8b995, "plan early in Frenet"), so requiring the ego to already
        be within on_spline_min_dist_thres_m of the path's start before allowing
        entry stalled TRAILING for seconds despite a valid, safe path being ready.
        on_spline is still used to gate OVERTAKE *sustain* (_check_availability,
        called from _check_overtaking_mode_sustainability) and dynamic entry
        (_check_overtaking_mode() -> _check_latest_wpnts()), both unchanged.
        """
        wpnts_data = self.cur_static_avoidance_wpnts
        if wpnts_data.frozen:
            self._static_path_dbg = {"exists": bool(wpnts_data.is_init), "ttl_ok": True, "raw_n": None, "age": None}
            return bool(wpnts_data.is_init)

        # raw_n / age are logged even on failure so a run can tell apart "topic never
        # arrives" (raw_n stays None/0 -- publisher/remap/QoS/network problem) from
        # "arrives but is always judged stale" (raw_n > 0, age >> latest_threshold --
        # almost always a clock mismatch between this node and the publisher, e.g. one
        # of the two running with use_sim_time and the other not).
        src = self.static_avoidance_wpnts
        raw_n = None if src is None else len(src.wpnts)
        if src is None or raw_n == 0:
            self._static_path_dbg = {"exists": bool(wpnts_data.is_init), "ttl_ok": False, "raw_n": raw_n, "age": None}
            return False
        age = self.now_sec() - time_to_float(src.header.stamp)
        if age > wpnts_data.latest_threshold:
            self._static_path_dbg = {"exists": bool(wpnts_data.is_init), "ttl_ok": False, "raw_n": raw_n, "age": age}
            return False

        wpnts_data.initialize_traj(src)
        self._static_path_dbg = {"exists": True, "ttl_ok": True, "raw_n": raw_n, "age": age}
        return True

    def _check_static_overtaking_mode(self) -> bool:
        path_available = self._static_path_available()
        path_safe = self._check_free_frenet(self.cur_static_avoidance_wpnts) if path_available else False
        on_spline = self._check_on_spline(self.cur_static_avoidance_wpnts)
        decision = path_available and path_safe
        reason = None if decision else ("PATH_BLOCKED" if path_available else "NO_STATIC_PATH")

        if decision:
            if self._static_first_valid_sec is None:
                self._static_first_valid_sec = self.now_sec()
            if not self.static_overtaking_mode:
                latency_ms = (self.now_sec() - self._static_first_valid_sec) * 1000.0
                transition_line = f"[STATIC_OT_TRANSITION] first_valid_path_to_overtake_ms={latency_ms:.1f}"
                self.get_logger().info(transition_line)
                self._dbg_log(transition_line)
        else:
            self._static_first_valid_sec = None

        # When a path exists but is judged unsafe, surface the actual clearance
        # (metres to spare, negative = overlapping) of the tightest blocking obstacle
        # -- answers "is it TRAILING because the gap is really too narrow?" directly,
        # instead of just a blocked/free bool.
        blocked_detail = ""
        if path_available and not path_safe:
            free_dbg = self.cur_static_avoidance_wpnts.free_dbg or {}
            blocked_obs = [o for o in free_dbg.get("obs", []) if o.get("blocked")]
            if blocked_obs:
                worst = min(blocked_obs, key=lambda o: o.get("free_dist") if o.get("free_dist") is not None else 0.0)
                blocked_detail = (
                    f" blocked_obs_id={worst['id']} gap={worst['gap']}m "
                    f"free_dist={worst.get('free_dist')}m branch={worst.get('branch')}"
                )

        dbg = self._static_path_dbg or {}
        raw_n = dbg.get("raw_n")
        age = dbg.get("age")
        static_ot_line = (
            f"[STATIC_OT] state={self.cur_state.value} "
            f"path_exists={int(dbg.get('exists', False))} "
            f"path_locked={int(self._src_cache(self.local_wpnts_src) is self.cur_static_avoidance_wpnts)} "
            f"path_ttl_ok={int(dbg.get('ttl_ok', False))} path_safe={int(path_safe)} "
            f"on_spline={int(on_spline)} speed={self.cur_vs:.2f} "
            f"raw_wpnts={'-' if raw_n is None else raw_n} "
            f"raw_age={'-' if age is None else f'{age:.2f}'}s "
            f"decision={'OVERTAKE' if decision else 'TRAILING'}"
            + (f" reason={reason}" if reason else "")
            + blocked_detail
        )
        self.get_logger().info(static_ot_line, throttle_duration_sec=0.2)
        if self.now_sec() - self._dbg_last_static_log_sec > 0.2:
            self._dbg_last_static_log_sec = self.now_sec()
            self._dbg_log(static_ot_line)

        if decision:
            self.static_overtaking_mode = True
            return True
        else:
            return False

    def _update_opponent_memory(self):
        """Remember a dynamic opponent that was ahead, for a few seconds after it
        is no longer reported.

        WHY THIS EXISTS. On 2026-08-22 the car rear-ended the opponent at
        4.6 m/s (state_machine_20260822_063122.log, t=1787348098.8-101.9). The
        chain was:

            opponent last seen at gap 5.42 m, cluster size 0.16 m -- one
            centimetre above perception's min_size_m rejection floor of 0.15
              -> tracker publishes nothing (ttl_dynamic is 40 frames @ 40 Hz
                 = 1.0 s, the dropout lasted 2.6 s / 10.7 m)
              -> len(cur_obstacles_in_interest) == 0
              -> NonObstacleTransition -> GB_TRACK
              -> full raceline velocity profile, 2.94 -> 4.60 m/s
              -> impact

        Nothing in the stack objected, because nothing remembers. The trailing
        speed cap only runs in TRAILING, and the controller's AEB
        (AEB_for_weird_local_wpnt) watches the local waypoint, not obstacles.

        Deliberately a TIME window and not dead reckoning. Propagating the last
        sighting forward on that run gives a projected gap of -0.95 m at the
        moment of impact -- "we already passed it" -- because opp_vs was pinned
        at a held 1.40 m/s. A position estimate that confident and that wrong is
        worse than no estimate: this only says "something was ahead recently, do
        not run the raceline flat out yet".
        """
        if self.cur_state == StateType.OVERTAKE:
            self._last_overtake_sec = self.now_sec()

        nearest_gap = None
        nearest_id = None
        for obs in self.cur_obstacles_in_interest:
            if obs.is_static or self.track_length <= 0.0:
                continue
            gap = (obs.s_center - self.cur_s) % self.track_length
            if gap > 0.5 * self.track_length:
                continue
            if nearest_gap is None or gap < nearest_gap:
                nearest_gap = gap
                nearest_id = int(obs.id)

        if nearest_gap is not None:
            self._last_dyn_seen_sec = self.now_sec()
            self._last_dyn_gap_m = nearest_gap
            self._last_dyn_id = nearest_id

    def _opponent_memory_active(self) -> bool:
        """True while a recently-lost opponent should still hold the speed down.

        Two exclusions, both learned from state_machine_20260822_091939.log, the
        first run in which the hold actually worked. Of 136 holds, only ~17 were
        the case this exists for:

          84 fired with state=OVERTAKE, capping the car to 3.63 m/s against a
             7.29 m/s raceline WHILE IT WAS PASSING. An opponent drawing level
             leaves the ahead-window, which reads identically to losing it.
          35 more fired within 2 s of leaving OVERTAKE, last seen 1.38 m ahead
             (median) -- i.e. a car we had just successfully passed.

        So: never hold while OVERTAKE is in progress -- there the committed
        avoidance path plus the per-loop _check_free_frenet re-validation is the
        guard, and braking mid-pass is the opposite of safe. And never hold for
        overtake_pass_grace_sec after leaving OVERTAKE, because a car that
        vanishes right after a pass is behind us.

        The 2026-08-22 rear-end still qualifies: last seen 5.42 m ahead, gone for
        2.6 s, and the previous OVERTAKE had ended 8.4 s earlier.
        """
        if self._last_dyn_seen_sec is None or self.dynamic_opponent_memory_sec <= 0.0:
            return False
        if self.cur_state == StateType.OVERTAKE:
            return False
        if (self._last_overtake_sec is not None
                and self.now_sec() - self._last_overtake_sec < self.overtake_pass_grace_sec):
            return False
        if len(self.cur_obstacles_in_interest) != 0:
            # Something is visible right now; the normal paths handle it.
            for obs in self.cur_obstacles_in_interest:
                if not obs.is_static:
                    return False
        return (self.now_sec() - self._last_dyn_seen_sec) <= self.dynamic_opponent_memory_sec

    def _log_opponent_memory_state(self):
        """Unconditional 1 Hz trace of every input the memory hold depends on.

        [OPP_MEMORY] never appeared in the 06:54 or 07:16 runs despite 33 sampled
        frames that should have qualified (obstacle list empty, within 3 s of a
        dynamic sighting, state GB_TRACK/OVERTAKE/LOSTLINE) -- and the same code,
        called directly with those values, does fire. Rather than guess at which
        term differs on the car, print all of them.

        Only while NO dynamic obstacle is visible, and only at 1 Hz, so it costs
        nothing during normal running and stops entirely once the opponent is
        back in view.
        """
        if any(not o.is_static for o in self.cur_obstacles_in_interest):
            return
        if self.now_sec() - self._dbg_last_memstate_log_sec < 1.0:
            return
        self._dbg_last_memstate_log_sec = self.now_sec()
        age = (None if self._last_dyn_seen_sec is None
               else self.now_sec() - self._last_dyn_seen_sec)
        self._dbg_log(
            f"[OPP_MEM_DBG] state={self.cur_state.value} src={self.local_wpnts_src.value} "
            f"n_obs={len(self.cur_obstacles_in_interest)} "
            f"last_gap={'-' if self._last_dyn_gap_m is None else f'{self._last_dyn_gap_m:.2f}'} "
            f"age={'-' if age is None else f'{age:.2f}'} "
            f"window={self.dynamic_opponent_memory_sec:.2f} "
            f"ebh={self.emergency_break_horizon:.2f} "
            f"track_len={self.track_length:.2f} "
            f"active={int(self._opponent_memory_active())}"
        )

    def _log_dynamic_ot_decision(self):
        """One throttled line answering "why is the car not overtaking?".

        Only emitted while there is a dynamic opponent in range, so a clear track
        and a purely static-obstacle run stay silent. Throttled to 5 Hz on a node
        that loops at 50: everything below the throttle -- including the two
        refreshes -- runs at most five times a second.
        """
        # Throttle FIRST, so the refresh below runs at 5 Hz, not at the loop's 50.
        if self.now_sec() - self._dbg_last_dynamic_log_sec <= 0.2:
            return

        # loop() clears these each iteration, because the transition path does not
        # always reach the gate: ObstacleTransition returns GB_TRACK early when the
        # raceline is free, _check_ot_sector() short-circuits _check_overtaking_mode
        # before the target is picked, and NonObstacleTransition never calls either.
        # Without the refresh this line would print the previous loop's target and
        # sector -- worst of all in the out-of-sector case, which is exactly the one
        # the log exists to explain. Both are cheap: a scan of the handful of
        # obstacles in interest, and a scan of the overtake zones.
        if self._dyn_ot_target is None:
            self._check_getting_closer(threshold_m=self.dynamic_overtake_max_gap_m)
        if self.ot_section_check is None:
            self._check_ot_sector()

        target = self._dyn_ot_target
        if target is None:
            return
        self._dbg_last_dynamic_log_sec = self.now_sec()

        avoid = self.cur_avoidance_wpnts
        path_age = (None if avoid.stamp is None
                    else self.now_sec() - time_to_float(avoid.stamp))
        pred_age = (None if self._prediction_stamp_sec is None
                    else self.now_sec() - self._prediction_stamp_sec)
        pred_valid = int(len(self.obstacles_prediction) != 0
                         and self.obstacles_prediction_id == target["id"])
        diag = self._planner_diag or {}

        # First failing precondition, in the order _check_overtaking_mode applies
        # them -- the answer to "which gate stopped it", not a list of every gate.
        if self.cur_state == StateType.OVERTAKE:
            decision, reason = "OVERTAKE", None
        elif not self.ot_section_check:
            decision, reason = "TRAILING", "NOT_OT_SECTOR"
        elif not target["rel_ok"]:
            decision, reason = "TRAILING", "REL_SPEED"
        elif self.use_force_trailing and self.force_trailing:
            decision, reason = "TRAILING", "FORCE_TRAILING"
        elif not avoid.is_init:
            # No diag at all means the lane_change planner itself is not
            # publishing -- either it is not running, or it has not reached its
            # first loop. Say that rather than the generic NO_PATH.
            decision, reason = "TRAILING", (diag.get("reason") or
                                            ("NO_PATH" if self._planner_diag else "NO_PLANNER_DIAG"))
        else:
            decision, reason = "TRAILING", "PATH_BLOCKED"

        line = (
            f"[DYNAMIC_OT] target={target['id']} gap={target['gap']:.2f} "
            f"ego_v={self.cur_vs:.2f} opp_v={target['opp_vs']:.2f} rel_v={target['rel_v']:.2f} "
            f"sector={int(self.ot_section_check)} force_trailing={int(self.force_trailing)} "
            f"pred_age={'-' if pred_age is None else f'{pred_age:.2f}'} pred_valid={pred_valid} "
            f"path={int(avoid.is_init)} "
            f"path_age={'-' if path_age is None else f'{path_age:.2f}'} "
            f"safe={int(bool((avoid.free_dbg or {}).get('is_free', False)))} "
            f"left={diag.get('left', '-')} right={diag.get('right', '-')} "
            f"state={self.cur_state.value} decision={decision}"
            + (f" reason={reason}" if reason else "")
        )
        self.get_logger().info(line, throttle_duration_sec=0.2)
        self._dbg_log(line)

    def _check_overtaking_mode_sustainability(self) -> bool:
        """Whether to STAY in OVERTAKE. Intentionally does not read force_trailing:
        that flag vetoes entry, not continuation (see _check_overtaking_mode)."""
        if self.static_overtaking_mode:
            if (
                self._check_availability(self.static_avoidance_wpnts, self.cur_static_avoidance_wpnts)
                and self._check_free_frenet(self.cur_static_avoidance_wpnts)
            ):
                return True
        else:
            if self._check_availability(self.avoidance_wpnts, self.cur_avoidance_wpnts):
                self.get_logger().debug("AVAILABLE")
                if self._check_free_frenet(self.cur_avoidance_wpnts):
                    return True
        return False

    ################
    # HELPER FUNCS #
    ################
    def update_velocity(self, wpnts_msg, safety_factor=1.0, speed_cap=None, speed_scale=1.0):
        """Recompute a physically-consistent velocity profile for `wpnts_msg`.

        `wpnts_msg` is either an object with a `.wpnts` list (OTWpntArray/WpntArray)
        or a plain `List[Wpnt]` (e.g. local_wpnts) -- both are mutated in place.

        `speed_cap`, if given, caps both the profile's ceiling (v_max) and its far
        end (v_end) at that value -- used by `_apply_trailing_speed_cap` to bring the
        cruise speed DOWN while TRAILING near an unavoidable-yet obstacle, without
        touching the braking curve (ax_max_machines/b_ax_max_machines) itself. That
        keeps this a normal, physically-smooth slow-down to a lower cruise speed --
        not an emergency stop (v_end still floors at 0 only if speed_cap does).

        `speed_scale` multiplies the finished profile, exactly as the sector tuner
        multiplies the raceline to produce /global_waypoints_scaled. It exists
        because that scaling never reached the avoidance path: the planner emits
        vx_mps = 0 for every point and this function rebuilds the profile from raw
        ggv/ax_max_machines, so only v_end (the last point) inherited the sector
        scaling. The raceline ran at 1.05-1.2x while the avoidance path ran at 1.0x.
        Still clipped to veh_params v_max, and NOT applied to speed_cap: a scale
        must never raise a cap that exists to slow the car down.
        """
        if self.ggv is None or self.gb_wpnts is None:
            return  # velocity replanning unavailable (no veh dyn info / no gb wpnts yet)
        wpnts = wpnts_msg.wpnts if hasattr(wpnts_msg, "wpnts") else wpnts_msg
        if len(wpnts) < 3:
            return
        kappa = np.array([wp.kappa_radpm for wp in wpnts])
        el_lengths = np.array([
            np.linalg.norm([
                wpnts[i + 1].x_m - wpnts[i].x_m,
                wpnts[i + 1].y_m - wpnts[i].y_m,
            ])
            for i in range(len(wpnts) - 1)
        ])
        # Bail if the path is degenerate: a zero-length segment or any non-finite input makes
        # calc_vel_profile divide by zero -> NaN velocities that propagate into the local path
        # and eventually the base_link TF. Leaving the original vx_mps untouched is the safe path.
        if (el_lengths <= 1e-6).any() or not np.all(np.isfinite(el_lengths)) \
                or not np.all(np.isfinite(kappa)):
            self.get_logger().warn(
                f"[{self.name}] degenerate path in update_velocity; keeping planner velocities",
                throttle_duration_sec=1.0,
            )
            return

        glb_start_idx = int(wpnts[-1].s_m / self.wpnt_dist)
        v_end = self.gb_wpnts.wpnts[glb_start_idx % len(self.gb_wpnts.wpnts)].vx_mps
        v_max = self.pars["veh_params"]["v_max"]
        if speed_cap is not None:
            v_end = min(v_end, speed_cap)
            v_max = min(v_max, speed_cap)

        ax_max_machines_sf = self.ax_max_machines.copy()
        b_ax_max_machines_sf = self.b_ax_max_machines.copy()
        ax_max_machines_sf[:, 1] *= safety_factor
        b_ax_max_machines_sf[:, 1] *= safety_factor

        vx_profile = calc_vel_profile(
            ax_max_machines=ax_max_machines_sf,
            kappa=kappa,
            el_lengths=el_lengths,
            closed=False,
            drag_coeff=self.pars["veh_params"]["dragcoeff"],
            m_veh=self.pars["veh_params"]["mass"],
            b_ax_max_machines=b_ax_max_machines_sf,
            ggv=self.ggv,
            v_max=v_max,
            filt_window=self.pars["vel_calc_opts"]["vel_profile_conv_filt_window"],
            dyn_model_exp=self.pars["vel_calc_opts"]["dyn_model_exp"],
            v_start=self.cur_vs,
            v_end=v_end,
        )

        if speed_scale != 1.0:
            vx_profile = np.minimum(vx_profile * float(speed_scale),
                                    self.pars["veh_params"]["v_max"])
            if speed_cap is not None:
                vx_profile = np.minimum(vx_profile, speed_cap)

        for i in range(len(vx_profile)):
            wpnts[i].vx_mps = vx_profile[i]

        ax_profile = tph.calc_ax_profile.calc_ax_profile(
            vx_profile=vx_profile, el_lengths=el_lengths, eq_length_output=False
        )
        for i in range(len(ax_profile)):
            wpnts[i].ax_mps2 = ax_profile[i]
        wpnts[len(ax_profile)].ax_mps2 = ax_profile[-1]

    def _apply_trailing_speed_cap(self, local_wpnts):
        """While TRAILING (obstacle recognised, no avoidance path committed yet),
        cut the cruise speed once the blocking obstacle is within
        emergency_break_horizon -- filling the gap between "obstacle detected" and
        "OVERTAKE/RECOVERY has a valid path", which otherwise runs GB_TRACK/RECOVERY
        at full raceline speed the whole time (states.GlobalTracking/get_recovery_wpts
        don't replan velocity). NOT an emergency stop: a lower cruise ceiling
        (trailing_speed_scale * raceline speed, floored at trailing_min_speed_mps)
        that calc_vel_profile still reaches via its normal decel curve.

        Returns local_wpnts unchanged if the cap does not apply. Otherwise returns a
        COPY with the reduced profile applied -- local_wpnts's Wpnt objects, for the
        GB_TRACK source, are the very same objects as self.gb_wpnts.wpnts (see
        update_waypoints: cur_gb_wpnts.list = self.gb_wpnts.wpnts, no copy), so
        mutating them in place would leak the trailing-reduced speed back into the
        shared global-waypoints message.
        """
        if not local_wpnts:
            return local_wpnts

        memory_hold = False
        if self.cur_state == StateType.TRAILING:
            if self.local_wpnts_src == StateType.GB_TRACK:
                gap = self.cur_gb_wpnts.closest_gap
            elif self.local_wpnts_src == StateType.RECOVERY:
                gap = self.cur_recovery_wpnts.closest_gap
            else:
                return local_wpnts
        elif self._opponent_memory_active():
            # An opponent was ahead within the last dynamic_opponent_memory_sec and
            # perception has since lost it. The state machine has already left
            # TRAILING (no obstacle -> GB_TRACK), which is exactly how the car came
            # to hit a stationary-ish opponent at 4.6 m/s on 2026-08-22. Keep the
            # cap on, using the LAST KNOWN gap, until the window expires or the
            # opponent is seen again. See _update_opponent_memory for why this does
            # not try to estimate where the opponent went.
            gap = self._last_dyn_gap_m
            memory_hold = True
        else:
            return local_wpnts

        if gap is None or gap > self.emergency_break_horizon:
            return local_wpnts
        if not self.num_glb_wpnts or self.gb_wpnts is None:
            return local_wpnts  # not initialised yet -- shouldn't be reachable in TRAILING, but don't /0

        idx = int(self.cur_s / self.wpnt_dist + 0.5) % self.num_glb_wpnts
        raceline_v = self.gb_wpnts.wpnts[idx].vx_mps

        # RAMP THE CAP WITH THE GAP, do not step it.
        #
        # This used to apply trailing_speed_scale in full the instant the gap
        # crossed emergency_break_horizon: at 8.01 m the car ran the raceline
        # flat out, at 7.99 m it was capped to 35% of it. On a 10.3 m/s straight
        # that is a 10.3 -> 3.6 m/s cliff triggered by a centimetre of gap, which
        # reads from the driver's seat as "it slows down hard, far too early" --
        # and a gap that noisy sits right on the edge and toggles.
        #
        # Now the scale interpolates: 1.0 (no cut at all) at the horizon, down to
        # trailing_speed_scale as the gap closes to zero. The car starts easing
        # off the moment the opponent is visible, which is the point of the wide
        # horizon, but the deep cut is spent where it is actually needed. Braking
        # distance is quadratic in speed, so most of the benefit is already there
        # by the time the scale is halfway down.
        #
        # Set trailing_speed_ramp false for the old step behaviour.
        scale = self.trailing_speed_scale
        if self.trailing_speed_ramp and self.emergency_break_horizon > 1e-6:
            t = min(max(gap / self.emergency_break_horizon, 0.0), 1.0)
            scale = self.trailing_speed_scale + (1.0 - self.trailing_speed_scale) * t
        cap = max(self.trailing_min_speed_mps, raceline_v * scale)

        if memory_hold and self.now_sec() - self._dbg_last_memory_log_sec > 0.5:
            self._dbg_last_memory_log_sec = self.now_sec()
            line = (
                f"[OPP_MEMORY] opponent id={self._last_dyn_id} lost "
                f"{self.now_sec() - self._last_dyn_seen_sec:.2f}s ago at gap={gap:.2f}m -- "
                f"holding speed cap {cap:.2f} m/s (raceline {raceline_v:.2f}) "
                f"state={self.cur_state.value}"
            )
            self.get_logger().warn(line, throttle_duration_sec=0.5)
            self._dbg_log(line)

        capped_wpnts = [copy.deepcopy(wp) for wp in local_wpnts]
        self.update_velocity(capped_wpnts, speed_cap=cap)
        return capped_wpnts

    def mincurv_splinification(self):
        coords = np.empty((len(self.cur_gb_wpnts.list), 4))
        for i, wpnt in enumerate(self.cur_gb_wpnts.list):
            coords[i, 0] = wpnt.s_m
            coords[i, 1] = wpnt.x_m
            coords[i, 2] = wpnt.y_m
            coords[i, 3] = wpnt.vx_mps
        self.mincurv_spline_x = Spline(coords[:, 0], coords[:, 1])
        self.mincurv_spline_y = Spline(coords[:, 0], coords[:, 2])
        self.mincurv_spline_v = Spline(coords[:, 0], coords[:, 3])
        self.get_logger().info(f"[{self.name}] Splinified Min Curve")

    def ot_splinification(self):
        coords = np.empty((len(self.overtake_wpnts), 5))
        for i, wpnt in enumerate(self.overtake_wpnts):
            coords[i, 0] = wpnt.s_m
            coords[i, 1] = wpnt.x_m
            coords[i, 2] = wpnt.y_m
            coords[i, 3] = wpnt.d_m
            coords[i, 4] = wpnt.vx_mps
        coords = coords[coords[:, 0].argsort()]
        # Drop non-finite rows and duplicate/non-increasing s: scipy Spline requires a
        # strictly increasing x or it raises / returns NaN. A reversed or seam-jumped
        # overtake path would otherwise poison every downstream spline eval with NaN.
        coords = coords[np.isfinite(coords).all(axis=1)]
        if len(coords) >= 2:
            keep = np.concatenate([[True], np.diff(coords[:, 0]) > 1e-6])
            coords = coords[keep]
        if len(coords) < 4:
            self.get_logger().warn(
                f"[{self.name}] overtake wpnts degenerate ({len(coords)} usable); skipping splinification",
                throttle_duration_sec=1.0,
            )
            return
        self.ot_spline_x = Spline(coords[:, 0], coords[:, 1])
        self.ot_spline_y = Spline(coords[:, 0], coords[:, 2])
        self.ot_spline_d = Spline(coords[:, 0], coords[:, 3])
        self.ot_spline_v = Spline(coords[:, 0], coords[:, 4])
        self.get_logger().info(f"[{self.name}] Splinified Overtaking Curve")

    def _find_nearest_ot_s(self) -> float:
        half_search_dim = 5
        idxs = [
            i % self.num_ot_points
            for i in range(self.cur_id_ot - half_search_dim, self.cur_id_ot + half_search_dim)
        ]
        ses = np.array([self.overtake_wpnts[i].s_m for i in idxs])
        dists = np.abs(self.cur_s - ses)
        chose_id = np.argmin(dists)
        s_ot = idxs[chose_id]
        s_ot %= self.num_ot_points
        return s_ot

    def get_splini_wpts(self) -> WpntArray:
        if self.static_overtaking_mode:
            wpnts = self.cur_static_avoidance_wpnts
        else:
            wpnts = self.cur_avoidance_wpnts

        # Never slice an invalidated cache: once _expire_stale_cache drops a path
        # (planner stopped emitting), is_init is False and its array is a frozen
        # old trajectory. Returning [] here makes the caller fall back to GB_TRACK
        # instead of emitting a stale/behind-the-car local path.
        if not wpnts.is_init:
            self._splini_dbg = {"static": bool(self.static_overtaking_mode), "invalid_cache": True}
            return []

        diff = np.linalg.norm(wpnts.array[:, 0:2] - self.current_position[:2], axis=1)
        min_idx = np.argmin(diff)
        avoidance_wpnts = wpnts.list[min_idx:min_idx + self.n_loc_wpnts]

        n_from_avoid = len(avoidance_wpnts)
        glb_extended = 0
        if len(avoidance_wpnts) < self.n_loc_wpnts:
            glb_start_idx = int(wpnts.list[-1].s_m / self.wpnt_dist) + 1
            extra_wpnts = [
                self.cur_gb_wpnts.list[(glb_start_idx + i) % len(self.cur_gb_wpnts.list)]
                for i in range(self.n_loc_wpnts - len(avoidance_wpnts))
            ]
            avoidance_wpnts.extend(extra_wpnts)
            glb_extended = len(extra_wpnts)

        # Record exactly how this OVERTAKE local path was assembled so the debug
        # snapshot can explain a frozen/behind-the-car first_s (argmin pick vs
        # cache extent vs global-fill), instead of guessing from the raw topic.
        self._splini_dbg = {
            "static": bool(self.static_overtaking_mode),
            "min_idx": int(min_idx),
            "min_dist": round(float(diff[min_idx]), 3),
            "pick_s": round(float(wpnts.array[min_idx, 2]), 3),
            "cache_n": int(len(wpnts.list)),
            "cache_s0": round(float(wpnts.array[0, 2]), 3),
            "cache_slast": round(float(wpnts.array[-1, 2]), 3),
            "n_from_avoid": int(n_from_avoid),
            "glb_extended": int(glb_extended),
        }
        return avoidance_wpnts

    def get_recovery_wpts(self) -> WpntArray:
        if self.cur_recovery_wpnts.is_init:
            diff = np.linalg.norm(self.cur_recovery_wpnts.array[:, 0:2] - self.current_position[:2], axis=1)
            min_idx = np.argmin(diff)
            wpnts = self.cur_recovery_wpnts.list[min_idx:min_idx + self.n_loc_wpnts]
            n_from_rec = len(wpnts)
            glb_extended = 0
            if len(wpnts) < self.n_loc_wpnts:
                glb_start_idx = int(self.cur_recovery_wpnts.list[-1].s_m / self.wpnt_dist)
                extra_wpnts = [
                    self.cur_gb_wpnts.list[(glb_start_idx + i) % len(self.cur_gb_wpnts.list)]
                    for i in range(self.n_loc_wpnts - len(wpnts))
                ]
                wpnts.extend(extra_wpnts)
                glb_extended = len(extra_wpnts)
            self._recovery_dbg = {
                "min_idx": int(min_idx),
                "min_dist": round(float(diff[min_idx]), 3),
                "pick_s": round(float(self.cur_recovery_wpnts.array[min_idx, 2]), 3),
                "cache_n": int(len(self.cur_recovery_wpnts.list)),
                "cache_s0": round(float(self.cur_recovery_wpnts.array[0, 2]), 3),
                "cache_slast": round(float(self.cur_recovery_wpnts.array[-1, 2]), 3),
                "n_from_rec": int(n_from_rec),
                "glb_extended": int(glb_extended),
            }
            return wpnts

    def get_start_wpts(self) -> WpntArray:
        if self.cur_start_wpnts.is_init:
            diff = np.linalg.norm(self.cur_start_wpnts.array[:, 0:2] - self.current_position[:2], axis=1)
            min_idx = np.argmin(diff)
            start_wpnts = self.cur_start_wpnts.list[min_idx:min_idx + self.n_loc_wpnts]
            if len(start_wpnts) < self.n_loc_wpnts:
                glb_start_idx = int(self.cur_start_wpnts.list[-1].s_m / self.wpnt_dist) + 1
                extra_wpnts = [
                    self.cur_gb_wpnts.list[(glb_start_idx + i) % len(self.cur_gb_wpnts.list)]
                    for i in range(self.n_loc_wpnts - len(start_wpnts))
                ]
                start_wpnts.extend(extra_wpnts)
            return start_wpnts
        else:
            self.get_logger().debug(f"[{self.name}] No valid avoidance waypoints, passing global waypoints")

    #######
    # VIZ #
    #######
    def _publish_debug(self, local_wpnts):
        # Emit a per-loop JSON snapshot on /state_machine/debug capturing the
        # full local_wpnts source-selection state. Purpose: catch a stale source
        # cache (e.g. cur_recovery_wpnts frozen at an old snapshot while the raw
        # recovery topic keeps advancing) leaking into local_wpnts, and the
        # controller-poisoning "car ran off the end of a frozen local path"
        # condition (idx near the tail -> empty curvature slice -> NaN).
        def s0(wpnts):
            return round(wpnts[0].s_m, 3) if wpnts else None

        first_s = local_wpnts[0].s_m if local_wpnts else None
        last_s = local_wpnts[-1].s_m if local_wpnts else None
        # frenet gap between car and where the emitted local path starts (wrap-safe)
        gap = None
        if first_s is not None and self.track_length:
            ds = (first_s - self.cur_s) % self.track_length
            gap = round(min(ds, self.track_length - ds), 3)

        rec = self.cur_recovery_wpnts
        avoid = self.cur_avoidance_wpnts

        static_avoid = self.cur_static_avoidance_wpnts # 주은 추가

        snap = {
            "t": round(self.now_sec(), 3),
            "src": self.local_wpnts_src.name,
            "state": self.cur_state.name,
            "cur_s": round(self.cur_s, 3),
            "cur_d": round(self.cur_d, 4),
            "cur_vs": round(self.cur_vs, 3),
            "close_to_raceline": bool(self._check_close_to_raceline(0.05)
                                      * self._check_close_to_raceline_heading(20)),
            "n_obs": len(self.cur_obstacles_in_interest),
            "local_first_s": None if first_s is None else round(first_s, 3),
            "local_last_s": None if last_s is None else round(last_s, 3),
            "local_n": len(local_wpnts) if local_wpnts else 0,
            "start_gap_m": gap,  # >~2m means a stale cache leaked in
            "recovery": {
                "topic_s": s0(self.recovery_wpnts.wpnts) if self.recovery_wpnts is not None else None,
                "cache_s": s0(rec.list),
                "cache_last_s": round(rec.list[-1].s_m, 3) if rec.list else None,
                "cache_age": (None if rec.stamp is None
                              else round(self.now_sec() - time_to_float(rec.stamp), 3)),
                # Wall-clock since the cache was last actually re-initialized with
                # fresh planner output, and total re-init count. If reinit_age keeps
                # growing while the topic advances, the cache is stale (never re-slotted).
                "reinit_age": (None if rec.last_init_sec is None
                               else round(self.now_sec() - rec.last_init_sec, 3)),
                "reinit_count": rec.init_count,
                "is_init": rec.is_init,
            },
            "avoidance": {
                "topic_s": s0(self.avoidance_wpnts.wpnts) if self.avoidance_wpnts is not None else None,
                "cache_s": s0(avoid.list),
                "cache_last_s": round(avoid.list[-1].s_m, 3) if avoid.list else None,
                "cache_age": (None if avoid.stamp is None
                              else round(self.now_sec() - time_to_float(avoid.stamp), 3)),
                "reinit_age": (None if avoid.last_init_sec is None
                               else round(self.now_sec() - avoid.last_init_sec, 3)),
                "reinit_count": avoid.init_count,
                "is_init": avoid.is_init,
            },
            #--------------- 주은 추가
            "static_avoidance": {
                "topic_s": s0(self.static_avoidance_wpnts.wpnts) if self.static_avoidance_wpnts is not None else None,
                "cache_s": s0(static_avoid.list),
                "cache_last_s": round(static_avoid.list[-1].s_m, 3) if static_avoid.list else None,
                "cache_age": (None if static_avoid.stamp is None
                            else round(self.now_sec() - time_to_float(static_avoid.stamp), 3)),
                "reinit_age": (None if static_avoid.last_init_sec is None
                            else round(self.now_sec() - static_avoid.last_init_sec, 3)),
                "reinit_count": static_avoid.init_count,
                "is_init": static_avoid.is_init,
            },
            #---------------

            # Internal slice detail from get_splini_wpts / get_recovery_wpts for
            # THIS loop (None if that source was not used). Shows the exact
            # min_idx, the s it picked, cache extent, and how many points came
            # from the avoidance/recovery cache vs global-fill -> pinpoints why
            # local_first_s sits where it does (argmin pick vs glb-extend).
            "splini_slice": self._splini_dbg,
            "recovery_slice": self._recovery_dbg,
            # Last free-check decisions this loop (why GB/recovery was judged free
            # or blocked). gb_free explains a "drove into an obstacle ahead" event.
            "gb_free": self.cur_gb_wpnts.free_dbg,

            #--------------- 주은 추가
            "static_free": static_avoid.free_dbg,
            #---------------

            # The dynamic overtake candidate this loop (nearest opponent ahead
            # within dynamic_overtake_max_gap_m, plus the relative-speed verdict),
            # or None when there is none. Replaces "getting_closer_static", which
            # called _check_getting_closer(7.0) for its value: that name never
            # matched what it measured, and now that the selector skips static
            # obstacles it would read False for every static-only run. Nothing in
            # this workspace consumed it. No extra work -- the value is whatever
            # _check_overtaking_mode already computed this loop.
            "dynamic_ot_target": self._dyn_ot_target,
            "planner_diag": self._planner_diag,


            "recovery_free": self.cur_recovery_wpnts.free_dbg,
        }
        self.debug_pub.publish(String(data=json.dumps(snap)))

    def _pub_local_wpnts(self, wpts):
        """Publish /local_waypoints (CONTROL) and, only if watched, its markers (VIZ).

        The two halves are deliberately separated and ordered control-first.

        WHY: this used to build one Marker per waypoint on every tick --
        n_loc_wpnts (80) x rate (50) = 4000 Marker objects a second. A Humble
        Marker is a nested message tree; measured on this stack, building and
        publishing 80 of them costs ~2.2 ms per call on an x86 dev box and 3-4x
        that on the Jetson, which put this single function at roughly 35% of one
        core -- most of the state_machine's whole CPU share. It ran whether or
        not anyone was looking, so a race with no RViz open paid all of it.

        Two fixes, both of which cost nothing in racing behaviour:
          1. gate on subscribers -- with pitwall closed this returns immediately,
             and with pitwall OPEN the output is byte-for-byte what it was.
             Unchecking the display in RViz now also unsubscribes, so it saves
             the car CPU too.
          2. one SPHERE_LIST instead of N SPHEREs -- same picture, one Marker.
             The z of each point still carries vx_mps, as before.

        NEVER gate self.loc_wpnt_pub: controller_manager zeroes speed after
        ~200 ms without /local_waypoints (its waypoint_safety_counter), so a
        subscriber check there would stop the car whenever RViz was closed.
        """
        loc_wpnts = WpntArray()
        loc_wpnts.wpnts = wpts if wpts is not None else []
        loc_wpnts.header.stamp = self.get_clock().now().to_msg()
        loc_wpnts.header.frame_id = "map"

        # ---- CONTROL PATH: unconditional, and first, so viz can never delay it.
        self.loc_wpnt_pub.publish(loc_wpnts)

        # ---- VIZ PATH: skipped entirely when nothing is subscribed, and
        # rate-limited to viz_rate_hz when something is.
        if self.vis_loc_wpnt_pub.get_subscription_count() == 0:
            return
        if self.viz_rate_hz <= 0.0:
            return
        _now = time.monotonic()
        if _now - self._last_viz_sec < 1.0 / self.viz_rate_hz:
            return
        self._last_viz_sec = _now

        mrk = Marker()
        mrk.header.frame_id = "map"
        mrk.header.stamp = loc_wpnts.header.stamp
        mrk.type = Marker.SPHERE_LIST
        mrk.action = Marker.ADD
        mrk.id = 0
        # For a SPHERE_LIST scale is the sphere diameter, shared by every point.
        mrk.scale.x = 0.15
        mrk.scale.y = 0.15
        mrk.scale.z = 0.15
        mrk.color.a = 1.0
        mrk.color.g = 1.0
        mrk.pose.orientation.w = 1.0
        mrk.points = [Point(x=w.x_m, y=w.y_m, z=w.vx_mps) for w in loc_wpnts.wpnts]

        # A single fixed-id marker is REPLACED on each publish, so the DELETEALL
        # that used to lead this array is no longer needed to clear stale spheres.
        loc_markers = MarkerArray()
        loc_markers.markers.append(mrk)
        self.vis_loc_wpnt_pub.publish(loc_markers)

    def visualize_state(self, state: str):
        if self.first_visualization:
            self.first_visualization = False
            x0 = self.cur_gb_wpnts.list[0].x_m
            y0 = self.cur_gb_wpnts.list[0].y_m
            x1 = self.cur_gb_wpnts.list[1].x_m
            y1 = self.cur_gb_wpnts.list[1].y_m
            xy_norm = (
                -np.array([y1 - y0, x0 - x1]) / np.linalg.norm([y1 - y0, x0 - x1])
                * 1.25 * self.cur_gb_wpnts.list[0].d_left
            )
            self.x_viz = x0 + xy_norm[0]
            self.y_viz = y0 + xy_norm[1]

        mrk = Marker()
        mrk.type = mrk.SPHERE
        mrk.id = 1
        mrk.header.frame_id = "map"
        mrk.header.stamp = self.get_clock().now().to_msg()
        mrk.color.a = 1.0
        mrk.pose.position.x = float(self.x_viz)
        mrk.pose.position.y = float(self.y_viz)
        mrk.pose.position.z = 0.0
        mrk.pose.orientation.w = 1.0
        mrk.scale.x = 1.0
        mrk.scale.y = 1.0
        mrk.scale.z = 1.0

        if state == "GB_TRACK":
            mrk.color.b = 1.0
        elif state == "OVERTAKE":
            mrk.color.r = 1.0
            mrk.color.g = 0.0
            mrk.color.b = 0.0
        elif state == "TRAILING":
            mrk.color.r = 1.0
            mrk.color.g = 1.0
            mrk.color.b = 0.0
        elif state == "ATTACK":
            mrk.color.r = 1.0
            mrk.color.g = 0.0
            mrk.color.b = 1.0
        elif state == "FTGONLY":
            mrk.color.r = 1.0
            mrk.color.g = 1.0
            mrk.color.b = 1.0
        elif state == "RECOVERY":
            mrk.color.r = 0.0
            mrk.color.g = 1.0
            mrk.color.b = 0.0
        else:
            mrk.color.r = 1.0
            mrk.color.g = 1.0
            mrk.color.b = 1.0
        self.state_mrk.publish(mrk)

    def publish_not_ready_marker(self):
        mrk = Marker()
        mrk.type = mrk.TEXT_VIEW_FACING
        mrk.id = 1
        mrk.header.frame_id = "map"
        mrk.header.stamp = self.get_clock().now().to_msg()
        mrk.color.a = 1.0
        mrk.color.r = 1.0
        mrk.color.g = 0.0
        mrk.color.b = 0.0
        mrk.pose.position.x = float(np.mean([wpnt.x_m for wpnt in self.cur_gb_wpnts.list]))
        mrk.pose.position.y = float(np.mean([wpnt.y_m for wpnt in self.cur_gb_wpnts.list]))
        mrk.pose.position.z = 1.0
        mrk.pose.orientation.w = 1.0
        mrk.scale.x = 4.69
        mrk.scale.y = 4.69
        mrk.scale.z = 4.69
        mrk.text = "BATTERY TOO LOW!!!"
        self.emergency_pub.publish(mrk)

    def update_waypoints(self):
        if not self.cur_gb_wpnts.is_init:
            self.cur_gb_wpnts.initialize_traj(self.gb_wpnts)
        else:
            self.cur_gb_wpnts.list = self.gb_wpnts.wpnts
        self.cur_obstacles_in_interest = self.obstacles_in_interest
        return

    def get_overtaking_target(self):
        if self.cur_gb_wpnts.closest_target is not None:
            return [self.cur_gb_wpnts.closest_target]
        if self.cur_recovery_wpnts.closest_target is not None:
            return [self.cur_recovery_wpnts.closest_target]
        else:
            return []

    def get_traling_target(self):
        if self.local_wpnts_src == StateType.GB_TRACK and self.cur_gb_wpnts.closest_target is not None:
            return [self.cur_gb_wpnts.closest_target]
        elif self.local_wpnts_src == StateType.RECOVERY and self.cur_recovery_wpnts.closest_target is not None:
            return [self.cur_recovery_wpnts.closest_target]
        elif self.local_wpnts_src == StateType.OVERTAKE and self.ot_closest_target is not None:
            return [self.ot_closest_target]
        else:
            return []

    def get_farthest_target(self, local_wpnts_src):
        # TRAILING must NOT hijack the src to OVERTAKE here: overtaking is gated by the
        # OVERTAKE state (sector/getting_closer/free_frenet). Pulling the raw avoidance
        # trajectory into local_wpnts while merely trailing would steer the car onto an
        # un-committed OT line -- exactly what the OT-blended recovery path exists to
        # avoid. Keep the src the transition chose (GB/RECOVERY); only pick the trailing
        # target (the farthest-ahead obstacle) off that same source.
        if local_wpnts_src == StateType.GB_TRACK and self.cur_gb_wpnts.closest_target is not None:
            return [self.cur_gb_wpnts.closest_target], local_wpnts_src

        if local_wpnts_src == StateType.RECOVERY and self.cur_recovery_wpnts.closest_target is not None:
            return [self.cur_recovery_wpnts.closest_target], local_wpnts_src

        return [], local_wpnts_src

    def check_ot_cloest_target(self):
        if self.gb_closest_target is not None and self.ot_closest_target is not None and \
                self.local_wpnts_src == StateType.GB_TRACK:
            if self.ot_closest_gap > self.gb_closest_gap:
                self.local_wpnts_src = StateType.OVERTAKE
        elif self.cur_recovery_wpnts.closest_target is not None and self.ot_closest_target is not None and \
                self.local_wpnts_src == StateType.RECOVERY:
            if self.ot_closest_gap > self.cur_recovery_wpnts.closest_gap:
                self.local_wpnts_src = StateType.OVERTAKE

    def save_params_to_yaml(self):
        # ROS1 dynamic_statemachine_server.save_yaml: persist the dynamic tunables to
        # state_machine_params.yaml, preserving the other keys.
        import yaml
        try:
            path = os.path.join(get_package_share_directory("stack_master"),
                                "config", "state_machine_params.yaml")
        except Exception:
            self.get_logger().error(f"[{self.name}] cannot locate state_machine_params.yaml")
            return
        keys = ["lateral_width_gb_m", "lateral_width_ot_m", "overtaking_ttl_sec",
                "splini_hyst_timer_sec", "splini_ttl", "pred_splini_ttl",
                "emergency_break_horizon", "trailing_speed_scale", "trailing_min_speed_mps",
                "ftg_speed_mps", "ftg_timer_sec",
                "ftg_active", "force_GBTRACK"]
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f) or {}
            block = data.setdefault("state_machine", {}).setdefault("ros__parameters", {})
            for k in keys:
                if self.has_parameter(k):
                    block[k] = self.get_parameter(k).value
            block["save_params"] = False
            with open(path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            self.get_logger().info(f"[{self.name}] saved params to {path}")
        except Exception as e:
            self.get_logger().error(f"[{self.name}] failed to save params: {e}")

    def _handle_momentary_params(self):
        # Act on the rqt buttons outside the on-set callback so set_parameters() is safe.
        if self._save_start_traj_requested:
            self._save_start_traj_requested = False
            self.save_start_traj_pub.publish(Bool(data=True))
            self.set_parameters([rclpy.parameter.Parameter('save_start_traj', rclpy.Parameter.Type.BOOL, False)])
        if self._save_params_requested:
            self._save_params_requested = False
            self.save_params_to_yaml()
            self.set_parameters([rclpy.parameter.Parameter('save_params', rclpy.Parameter.Type.BOOL, False)])

    #############
    # MAIN LOOP #
    #############
    def loop(self):
        self._splini_dbg = None
        self._recovery_dbg = None
        # Per-loop, not persistent: see _log_dynamic_ot_decision. None means "not
        # evaluated yet this loop", which is different from "evaluated, no target".
        self._dyn_ot_target = None
        self.ot_section_check = None
        self._handle_momentary_params()
        if self.measuring:
            start = time.perf_counter()

        self.update_waypoints()
        self.gb_closest_target = None
        self.ot_closest_target = None
        need_vel_planner = False

        self.cur_gb_wpnts.closest_target = None
        self.cur_recovery_wpnts.closest_target = None
        self.cur_avoidance_wpnts.closest_target = None
        self.cur_static_avoidance_wpnts.closest_target = None
        self.cur_start_wpnts.closest_target = None

        # Expire any planner-output cache whose planner stopped emitting for >1 s, so
        # a frozen path (old avoidance/static/recovery output) can't keep being sliced
        # into local_wpnts or keep passing the sustainability/free checks. A live
        # planner re-inits every loop, so a path actively being driven never expires.
        self._expire_stale_cache(self.cur_avoidance_wpnts, 2.0)
        self._expire_stale_cache(self.cur_static_avoidance_wpnts, 2.0)
        self._expire_stale_cache(self.cur_recovery_wpnts, 2.0)

        # Obstacle-recognition snapshot for the debug log: is perception/tracking
        # actually reporting the obstacle(s) the car is reacting to, with real gap/d?
        # Throttled independently of the [STATIC_OT] line (fires even with 0 obstacles,
        # so "nothing was ever detected" is distinguishable from "detected but blocked").
        if self.now_sec() - self._dbg_last_obs_log_sec > 0.5:
            self._dbg_last_obs_log_sec = self.now_sec()
            if len(self.cur_obstacles_in_interest) != 0:
                obs_summary = "; ".join(
                    f"id={o.id} static={int(o.is_static)} "
                    f"gap={(o.s_start - self.cur_s) % self.track_length:.2f}m "
                    f"d={o.d_center:.2f} size={o.size:.2f}"
                    for o in self.cur_obstacles_in_interest
                )
            else:
                obs_summary = "none"
            self._dbg_log(
                f"[OBSTACLES] cur_s={self.cur_s:.2f} cur_d={self.cur_d:.2f} vs={self.cur_vs:.2f} "
                f"n={len(self.cur_obstacles_in_interest)} [{obs_summary}]"
            )

        # safety check
        if self.cur_volt < self.volt_threshold:
            self.get_logger().error(
                f"[{self.name}] VOLTS TOO LOW, STOP THE CAR", throttle_duration_sec=1.0
            )
            self.publish_not_ready_marker()

        if self.force_gbtrack_state:
            self.cur_state = StateType.GB_TRACK
            self.local_wpnts_src = StateType.GB_TRACK
        elif self._check_only_ftg_zone():
            self.cur_state = StateType.FTGONLY
            self.local_wpnts_src = StateType.FTGONLY
            self.get_logger().warn(f"[{self.name}] FTGONLY sector !!!")
        else:
            self.cur_state, self.local_wpnts_src = self.state_transitions[self.cur_state](self)

        self._update_opponent_memory()
        self._log_opponent_memory_state()
        self._log_dynamic_ot_decision()

        if self.cur_state.value != self._dbg_last_state_value:
            self._dbg_log(
                f"[STATE_CHANGE] {self._dbg_last_state_value} -> {self.cur_state.value} "
                f"src={self.local_wpnts_src.value} s={self.cur_s:.2f} d={self.cur_d:.2f} vs={self.cur_vs:.2f}"
            )
            self._dbg_last_state_value = self.cur_state.value

        if self.cur_state == StateType.TRAILING:
            # NOTE: check_ot_cloest_target() intentionally NOT called -- it promoted the
            # src to OVERTAKE while merely trailing (un-committed OT line). See
            # get_farthest_target for the rationale; overtaking is gated by the state.
            self.behavior_strategy.trailing_targets, self.local_wpnts_src = \
                self.get_farthest_target(self.local_wpnts_src)
        else:
            self.behavior_strategy.trailing_targets = []

        self.behavior_strategy.overtaking_targets = self.get_overtaking_target()

        # Rule 2 (source change): when the source switches to a different cache (e.g.
        # RECOVERY->GB as the car reaches the raceline), drop the cache we just left so
        # its output can't linger and be re-selected as a ghost. A live planner re-fills
        # its cache via _check_latest_wpnts next time that source is needed, so this only
        # discards a path we actually stopped driving.
        cur_src_cache = self._src_cache(self.local_wpnts_src)
        if self._prev_src_cache is not None and self._prev_src_cache is not cur_src_cache:
            self._prev_src_cache.is_init = False
            self._prev_src_cache.closest_target = None
        self._prev_src_cache = cur_src_cache

        # Freeze the recovery/blended path while it is the active source: capture it on
        # entry (the transition already re-inited the cache with fresh output this loop)
        # and hold that single path until we leave RECOVERY, so the controller target
        # stops jumping as recovery_spliner re-anchors the blended path every frame.
        self._hold_recovery_freeze()

        # Post-OVERTAKE grace: keep the blended-recovery source eligible for a short
        # window after leaving OVERTAKE (see _select_recovery_source), so the OT->GB
        # blend can smooth the return instead of snapping to GB the instant OVERTAKE
        # ends. Refresh while overtaking, decrement once we are out.
        if self.cur_state == StateType.OVERTAKE:
            self._blended_grace_count = self.blended_recovery_grace_loops
        elif self._blended_grace_count > 0:
            self._blended_grace_count -= 1

        local_wpnts = self.states[self.local_wpnts_src](self)

        # Safety net: never publish an empty local path (an invalidated OT/recovery
        # cache makes its slice return []). An empty WpntArray crashes the controller
        # (1-D waypoint array indexed as 2-D). Fill from the global raceline, which is
        # regenerated at the car every loop. Only the PATH source is swapped -- cur_state
        # is left as the transition decided (e.g. TRAILING keeps trailing/braking), so
        # this never turns "obstacle ahead" into a full-speed GB run.
        if not local_wpnts:
            self.local_wpnts_src = StateType.GB_TRACK
            local_wpnts = self.states[StateType.GB_TRACK](self)

        local_wpnts = self._apply_trailing_speed_cap(local_wpnts)

        self._publish_debug(local_wpnts)

        if self.cur_state == StateType.LOSTLINE:
            self.cur_state = StateType.GB_TRACK

        need_vel_planner = False
        self.behavior_strategy.header.stamp = self.get_clock().now().to_msg()
        self.behavior_strategy.local_wpnts = local_wpnts if local_wpnts is not None else []
        self.behavior_strategy.state = self.cur_state.value
        self.behavior_strategy.need_vel_planner = need_vel_planner

        self.behavior_strategy_pub.publish(self.behavior_strategy)

        self.state_pub.publish(String(data=self.cur_state.value))
        self.visualize_state(state=self.cur_state.value)

        self._pub_local_wpnts(local_wpnts)

        if self.cur_state != StateType.TRAILING and self.cur_state != StateType.ATTACK:
            self.ftg_counter = 0

        overtaking_target_mrk = Marker()
        overtaking_target_mrk.header.frame_id = "map"  # set always so the DELETEALL marker isn't dropped by RViz (empty frame)
        if len(self.behavior_strategy.overtaking_targets) != 0:
            overtaking_target_mrk.type = Marker.SPHERE
            overtaking_target_mrk.scale.x = 0.5
            overtaking_target_mrk.scale.y = 0.5
            overtaking_target_mrk.scale.z = 0.5
            overtaking_target_mrk.color.a = 1.0
            overtaking_target_mrk.color.b = 1.0
            overtaking_target_mrk.pose.position.x = self.behavior_strategy.overtaking_targets[0].x_m
            overtaking_target_mrk.pose.position.y = self.behavior_strategy.overtaking_targets[0].y_m
            overtaking_target_mrk.pose.orientation.w = 1.0
        else:
            overtaking_target_mrk.action = Marker.DELETEALL
        self.overtaking_marker_pub.publish(overtaking_target_mrk)

        trailing_target_mrk = Marker()
        trailing_target_mrk.header.frame_id = "map"  # set always so the DELETEALL marker isn't dropped by RViz (empty frame)
        if len(self.behavior_strategy.trailing_targets) != 0:
            trailing_target_mrk.type = Marker.SPHERE
            trailing_target_mrk.scale.x = 0.5
            trailing_target_mrk.scale.y = 0.5
            trailing_target_mrk.scale.z = 0.5
            trailing_target_mrk.color.a = 1.0
            trailing_target_mrk.color.g = 1.0
            trailing_target_mrk.pose.position.x = self.behavior_strategy.trailing_targets[0].x_m
            trailing_target_mrk.pose.position.y = self.behavior_strategy.trailing_targets[0].y_m
            trailing_target_mrk.pose.orientation.w = 1.0
        else:
            trailing_target_mrk.action = Marker.DELETEALL
        self.trailing_marker_pub.publish(trailing_target_mrk)

        if self.measuring:
            end = time.perf_counter()
            self.latency_pub.publish(Float32(data=1.0 / (end - start)))


# defined as entry point in setup.py:
def main(args=None):
    rclpy.init(args=args)
    state_machine = StateMachine()
    try:
        rclpy.spin(state_machine)
    except KeyboardInterrupt:
        pass
    state_machine.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
