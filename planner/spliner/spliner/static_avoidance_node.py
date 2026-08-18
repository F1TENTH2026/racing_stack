#!/usr/bin/env python3
"""
Static-obstacle avoidance planner.

Plans an evasion path around up to three visible STATIC obstacles at once, in
Frenet space, anchored at the car's current pose.

Two properties matter and both are deliberate:

1. It reads `/tracking/obstacles` DIRECTLY. The previous version waited for the
   state machine to nominate an obstacle on `/behavior_strategy`
   (`overtaking_targets`), which is only populated from
   `cur_gb_wpnts.closest_target` -- and that field is cleared at the top of every
   state-machine loop and only re-filled by `_check_free_frenet(cur_gb_wpnts)`,
   which `OvertakingTransition` never calls. So the moment the car entered
   OVERTAKE the nomination vanished, this planner lost its obstacle, published an
   empty path, and the state machine dropped back out of OVERTAKE. Reading the
   tracker directly breaks that loop. `/behavior_strategy` is still consumed, but
   only as a *hint* for which candidate to prefer, so the two agree when the state
   machine does have an opinion.

2. It plans in FRENET and converts to cartesian, instead of fitting a cartesian
   BPoly through {ego, apex, return}. A three-point cartesian spline from the car
   to an apex 8 m away is essentially a straight chord: on any curved section it
   cuts the corner, leaves the eroded map, and every candidate is rejected. The
   rejection disappears only once the car is close enough for the chord to be
   short -- which is exactly the "crawl up to the obstacle, then plan" behaviour
   this file used to have. A Frenet path follows the raceline curvature by
   construction, so a 10 m lookahead is as plannable as a 2 m one.

Subscribes:
    - `/tracking/obstacles`        : tracked obstacles (static ones are the input)
    - `/behavior_strategy`         : optional hint for which obstacle to prefer
    - `/car_state/odom_frenet`     : ego state in Frenet coordinates
    - `/car_state/odom`            : ego state in cartesian coordinates (heading)
    - `/global_waypoints`          : global waypoints (Frenet converter reference)
    - `/global_waypoints_scaled`   : scaled global waypoints (bounds + speeds)

Publishes:
    - `/planner/avoidance/otwpnts` : the evasion trajectory (remapped to
                                     /planner/avoidance/static_otwpnts by the launch file)
    - `/planner/avoidance/markers` : trajectory visualization
    - `/planner/avoidance/latency` : planning callback duration [s] (when measure:=true)
"""
import os
import time
import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from f110_msgs.msg import BehaviorStrategy, Obstacle, ObstacleArray, OTWpntArray, Wpnt, WpntArray
from frenet_conversion.frenet_converter import FrenetConverter
from grid_filter.grid_filter import GridFilter
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from scipy.interpolate import CubicSpline
from std_msgs.msg import Float32
from transforms3d.euler import quat2euler
from visualization_msgs.msg import Marker, MarkerArray

# --- Fixed geometry. NOT tunables: they define the collision invariant. ---
SAFETY_MARGIN_M = 0.025     # per side, on top of half the car width
MIN_SAMPLES = 4             # below this the curvature math is degenerate
MAX_ENTRY_SLOPE = 0.4       # [m/m] cap on dd/ds at the ego anchor, so a bad heading
                            # estimate cannot whip the spline sideways

# Relaxation ladder: (evasion scale, return-distance scale, side key).
# Tried in order; the first candidate that passes every safety check wins. The
# effective clearance is max(scale * evasion_distance, floor), where the floor is
# ego_width_m/2 + min_free_dist_m -- no rung can plan closer than that.
RELAXATION_LADDER = (
    (1.0, 1.0, "preferred"),
    (1.0, 1.0, "opposite"),
    (0.7, 1.4, "wider"),
    (0.0, 1.4, "wider"),
)


def _opposite(side: str) -> str:
    return "right" if side == "left" else "left"


def _default_debug_log_dir() -> Path:
    """<repo>/logfile if this source file lives inside a git checkout (true for a
    symlink-install, which is how this workspace is normally built -- __file__
    resolves through the symlink to the tracked source), so debug logs land
    somewhere `git add logfile/` already picks up instead of needing a manual copy
    out of the home directory after every run. Falls back to a fixed home-directory
    path if the source isn't inside a repo (e.g. a non-symlink install, where
    __file__ points into install/ instead of src/). Kept identical to
    state_machine_node.py's helper of the same name -- separate packages, no shared
    import between them.
    """
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / ".git").exists():
            return parent / "logfile"
    return Path("~/roboracer_debug_logs").expanduser()


class StaticObstacleSpliner(Node):
    """Plans one same-side evasion path around visible static obstacles."""

    def __init__(self):
        super().__init__("static_avoidance_planner")
        self.name = "static_avoidance_planner"

        # --- inputs ---
        self.obstacles: List[Obstacle] = []
        self.hint_obs: Optional[Obstacle] = None
        self.gb_wpnts: Optional[WpntArray] = None
        self.gb_scaled_wpnts: Optional[WpntArray] = None
        self.cur_s = None
        self.cur_d = None
        self.cur_vs = 0.0
        self.cur_x = None
        self.cur_y = None
        self.cur_yaw = 0.0

        # --- derived global-line arrays (rebuilt on every scaled update) ---
        self.gb_s = None
        self.gb_d_left = None
        self.gb_d_right = None
        self.gb_vx = None
        self.gb_ax = None
        self.gb_psi = None
        self.gb_x = None
        self.gb_y = None
        self.gb_vmax = 1.0
        self.max_s = None
        self.wpnt_dist = 0.1

        # --- planner memory ---
        self.last_side: Optional[str] = None
        self.last_side_obs_id: Optional[int] = None
        self.last_good_path: Optional[OTWpntArray] = None
        self.last_good_obs_id: Optional[int] = None
        self.last_good_generated = 0
        self.last_good_origin_s = 0.0    # cur_s when the held path was built
        self.last_good_reach = 0.0       # how far ahead of that s the path reached

        # Per-run debug log file (logfile/static_avoidance_*.log): off by default so
        # a normal race doesn't pay disk-write overhead for output nobody reads; pass
        # debug:=true on race.launch.xml (-> debug_log_enabled:=true here) to get it.
        self.debug_log_enabled = bool(
            self.declare_parameter("debug_log_enabled", False).value)
        self._setup_debug_log_file()

        self.declare_all_parameters()
        self._load_parameters()

        self.map_filter = GridFilter(node=self, map_topic="/map", debug=False)
        self.map_filter.set_erosion_kernel_size(self.kernel_size)
        # Registered only once map_filter exists: the callback re-erodes on a
        # kernel_size change and would otherwise touch an attribute that is not there.
        self.add_on_set_parameters_callback(self._parameter_cb)

        self.create_subscription(ObstacleArray, "/tracking/obstacles", self.obstacles_cb, 10)
        self.create_subscription(BehaviorStrategy, "/behavior_strategy", self.behavior_cb, 10)
        self.create_subscription(Odometry, "/car_state/odom_frenet", self.state_frenet_cb, 10)
        self.create_subscription(Odometry, "/car_state/odom", self.state_cb, 10)
        self.create_subscription(WpntArray, "/global_waypoints", self.gb_cb, 10)
        self.create_subscription(WpntArray, "/global_waypoints_scaled", self.gb_scaled_cb, 10)

        self.evasion_pub = self.create_publisher(OTWpntArray, "/planner/avoidance/otwpnts", 10)
        self.mrks_pub = self.create_publisher(MarkerArray, "/planner/avoidance/markers", 10)
        self.latency_pub = self.create_publisher(Float32, "/planner/avoidance/latency", 10)

        self.wait_for_messages()
        self.converter = self.initialize_converter()
        # The timer period is fixed at construction (rclpy cannot retime a timer),
        # so rate_hz is read once here even though the rest is live-tunable.
        self.create_timer(1.0 / max(1.0, self.rate_hz), self.loop)

    ##############
    # PARAMETERS #
    ##############
    def declare_all_parameters(self):
        defaults = {
            # --- cadence ---
            "rate_hz": 20.0,
            "log_period_s": 0.5,
            "measure": True,
            # --- what to plan around ---
            "lookahead": 10.0,
            "keep_behind_m": 1.0,
            "trajectory_threshold": 0.6,
            "raceline_clearance_m": 0.35,
            "max_group_obstacles": 3,
            # --- path shape ---
            "evasion_distance": 0.30,
            "spline_resolution": 0.10,
            "pre_dist_gain": 1.0,
            "pre_dist_min": 2.0,
            "pre_dist_max": 4.0,
            "post_dist_gain": 0.8,
            "post_dist_min": 1.5,
            "post_dist_max": 4.0,
            "tail_m": 4.0,
            "min_path_end_m": 11.0,
            "min_apex_lead_m": 0.5,
            # --- safety ---
            "boundary_margin": 0.19,
            "ego_width_m": 0.29,
            "min_free_dist_m": 0.16,
            "ego_grace_m": 1.0,
            "use_map_filter": True,
            "kernel_size": 4,
            "max_speed_mps": 2.0,
            # --- stability ---
            "path_hold_s": 0.25,
            "side_hysteresis_m": 0.10,
        }
        for key, value in defaults.items():
            self.declare_parameter(key, value)

    _PARAM_ATTRS = {
        "rate_hz": "rate_hz",
        "log_period_s": "log_period_s",
        "measure": "measure",
        "lookahead": "lookahead",
        "keep_behind_m": "keep_behind_m",
        "trajectory_threshold": "trajectory_threshold",
        "raceline_clearance_m": "raceline_clearance_m",
        "max_group_obstacles": "max_group_obstacles",
        "evasion_distance": "evasion_distance",
        "spline_resolution": "resolution",
        "pre_dist_gain": "pre_dist_gain",
        "pre_dist_min": "pre_dist_min",
        "pre_dist_max": "pre_dist_max",
        "post_dist_gain": "post_dist_gain",
        "post_dist_min": "post_dist_min",
        "post_dist_max": "post_dist_max",
        "tail_m": "tail_m",
        "min_path_end_m": "min_path_end_m",
        "min_apex_lead_m": "min_apex_lead_m",
        "boundary_margin": "boundary_margin",
        "ego_width_m": "ego_width_m",
        "min_free_dist_m": "min_free_dist_m",
        "ego_grace_m": "ego_grace_m",
        "use_map_filter": "use_map_filter",
        "kernel_size": "kernel_size",
        "max_speed_mps": "max_speed_mps",
        "path_hold_s": "path_hold_s",
        "side_hysteresis_m": "side_hysteresis_m",
    }

    def _load_parameters(self):
        for name, attr in self._PARAM_ATTRS.items():
            setattr(self, attr, self.get_parameter(name).value)

    def _parameter_cb(self, params):
        for param in params:
            attr = self._PARAM_ATTRS.get(param.name)
            if attr is None:
                continue
            setattr(self, attr, param.value)
            if param.name == "kernel_size":
                self.map_filter.set_erosion_kernel_size(int(param.value))
        self.get_logger().info(
            f"[{self.name}] lookahead {self.lookahead:.1f} m, evasion {self.evasion_distance:.2f} m "
            f"(floor {self.min_evasion_m:.3f} m), boundary margin {self.boundary_margin:.2f} m, "
            f"path_hold {self.path_hold_s:.2f} s, side hysteresis {self.side_hysteresis_m:.2f} m, "
            f"group max {int(self.max_group_obstacles)}"
        )
        return SetParametersResult(successful=True)

    # A third of the lap. Both horizons below are capped by it because `lookahead`
    # and `min_path_end_m` are lengths in metres, and what they MEAN depends on the
    # track: 10 m is a modest look ahead on a 60 m circuit and nearly half the lap
    # on this 22.3 m one. A path spanning half the lap breaks the arithmetic that
    # assumes "ahead" and "behind" are separable -- every (s - cur_s) % max_s
    # comparison, here and in the state machine, wraps at max_s/2.
    _TRACK_FRACTION = 1.0 / 3.0

    @property
    def _lookahead_eff(self) -> float:
        return min(self.lookahead, self.max_s * self._TRACK_FRACTION)

    @property
    def _path_end_eff(self) -> float:
        return min(self.min_path_end_m, self.max_s * self._TRACK_FRACTION)

    @property
    def min_evasion_m(self) -> float:
        """Hard floor on the clearance from the obstacle EDGE to the path.

        The path is the car's centreline, so half the car plus the free gap the
        state machine insists on (`min_free_dist_m`, which must stay >= the state
        machine's `static_avoidance_planner.lateral_width_m`) is the least that can
        ever be planned. No relaxation rung goes below this.
        """
        return self.ego_width_m / 2 + max(self.min_free_dist_m, SAFETY_MARGIN_M)

    #############
    # CALLBACKS #
    #############
    def obstacles_cb(self, data: ObstacleArray):
        self.obstacles = list(data.obstacles)

    def behavior_cb(self, data: BehaviorStrategy):
        # Hint only: it says which obstacle the state machine currently blames for
        # blocking the raceline. Used to break ties between candidates so both ends
        # agree, never to decide WHETHER to plan -- see the module docstring.
        self.hint_obs = data.overtaking_targets[0] if len(data.overtaking_targets) else None

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
        if not data.wpnts:
            return
        self.gb_wpnts = data
        self.max_s = data.wpnts[-1].s_m
        if len(data.wpnts) > 1:
            self.wpnt_dist = data.wpnts[1].s_m - data.wpnts[0].s_m

    def gb_scaled_cb(self, data: WpntArray):
        if not data.wpnts:
            return
        self.gb_scaled_wpnts = data
        self.gb_s = np.array([w.s_m for w in data.wpnts])
        self.gb_d_left = np.array([w.d_left for w in data.wpnts])
        self.gb_d_right = np.array([w.d_right for w in data.wpnts])
        self.gb_vx = np.array([w.vx_mps for w in data.wpnts])
        self.gb_ax = np.array([w.ax_mps2 for w in data.wpnts])
        self.gb_psi = np.array([w.psi_rad for w in data.wpnts])
        self.gb_x = np.array([w.x_m for w in data.wpnts])
        self.gb_y = np.array([w.y_m for w in data.wpnts])
        self.gb_vmax = max(float(np.max(self.gb_vx)), 1e-3)

    #########
    # SETUP #
    #########
    def wait_for_messages(self):
        self.get_logger().info(f"[{self.name}] Waiting for messages and services...")
        while None in (self.cur_s, self.cur_x, self.gb_wpnts, self.gb_scaled_wpnts):
            rclpy.spin_once(self)
        self.get_logger().info(f"[{self.name}] Ready!")

    def initialize_converter(self) -> FrenetConverter:
        wpnts = self.gb_wpnts.wpnts
        converter = FrenetConverter(
            np.array([w.x_m for w in wpnts]),
            np.array([w.y_m for w in wpnts]),
            np.array([w.psi_rad for w in wpnts]),
        )
        self.get_logger().info(f"[{self.name}] initialized FrenetConverter object")
        return converter

    #############
    # MAIN LOOP #
    #############
    def loop(self):
        started = time.perf_counter()
        dbg = {}
        path = self._plan(dbg)

        cache_used = False
        if path.wpnts:
            # New valid path: it becomes the held path. Its stamp is now, so the
            # state machine sees it as fresh.
            self.last_good_path = path
            self.last_good_obs_id = dbg.get("obs_id")
            self.last_good_generated = int(dbg.get("generated_count", len(path.wpnts)))
            self.last_good_origin_s = self.cur_s
            self.last_good_reach = float(dbg.get("reach", 0.0))
            dbg["path_age"] = 0.0
        else:
            held, reason = self._usable_cache(dbg)
            if held is not None:
                # Republish the held path WITH ITS ORIGINAL STAMP. That is the point:
                # it is not claimed to be fresh, and the state machine's own
                # latest_threshold (0.25 s) still decides whether it is too old to
                # use. This only stops a single dropped planning frame from being
                # published as a positive assertion that there is nothing to avoid,
                # which is what dropped the car out of OVERTAKE mid-manoeuvre.
                path = held
                cache_used = True
                dbg["path_age"] = self._age(held)
            else:
                self.last_good_path = None
                self.last_good_obs_id = None
                dbg["cache_reason"] = reason

        dbg["cache_used"] = cache_used
        dbg["path_points"] = len(path.wpnts)
        dbg["planning_time_ms"] = (time.perf_counter() - started) * 1e3
        self._log(dbg)

        self.evasion_pub.publish(path)
        self.mrks_pub.publish(self._markers(path))
        if self.measure:
            self.latency_pub.publish(Float32(data=float(time.perf_counter() - started)))

    ###############
    # PATH CACHE  #
    ###############
    def _age(self, msg: OTWpntArray) -> float:
        stamp = msg.header.stamp
        return self.get_clock().now().nanoseconds * 1e-9 - (stamp.sec + stamp.nanosec * 1e-9)

    def _usable_cache(self, dbg) -> Tuple[Optional[OTWpntArray], str]:
        """The held path, or None plus the reason it was discarded.

        A held path is never kept indefinitely. It is dropped as soon as any of
        the reasons it existed stops being true.
        """
        path = self.last_good_path
        if path is None or not path.wpnts:
            return None, "no_cache"
        if not dbg.get("obs_present", False):
            # Case 4: the obstacle is gone. Nothing to avoid, return to the raceline.
            return None, "obstacle_gone"
        if self.last_good_obs_id is not None and dbg.get("obs_id") is not None \
                and self.last_good_obs_id != dbg["obs_id"]:
            return None, "different_obstacle"
        age = self._age(path)
        if age > self.path_hold_s:
            return None, "cache_expired"
        # How much of the held path is still ahead. Measured from how far the car has
        # travelled SINCE the path was built, not as (path_end_s - cur_s) % max_s:
        # that wraps at max_s/2, so on a short track every path longer than half a
        # lap read as "already passed" and the hold never applied -- which is
        # precisely the far-obstacle case the hold exists for.
        travelled = (self.cur_s - self.last_good_origin_s + self.max_s / 2) % self.max_s \
            - self.max_s / 2
        remaining = self.last_good_reach - travelled
        if remaining < self.min_apex_lead_m:
            return None, "path_passed"
        # Re-check the held path against the obstacles as they are measured NOW: a
        # cached path that has become a collision is worse than no path at all.
        n = max(MIN_SAMPLES, min(self.last_good_generated, len(path.wpnts)))
        cached_u = np.array([self.cur_s + self._signed_gap(w.s_m) for w in path.wpnts[:n]])
        if not self._path_clear_of(cached_u, np.array([w.d_m for w in path.wpnts[:n]]),
                                   dbg.get("candidates", [])):
            return None, "cache_collision"
        return path, ""

    ############
    # PLANNING #
    ############
    def _plan(self, dbg) -> OTWpntArray:
        out = OTWpntArray()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = "map"

        if self.max_s is None or self.gb_s is None or self.max_s <= 0.0:
            dbg["reason"] = "no_global_line"
            return out

        candidates = self._candidates()
        dbg["candidates"] = candidates
        dbg["ego_s"] = self.cur_s
        if not candidates:
            # No obstacle in range: forget the side latch so the next obstacle is
            # judged on its own geometry.
            self.last_side = None
            self.last_side_obs_id = None
            dbg["obs_present"] = False
            dbg["reason"] = "no_static_obstacle_in_range"
            return out

        group = self._pick_target_group(candidates)
        group_key = tuple(sorted(int(o.id) for o in group))
        first = group[0]
        first_gap = self._signed_gap(first.s_center)
        last_gap = max(self._signed_gap(o.s_center) for o in group)
        dbg.update({"obs_present": True, "obs_id": int(first.id), "obs_dist": first_gap,
                    "obs_s": first.s_center, "obs_d": first.d_center,
                    "group_ids": list(group_key), "group_size": len(group),
                    "group_span": max(0.0, last_gap - first_gap)})

        # Nothing to avoid: the raceline itself already passes the obstacle with
        # room to spare. Publishing the raceline as an "avoidance path" would make
        # the state machine measure it against the same obstacle and refuse it,
        # forever. Say nothing instead. The threshold must be >= the state
        # machine's own "raceline is blocked" test
        # (gb_ego_width_m/2 + global_tracking.lateral_width_m) or the car trails an
        # obstacle the state machine thinks needs avoiding and this planner does not.
        blockers = []
        for obstacle in group:
            near_edge = obstacle.d_right if obstacle.d_center >= 0.0 else -obstacle.d_left
            if near_edge < self.raceline_clearance_m:
                blockers.append(obstacle)
        if not blockers:
            dbg["reason"] = "raceline_already_clear"
            return out

        # The apex is pushed ahead of the car once we are alongside the obstacle,
        # instead of dropping the path: an empty array here makes the state machine
        # leave OVERTAKE mid-manoeuvre and snap back onto the raceline, straight
        # into the obstacle it was avoiding.
        first_u = self.cur_s + max(min(self._signed_gap(o.s_center) for o in blockers),
                                   self.min_apex_lead_m)
        last_u = self.cur_s + max(max(self._signed_gap(o.s_center) for o in blockers),
                                  self.min_apex_lead_m)

        speed = max(abs(self.cur_vs), 1.0)
        pre_dist = float(np.clip(self.pre_dist_gain * speed, self.pre_dist_min, self.pre_dist_max))
        base_post = float(np.clip(self.post_dist_gain * speed, self.post_dist_min, self.post_dist_max))

        left_room, right_room = self._group_side_rooms(
            blockers, first_u, last_u, pre_dist, base_post, self.evasion_distance)
        dbg["left_clearance"] = left_room
        dbg["right_clearance"] = right_room

        preferred = self._preferred_group_side(group_key, left_room, right_room)
        wider = "left" if left_room >= right_room else "right"
        sides = {"preferred": preferred, "opposite": _opposite(preferred), "wider": wider}

        tried = set()
        last_reason = "no_valid_candidate"
        for rung, (evasion_scale, post_scale, side_key) in enumerate(RELAXATION_LADDER):
            side = sides[side_key]
            evasion = max(evasion_scale * self.evasion_distance, self.min_evasion_m)
            post_dist = min(base_post * post_scale, self.max_s / 3)
            key = (side, round(evasion, 3), round(post_dist, 3))
            if key in tried:
                continue
            tried.add(key)
            apex_d = (max(o.d_left for o in blockers) + evasion) if side == "left" \
                else (min(o.d_right for o in blockers) - evasion)
            built, reason = self._build_group_path(
                blockers, candidates, first_u, last_u, apex_d, pre_dist, post_dist)
            if built is None:
                last_reason = reason
                continue

            sample_s, sample_d, xy, psi, kappa, generated_count, reach = built
            dbg["generated_count"] = generated_count
            dbg["reach"] = reach
            if rung > 0:
                self.get_logger().info(
                    f"[{self.name}] evasion found on ladder rung {rung} "
                    f"(side={side}, clearance={evasion:.2f} m, return={post_dist:.1f} m)",
                    throttle_duration_sec=2.0,
                )
            self.last_side = side
            self.last_side_obs_id = group_key
            dbg.update({"side": side.upper(), "candidate_valid": True,
                        "apex_d": apex_d, "rung": rung})
            self._fill_msg(out, sample_s, sample_d, xy, psi, kappa, side)
            return out

        dbg.update({"candidate_valid": False, "reason": last_reason,
                    "side": (self.last_side or "-").upper()})
        return out

    def _candidates(self) -> List[Obstacle]:
        """Static obstacles close enough, and close enough to the line, to matter.

        `keep_behind_m` keeps an obstacle the car is currently passing in the list.
        Dropping it the instant its s went behind the car ended the manoeuvre
        half-way through, before the return leg had been driven.
        """
        out = []
        for obs in self.obstacles:
            if not obs.is_static:
                continue    # dynamic opponents belong to the overtaking planner
            gap = self._signed_gap(obs.s_center)
            if -self.keep_behind_m <= gap < self._lookahead_eff \
                    and abs(obs.d_center) < self.trajectory_threshold:
                out.append(obs)
        return out

    def _signed_gap(self, s: float) -> float:
        """Longitudinal distance to `s`, negative when it is behind the car."""
        return (s - self.cur_s + self.max_s / 2) % self.max_s - self.max_s / 2

    def _pick_target_group(self, candidates: List[Obstacle]) -> List[Obstacle]:
        """Return up to three visible obstacles in travel order.

        A state-machine hint no longer switches planning from A to B: if the
        hinted obstacle is visible it is included with the other visible
        obstacles, and one path is built for the complete group.
        """
        ordered = sorted(candidates, key=lambda o: self._signed_gap(o.s_center))
        limit = max(1, min(3, int(self.max_group_obstacles)))
        return ordered[:limit]

    def _gb_idx(self, s):
        """Index into the scaled global line for one or more (wrapped) s values.

        Searched rather than computed as s/wpnt_dist: that only lands on the right
        waypoint while the raceline is exactly uniformly spaced, and a line that is
        not silently offsets every bound and speed lookup along the path.
        """
        return np.clip(np.searchsorted(self.gb_s, np.asarray(s)), 0, len(self.gb_s) - 1)

    def _side_rooms(self, obs, candidates, apex_u, pre_dist, post_dist):
        """Free width either side of the obstacle, over the whole corridor the path
        drives through, plus whichever side is blocked by a *different* obstacle.

        Scanning the corridor rather than only the obstacle's own waypoint matters:
        a side that is wide at the apex and narrows two metres later is not usable,
        and picking it is how the planner used to talk itself into a path that the
        boundary check then threw away.
        """
        span = np.arange(apex_u - pre_dist, apex_u + post_dist, self.wpnt_dist) % self.max_s
        idx = self._gb_idx(span)
        min_d_left = float(np.min(self.gb_d_left[idx]))
        min_d_right = float(np.min(self.gb_d_right[idx]))

        left_apex = obs.d_left + self.evasion_distance
        right_apex = obs.d_right - self.evasion_distance
        left_room = min_d_left - left_apex
        right_room = min_d_right + right_apex

        # Do not dodge into the next obstacle: only the nearest one shapes the path,
        # but the path still has to get past everything inside its own span, and
        # obstacles staggered on opposite sides are exactly the case where dodging
        # the first aims at the second.
        span_start = self._signed_gap(obs.s_center) - pre_dist
        span_end = self._signed_gap(obs.s_center) + post_dist

        def occupied_by(apex_d):
            for other in candidates:
                if other.id == obs.id:
                    continue
                ahead = self._signed_gap(other.s_center)
                if not span_start <= ahead <= span_end:
                    continue
                if (other.d_right - self.evasion_distance < apex_d
                        < other.d_left + self.evasion_distance):
                    return other
            return None

        return left_room, right_room, occupied_by(left_apex), occupied_by(right_apex)

    def _group_side_rooms(self, group, first_u, last_u, pre_dist, post_dist, evasion):
        """Track room remaining after holding one side around the whole group."""
        span = np.arange(first_u - pre_dist, last_u + post_dist, self.wpnt_dist) % self.max_s
        idx = self._gb_idx(span)
        left_apex = max(o.d_left for o in group) + evasion
        right_apex = min(o.d_right for o in group) - evasion
        return (float(np.min(self.gb_d_left[idx])) - left_apex,
                float(np.min(self.gb_d_right[idx])) + right_apex)

    def _preferred_group_side(self, group_key, left_room, right_room) -> str:
        """Sticky same-side choice for a complete obstacle group."""
        bias = 0.0
        if self.last_side_obs_id == group_key:
            bias = self.side_hysteresis_m if self.last_side == "left" else -self.side_hysteresis_m
        return "left" if left_room + bias >= right_room else "right"

    def _preferred_side(self, obs, left_room, right_room, left_taken, right_taken) -> str:
        """Which side to try first, with hysteresis.

        Left and right room are often within a couple of centimetres of each other
        and the obstacle's measured edges move by about that much between frames, so
        a bare comparison flips sides several times a second. Every flip is a
        completely different path, the state machine sees a path the car is not on,
        and avoidance never settles. Requiring the other side to win by
        `side_hysteresis_m` makes the choice sticky without making it permanent: a
        genuinely wider gap still takes it.
        """
        left_ok = left_room >= self.boundary_margin and left_taken is None
        right_ok = right_room >= self.boundary_margin and right_taken is None

        if left_ok and not right_ok:
            return "left"
        if right_ok and not left_ok:
            return "right"

        bias = 0.0
        if self.last_side_obs_id == obs.id:
            if self.last_side == "left":
                bias = self.side_hysteresis_m
            elif self.last_side == "right":
                bias = -self.side_hysteresis_m
        return "left" if left_room + bias >= right_room else "right"

    def _build_path(self, obs, candidates, apex_u, apex_d, pre_dist, post_dist):
        """Build and validate one candidate. Returns (result, "") or (None, reason).

        The path is a cubic in (s, d) that STARTS AT THE CAR: the first control
        point is the ego's own (s, d) and the initial slope is the car's heading
        error. That is what makes the state machine's `_check_on_spline` (which
        needs a path point within `on_spline_min_dist_thres_m` of the car) true the
        moment the path is published, instead of only once the car has driven up to
        a path that began several metres ahead of it.
        """
        control_s = [self.cur_s]
        control_d = [self.cur_d]
        move_start = apex_u - pre_dist
        if move_start > self.cur_s + max(0.5, self.wpnt_dist * 2):
            # Far obstacle: hold the current lateral offset, then move over. Without
            # this the car drifts sideways for the whole approach and spends track
            # width it may need later.
            control_s.append(move_start)
            control_d.append(self.cur_d)
        control_s.append(apex_u)
        control_d.append(apex_d)
        control_s.append(apex_u + post_dist)
        control_d.append(0.0)

        control_s = np.array(control_s, dtype=float)
        control_d = np.array(control_d, dtype=float)
        if np.any(np.diff(control_s) <= 1e-3):
            return None, "degenerate_control_points"

        # Match the car's heading at the anchor so the manoeuvre starts without a
        # steering step. d'(s) = tan(psi_ego - psi_line), capped: a bad heading
        # estimate must not be able to whip the spline sideways.
        psi_ref = float(self.gb_psi[self._gb_idx(self.cur_s % self.max_s)])
        heading_err = np.arctan2(np.sin(self.cur_yaw - psi_ref), np.cos(self.cur_yaw - psi_ref))
        slope0 = float(np.clip(np.tan(np.clip(heading_err, -1.2, 1.2)),
                               -MAX_ENTRY_SLOPE, MAX_ENTRY_SLOPE))
        try:
            spline = CubicSpline(control_s, control_d, bc_type=((1, slope0), (1, 0.0)))
        except ValueError:
            return None, "spline_fit_failed"

        sample_u = np.arange(control_s[0], control_s[-1], self.resolution)
        if len(sample_u) < MIN_SAMPLES:
            return None, "path_too_short"
        # Clip the cubic's overshoot: it may not swing further out than the apex it
        # was built for, nor cross to the far side of the raceline.
        lo = min(0.0, apex_d, self.cur_d)
        hi = max(0.0, apex_d, self.cur_d)
        sample_d = np.clip(spline(sample_u), lo, hi)

        # Tail along the global line, so the state machine can measure obstacles
        # past the manoeuvre. Long enough to cover its own max_horizon: a static
        # obstacle that lies beyond the path's end is treated as blocking
        # (`static/beyond_path` in _check_free_frenet).
        end_u = max(sample_u[-1] + self.tail_m, self.cur_s + self._path_end_eff)
        # + half a step so `end_u` itself is included: arange's exclusive stop left
        # the path one resolution short of min_path_end_m, which is the difference
        # between covering the state machine's max_horizon and not.
        tail_u = np.arange(sample_u[-1] + self.resolution, end_u + self.resolution / 2,
                           self.resolution)
        generated_count = len(sample_u)
        sample_u = np.concatenate([sample_u, tail_u])
        sample_d = np.concatenate([sample_d, np.zeros(len(tail_u))])

        s_along = sample_u - self.cur_s          # distance travelled along the path
        sample_s = sample_u % self.max_s

        # CLAMP to the track, do not reject on it. Rejecting was a regression against
        # the old planner, which clamped the apex to the wall (`min(d_apex,
        # gb_wp.d_left)`) and never had a Frenet boundary test at all. On this track
        # -- half-widths 0.49..1.20 m, median 0.75 -- an apex of 0.55 m plus a 0.19 m
        # margin needs 0.74 m, so a hard reject threw away every candidate wherever
        # the track is merely average width.
        #
        # Clamping is safe in a way rejecting is not: it only ever moves the path
        # TOWARDS the raceline, away from the wall. What it can break is the
        # clearance to the obstacle -- so that is re-checked below, on the clamped
        # path, and a clamp that eats the clearance is what makes the gap genuinely
        # impassable (Case 5) rather than merely tight.
        clamped = self._clamp_to_track(sample_s, sample_d, s_along)
        was_clamped = bool(np.any(np.abs(clamped - sample_d) > 1e-6))
        sample_d = clamped

        # Only over the part this planner shaped. The tail is the global line, and
        # an obstacle sitting on it is not something a single-apex path can solve --
        # the state machine judges that (and refuses OVERTAKE) on its own.
        if not self._path_clear_of(sample_u[:generated_count], sample_d[:generated_count],
                                   candidates):
            return None, "track_too_narrow" if was_clamped else "obstacle_clearance"

        xy = self.converter.get_cartesian(sample_s, sample_d)
        px = np.asarray(xy[0], dtype=float)
        py = np.asarray(xy[1], dtype=float)

        if self.use_map_filter and self.map_filter.eroded_image is not None:
            # Validate only what this planner generated; the tail is the already
            # accepted global line, and rejecting a whole manoeuvre over one eroded
            # pixel 10 m downstream made static avoidance fail at random.
            for i in range(generated_count):
                if not self.map_filter.is_point_inside(px[i], py[i]):
                    return None, "map_occupied"

        # Heading and curvature OF THIS PATH, not of the line underneath it: the
        # controller sets its lookahead and cuts speed from kappa, and the state
        # machine replans the velocity profile from it. psi = atan2(dy, dx) is the
        # convention the global waypoints already carry (gb_optimizer.conv_psi).
        dx, dy = np.gradient(px), np.gradient(py)
        ddx, ddy = np.gradient(dx), np.gradient(dy)
        psi = np.arctan2(dy, dx)
        denom = np.power(dx * dx + dy * dy, 1.5)
        kappa = np.divide(dx * ddy - dy * ddx, denom,
                          out=np.zeros_like(denom), where=denom > 1e-9)
        return (sample_s, sample_d, np.vstack([px, py]), psi, kappa, generated_count,
                float(sample_u[-1] - self.cur_s)), ""

    def _build_group_path(self, group, candidates, first_u, last_u,
                          apex_d, pre_dist, post_dist):
        """Build one continuous same-side path around every obstacle in group."""
        # Reuse the proven single-path implementation for a one-obstacle group.
        if len(group) == 1:
            return self._build_path(
                group[0], candidates, first_u, apex_d, pre_dist, post_dist)

        control_s = [self.cur_s]
        control_d = [self.cur_d]
        move_start = first_u - pre_dist
        if move_start > self.cur_s + max(0.5, self.wpnt_dist * 2):
            control_s.append(move_start)
            control_d.append(self.cur_d)
        control_s.append(first_u)
        control_d.append(apex_d)
        if last_u > first_u + max(1e-3, self.resolution):
            control_s.append(last_u)
            control_d.append(apex_d)
        control_s.append(last_u + post_dist)
        control_d.append(0.0)

        control_s = np.asarray(control_s, dtype=float)
        control_d = np.asarray(control_d, dtype=float)
        if np.any(np.diff(control_s) <= 1e-3):
            return None, "degenerate_group_control_points"

        psi_ref = float(self.gb_psi[self._gb_idx(self.cur_s % self.max_s)])
        heading_err = np.arctan2(np.sin(self.cur_yaw - psi_ref), np.cos(self.cur_yaw - psi_ref))
        slope0 = float(np.clip(np.tan(np.clip(heading_err, -1.2, 1.2)),
                               -MAX_ENTRY_SLOPE, MAX_ENTRY_SLOPE))
        try:
            spline = CubicSpline(control_s, control_d, bc_type=((1, slope0), (1, 0.0)))
        except ValueError:
            return None, "group_spline_fit_failed"

        sample_u = np.arange(control_s[0], control_s[-1], self.resolution)
        if len(sample_u) < MIN_SAMPLES:
            return None, "path_too_short"
        lo = min(0.0, apex_d, self.cur_d)
        hi = max(0.0, apex_d, self.cur_d)
        sample_d = np.clip(spline(sample_u), lo, hi)

        end_u = max(sample_u[-1] + self.tail_m, self.cur_s + self._path_end_eff)
        tail_u = np.arange(sample_u[-1] + self.resolution,
                           end_u + self.resolution / 2, self.resolution)
        generated_count = len(sample_u)
        sample_u = np.concatenate([sample_u, tail_u])
        sample_d = np.concatenate([sample_d, np.zeros(len(tail_u))])
        sample_s = sample_u % self.max_s
        s_along = sample_u - self.cur_s

        clamped = self._clamp_to_track(sample_s, sample_d, s_along)
        was_clamped = bool(np.any(np.abs(clamped - sample_d) > 1e-6))
        sample_d = clamped
        if not self._path_clear_of(sample_u[:generated_count], sample_d[:generated_count],
                                   candidates):
            return None, "group_track_too_narrow" if was_clamped else "group_obstacle_clearance"

        xy = self.converter.get_cartesian(sample_s, sample_d)
        px, py = np.asarray(xy[0], dtype=float), np.asarray(xy[1], dtype=float)
        if self.use_map_filter and self.map_filter.eroded_image is not None:
            for i in range(generated_count):
                if not self.map_filter.is_point_inside(px[i], py[i]):
                    return None, "group_map_occupied"

        dx, dy = np.gradient(px), np.gradient(py)
        ddx, ddy = np.gradient(dx), np.gradient(dy)
        psi = np.arctan2(dy, dx)
        denom = np.power(dx * dx + dy * dy, 1.5)
        kappa = np.divide(dx * ddy - dy * ddx, denom,
                          out=np.zeros_like(denom), where=denom > 1e-9)
        return (sample_s, sample_d, np.vstack([px, py]), psi, kappa, generated_count,
                float(sample_u[-1] - self.cur_s)), ""

    def _clamp_to_track(self, sample_s, sample_d, s_along):
        """Hold the path's centreline `boundary_margin` clear of the track edge.

        Samples within `ego_grace_m` of the car keep at least the car's own current
        offset: where the car IS is a fact no path can undo, and pulling the first
        few centimetres of the path away from where the car already stands would
        just put a step at the start of it.
        """
        idx = self._gb_idx(sample_s)
        available = np.where(sample_d >= 0.0, self.gb_d_left[idx], self.gb_d_right[idx])
        allowed = np.maximum(available - self.boundary_margin, 0.0)
        grace = s_along < self.ego_grace_m
        allowed = np.where(grace, np.maximum(allowed, abs(self.cur_d)), allowed)
        return np.clip(sample_d, -allowed, allowed)

    def _path_clear_of(self, sample_u, sample_d, obstacles) -> bool:
        """The collision invariant, checked on the FINISHED path.

        `sample_u` is UNWRAPPED s (monotonically increasing from the car), so an
        obstacle either falls inside the path's span or it does not -- with wrapped
        s the nearest-sample search silently snaps an out-of-range obstacle onto an
        endpoint and rejects paths that never go near it.

        Deliberately the same arithmetic the state machine uses in
        `_check_free_frenet` (nearest path sample by s, then
        |d_path - d_obs| - size/2 - ego_width/2), so a path this planner accepts is
        one the state machine will accept too. `min_free_dist_m` must stay >= the
        state machine's `static_avoidance_planner.lateral_width_m`.
        """
        if len(sample_u) == 0:
            return False
        for obs in obstacles:
            obs_u = self.cur_s + self._signed_gap(obs.s_center)
            if not sample_u[0] <= obs_u <= sample_u[-1]:
                continue    # the path does not span it; not this check's business
            j = int(np.argmin(np.abs(sample_u - obs_u)))
            free_dist = abs(sample_d[j] - obs.d_center) - obs.size / 2 - self.ego_width_m / 2

            required = self.min_free_dist_m
            if abs(obs_u - self.cur_s) <= obs.size / 2 + self.min_apex_lead_m:
                # The car is already alongside this obstacle. Its lateral offset
                # right now is a fact no path can undo, so the requirement becomes
                # "do not get any closer than we already are". Without this the
                # planner refuses every path exactly while the car is passing, the
                # state machine leaves OVERTAKE mid-manoeuvre, and the car snaps
                # back onto the raceline into the obstacle it was avoiding.
                ego_free = abs(self.cur_d - obs.d_center) - obs.size / 2 - self.ego_width_m / 2
                required = min(required, ego_free)
            # 1 mm of numerical slack: the last relaxation rung plans exactly at the
            # floor and sample discretisation would otherwise reject it on rounding.
            if free_dist < required - 1e-3:
                return False
        return True

    def _fill_msg(self, out, sample_s, sample_d, xy, psi, kappa, side):
        idx = self._gb_idx(sample_s)
        out.ot_side = side
        out.ot_line = side
        for i in range(len(sample_s)):
            k = int(idx[i])
            out.wpnts.append(Wpnt(
                id=i,
                s_m=float(sample_s[i]),
                d_m=float(sample_d[i]),
                x_m=float(xy[0, i]),
                y_m=float(xy[1, i]),
                d_left=float(self.gb_d_left[k]),
                d_right=float(self.gb_d_right[k]),
                psi_rad=float(psi[i]),
                kappa_radpm=float(kappa[i]),
                # Seed only: the state machine replans the profile from kappa and the
                # ggv (update_velocity). Capped so a path that never reaches the
                # planner keeps the conservative speed this branch always used.
                vx_mps=float(min(self.gb_vx[k], self.max_speed_mps)),
                ax_mps2=float(self.gb_ax[k]),
            ))

    ###########
    # LOGGING #
    ###########
    def _setup_debug_log_file(self):
        """Per-run plain-text mirror of _log()'s [STATIC_AVOID] diagnostics (candidate
        clearance, which rung was accepted, cache/hold state, planning latency) --
        independent of rclpy's own per-process log, which rcutils names by PID under
        ~/.ros/log and is painful to locate after a run. Same convention as
        state_machine's _setup_debug_log_file: one file per run + a "latest" symlink,
        directory overridable via RACE_DEBUG_LOG_DIR. Never crashes the node on failure.
        Gated on debug_log_enabled (race.launch.xml debug:=true) -- disabled, this
        only sets up the tracking attr so _dbg_log()'s `_dbg_fh is None` check
        no-ops the rest of the class out without touching the disk.
        """
        self._dbg_fh = None
        self._dbg_last_log_sec = 0.0
        if not self.debug_log_enabled:
            return
        try:
            log_dir = Path(os.environ.get("RACE_DEBUG_LOG_DIR", str(_default_debug_log_dir()))).expanduser()
            log_dir.mkdir(parents=True, exist_ok=True)
            run_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = log_dir / f"static_avoidance_{run_stamp}.log"
            self._dbg_fh = open(log_path, "a", buffering=1)
            latest = log_dir / "latest_static_avoidance.log"
            if latest.is_symlink() or latest.exists():
                latest.unlink()
            latest.symlink_to(log_path.name)
            self._dbg_fh.write(f"# static_avoidance_node debug log started {run_stamp}\n")
            self.get_logger().info(f"[{self.name}] debug log: {log_path} (latest: {latest})")
        except OSError as e:
            self.get_logger().warn(f"[{self.name}] could not open debug log file: {e}")

    def _dbg_log(self, msg: str) -> None:
        if self._dbg_fh is None:
            return
        try:
            now = self.get_clock().now().nanoseconds * 1e-9
            self._dbg_fh.write(f"{now:.3f} {msg}\n")
        except OSError:
            pass

    def _log(self, dbg):
        def fmt(key, spec=".2f", default="-"):
            value = dbg.get(key)
            if value is None:
                return default
            return format(value, spec) if isinstance(value, float) else str(value)

        lines = [f"[STATIC_AVOID] obs_id={fmt('obs_id')} obs_dist={fmt('obs_dist')}m "
                 f"obs_s={fmt('obs_s')} ego_s={fmt('ego_s')} side={fmt('side')}"]
        lines.append(f"  left_clearance={fmt('left_clearance')} right_clearance={fmt('right_clearance')} "
                     f"candidate_valid={str(dbg.get('candidate_valid', False)).lower()} "
                     f"rung={fmt('rung')}")
        lines.append(f"  cache_used={str(dbg.get('cache_used', False)).lower()} "
                     f"path_age={fmt('path_age', '.3f')} path_points={dbg.get('path_points', 0)} "
                     f"planning_time_ms={fmt('planning_time_ms', '.1f')}")
        if dbg.get("reason"):
            lines.append(f"  reason={dbg['reason']}")
        if dbg.get("cache_reason"):
            lines.append(f"  cache_reason={dbg['cache_reason']}")
        period = max(0.05, float(self.log_period_s))
        self.get_logger().info("\n".join(lines), throttle_duration_sec=period)
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self._dbg_last_log_sec > period:
            self._dbg_last_log_sec = now
            self._dbg_log(" | ".join(line.strip() for line in lines))

    ###############
    # MARKERS/VIZ #
    ###############
    def _markers(self, path: OTWpntArray) -> MarkerArray:
        markers = MarkerArray()
        if not path.wpnts:
            delete = Marker()
            delete.header.frame_id = "map"
            delete.header.stamp = self.get_clock().now().to_msg()
            delete.action = Marker.DELETEALL
            markers.markers.append(delete)
            return markers
        for wpnt in path.wpnts:
            mrk = Marker()
            mrk.header.frame_id = "map"
            mrk.header.stamp = self.get_clock().now().to_msg()
            mrk.ns = "static_avoidance"
            mrk.id = wpnt.id
            mrk.type = Marker.CYLINDER
            mrk.action = Marker.ADD
            mrk.pose.position.x = wpnt.x_m
            mrk.pose.position.y = wpnt.y_m
            mrk.pose.position.z = float(wpnt.vx_mps / self.gb_vmax / 2)
            mrk.pose.orientation.w = 1.0
            mrk.scale.x = 0.1
            mrk.scale.y = 0.1
            mrk.scale.z = max(float(wpnt.vx_mps / self.gb_vmax), 1e-3)
            mrk.color.a = 1.0
            mrk.color.b = 0.75
            mrk.color.r = 0.75
            markers.markers.append(mrk)
        return markers


def main(args=None):
    rclpy.init(args=args)
    node = StaticObstacleSpliner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
