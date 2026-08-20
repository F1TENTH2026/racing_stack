"""Geometry and stability tests for the static-obstacle avoidance planner.

Drives StaticObstacleSpliner._plan / .loop by hand on a synthetic track, with no
ROS graph: the node is built with __new__ and its clock, logger, map filter and
publishers are stubbed. What is exercised is the part that used to fail on the
car -- when a path is produced, where it starts, and whether it survives a single
dropped planning frame.

    colcon test --packages-select spliner
    pytest planner/spliner/test/test_static_avoidance.py -v
"""
import math
import types

import numpy as np
import pytest
from f110_msgs.msg import Obstacle, Wpnt, WpntArray
from frenet_conversion.frenet_converter import FrenetConverter

from spliner.static_avoidance_node import StaticObstacleSpliner

HALF_WIDTH = 0.9
OBS_SIZE = 0.4
EGO_WIDTH = 0.29
MIN_FREE_DIST = 0.10


class _Clock:
    def __init__(self):
        self.t = 100.0

    def now(self):
        outer = self

        class _Now:
            nanoseconds = 0

            def __init__(self):
                self.nanoseconds = int(outer.t * 1e9)

            def to_msg(self):
                from builtin_interfaces.msg import Time
                return Time(sec=int(outer.t), nanosec=int((outer.t % 1) * 1e9))

        return _Now()


class _Logger:
    def _noop(self, msg, **kwargs):
        pass

    info = warn = warning = error = debug = _noop


class _MapFilter:
    """Accepts everything: the map image is not what these tests are about."""
    eroded_image = None

    def is_point_inside(self, x, y):
        return True

    def set_erosion_kernel_size(self, size):
        pass


def make_track(n=1200, wpnt_dist=0.1, half_width=HALF_WIDTH, curve=True):
    """60 m of straight, then a constant-radius curve."""
    xs, ys, psis = [], [], []
    x = y = psi = 0.0
    radius = 12.0
    for i in range(n):
        xs.append(x)
        ys.append(y)
        psis.append(psi)
        x += wpnt_dist * math.cos(psi)
        y += wpnt_dist * math.sin(psi)
        if curve and i * wpnt_dist > 60.0:
            psi += wpnt_dist / radius
    track = WpntArray()
    for i in range(n):
        track.wpnts.append(Wpnt(
            id=i, s_m=i * wpnt_dist, d_m=0.0, x_m=xs[i], y_m=ys[i],
            d_left=half_width, d_right=half_width, psi_rad=psis[i],
            kappa_radpm=0.0, vx_mps=5.0, ax_mps2=0.0))
    return track


def build_node(track, cur_s, cur_d=0.0, cur_vs=3.0):
    node = StaticObstacleSpliner.__new__(StaticObstacleSpliner)
    node.name = "static_avoidance_planner"
    node._clock = _Clock()
    node.get_clock = lambda: node._clock
    node._logger = _Logger()
    node.get_logger = lambda: node._logger
    node.map_filter = _MapFilter()
    # __new__ skips __init__, so _setup_debug_log_file() (which normally sets these)
    # never runs; _log() reads _dbg_last_log_sec unconditionally regardless of
    # whether the debug log file opened.
    node._dbg_fh = None
    node._dbg_last_log_sec = 0.0

    # Mirrors stack_master/config/static_avoidance_planner.yaml.
    for key, value in dict(
        rate_hz=20.0, log_period_s=0.5, measure=False,
        lookahead=10.0, keep_behind_m=1.0, trajectory_threshold=0.6,
        raceline_clearance_m=0.35, max_group_obstacles=3,
        evasion_distance=0.30, resolution=0.10,
        pre_dist_gain=1.6, pre_dist_min=2.0, pre_dist_max=8.0,
        pre_dist_kappa_max=0.30, pre_dist_corner_min_m=4.0,
        post_dist_gain=0.8, post_dist_min=1.5, post_dist_max=4.0,
        tail_m=4.0, min_path_end_m=10.0, min_apex_lead_m=0.5,
        boundary_margin=0.19, ego_width_m=EGO_WIDTH, ego_length_m=0.52,
        min_free_dist_m=0.10, obstacle_uncertainty_m=0.02,
        obstacle_speed_margin_gain_s=0.008, wall_clearance_m=0.05,
        wall_speed_margin_gain_s=0.005, margin_speed_cap_mps=6.0,
        min_passage_speed_mps=1.0, comfortable_decel_mps2=3.0,
        planning_reaction_s=0.25,
        ego_grace_m=1.0, use_map_filter=True, kernel_size=4, max_speed_mps=3.0,
        path_hold_s=0.25, side_hysteresis_m=0.10,
    ).items():
        setattr(node, key, value)

    node.obstacles = []
    node.hint_obs = None
    node.gb_wpnts = node.gb_scaled_wpnts = track
    node.max_s = track.wpnts[-1].s_m
    node.wpnt_dist = track.wpnts[1].s_m - track.wpnts[0].s_m
    node.gb_s = np.array([w.s_m for w in track.wpnts])
    node.gb_d_left = np.array([w.d_left for w in track.wpnts])
    node.gb_d_right = np.array([w.d_right for w in track.wpnts])
    node.gb_vx = np.array([w.vx_mps for w in track.wpnts])
    node.gb_ax = np.array([w.ax_mps2 for w in track.wpnts])
    node.gb_psi = np.array([w.psi_rad for w in track.wpnts])
    node.gb_kappa = np.array([w.kappa_radpm for w in track.wpnts])
    node.gb_vmax = 5.0
    node.converter = FrenetConverter(
        np.array([w.x_m for w in track.wpnts]),
        np.array([w.y_m for w in track.wpnts]),
        np.array([w.psi_rad for w in track.wpnts]))

    node.cur_s, node.cur_d, node.cur_vs = cur_s, cur_d, cur_vs
    node.cur_yaw = float(node.gb_psi[int(cur_s / node.wpnt_dist)])
    node.cur_x = node.cur_y = 0.0
    node.last_side = node.last_side_obs_id = None
    node.latched_group = node.latched_move_start = None
    node.last_good_path = node.last_good_obs_id = None
    node.last_good_generated = 0
    node.last_good_origin_s = node.last_good_reach = 0.0
    node._margin_speed_override = None

    node.published = []
    node.evasion_pub = types.SimpleNamespace(publish=node.published.append)
    node.mrks_pub = types.SimpleNamespace(publish=lambda msg: None)
    node.latency_pub = types.SimpleNamespace(publish=lambda msg: None)
    return node


def obstacle(oid, s, d, size=OBS_SIZE):
    obs = Obstacle()
    obs.id = oid
    obs.s_center, obs.d_center, obs.size = s, d, size
    obs.s_start, obs.s_end = s - size / 2, s + size / 2
    obs.d_left, obs.d_right = d + size / 2, d - size / 2
    obs.is_static = True
    return obs


def close_the_walls(node):
    """Make every candidate infeasible without removing the obstacle."""
    node.gb_d_left = np.full_like(node.gb_d_left, 0.20)
    node.gb_d_right = np.full_like(node.gb_d_right, 0.20)


def test_safety_margin_grows_with_speed():
    node = build_node(make_track(), cur_s=20.0, cur_vs=1.0)
    slow_obs = node._required_obstacle_gap()
    slow_wall = node._required_wall_gap()
    node.cur_vs = 6.0
    assert node._required_obstacle_gap() > slow_obs
    assert node._required_wall_gap() > slow_wall


def test_passage_speed_requires_reachable_braking_distance():
    node = build_node(make_track(), cur_s=20.0, cur_vs=6.0)
    assert not node._can_slow_for(2.0, obstacle_gap=4.0)
    assert node._can_slow_for(2.0, obstacle_gap=9.0)


def test_rectangular_footprint_detects_corner_overlap():
    node = build_node(make_track(), cur_s=20.0, cur_vs=3.0)
    obs = obstacle(99, 22.0, 0.0)
    sample_u = np.array([21.55, 21.65, 21.75])
    sample_d = np.full(3, 0.50)

    # A width-only point-mass check would accept 13.5 cm here. With a 0.35 rad
    # heading error, the 0.52 m body projects farther laterally and must reject.
    point_mass_free = sample_d[1] - obs.size / 2 - EGO_WIDTH / 2
    assert point_mass_free > node._required_obstacle_gap()
    assert not node._path_clear_of(
        sample_u, sample_d, [obs], np.full(3, 0.35))


# ---------------------------------------------------------------- Case 1
@pytest.mark.parametrize("distance", [3.0, 5.0, 8.0, 9.8])
def test_far_obstacle_is_planned_immediately(distance):
    """A path exists on the FIRST tick, with no approach needed first."""
    node = build_node(make_track(), cur_s=20.0)
    node.obstacles = [obstacle(3, 20.0 + distance, 0.05)]
    dbg = {}
    path = node._plan(dbg)

    assert path.wpnts, f"no path for an obstacle {distance} m ahead: {dbg.get('reason')}"
    # It starts AT the car, so the state machine's _check_on_spline
    # (on_spline_min_dist_thres_m = 0.5 m) is true the moment it is published.
    assert abs(path.wpnts[0].s_m - node.cur_s) < 0.11
    assert abs(path.wpnts[0].d_m - node.cur_d) < 0.01
    # It reaches past the state machine's max_horizon (10 m), so an obstacle
    # inside that horizon cannot read as "beyond the path".
    assert path.wpnts[-1].s_m - node.cur_s >= node.min_path_end_m - 1e-6


def test_far_obstacle_on_a_curve():
    """The failure the cartesian 3-point spline had: a long chord cuts the corner."""
    node = build_node(make_track(), cur_s=75.0)
    node.obstacles = [obstacle(3, 84.0, 0.05)]
    dbg = {}
    assert node._plan(dbg).wpnts, dbg.get("reason")


# ---------------------------------------------------------------- Case 2
def test_single_failed_cycle_holds_the_last_good_path():
    node = build_node(make_track(), cur_s=20.0)
    node.obstacles = [obstacle(3, 28.0, 0.05)]

    node.loop()
    first = node.published[-1]
    assert first.wpnts

    saved = node.gb_d_left.copy(), node.gb_d_right.copy()
    close_the_walls(node)
    node._clock.t += 0.05
    node.loop()
    held = node.published[-1]
    # Same object, so the ORIGINAL stamp is republished: not claimed to be fresh,
    # but not an empty array either.
    assert held is first

    node.gb_d_left, node.gb_d_right = saved
    node._clock.t += 0.05
    node.loop()
    assert node.published[-1].wpnts and node.published[-1] is not first


def test_hold_expires_after_path_hold_s():
    node = build_node(make_track(), cur_s=20.0)
    node.obstacles = [obstacle(3, 28.0, 0.05)]
    node.loop()
    assert node.published[-1].wpnts

    close_the_walls(node)
    node._clock.t += node.path_hold_s + 0.05
    node.loop()
    assert not node.published[-1].wpnts


# ---------------------------------------------------------------- Case 3
def test_side_does_not_oscillate_on_a_symmetric_gap():
    node = build_node(make_track(), cur_s=20.0)
    rng = np.random.default_rng(0)
    sides = []
    for _ in range(40):
        node.obstacles = [obstacle(7, 28.0, float(rng.normal(0.0, 0.02)))]
        dbg = {}
        node._plan(dbg)
        if dbg.get("side"):
            sides.append(dbg["side"])
    assert len(sides) > 30
    assert all(a == b for a, b in zip(sides, sides[1:])), sides


def test_hysteresis_still_yields_to_a_clearly_wider_side():
    node = build_node(make_track(), cur_s=20.0)
    node.obstacles = [obstacle(7, 28.0, 0.0)]
    node._plan({})
    node.obstacles = [obstacle(7, 28.0, -0.45)]      # well right of the line
    dbg = {}
    node._plan(dbg)
    assert dbg.get("side") == "LEFT"


# ---------------------------------------------------------------- Case 4
def test_cache_is_dropped_when_the_obstacle_disappears():
    node = build_node(make_track(), cur_s=20.0)
    node.obstacles = [obstacle(3, 28.0, 0.05)]
    node.loop()
    assert node.published[-1].wpnts

    node.obstacles = []
    node._clock.t += 0.05                            # well inside path_hold_s
    node.loop()
    assert not node.published[-1].wpnts


# ---------------------------------------------------------------- Case 5
def test_impassable_gap_produces_no_path():
    node = build_node(make_track(half_width=0.42), cur_s=20.0)
    node.obstacles = [obstacle(3, 28.0, 0.0, size=0.5)]
    dbg = {}
    assert not node._plan(dbg).wpnts
    assert dbg.get("candidate_valid") is False
    assert dbg.get("reason")


# ---------------------------------------------------------------- invariants
def test_nothing_is_published_when_the_raceline_already_clears_it():
    node = build_node(make_track(), cur_s=20.0)
    node.obstacles = [obstacle(3, 28.0, 0.55)]       # inside trajectory_threshold
    dbg = {}
    assert not node._plan(dbg).wpnts
    assert dbg["reason"] == "raceline_already_clear"


def test_published_path_clears_the_obstacle():
    node = build_node(make_track(), cur_s=20.0)
    node.obstacles = [obstacle(3, 27.0, 0.10)]
    path = node._plan({})
    assert path.wpnts

    s = np.array([w.s_m for w in path.wpnts])
    d = np.array([w.d_m for w in path.wpnts])
    nearest = int(np.argmin(np.abs(s - 27.0)))
    free_dist = abs(d[nearest] - 0.10) - OBS_SIZE / 2 - EGO_WIDTH / 2
    assert free_dist >= MIN_FREE_DIST - 1e-9


def test_published_path_stays_inside_the_track():
    node = build_node(make_track(), cur_s=20.0)
    node.obstacles = [obstacle(3, 27.0, 0.10)]
    path = node._plan({})
    assert path.wpnts
    for wpnt in path.wpnts:
        bound = wpnt.d_left if wpnt.d_m >= 0.0 else wpnt.d_right
        assert abs(wpnt.d_m) <= bound - node.boundary_margin + 1e-9


def test_path_is_kept_while_passing_the_obstacle():
    """Dropping it here makes the state machine leave OVERTAKE mid-manoeuvre."""
    for cur_s, cur_d in ((27.9, 0.35), (28.6, 0.40)):
        node = build_node(make_track(), cur_s=cur_s, cur_d=cur_d, cur_vs=2.0)
        node.obstacles = [obstacle(3, 28.0, 0.10)]
        dbg = {}
        assert node._plan(dbg).wpnts, f"dropped at s={cur_s}: {dbg.get('reason')}"


# ---------------------------------------------------------------- short tracks
# 0814test2 is 22.31 m round. `lookahead: 10` and `min_path_end_m: 11` were picked
# for a track twice that size and are close to half a lap here, which is where the
# (s - cur_s) % max_s arithmetic stops being able to tell ahead from behind.
SHORT_TRACK_N = 223            # 22.3 m at 0.1 m spacing


def test_horizons_are_capped_to_a_fraction_of_the_lap():
    track = make_track(n=SHORT_TRACK_N, curve=False)
    node = build_node(track, cur_s=5.0)
    node.lookahead, node.min_path_end_m = 10.0, 11.0

    assert node._lookahead_eff == pytest.approx(node.max_s / 3, abs=1e-6)
    assert node._path_end_eff == pytest.approx(node.max_s / 3, abs=1e-6)

    node.obstacles = [obstacle(3, 9.0, 0.05)]
    path = node._plan({})
    assert path.wpnts
    reach = (path.wpnts[-1].s_m - node.cur_s) % node.max_s
    assert reach < node.max_s / 2, "a path spanning half a lap breaks ahead/behind"


def test_hold_survives_a_far_obstacle_on_a_short_track():
    """The regression: (path_end - cur_s) % max_s wraps at max_s/2, so a long path
    on a short track read as 'already passed' and the hold never applied."""
    track = make_track(n=SHORT_TRACK_N, curve=False)
    node = build_node(track, cur_s=2.0, cur_vs=2.5)
    node.obstacles = [obstacle(3, 8.0, 0.05)]

    node.loop()
    first = node.published[-1]
    assert first.wpnts

    close_the_walls(node)
    node.cur_s += 0.12                       # the car keeps moving during the drop
    node._clock.t += 0.05
    node.loop()
    assert node.published[-1] is first, "held path discarded as path_passed"


def test_dynamic_obstacles_are_ignored():
    node = build_node(make_track(), cur_s=20.0)
    moving = obstacle(3, 28.0, 0.05)
    moving.is_static = False
    node.obstacles = [moving]
    dbg = {}
    assert not node._plan(dbg).wpnts
    assert dbg["reason"] == "no_static_obstacle_in_range"


# -------------------------------------------------------- multi-obstacle group
def test_three_visible_obstacles_shape_one_path_and_all_are_clear():
    node = build_node(make_track(), cur_s=20.0, cur_vs=3.0)
    node.obstacles = [
        obstacle(11, 25.0, 0.05),
        obstacle(12, 27.0, 0.10),
        obstacle(13, 29.0, 0.00),
    ]
    dbg = {}
    path = node._plan(dbg)

    assert path.wpnts, dbg.get("reason")
    assert dbg["group_size"] == 3
    assert dbg["group_ids"] == [11, 12, 13]
    s = np.array([w.s_m for w in path.wpnts])
    d = np.array([w.d_m for w in path.wpnts])
    for obs in node.obstacles:
        j = int(np.argmin(np.abs(s - obs.s_center)))
        free = abs(d[j] - obs.d_center) - obs.size / 2 - EGO_WIDTH / 2
        assert free >= MIN_FREE_DIST - 1e-3


def test_group_does_not_return_to_raceline_between_obstacles():
    node = build_node(make_track(), cur_s=20.0, cur_vs=3.0)
    node.obstacles = [obstacle(21, 25.0, 0.0), obstacle(22, 28.0, 0.0)]
    path = node._plan({})
    assert path.wpnts

    between = [abs(w.d_m) for w in path.wpnts if 25.2 <= w.s_m <= 27.8]
    assert between
    assert min(between) >= node.min_evasion_m - 1e-3


def test_group_returns_only_after_last_obstacle():
    node = build_node(make_track(), cur_s=20.0, cur_vs=3.0)
    node.obstacles = [obstacle(31, 25.0, 0.0), obstacle(32, 28.0, 0.0)]
    path = node._plan({})
    assert path.wpnts

    after_last = [w for w in path.wpnts if w.s_m >= 30.0]
    assert after_last
    assert abs(path.wpnts[-1].d_m) < 1e-9
