# MIT License

# Copyright (c) 2020 Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import yaml
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.parameter import ParameterType
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from std_msgs.msg import Empty, Float32, Bool
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Transform
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker, MarkerArray
from f110_msgs.msg import ObstacleArray, WpntArray
from tf2_ros import TransformBroadcaster

# Make f110_gym import & scan without numba (numpy>=2.1 breaks numba 0.60).
# Installs a numba shim and swaps ScanSimulator2D -> range_libc RayMarching.
# MUST run before gym.make('f110_gym:...').
from f1tenth_gym_ros import numba_free  # noqa: F401

import pitwall  # one-line telemetry: pitwall.event(...) -> /pitwall/events

import gymnasium as gym
import numpy as np
import cv2
import threading
import sys
import math
from transforms3d import euler
import os.path


class GymBridge(Node):
    def _param(self, name, default):
        """Read a parameter, declaring `default` only if it was not already
        auto-declared from a params-file override. The node is created with
        automatically_declare_parameters_from_overrides=True, so any override
        (e.g. use_external_opponent from the launch file) is pre-declared and a
        second declare_parameter() would raise ParameterAlreadyDeclared."""
        if not self.has_parameter(name):
            self.declare_parameter(name, default)
        return self.get_parameter(name).value

    def __init__(self):
        super().__init__('gym_bridge',
                         automatically_declare_parameters_from_overrides=True)

        # Bind pitwall telemetry to this node (reuses its DDS participant).
        pitwall.init(self)
        # rising-edge latches so each collision is reported once, not every tick.
        self._wall_hit_latch = False
        self._opp_hit_latch = False
        self._opp_collision_now = False

        self.set_descriptor(name='ego_namespace', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING))
        self.set_descriptor(name='ego_odom_topic', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING))
        self.set_descriptor(name='ego_opp_odom_topic', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING))
        self.set_descriptor(name='ego_scan_topic', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING))
        self.set_descriptor(name='ego_drive_topic', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING))
        self.set_descriptor(name='opp_namespace', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING))
        self.set_descriptor(name='opp_odom_topic', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING))
        self.set_descriptor(name='opp_ego_odom_topic', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING))
        self.set_descriptor(name='opp_scan_topic', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING))
        self.set_descriptor(name='opp_drive_topic', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING))

        self.set_descriptor(name='scan_distance_to_base_link', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE, description="Transforms related"))
        self.set_descriptor(name='scan_fov', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE, description="laserscan related"))
        self.set_descriptor(name='scan_beams', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_INTEGER, description="laserscan related"))

        self.set_descriptor(name='map_path', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING))
        self.set_descriptor(name='map_img_ext', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING))

        self.set_descriptor(name='num_agent', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_INTEGER))

        self.set_descriptor(name='dynamics', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING, description="The path to the vehicle dynamics yaml file"))

        self.set_descriptor(name='sx', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE, description="Starting X position of ego"))
        self.set_descriptor(name='sy', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE, description="Starting Y position of ego"))
        self.set_descriptor(name='stheta', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE, description="Starting heading of ego"))

        self.set_descriptor(name='sx1', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE, description="Starting X position of opponent"))
        self.set_descriptor(name='sy1', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE, description="Starting Y position of opponent"))
        self.set_descriptor(name='stheta1', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE, description="Starting heading of opponent"))

        self.set_descriptor(name='kb_teleop', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_BOOL, description="Whether teleop is enabled"))

        # check num_agents
        num_agents = self.get_parameter('num_agent').value
        if num_agents < 1 or num_agents > 2:
            raise ValueError('num_agents should be either 1 or 2.')
        elif type(num_agents) != int:
            raise ValueError('num_agents should be an int.')

        # get vehicle dynamics params
        sim_params_yaml = self.get_parameter('dynamics').value
        sim_param_data = yaml.safe_load(open(sim_params_yaml, 'r'))
        sim_params = {key: float(value)
                      for key, value in sim_param_data.items()}

        # get scan parameters
        scan_fov = self.get_parameter('scan_fov').value
        scan_beams = self.get_parameter('scan_beams').value
        self.scan_fov = scan_fov
        self.scan_beams = scan_beams

        # env backend
        map_yaml_path = os.path.abspath(self.get_parameter('map_path').value)
        # Physics/dynamics step and lidar scan are phase-locked: the scan is
        # decimated from the physics step (every scan_decim steps), so scan_hz must
        # be an integer divisor of sim_step_hz. See the single-timer setup below.
        self.sim_step_hz = float(self._param('sim_step_hz', 80.0))
        self.scan_hz = float(self._param('scan_hz', 40.0))
        # When true, the opponent is owned by the external `opponent` package
        # (kinematic virtual car + scan_augmentor) so the SAME method works on the
        # real car. gym_bridge then runs EGO-ONLY: no opp subs/pubs/TF/overlay,
        # and its ego scan is meant to be remapped to /scan_raw for the augmentor.
        self.external_opp = bool(self._param('use_external_opponent', False))

        # Always build the env with 2-agent CAPACITY so an opponent can be
        # spawned/removed at runtime (gym fixes the agent count at make-time).
        # Whether an opponent actually exists is the runtime flag self.has_opp;
        # when absent the 2nd car is parked far off-map and ignored.
        # timestep = 1/sim_step_hz keeps sim-time aligned with wall-clock.
        self.env = gym.make('f110_gym:f110-v0',
                            map=map_yaml_path.split('.')[0],
                            map_ext=self.get_parameter('map_img_ext').value,
                            params=sim_params,
                            num_agents=2,
                            timestep=1.0 / self.sim_step_hz,
                            num_beams=scan_beams,
                            scan_fov=scan_fov,
                            disable_env_checker=True)

        sx = self.get_parameter('sx').value
        sy = self.get_parameter('sy').value
        stheta = self.get_parameter('stheta').value
        self.ego_pose = [sx, sy, stheta]
        self.ego_speed = [0.0, 0.0, 0.0]
        self.ego_requested_speed = 0.0
        self.ego_steer = 0.0
        self.ego_collision = False
        ego_scan_topic = self.get_parameter('ego_scan_topic').value
        ego_drive_topic = self.get_parameter('ego_drive_topic').value
        self.angle_min = -scan_fov / 2.
        self.angle_max = scan_fov / 2.
        self.angle_inc = scan_fov / (scan_beams - 1)
        self.ego_namespace = self.get_parameter('ego_namespace').value
        ego_odom_topic = self.ego_namespace + '/' + \
            self.get_parameter('ego_odom_topic').value
        self.scan_distance_to_base_link = self.get_parameter(
            'scan_distance_to_base_link').value
        self.ts = self.get_clock().now().to_msg()

        # Opponent state (always allocated; presence is the runtime flag has_opp).
        self.OPP_PARK_POSE = [1000.0, 1000.0, 0.0]  # off-map "garage" when absent
        self.has_opp = (num_agents == 2) and not self.external_opp   # external pkg owns the opp
        # gym pose is at base_link (rear axle); shift the overlay box forward to
        # the vehicle centre (mid-wheelbase) so the footprint matches the car body.
        self.opp_center_offset = 0.5 * (sim_params.get('lf', 0.16) + sim_params.get('lr', 0.17))
        # per-vehicle lidar enable (panel toggles) — lets you skip unused scans
        self.ego_lidar_on = True
        self.opp_lidar_on = True
        self.opp_namespace = self.get_parameter('opp_namespace').value
        sx1 = self.get_parameter('sx1').value
        sy1 = self.get_parameter('sy1').value
        stheta1 = self.get_parameter('stheta1').value
        self.opp_pose = [sx1, sy1, stheta1]
        self.opp_speed = [0.0, 0.0, 0.0]
        self.opp_requested_speed = 0.0
        self.opp_steer = 0.0
        self.opp_collision = False
        self.opp_max_speed = float(sim_params.get('v_max', 8.0))
        # geometric collision stop vs virtual obstacles (overlay arch has no gym
        # collision for them): ego footprint vs f110_msgs/Obstacle boxes.
        self.ego_half = 0.5 * float(sim_params.get('length', 0.58))
        self.ego_collision = False
        self._dyn_obs = []     # [(x, y, half), ...]
        self._stat_obs = []

        init_opp = [sx1, sy1, stheta1] if self.has_opp else self.OPP_PARK_POSE
        self.obs, _, self.done, _ = self.env.reset(
            poses=np.array([[sx, sy, stheta], init_opp]))
        self.ego_scan = list(self.obs['scans'][0])
        self.opp_scan = list(self.obs['scans'][1])

        opp_scan_topic = self.get_parameter('opp_scan_topic').value
        opp_odom_topic = self.opp_namespace + '/' + \
            self.get_parameter('opp_odom_topic').value
        opp_drive_topic = self.get_parameter('opp_drive_topic').value
        ego_opp_odom_topic = self.ego_namespace + '/' + \
            self.get_parameter('ego_opp_odom_topic').value
        opp_ego_odom_topic = self.opp_namespace + '/' + \
            self.get_parameter('opp_ego_odom_topic').value

        # Single phase-locked timer. Physics steps at sim_step_hz and the lidar
        # scan is decimated from it (published every scan_decim steps from inside
        # the SAME callback), so the scan is raycast from -- and stamped with --
        # the exact physics step whose map->base_link TF was just published.
        # Previously a separate scan_timer running on the MultiThreadedExecutor +
        # a shared ReentrantCallbackGroup let the scan read self.obs/self.ts while
        # the drive step was mid-write, so scan geometry and TF could come from
        # different steps -> the scan slid against the map while moving.
        # sim_step_hz must be an integer multiple of scan_hz; round + warn if not.
        self.scan_decim = max(1, int(round(self.sim_step_hz / self.scan_hz)))
        eff_scan_hz = self.sim_step_hz / self.scan_decim
        if abs(eff_scan_hz - self.scan_hz) > 1e-6:
            self.get_logger().warn(
                f'[GymBridge] scan_hz {self.scan_hz} is not an integer divisor of '
                f'sim_step_hz {self.sim_step_hz}; using {eff_scan_hz:.3f} Hz '
                f'(scan every {self.scan_decim} physics steps) to stay phase-locked.')
        self.scan_hz = eff_scan_hz
        self._scan_tick = 0
        # MutuallyExclusive so the physics+scan step never overlaps itself.
        cb_group1 = MutuallyExclusiveCallbackGroup()
        self.drive_timer = self.create_timer(
            1.0 / self.sim_step_hz, self.drive_timer_callback, callback_group=cb_group1)

        # transform broadcaster
        self.br = TransformBroadcaster(self)

        # publishers
        self.ego_scan_pub = self.create_publisher(
            LaserScan, ego_scan_topic, 10)
        self.ego_odom_pub = self.create_publisher(Odometry, ego_odom_topic, 10)
        self.fake_vesc_odom_pub = self.create_publisher(Odometry, '/vesc/odom', 10)
        self.ego_drive_published = False
        self.opp_drive_published = False
        # opponent publishers exist only when gym owns the opponent (internal mode)
        if not self.external_opp:
            self.opp_scan_pub = self.create_publisher(
                LaserScan, opp_scan_topic, 10)
            self.ego_opp_odom_pub = self.create_publisher(
                Odometry, ego_opp_odom_topic, 10)
            self.opp_odom_pub = self.create_publisher(
                Odometry, opp_odom_topic, 10)
            self.opp_ego_odom_pub = self.create_publisher(
                Odometry, opp_ego_odom_topic, 10)

        # QoS Profiles
        best_effort_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10)

        # subscribers
        self.ego_drive_sub = self.create_subscription(
            AckermannDriveStamped,
            ego_drive_topic,
            self.drive_callback,
            10)
        self.ego_reset_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/initialpose',
            self.ego_reset_callback,
            qos_profile=best_effort_qos_profile)
        # Raceline for crash recovery: if the ego stays wedged against a virtual
        # obstacle, respawn it on the nearest raceline point so an endurance run
        # never dead-locks (the collision-stop alone would freeze it forever).
        self._raceline = None        # (N,3) [x, y, psi]
        self._collision_ticks = 0
        self.create_subscription(WpntArray, '/global_waypoints',
                                 self._raceline_callback, 10)
        # opponent control — only when gym owns the opponent. With the external
        # `opponent` package these topics (/goal_pose, /opp_drive, /sim/remove_opponent,
        # /sim/opp_speed_delta) are owned by that package instead.
        if not self.external_opp:
            self.opp_drive_sub = self.create_subscription(
                AckermannDriveStamped,
                opp_drive_topic,
                self.opp_drive_callback,
                10)
            # RViz "2D Goal Pose" tool: spawns the opponent if absent, else moves it
            self.opp_reset_sub = self.create_subscription(
                PoseStamped,
                '/goal_pose',
                self.opp_reset_callback,
                10)
            # RViz control-panel buttons
            self.remove_opp_sub = self.create_subscription(
                Empty, '/sim/remove_opponent', self.remove_opponent_callback, 10)
            self.opp_speed_sub = self.create_subscription(
                Float32, '/sim/opp_speed_delta', self.opp_speed_delta_callback, 10)
            self.opp_lidar_sub = self.create_subscription(
                Bool, '/sim/opp_lidar_enable', self.opp_lidar_callback, 10)
        # ego lidar toggle is always available
        self.ego_lidar_sub = self.create_subscription(
            Bool, '/sim/ego_lidar_enable', self.ego_lidar_callback, 10)
        # virtual obstacles (opponent + static) for the ego collision-stop check
        self.create_subscription(
            ObstacleArray, '/sim/dynamic_obstacles',
            lambda m: setattr(self, '_dyn_obs', [(o.x_m, o.y_m, 0.5 * o.size) for o in m.obstacles]), 10)
        self.create_subscription(
            ObstacleArray, '/sim/static_obstacles',
            lambda m: setattr(self, '_stat_obs', [(o.x_m, o.y_m, 0.5 * o.size) for o in m.obstacles]), 10)

        if self.get_parameter('kb_teleop').value:
            self.teleop_sub = self.create_subscription(
                Twist,
                '/cmd_vel',
                self.teleop_callback,
                10)

        # ===== Obstacle system =====
        self._obstacle_lock = threading.Lock()
        self.static_obstacles = []
        self.dynamic_obstacle = None
        self.map_needs_update = False

        # Load base map for obstacle rendering
        self.base_map_img = None
        self.current_map_img = None
        self.map_resolution = None
        self.map_origin_x = None
        self.map_origin_y = None
        self.map_height = None
        self.map_width = None
        self._load_base_map(map_yaml_path)
        self._init_raycaster(map_yaml_path)

        # /map publisher (TRANSIENT_LOCAL for RViz latched behavior)
        map_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', map_qos)

        # Obstacle subscribers
        self.static_obs_sub = self.create_subscription(
            MarkerArray, '/static_obstacles', self._static_obstacles_cb, 10)
        self.dynamic_obs_sub = self.create_subscription(
            Marker, '/dynamic_obstacle_state', self._dynamic_obstacle_cb, 10)

        # Publish initial map (no obstacles)
        if self.base_map_img is not None:
            self._publish_occupancy_grid()
            self.get_logger().info('[GymBridge] Initial /map published')

    def drive_callback(self, drive_msg):
        self.ego_requested_speed = drive_msg.drive.speed
        self.ego_steer = drive_msg.drive.steering_angle
        self.ego_drive_published = True

    def opp_drive_callback(self, drive_msg):
        self.opp_requested_speed = drive_msg.drive.speed
        self.opp_steer = drive_msg.drive.steering_angle
        self.opp_drive_published = True

    def ego_reset_callback(self, pose_msg):
        rx = pose_msg.pose.pose.position.x
        ry = pose_msg.pose.pose.position.y
        rqx = pose_msg.pose.pose.orientation.x
        rqy = pose_msg.pose.pose.orientation.y
        rqz = pose_msg.pose.pose.orientation.z
        rqw = pose_msg.pose.pose.orientation.w
        _, _, rtheta = euler.quat2euler([rqw, rqx, rqy, rqz], axes='sxyz')
        # env always has 2-agent capacity -> reset must pass 2 poses. Keep the
        # opponent where it is (or parked off-map when absent).
        opp_pose = ([self.obs['poses_x'][1], self.obs['poses_y'][1], self.obs['poses_theta'][1]]
                    if self.has_opp else self.OPP_PARK_POSE)
        self.obs, _, self.done, _ = self.env.reset(
            poses=np.array([[rx, ry, rtheta], opp_pose]))

    def opp_reset_callback(self, pose_msg):
        """RViz '2D Goal Pose' (/goal_pose): spawn the opponent if it doesn't
        exist yet, otherwise just relocate it. Ego is left untouched."""
        rx = pose_msg.pose.position.x
        ry = pose_msg.pose.position.y
        rqx = pose_msg.pose.orientation.x
        rqy = pose_msg.pose.orientation.y
        rqz = pose_msg.pose.orientation.z
        rqw = pose_msg.pose.orientation.w
        _, _, rtheta = euler.quat2euler([rqw, rqx, rqy, rqz], axes='sxyz')
        self._place_opponent(rx, ry, rtheta)
        if not self.has_opp:
            self.has_opp = True
            self.get_logger().info(
                f'[GymBridge] opponent SPAWNED at ({rx:.2f}, {ry:.2f}, {rtheta:.2f})')
        else:
            self.get_logger().info(
                f'[GymBridge] opponent moved to ({rx:.2f}, {ry:.2f}, {rtheta:.2f})')

    def _place_opponent(self, x, y, theta):
        """Directly set the 2nd gym agent's state (no env.reset -> ego undisturbed)."""
        agent = self.env.unwrapped.sim.agents[1]
        agent.state[:] = 0.0
        agent.state[0] = x
        agent.state[1] = y
        agent.state[4] = theta
        agent.in_collision = False
        self.opp_pose = [x, y, theta]
        self.opp_requested_speed = 0.0
        self.opp_steer = 0.0
        self.opp_collision = False

    def remove_opponent_callback(self, msg):
        """RViz panel 'Remove opponent': park the 2nd car off-map and stop
        computing/publishing it."""
        if not self.has_opp:
            return
        self.has_opp = False
        self._place_opponent(*self.OPP_PARK_POSE)
        self.get_logger().info('[GymBridge] opponent REMOVED (parked off-map)')

    def opp_speed_delta_callback(self, msg):
        """RViz panel speed +/-: nudge the opponent's target speed."""
        if not self.has_opp:
            self.get_logger().warn('[GymBridge] no opponent present; spawn one first')
            return
        self.opp_requested_speed = float(
            np.clip(self.opp_requested_speed + msg.data, 0.0, self.opp_max_speed))
        self.opp_drive_published = True
        self.get_logger().info(
            f'[GymBridge] opponent target speed -> {self.opp_requested_speed:.2f} m/s')

    def ego_lidar_callback(self, msg):
        self.ego_lidar_on = bool(msg.data)
        self.get_logger().info(f'[GymBridge] ego lidar {"ON" if self.ego_lidar_on else "OFF"}')

    def opp_lidar_callback(self, msg):
        self.opp_lidar_on = bool(msg.data)
        self.get_logger().info(f'[GymBridge] opp lidar {"ON" if self.opp_lidar_on else "OFF"} '
                               '(FTG needs it ON)')

    def teleop_callback(self, twist_msg):
        if not self.ego_drive_published:
            self.ego_drive_published = True

        self.ego_requested_speed = twist_msg.linear.x

        if twist_msg.angular.z > 0.0:
            self.ego_steer = 0.3
        elif twist_msg.angular.z < 0.0:
            self.ego_steer = -0.3
        else:
            self.ego_steer = 0.0

    # ===== Obstacle system methods =====

    def _load_base_map(self, map_yaml_path):
        """Load base map image and metadata from YAML + image file."""
        try:
            with open(map_yaml_path, 'r') as f:
                map_data = yaml.safe_load(f)

            map_dir = os.path.dirname(map_yaml_path)
            img_filename = map_data['image']
            img_path = os.path.join(map_dir, img_filename)

            self.base_map_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if self.base_map_img is None:
                self.get_logger().error(f'[GymBridge] Failed to load map image: {img_path}')
                return

            self.map_resolution = map_data['resolution']
            self.map_origin_x = map_data['origin'][0]
            self.map_origin_y = map_data['origin'][1]
            self.map_height = self.base_map_img.shape[0]
            self.map_width = self.base_map_img.shape[1]
            self.current_map_img = self.base_map_img.copy()

            self.get_logger().info(f'[GymBridge] Base map loaded: {self.base_map_img.shape}, '
                                   f'res={self.map_resolution}')
        except Exception as e:
            self.get_logger().error(f'[GymBridge] Failed to load base map: {e}')

    def _init_raycaster(self, map_yaml_path):
        """Build the standalone RaycastEngine so the published scan comes from it
        (rm = exact, NaN on miss) and obstacles/opponent are applied via OVERLAY
        (min(static, ray)) — NO distance-transform rebuild on obstacle change."""
        # NOTE: the sim does SINGLE scans (ego/opp) -> no batch parallelism here.
        # raycaster_threads belongs to the particle-filter (many poses), not the sim.
        self.use_raycaster = self._param('use_raycaster', True)
        self.rc_backend = self._param('raycaster_backend', 'rm')
        self.rc_max_range = self._param('raycaster_max_range', 10.0)
        self.rc_nan_on_miss = self._param('raycaster_nan_on_miss', True)
        # additive Gaussian range noise per beam (sim realism) — matches f1tenth_gym's std_dev=0.01 m
        self.rc_noise_std = self._param('raycaster_scan_noise_std', 0.01)
        self._scan_rng = np.random.default_rng()
        self.engine = None
        if not self.use_raycaster:
            self.get_logger().info('[GymBridge] use_raycaster=false -> gym internal numba scan + DT rebuild')
            return
        try:
            rc_path = os.environ.get('RAYCASTER_PATH') or os.environ.get('RAYCASTER_DIR') or numba_free._find_raycaster_dir()
            for p in (rc_path, os.path.join(rc_path, 'range_libc', 'pywrapper')):
                if p not in sys.path:
                    sys.path.insert(0, p)
            from raycaster import RaycastEngine
            occ, res, origin = RaycastEngine.load_map_yaml(map_yaml_path)
            self.engine = RaycastEngine(self.rc_backend, max_range_m=self.rc_max_range).set_map(occ, res, origin)
            self.get_logger().info(
                f"[GymBridge] raycaster '{self.rc_backend}' built; "
                f"obstacles+opponent via OVERLAY post-process, no DT rebuild")
        except Exception as e:
            self.get_logger().error(f'[GymBridge] raycaster init failed: {e} -> fallback to gym scan')
            self.engine = None
            self.use_raycaster = False

    def _lidar_pose(self, agent):
        """Lidar pose (raycaster scans from here) = base_link + forward offset."""
        x, y, th = (self.obs['poses_x'][agent], self.obs['poses_y'][agent],
                    self.obs['poses_theta'][agent])
        d = self.scan_distance_to_base_link
        return np.array([x + d * math.cos(th), y + d * math.sin(th), th])

    def _raycast_scan(self, agent, other_agent):
        """Overlay scan for one agent: static map + the OTHER car (box) + static
        obstacles (circles). NaN where a beam returns nothing (real-sensor style)."""
        with self._obstacle_lock:
            # snap half-side to the grid (same cell count the map renders) so the
            # raycaster footprint matches the occupancy grid exactly
            obstacles = [[o['x'], o['y'],
                          max(1, int(round(o['radius'] / self.map_resolution))) * self.map_resolution]
                         for o in self.static_obstacles] or None
        opp = None
        if other_agent is not None:
            oth = self.obs['poses_theta'][other_agent]
            c = self.opp_center_offset   # base_link(rear axle) -> vehicle centre
            opp = [[self.obs['poses_x'][other_agent] + c * math.cos(oth),
                    self.obs['poses_y'][other_agent] + c * math.sin(oth),
                    oth]]
        miss = np.nan if self.rc_nan_on_miss else None
        scan = self.engine.scan_with_dynamics(
            self._lidar_pose(agent), self.scan_beams, self.scan_fov,
            opp_poses=opp, obstacles=obstacles, miss=miss)
        if self.rc_noise_std > 0.0:                          # gym-equivalent white noise (NaN beams stay NaN)
            scan = scan + self._scan_rng.normal(0.0, self.rc_noise_std, scan.shape)
        return scan.astype(np.float64).tolist()

    def _static_obstacles_cb(self, msg: MarkerArray):
        """Callback for static obstacle positions."""
        with self._obstacle_lock:
            new_obstacles = []
            for marker in msg.markers:
                if marker.action == Marker.DELETEALL:
                    new_obstacles = []
                    break
                if marker.action == Marker.ADD:
                    radius_m = marker.scale.x / 2.0 if marker.scale.x > 0 else 0.25
                    new_obstacles.append({
                        'x': marker.pose.position.x,
                        'y': marker.pose.position.y,
                        'radius': radius_m
                    })
            self.static_obstacles = new_obstacles
            self.map_needs_update = True

    def _dynamic_obstacle_cb(self, msg: Marker):
        """Callback for dynamic obstacle state."""
        with self._obstacle_lock:
            if msg.action == Marker.DELETE or msg.action == Marker.DELETEALL:
                self.dynamic_obstacle = None
            else:
                self.dynamic_obstacle = msg
            self.map_needs_update = True

    def _meters_to_pixels(self, x_m, y_m):
        """Convert world meters to pixel coordinates (origin top-left)."""
        x_px = int((x_m - self.map_origin_x) / self.map_resolution)
        y_px = int((y_m - self.map_origin_y) / self.map_resolution)
        y_px = self.map_height - y_px  # flip Y (image origin is top-left)
        return x_px, y_px

    def _render_obstacles_on_map(self):
        """Draw all obstacles on a fresh copy of the base map."""
        self.current_map_img = self.base_map_img.copy()

        # Static obstacles (circles)
        for obs in self.static_obstacles:
            cx, cy = self._meters_to_pixels(obs['x'], obs['y'])
            half = max(1, int(round(obs['radius'] / self.map_resolution)))   # square half-side (cells)
            # draw a 2*half-cell span (even) so the footprint == raycaster's 2*r box
            cv2.rectangle(self.current_map_img, (cx - half, cy - half), (cx + half - 1, cy + half - 1), 0, -1)

        # Dynamic obstacle (rotated rectangle)
        if self.dynamic_obstacle is not None:
            dyn = self.dynamic_obstacle
            x_m = dyn.pose.position.x
            y_m = dyn.pose.position.y
            qz = dyn.pose.orientation.z
            qw = dyn.pose.orientation.w
            heading = 2.0 * np.arctan2(qz, qw)

            length_m = dyn.scale.x
            width_m = dyn.scale.y
            center_px = self._meters_to_pixels(x_m, y_m)
            length_px = int(length_m / self.map_resolution)
            width_px = int(width_m / self.map_resolution)

            half_l = length_px / 2.0
            half_w = width_px / 2.0
            corners_local = np.array([
                [-half_w, -half_l],
                [-half_w, half_l],
                [half_w, half_l],
                [half_w, -half_l]
            ])
            # Rotation for image coordinates (Y flipped)
            h_adj = heading - np.pi / 2.0
            cos_h = np.cos(h_adj)
            sin_h = np.sin(h_adj)
            rot = np.array([[cos_h, sin_h], [-sin_h, cos_h]])
            corners_px = (rot @ corners_local.T).T + np.array(center_px)
            corners_px = corners_px.astype(np.int32)
            cv2.fillPoly(self.current_map_img, [corners_px], 0)

    def _update_gym_map(self):
        """Render obstacles, update gym DT, and publish /map."""
        if self.base_map_img is None:
            return

        with self._obstacle_lock:
            self._render_obstacles_on_map()
            self.map_needs_update = False

        # With the raycaster, obstacles are applied via OVERLAY (per-scan, ~0.1 ms) so we
        # SKIP the gym distance-transform rebuild (numba ~12 ms / glt ~880 ms hitch each change).
        if not (self.use_raycaster and self.engine is not None):
            flipped = np.flipud(self.current_map_img)   # gym expects bottom-left origin
            self.env.unwrapped.update_map_from_array(
                flipped, self.map_resolution,
                self.map_origin_x, self.map_origin_y)

        # Publish updated /map (display only)
        self._publish_occupancy_grid()

    def _publish_occupancy_grid(self):
        """Convert current map image to OccupancyGrid and publish."""
        if self.current_map_img is None:
            return

        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = 'map'
        grid_msg.info.resolution = self.map_resolution
        grid_msg.info.width = self.map_width
        grid_msg.info.height = self.map_height
        grid_msg.info.origin.position.x = self.map_origin_x
        grid_msg.info.origin.position.y = self.map_origin_y
        grid_msg.info.origin.position.z = 0.0
        grid_msg.info.origin.orientation.w = 1.0

        # OccupancyGrid origin is bottom-left, image origin is top-left
        flipped_img = np.flipud(self.current_map_img)
        occupancy = np.zeros(flipped_img.shape, dtype=np.int8)
        occupancy[flipped_img < 128] = 100   # occupied
        occupancy[flipped_img >= 128] = 0     # free
        grid_msg.data = occupancy.flatten().tolist()

        self.map_pub.publish(grid_msg)

    def drive_timer_callback(self):
        # ---- physics / dynamics step (runs at sim_step_hz) ----
        if self.map_needs_update:
            self._update_gym_map()

        # Env always has 2-agent capacity -> always step a (2, 2) action.
        # When the opponent is absent it stays parked off-map with zero command.
        # While in collision with a virtual obstacle, the ego is held at 0 speed.
        ego_speed = 0.0 if self.ego_collision else self.ego_requested_speed
        ego_act = [self.ego_steer, ego_speed]
        opp_act = [self.opp_steer, self.opp_requested_speed] if self.has_opp else [0.0, 0.0]
        self.obs, _, self.done, _ = self.env.step(np.array([ego_act, opp_act]))
        self.ts = self.get_clock().now().to_msg()
        self._update_state()
        self._check_collision()
        self._report_collisions()
        # odom + tf track the physics rate
        self._publish_odom(self.ts)
        self._publish_transforms(self.ts)
        self._publish_wheel_transforms(self.ts)
        # lidar phase-locked to the physics step: every scan_decim steps publish a
        # scan raycast from THIS step's obs and stamped with THIS step's self.ts,
        # so the scan and its TF are always coherent (no inter-timer drift/race).
        self._scan_tick += 1
        if self._scan_tick >= self.scan_decim:
            self._scan_tick = 0
            self._publish_scans(self.ts)

    def _check_collision(self):
        """Geometric ego-vs-virtual-obstacle collision (opponent + static).
        Overlay obstacles aren't in the gym physics, so stop the ego here."""
        ex, ey = self.obs['poses_x'][0], self.obs['poses_y'][0]
        # Split opponent (dynamic) vs static so telemetry can distinguish them.
        opp_hit = any(math.hypot(ex - x, ey - y) < self.ego_half + half
                      for x, y, half in self._dyn_obs)
        stat_hit = any(math.hypot(ex - x, ey - y) < self.ego_half + half
                       for x, y, half in self._stat_obs)
        self._opp_collision_now = opp_hit
        hit = opp_hit or stat_hit
        if hit:
            self.env.unwrapped.sim.agents[0].state[3] = 0.0   # zero ego speed now
            self.ego_requested_speed = 0.0
            if not self.ego_collision:
                self.get_logger().warn('[GymBridge] ego collision with virtual obstacle -> stopped')
            self._collision_ticks += 1
            # Wedged against the obstacle for too long -> respawn on the raceline
            # so the run recovers instead of dead-locking (speed is forced to 0
            # every tick, so the ego could never drive itself free).
            if self._collision_ticks > 150 and self._raceline is not None:
                self._respawn_on_raceline(ex, ey)
                self._collision_ticks = 0
                hit = False
        else:
            self._collision_ticks = 0
        self.ego_collision = hit

    def _report_collisions(self):
        """Emit one pitwall telemetry event per collision onset (rising edge).
        Wall = gym's per-agent collision flag (ego footprint vs map walls);
        opponent = ego footprint vs the opponent virtual obstacle (overlay arch)."""
        wall_hit = bool(self.obs['collisions'][0]) if 'collisions' in self.obs else False
        if wall_hit and not self._wall_hit_latch:
            pitwall.event('ego: WALL collision')
        self._wall_hit_latch = wall_hit

        opp_hit = self._opp_collision_now
        if opp_hit and not self._opp_hit_latch:
            pitwall.event('ego: OPPONENT collision')
        self._opp_hit_latch = opp_hit

    def _raceline_callback(self, msg: WpntArray):
        if len(msg.wpnts) >= 3:
            self._raceline = np.array([[w.x_m, w.y_m, w.psi_rad] for w in msg.wpnts])

    def _respawn_on_raceline(self, ex, ey):
        rl = self._raceline
        i = int(np.argmin((rl[:, 0] - ex) ** 2 + (rl[:, 1] - ey) ** 2))
        rx, ry, rpsi = float(rl[i, 0]), float(rl[i, 1]), float(rl[i, 2])
        opp_pose = ([self.obs['poses_x'][1], self.obs['poses_y'][1], self.obs['poses_theta'][1]]
                    if self.has_opp else self.OPP_PARK_POSE)
        self.obs, _, self.done, _ = self.env.reset(
            poses=np.array([[rx, ry, rpsi], opp_pose]))
        self.ego_requested_speed = 0.0
        self.get_logger().warn(
            f'[GymBridge] crash recovery -> respawn on raceline at ({rx:.1f}, {ry:.1f})')

    def _publish_scans(self, stamp):
        # ---- lidar scans (phase-locked: called from drive_timer_callback every
        # scan_decim physics steps, running in that same callback) ----
        self._update_scans()
        # `stamp` is the self.ts of the physics step we are raycasting from, the
        # SAME stamp the map->base_link TF for this pose was published with. Since
        # this runs inside the drive step (one thread, no step can interleave), the
        # scan geometry and its TF are always coherent, so the scan no longer slides
        # against the map while moving.
        if self.ego_lidar_on:
            scan = LaserScan()
            scan.header.stamp = stamp
            scan.header.frame_id = (self.ego_namespace + '/laser') if self.ego_namespace else 'laser'
            scan.angle_min = self.angle_min
            scan.angle_max = self.angle_max
            scan.angle_increment = self.angle_inc
            scan.range_min = 0.
            scan.range_max = 30.
            scan.ranges = self.ego_scan
            scan.intensities = self.ego_scan  # Use range as intensity for rainbow coloring
            self.ego_scan_pub.publish(scan)

        if self.has_opp and self.opp_lidar_on:
            opp_scan = LaserScan()
            opp_scan.header.stamp = stamp
            opp_scan.header.frame_id = self.opp_namespace + '/laser'
            opp_scan.angle_min = self.angle_min
            opp_scan.angle_max = self.angle_max
            opp_scan.angle_increment = self.angle_inc
            opp_scan.range_min = 0.
            opp_scan.range_max = 30.
            opp_scan.ranges = self.opp_scan
            self.opp_scan_pub.publish(opp_scan)

    def _update_state(self):
        # cheap per-step state (poses/speeds) at physics rate.
        # opponent pose ALWAYS tracked from obs (parked pose 1000 -> TF follows
        # the model off-screen when removed).
        self.opp_pose[0] = self.obs['poses_x'][1]
        self.opp_pose[1] = self.obs['poses_y'][1]
        self.opp_pose[2] = self.obs['poses_theta'][1]
        if self.has_opp:
            self.opp_speed[0] = self.obs['linear_vels_x'][1]
            self.opp_speed[1] = self.obs['linear_vels_y'][1]
            self.opp_speed[2] = self.obs['ang_vels_z'][1]

        self.ego_pose[0] = self.obs['poses_x'][0]
        self.ego_pose[1] = self.obs['poses_y'][0]
        self.ego_pose[2] = self.obs['poses_theta'][0]
        self.ego_speed[0] = self.obs['linear_vels_x'][0]
        self.ego_speed[1] = self.obs['linear_vels_y'][0]
        self.ego_speed[2] = self.obs['ang_vels_z'][0]

    def _update_scans(self):
        # raycaster scans at scan rate (reads the latest stepped obs).
        if self.ego_lidar_on:
            if self.use_raycaster and self.engine is not None:
                self.ego_scan = self._raycast_scan(0, 1 if self.has_opp else None)
            else:
                self.ego_scan = list(self.obs['scans'][0])
        if self.has_opp and self.opp_lidar_on:
            if self.use_raycaster and self.engine is not None:
                self.opp_scan = self._raycast_scan(1, 0)
            else:
                self.opp_scan = list(self.obs['scans'][1])

    def _publish_odom(self, ts):
        ego_odom = Odometry()
        ego_odom.header.stamp = ts
        ego_odom.header.frame_id = 'map'
        ego_odom.child_frame_id = (self.ego_namespace + '/base_link') if self.ego_namespace else 'base_link'
        ego_odom.pose.pose.position.x = self.ego_pose[0]
        ego_odom.pose.pose.position.y = self.ego_pose[1]
        ego_quat = euler.euler2quat(0., 0., self.ego_pose[2], axes='sxyz')
        ego_odom.pose.pose.orientation.x = ego_quat[1]
        ego_odom.pose.pose.orientation.y = ego_quat[2]
        ego_odom.pose.pose.orientation.z = ego_quat[3]
        ego_odom.pose.pose.orientation.w = ego_quat[0]
        ego_odom.twist.twist.linear.x = self.ego_speed[0]
        ego_odom.twist.twist.linear.y = self.ego_speed[1]
        ego_odom.twist.twist.angular.z = self.ego_speed[2]
        self.ego_odom_pub.publish(ego_odom)
        self.fake_vesc_odom_pub.publish(ego_odom)

        if self.has_opp:
            opp_odom = Odometry()
            opp_odom.header.stamp = ts
            opp_odom.header.frame_id = 'map'
            opp_odom.child_frame_id = self.opp_namespace + '/base_link'
            opp_odom.pose.pose.position.x = self.opp_pose[0]
            opp_odom.pose.pose.position.y = self.opp_pose[1]
            opp_quat = euler.euler2quat(0., 0., self.opp_pose[2], axes='sxyz')
            opp_odom.pose.pose.orientation.x = opp_quat[1]
            opp_odom.pose.pose.orientation.y = opp_quat[2]
            opp_odom.pose.pose.orientation.z = opp_quat[3]
            opp_odom.pose.pose.orientation.w = opp_quat[0]
            opp_odom.twist.twist.linear.x = self.opp_speed[0]
            opp_odom.twist.twist.linear.y = self.opp_speed[1]
            opp_odom.twist.twist.angular.z = self.opp_speed[2]
            self.opp_odom_pub.publish(opp_odom)
            self.opp_ego_odom_pub.publish(ego_odom)
            self.ego_opp_odom_pub.publish(opp_odom)

    def _publish_transforms(self, ts):
        ego_t = Transform()
        ego_t.translation.x = self.ego_pose[0]
        ego_t.translation.y = self.ego_pose[1]
        ego_t.translation.z = 0.0
        ego_quat = euler.euler2quat(0.0, 0.0, self.ego_pose[2], axes='sxyz')
        ego_t.rotation.x = ego_quat[1]
        ego_t.rotation.y = ego_quat[2]
        ego_t.rotation.z = ego_quat[3]
        ego_t.rotation.w = ego_quat[0]

        ego_ts = TransformStamped()
        ego_ts.transform = ego_t
        ego_ts.header.stamp = ts
        ego_ts.header.frame_id = 'map'
        ego_ts.child_frame_id = (self.ego_namespace + '/base_link') if self.ego_namespace else 'base_link'
        self.br.sendTransform(ego_ts)

        # Opponent base_link TF (only in internal-opponent mode; the external
        # `opponent` package publishes it otherwise). Parked off-map (~1000 m)
        # when removed, so the RViz model follows off-screen instead of freezing.
        if self.external_opp:
            return
        opp_t = Transform()
        opp_t.translation.x = self.opp_pose[0]
        opp_t.translation.y = self.opp_pose[1]
        opp_t.translation.z = 0.0
        opp_quat = euler.euler2quat(
            0.0, 0.0, self.opp_pose[2], axes='sxyz')
        opp_t.rotation.x = opp_quat[1]
        opp_t.rotation.y = opp_quat[2]
        opp_t.rotation.z = opp_quat[3]
        opp_t.rotation.w = opp_quat[0]

        opp_ts = TransformStamped()
        opp_ts.transform = opp_t
        opp_ts.header.stamp = ts
        opp_ts.header.frame_id = 'map'
        opp_ts.child_frame_id = self.opp_namespace + '/base_link'
        self.br.sendTransform(opp_ts)

    def _publish_wheel_transforms(self, ts):
        ego_wheel_ts = TransformStamped()
        ego_wheel_quat = euler.euler2quat(0., 0., self.ego_steer, axes='sxyz')
        ego_wheel_ts.transform.rotation.x = ego_wheel_quat[1]
        ego_wheel_ts.transform.rotation.y = ego_wheel_quat[2]
        ego_wheel_ts.transform.rotation.z = ego_wheel_quat[3]
        ego_wheel_ts.transform.rotation.w = ego_wheel_quat[0]
        ego_wheel_ts.header.stamp = ts
        ego_wheel_ts.header.frame_id = (self.ego_namespace + '/front_left_hinge') if self.ego_namespace else 'front_left_hinge'
        ego_wheel_ts.child_frame_id = (self.ego_namespace + '/front_left_wheel') if self.ego_namespace else 'front_left_wheel'
        self.br.sendTransform(ego_wheel_ts)
        ego_wheel_ts.header.frame_id = (self.ego_namespace + '/front_right_hinge') if self.ego_namespace else 'front_right_hinge'
        ego_wheel_ts.child_frame_id = (self.ego_namespace + '/front_right_wheel') if self.ego_namespace else 'front_right_wheel'
        self.br.sendTransform(ego_wheel_ts)

        if self.has_opp:
            opp_wheel_ts = TransformStamped()
            opp_wheel_quat = euler.euler2quat(
                0., 0., self.opp_steer, axes='sxyz')
            opp_wheel_ts.transform.rotation.x = opp_wheel_quat[1]
            opp_wheel_ts.transform.rotation.y = opp_wheel_quat[2]
            opp_wheel_ts.transform.rotation.z = opp_wheel_quat[3]
            opp_wheel_ts.transform.rotation.w = opp_wheel_quat[0]
            opp_wheel_ts.header.stamp = ts
            opp_wheel_ts.header.frame_id = self.opp_namespace + '/front_left_hinge'
            opp_wheel_ts.child_frame_id = self.opp_namespace + '/front_left_wheel'
            self.br.sendTransform(opp_wheel_ts)
            opp_wheel_ts.header.frame_id = self.opp_namespace + '/front_right_hinge'
            opp_wheel_ts.child_frame_id = self.opp_namespace + '/front_right_wheel'
            self.br.sendTransform(opp_wheel_ts)


def main(args=None):
    rclpy.init(args=args)
    gym_bridge = GymBridge()
    
    executor = MultiThreadedExecutor()
    executor.add_node(gym_bridge)

    try:
        executor.spin()
    except KeyboardInterrupt:
        gym_bridge.get_logger().info('Exiting gym_bridge')
    

    gym_bridge.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
