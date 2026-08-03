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
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Transform
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformBroadcaster

import gymnasium as gym
import numpy as np
import cv2
import threading
from transforms3d import euler
import os.path


class GymBridge(Node):
    def __init__(self):
        super().__init__('gym_bridge',
                         automatically_declare_parameters_from_overrides=True)

        self.set_descriptor(name='ego_namespace', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING))
        self.set_descriptor(name='ego_odom_topic', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING))
        self.set_descriptor(name='ego_pose_topic', descriptor=ParameterDescriptor(
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

        self.set_descriptor(name='sim_params', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING, description="The path to the sim_params yaml file"))

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

        if not self.has_parameter('publish_tf'):
            self.declare_parameter('publish_tf', True)
        self.set_descriptor(name='publish_tf', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_BOOL, description="Whether to publish map->base_link TF"))

        # check num_agents
        num_agents = self.get_parameter('num_agent').value
        if num_agents < 1 or num_agents > 2:
            raise ValueError('num_agents should be either 1 or 2.')
        elif type(num_agents) != int:
            raise ValueError('num_agents should be an int.')

        # get sim params
        sim_params_yaml = self.get_parameter('sim_params').value
        sim_param_data = yaml.safe_load(open(sim_params_yaml, 'r'))
        sim_params = {key: float(value)
                      for key, value in sim_param_data.items()}

        # get scan parameters
        scan_fov = self.get_parameter('scan_fov').value
        scan_beams = self.get_parameter('scan_beams').value

        # env backend
        map_yaml_path = os.path.abspath(self.get_parameter('map_path').value)
        self.env = gym.make('f110_gym:f110-v0',
                            map=map_yaml_path.split('.')[0],
                            map_ext=self.get_parameter('map_img_ext').value,
                            params=sim_params,
                            num_agents=num_agents,
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
        ego_pose_topic = self.ego_namespace + '/' + \
            self.get_parameter('ego_pose_topic').value
        self.scan_distance_to_base_link = self.get_parameter(
            'scan_distance_to_base_link').value
        self.ts = self.get_clock().now().to_msg()
        self.publish_tf = self.get_parameter('publish_tf').value
        self._last_published_ts = None
        self._ts_lock = threading.Lock()

        if num_agents == 2:
            self.has_opp = True
            self.opp_namespace = self.get_parameter('opp_namespace').value
            sx1 = self.get_parameter('sx1').value
            sy1 = self.get_parameter('sy1').value
            stheta1 = self.get_parameter('stheta1').value
            self.opp_pose = [sx1, sy1, stheta1]
            self.opp_speed = [0.0, 0.0, 0.0]
            self.opp_requested_speed = 0.0
            self.opp_steer = 0.0
            self.opp_collision = False
            self.obs, _, self.done, _ = self.env.reset(
                poses=np.array([[sx, sy, stheta], [sx1, sy1, stheta1]]))
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
        else:
            self.has_opp = False
            self.obs, _, self.done, _ = self.env.reset(
                poses=np.array([[sx, sy, stheta]]))
            self.ego_scan = list(self.obs['scans'][0])

        # sim physical step timer
        cb_group1= ReentrantCallbackGroup()
        self.drive_timer = self.create_timer(0.01, self.drive_timer_callback, callback_group=cb_group1)
        # topic publishing timer
        self.timer = self.create_timer(0.01, self.timer_callback, callback_group=cb_group1)

        # transform broadcaster
        self.br = TransformBroadcaster(self)

        # publishers
        self.ego_scan_pub = self.create_publisher(
            LaserScan, ego_scan_topic, 10)
        self.ego_odom_pub = self.create_publisher(Odometry, ego_odom_topic, 10)
        self.ego_pose_pub = self.create_publisher(PoseStamped, ego_pose_topic, 10)
        self.ego_drive_published = False
        if num_agents == 2:
            self.opp_scan_pub = self.create_publisher(
                LaserScan, opp_scan_topic, 10)
            self.ego_opp_odom_pub = self.create_publisher(
                Odometry, ego_opp_odom_topic, 10)
            self.opp_odom_pub = self.create_publisher(
                Odometry, opp_odom_topic, 10)
            self.opp_ego_odom_pub = self.create_publisher(
                Odometry, opp_ego_odom_topic, 10)
            self.opp_drive_published = False

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
        if num_agents == 2:
            self.opp_drive_sub = self.create_subscription(
                AckermannDriveStamped,
                opp_drive_topic,
                self.opp_drive_callback,
                10)
            self.opp_reset_sub = self.create_subscription(
                PoseStamped,
                '/goal_pose',
                self.opp_reset_callback,
                10)

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
        if self.has_opp:
            opp_pose = [self.obs['poses_x'][1], self.obs['poses_y']
                        [1], self.obs['poses_theta'][1]]
            self.obs, _, self.done, _ = self.env.reset(
                poses=np.array([[rx, ry, rtheta], opp_pose]))
        else:
            self.obs, _, self.done, _ = self.env.reset(
                poses=np.array([[rx, ry, rtheta]]))

    def opp_reset_callback(self, pose_msg):
        if self.has_opp:
            rx = pose_msg.pose.position.x
            ry = pose_msg.pose.position.y
            rqx = pose_msg.pose.orientation.x
            rqy = pose_msg.pose.orientation.y
            rqz = pose_msg.pose.orientation.z
            rqw = pose_msg.pose.orientation.w
            _, _, rtheta = euler.quat2euler([rqw, rqx, rqy, rqz], axes='sxyz')
            self.obs, _, self.done, _ = self.env.reset(
                poses=np.array([list(self.ego_pose), [rx, ry, rtheta]]))

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
            center_px = self._meters_to_pixels(obs['x'], obs['y'])
            radius_px = max(1, int(obs['radius'] / self.map_resolution))
            cv2.circle(self.current_map_img, center_px, radius_px, 0, -1)

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

        # Update gym environment's distance transform
        # The gym expects the image with origin at bottom-left (flipped)
        flipped = np.flipud(self.current_map_img)
        self.env.unwrapped.update_map_from_array(
            flipped, self.map_resolution,
            self.map_origin_x, self.map_origin_y)

        # Publish updated /map
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
        # Update map if obstacles changed (DT recalculation)
        if self.map_needs_update:
            self._update_gym_map()

        # Always step the simulation to generate new scan noise
        if not self.has_opp:
            self.obs, _, self.done, _ = self.env.step(
                np.array([[self.ego_steer, self.ego_requested_speed]]))
        elif self.has_opp and self.opp_drive_published:
            self.obs, _, self.done, _ = self.env.step(np.array(
                [[self.ego_steer, self.ego_requested_speed], [self.opp_steer, self.opp_requested_speed]]))
        with self._ts_lock:
            self.ts = self.get_clock().now().to_msg()
        self._update_sim_state()

    def timer_callback(self):
        # Capture ts snapshot inside lock to guarantee we publish exactly this timestamp.
        # self.ts can be updated by drive_timer_callback concurrently, so we must
        # snapshot it atomically and use only the snapshot → no duplicate timestamps.
        with self._ts_lock:
            ts = self.ts  # snapshot
            ts_key = (ts.sec, ts.nanosec)
            if ts_key == self._last_published_ts:
                return
            self._last_published_ts = ts_key

        # pub scans (use snapshot ts, not self.ts)
        scan = LaserScan()
        scan.header.stamp = ts
        scan.header.frame_id = (self.ego_namespace + '/laser') if self.ego_namespace else 'laser'
        scan.angle_min = self.angle_min
        scan.angle_max = self.angle_max
        scan.angle_increment = self.angle_inc
        scan.range_min = 0.
        scan.range_max = 30.
        scan.ranges = self.ego_scan
        scan.intensities = self.ego_scan  # Use range as intensity for rainbow coloring
        self.ego_scan_pub.publish(scan)

        if self.has_opp:
            opp_scan = LaserScan()
            opp_scan.header.stamp = ts
            opp_scan.header.frame_id = self.opp_namespace + '/laser'
            opp_scan.angle_min = self.angle_min
            opp_scan.angle_max = self.angle_max
            opp_scan.angle_increment = self.angle_inc
            opp_scan.range_min = 0.
            opp_scan.range_max = 30.
            opp_scan.ranges = self.opp_scan
            self.opp_scan_pub.publish(opp_scan)

        self._publish_odom(ts)
        if self.publish_tf:
            self._publish_transforms(ts)
            self._publish_wheel_transforms(ts)

    def _update_sim_state(self):
        self.ego_scan = list(self.obs['scans'][0])
        if self.has_opp:
            self.opp_scan = list(self.obs['scans'][1])
            self.opp_pose[0] = self.obs['poses_x'][1]
            self.opp_pose[1] = self.obs['poses_y'][1]
            self.opp_pose[2] = self.obs['poses_theta'][1]
            self.opp_speed[0] = self.obs['linear_vels_x'][1]
            self.opp_speed[1] = self.obs['linear_vels_y'][1]
            self.opp_speed[2] = self.obs['ang_vels_z'][1]

        self.ego_pose[0] = self.obs['poses_x'][0]
        self.ego_pose[1] = self.obs['poses_y'][0]
        self.ego_pose[2] = self.obs['poses_theta'][0]
        self.ego_speed[0] = self.obs['linear_vels_x'][0]
        self.ego_speed[1] = self.obs['linear_vels_y'][0]
        self.ego_speed[2] = self.obs['ang_vels_z'][0]

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
        
        # publish pose
        pose_msg = PoseStamped()
        pose_msg.header = ego_odom.header
        pose_msg.pose = ego_odom.pose.pose
        self.ego_pose_pub.publish(pose_msg)

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
        ego_ts.child_frame_id = self.ego_namespace + '/base_link'
        self.br.sendTransform(ego_ts)

        if self.has_opp:
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
        ego_wheel_ts.header.frame_id = self.ego_namespace + '/front_left_hinge'
        ego_wheel_ts.child_frame_id = self.ego_namespace + '/front_left_wheel'
        self.br.sendTransform(ego_wheel_ts)
        ego_wheel_ts.header.frame_id = self.ego_namespace + '/front_right_hinge'
        ego_wheel_ts.child_frame_id = self.ego_namespace + '/front_right_wheel'
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