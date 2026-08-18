# MIT License

# Copyright (c) 2020 Hongrui Zheng, Corey Walsh

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the 'Software'), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ros2 python
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor

# libraries
import numpy as np
import range_libc
import time
from threading import Lock
from particle_filter import utils as Utils

# TF
# import tf.transformations
# import tf
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
import tf_transformations

# messages
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String, Header, Float32MultiArray
from sensor_msgs.msg import Imu, LaserScan
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Pose, PoseStamped, PoseArray, Quaternion, PolygonStamped, Polygon, Point32, PoseWithCovarianceStamped, PointStamped, TransformStamped
from nav_msgs.msg import Odometry
from nav_msgs.srv import GetMap

'''
These flags indicate several variants of the sensor model. Only one of them is used at a time.
'''
VAR_NO_EVAL_SENSOR_MODEL = 0
VAR_CALC_RANGE_MANY_EVAL_SENSOR = 1
VAR_REPEAT_ANGLES_EVAL_SENSOR = 2
VAR_REPEAT_ANGLES_EVAL_SENSOR_ONE_SHOT = 3
VAR_RADIAL_CDDT_OPTIMIZATIONS = 4


class ParticleFiler(Node):
    '''
    This class implements Monte Carlo Localization based on odometry and a laser scanner.
    '''

    def __init__(self):
        super().__init__('particle_filter')

        # declare parameters
        #
        # dynamic_typing because every one of these gets its value and its type
        # from pf.yaml. Declaring a bare name means the same thing, but Humble
        # deprecated it and prints a six-line warning for each - twenty-two of
        # them, which buried the startup lines worth reading.
        dyn = ParameterDescriptor(dynamic_typing=True)
        self.declare_parameter('angle_step', descriptor=dyn)
        self.declare_parameter('max_particles', descriptor=dyn)
        self.declare_parameter('max_viz_particles', descriptor=dyn)
        self.declare_parameter('squash_factor', descriptor=dyn)
        self.declare_parameter('max_range', descriptor=dyn)
        self.declare_parameter('theta_discretization', descriptor=dyn)
        self.declare_parameter('range_method', descriptor=dyn)
        self.declare_parameter('rangelib_variant', descriptor=dyn)
        self.declare_parameter('fine_timing', descriptor=dyn)
        self.declare_parameter('publish_odom', descriptor=dyn)
        # Default matters: the parameter must not be required. False is also the
        # right default here - the EKF downstream owns map -> base_link, and if
        # the filter published map -> laser as well, laser would have two parents
        # (the URDF already gives base_link -> laser) and TF would reject it.
        self.declare_parameter('publish_tf', False)
        # Frame names come from the racecar description, not from upstream's defaults.
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'ego_racecar/base_link')
        self.declare_parameter('laser_frame', 'ego_racecar/laser')
        self.declare_parameter('viz', descriptor=dyn)
        self.declare_parameter('z_short', descriptor=dyn)
        self.declare_parameter('z_max', descriptor=dyn)
        self.declare_parameter('z_rand', descriptor=dyn)
        self.declare_parameter('z_hit', descriptor=dyn)
        self.declare_parameter('sigma_hit', descriptor=dyn)
        self.declare_parameter('motion_dispersion_x', descriptor=dyn)
        self.declare_parameter('motion_dispersion_y', descriptor=dyn)
        self.declare_parameter('motion_dispersion_theta', descriptor=dyn)
        self.declare_parameter('scan_topic', descriptor=dyn)
        self.declare_parameter('odometry_topic', descriptor=dyn)
        # Where the motion model's YAW RATE comes from. Empty keeps it on the
        # odometry topic; see imuCB for why that is the wrong source on this car.
        self.declare_parameter('imu_topic', '')
        # [Hz] How often the viz topics are drawn, independent of the scan
        # rate. 0 draws on every iteration, the old behaviour.
        self.declare_parameter('viz_rate_hz', 10.0)

        # parameters
        self.ANGLE_STEP           = self.get_parameter('angle_step').value
        self.MAX_PARTICLES        = self.get_parameter('max_particles').value
        self.MAX_VIZ_PARTICLES    = self.get_parameter('max_viz_particles').value
        self.INV_SQUASH_FACTOR    = 1.0 / self.get_parameter('squash_factor').value
        self.MAX_RANGE_METERS     = self.get_parameter('max_range').value
        self.THETA_DISCRETIZATION = self.get_parameter('theta_discretization').value
        self.WHICH_RM             = self.get_parameter('range_method').value
        self.RANGELIB_VAR         = self.get_parameter('rangelib_variant').value
        self.SHOW_FINE_TIMING     = self.get_parameter('fine_timing').value
        self.PUBLISH_ODOM         = self.get_parameter('publish_odom').value
        self.PUBLISH_TF           = self.get_parameter('publish_tf').value
        self.MAP_FRAME            = self.get_parameter('map_frame').value
        self.BASE_FRAME           = self.get_parameter('base_frame').value
        self.LASER_FRAME          = self.get_parameter('laser_frame').value
        self.DO_VIZ               = self.get_parameter('viz').value
        self.VIZ_RATE_HZ          = float(self.get_parameter('viz_rate_hz').value)

        # sensor model constants
        self.Z_SHORT   = self.get_parameter('z_short').value
        self.Z_MAX     = self.get_parameter('z_max').value
        self.Z_RAND    = self.get_parameter('z_rand').value
        self.Z_HIT     = self.get_parameter('z_hit').value
        self.SIGMA_HIT = self.get_parameter('sigma_hit').value

        # motion model constants
        self.MOTION_DISPERSION_X     = self.get_parameter('motion_dispersion_x').value
        self.MOTION_DISPERSION_Y     = self.get_parameter('motion_dispersion_y').value
        self.MOTION_DISPERSION_THETA = self.get_parameter('motion_dispersion_theta').value
        
        # various data containers used in the MCL algorithm
        self.MAX_RANGE_PX = None
        self.odometry_data = np.array([0.0, 0.0, 0.0])
        self.laser = None
        self.iters = 0
        self.map_info = None
        self.map_initialized = False
        self.lidar_initialized = False
        # velocity motion model is integrated in lidarCB, so the filter no longer
        # requires odom to have arrived; velocities default to 0 (= no motion).
        self.odom_initialized = True
        self.last_pose = None
        self.laser_angles = None
        self.downsampled_angles = None
        self.range_method = None
        self.last_time = None
        self.last_stamp = None
        self.first_sensor_update = True
        self.last_viz_sec = 0.0
        self.state_lock = Lock()

        # cache this to avoid memory allocation in motion model
        self.local_deltas = np.zeros((self.MAX_PARTICLES, 3))

        # cache this for the sensor model computation
        self.queries = None
        self.ranges = None
        self.tiled_angles = None
        self.sensor_model_table = None

        # particle poses and weights
        self.inferred_pose = None
        self.particle_indices = np.arange(self.MAX_PARTICLES)
        self.particles = np.zeros((self.MAX_PARTICLES, 3))
        self.weights = np.ones(self.MAX_PARTICLES) / float(self.MAX_PARTICLES)

        # initialize the state
        self.smoothing = Utils.CircularArray(10)
        self.timer = Utils.Timer(10)
        # map service client
        self.map_client = self.create_client(GetMap, '/map_server/map')
        self.get_omap()
        self.precompute_sensor_model()
        self.initialize_global()

        # keep track of velocity from input odom; integrated over the scan dt
        self.current_speed   = 0.0   # twist.linear.x  (body forward velocity)
        self.current_angular = 0.0   # twist.angular.z (yaw rate)
        self.last_scan_time  = None  # float secs of the previous scan, for dt

        # Pub Subs
        # these topics are for visualization
        self.pose_pub = self.create_publisher(PoseStamped, '/pf/viz/inferred_pose', 1)
        self.particle_pub = self.create_publisher(PoseArray, '/pf/viz/particles', 1)
        self.pub_fake_scan = self.create_publisher(LaserScan, '/pf/viz/fake_scan', 1)
        self.rect_pub = self.create_publisher(PolygonStamped, '/pf/viz/poly1', 1)

        if self.PUBLISH_ODOM:
            self.odom_pub = self.create_publisher(Odometry, '/pf/pose/odom', 1)
            # MCL estimates the LASER pose; /pf/pose/odom must report the BASE_LINK
            # pose (robot_localization fuses an odom pose as base_link's and ignores
            # child_frame_id for the pose). Look up the static base_link->laser
            # transform (lazily, once TF is up) to convert laser pose -> base_link.
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
            self.bl2laser = None          # (dx, dy, dyaw): laser expressed in base_link
            self._bl2laser_warned = False

        # these topics are for coordinate space things
        if self.PUBLISH_TF:
            self.pub_tf = TransformBroadcaster(self)    

        # these topics are to receive data from the racecar
        self.laser_sub = self.create_subscription(
            LaserScan,
            self.get_parameter('scan_topic').value,
            self.lidarCB,
            qos_profile_sensor_data)
        # Before odom_sub: odomCB reads it to decide whether it owns the yaw rate.
        self.imu_sub = None
        self.odom_sub = self.create_subscription(
            Odometry,
            self.get_parameter('odometry_topic').value,
            self.odomCB,
            1)
        imu_topic = self.get_parameter('imu_topic').value
        if imu_topic:
            self.imu_sub = self.create_subscription(
                Imu, imu_topic, self.imuCB, qos_profile_sensor_data)
            self.get_logger().info(f'yaw rate from {imu_topic}, not the odometry twist')
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/initialpose',
            self.clicked_pose,
            1)
        self.click_sub = self.create_subscription(
            PointStamped,
            '/clicked_point',
            self.clicked_pose,
            1)

        self.get_logger().info('Finished initializing, waiting on messages...')

    def get_omap(self):
        '''
        Fetch the occupancy grid map from the map_server instance, and initialize the correct
        RangeLibc method. Also stores a matrix which indicates the permissible region of the map
        '''

        while not self.map_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Get map service not available, waiting...')
        req = GetMap.Request()
        future = self.map_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        map_msg = future.result().map
        self.map_info = map_msg.info

        oMap = range_libc.PyOMap(map_msg)
        self.MAX_RANGE_PX = int(self.MAX_RANGE_METERS / self.map_info.resolution)

        # initialize range method
        self.get_logger().info('Initializing range method: ' + self.WHICH_RM)
        if self.WHICH_RM == 'bl':
            self.range_method = range_libc.PyBresenhamsLine(oMap, self.MAX_RANGE_PX)
        elif 'cddt' in self.WHICH_RM:
            self.range_method = range_libc.PyCDDTCast(oMap, self.MAX_RANGE_PX, self.THETA_DISCRETIZATION)
            if self.WHICH_RM == 'pcddt':
                self.get_logger().info('Pruning...')
                self.range_method.prune()
        elif self.WHICH_RM == 'rm':
            self.range_method = range_libc.PyRayMarching(oMap, self.MAX_RANGE_PX)
        elif self.WHICH_RM == 'rmgpu':
            self.range_method = range_libc.PyRayMarchingGPU(oMap, self.MAX_RANGE_PX)
        elif self.WHICH_RM == 'glt':
            self.range_method = range_libc.PyGiantLUTCast(oMap, self.MAX_RANGE_PX, self.THETA_DISCRETIZATION)
        self.get_logger().info('Done loading map')

         # 0: permissible, -1: unmapped, 100: blocked
        array_255 = np.array(map_msg.data).reshape((map_msg.info.height, map_msg.info.width))

        # 0: not permissible, 1: permissible
        self.permissible_region = np.zeros_like(array_255, dtype=bool)
        self.permissible_region[array_255==0] = 1
        self.map_initialized = True

    def _ensure_bl2laser(self):
        '''Lazily look up the static base_link->laser transform (laser pose in the
        base_link frame) and cache it. Returns True once available.'''
        if self.bl2laser is not None:
            return True
        try:
            tf = self.tf_buffer.lookup_transform(self.BASE_FRAME, self.LASER_FRAME, rclpy.time.Time())
            t = tf.transform.translation
            q = tf.transform.rotation
            yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
            self.bl2laser = (t.x, t.y, yaw)
            self.get_logger().info(
                f"[pf] base_link->laser offset = ({t.x:.3f}, {t.y:.3f}, {yaw:.3f} rad)")
            return True
        except Exception as e:
            if not self._bl2laser_warned:
                self.get_logger().warn(
                    f"[pf] base_link->laser TF not available yet ({e}); "
                    "publishing laser pose on /pf/pose/odom until it appears")
                self._bl2laser_warned = True
            return False

    def _laser_pose_to_base_link(self, pose):
        '''Convert a (x, y, yaw) LASER pose in map to the BASE_LINK pose in map,
        using the cached static base_link->laser offset.
        T_map_base = T_map_laser . (T_base_laser)^-1'''
        dx, dy, dyaw = self.bl2laser
        xl, yl, thl = pose[0], pose[1], pose[2]
        c, s = np.cos(dyaw), np.sin(dyaw)
        ix = -(c * dx + s * dy)            # inverse of base_link->laser
        iy = -(-s * dx + c * dy)
        ith = -dyaw
        cl, sl = np.cos(thl), np.sin(thl)
        xb = xl + cl * ix - sl * iy
        yb = yl + sl * ix + cl * iy
        thb = thl + ith
        return xb, yb, thb

    def publish_tf(self, pose, stamp=None):
        ''' Publish a tf for the car. This tells ROS where the car is with respect to the map. '''
        if stamp == None:
            stamp = self.get_clock().now().to_msg()
        if self.PUBLISH_TF:
            t = TransformStamped()
            # header
            t.header.stamp = stamp
            t.header.frame_id = self.MAP_FRAME
            t.child_frame_id = self.LASER_FRAME
            # translation
            t.transform.translation.x = pose[0]
            t.transform.translation.y = pose[1]
            t.transform.translation.z = 0.0
            q = tf_transformations.quaternion_from_euler(0., 0., pose[2])
            # rotation
            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]
            self.pub_tf.sendTransform(t)
        # also publish odometry to facilitate getting the localization pose
        if self.PUBLISH_ODOM:
            # pose is the LASER pose in map; convert to BASE_LINK so consumers
            # (e.g. robot_localization, which fuses an odom pose as base_link's) get
            # the correct frame. Falls back to the laser pose until the TF is up.
            if self._ensure_bl2laser():
                bx, by, byaw = self._laser_pose_to_base_link(pose)
            else:
                bx, by, byaw = pose[0], pose[1], pose[2]
            odom = Odometry()
            odom.header.stamp = self.get_clock().now().to_msg()
            odom.header.frame_id = self.MAP_FRAME
            odom.child_frame_id = self.BASE_FRAME
            odom.pose.pose.position.x = bx
            odom.pose.pose.position.y = by
            odom.pose.pose.orientation = Utils.angle_to_quaternion(byaw)
            # The particle spread is 3x3 over (x, y, yaw); pose.covariance is a
            # 6x6 over (x, y, z, roll, pitch, yaw) flattened row-major. Writing
            # the 3x3 into the first nine slots scatters it - yaw variance lands
            # in the x-z slot and index 35, the one robot_localization reads for
            # heading, stays zero. A zero variance claims the heading is known
            # exactly, which is why the filter appeared to lock and stop moving.
            cov = np.cov(self.particles, rowvar=False, ddof=0, aweights=self.weights)
            idx = (0, 1, 5)  # x, y, yaw within the 6x6
            for r in range(3):
                for c in range(3):
                    odom.pose.covariance[idx[r] * 6 + idx[c]] = float(cov[r, c])
            # z, roll and pitch are not observed in 2D. Large rather than zero,
            # so nothing downstream mistakes them for certainty.
            for i in (2, 3, 4):
                odom.pose.covariance[i * 6 + i] = 1e6
            odom.twist.twist.linear.x = self.current_speed
            self.odom_pub.publish(odom)
        
        return

    def visualize(self):
        '''
        Publish various visualization messages.
        '''
        if not self.DO_VIZ:
            return

        # Drawing is not free and it is charged to this node. Every particle is
        # a Pose and a Quaternion built in python, and the fake scan below is a
        # whole extra calc_range_many pass - all of it at the scan rate, all of
        # it inside the loop whose duration is reported as `possible`. Turning
        # the full cloud on at 40 Hz would visibly depress the number being
        # measured. Ten a second is plenty to watch a cloud with.
        if self.VIZ_RATE_HZ > 0.0:
            now = time.time()
            if now - self.last_viz_sec < 1.0 / self.VIZ_RATE_HZ:
                return
            self.last_viz_sec = now

        if self.pose_pub.get_subscription_count() > 0 and isinstance(self.inferred_pose, np.ndarray):
            # Publish the inferred pose for visualization
            ps = PoseStamped()
            ps.header.stamp = self.get_clock().now().to_msg()
            ps.header.frame_id = self.MAP_FRAME
            ps.pose.position.x = self.inferred_pose[0]
            ps.pose.position.y = self.inferred_pose[1]
            ps.pose.orientation = Utils.angle_to_quaternion(self.inferred_pose[2])
            self.pose_pub.publish(ps)

        if self.particle_pub.get_subscription_count() > 0:
            # UNIFORM downsampling, and max_viz_particles: 0 draws every one.
            #
            # This used to sample with p=self.weights, which draws the sixty
            # BEST particles and therefore always looks tight - it showed a
            # converged cloud through an entire run that ended in a wall with
            # the cloud actually falling apart. The inferred pose is already
            # published separately, so a weighted sample showed nothing that
            # was not already on screen and hid the one thing worth seeing.
            if 0 < self.MAX_VIZ_PARTICLES < self.MAX_PARTICLES:
                indices = np.random.choice(
                    self.particle_indices, self.MAX_VIZ_PARTICLES, replace=False)
                self.publish_particles(self.particles[indices, :])
            else:
                self.publish_particles(self.particles)

        if self.pub_fake_scan.get_subscription_count() > 0 and isinstance(self.ranges, np.ndarray):
            # generate the scan from the point of view of the inferred position for visualization
            self.viz_queries[:,0] = self.inferred_pose[0]
            self.viz_queries[:,1] = self.inferred_pose[1]
            self.viz_queries[:,2] = self.downsampled_angles + self.inferred_pose[2]
            self.range_method.calc_range_many(self.viz_queries, self.viz_ranges)
            self.publish_scan(self.downsampled_angles, self.viz_ranges)

    def publish_particles(self, particles):
        # publish the given particles as a PoseArray object
        pa = PoseArray()
        pa.header.stamp = self.get_clock().now().to_msg()
        pa.header.frame_id = self.MAP_FRAME
        pa.poses = Utils.particles_to_poses(particles)
        self.particle_pub.publish(pa)

    def publish_scan(self, angles, ranges):
        # publish the given angels and ranges as a laser scan message
        ls = LaserScan()
        ls.header.stamp = self.last_stamp
        ls.header.frame_id = self.LASER_FRAME
        ls.angle_min = float(np.min(angles))
        ls.angle_max = float(np.max(angles))
        ls.angle_increment = float(np.abs(angles[0] - angles[1]))
        ls.range_min = 0.0
        ls.range_max = float(np.max(ranges))
        ls.ranges = ranges.tolist()
        self.pub_fake_scan.publish(ls)

    def lidarCB(self, msg):
        '''
        Initializes reused buffers, and stores the relevant laser scanner data for later use.
        '''
        if not isinstance(self.laser_angles, np.ndarray):
            self.get_logger().info('...Received first LiDAR message')
            self.laser_angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
            self.downsampled_angles = np.copy(self.laser_angles[0::self.ANGLE_STEP]).astype(np.float32)
            self.viz_queries = np.zeros((self.downsampled_angles.shape[0],3), dtype=np.float32)
            self.viz_ranges = np.zeros(self.downsampled_angles.shape[0], dtype=np.float32)
            self.get_logger().info(str(self.downsampled_angles.shape[0]))

        # store the necessary scanner information for later processing
        self.downsampled_ranges = np.array(msg.ranges[::self.ANGLE_STEP])
        self.lidar_initialized = True

        # Velocity motion model: integrate the latest odom twist over the interval
        # since the previous scan -> body-frame delta [dx, dy(~0 on a car), dtheta].
        # update() consumes self.odometry_data and zeroes it, so we recompute it
        # fresh from velocity * dt on every scan.
        scan_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_scan_time is not None:
            dt = scan_time - self.last_scan_time
            if dt > 0.0:
                self.odometry_data = np.array(
                    [self.current_speed * dt, 0.0, self.current_angular * dt])
        self.last_scan_time = scan_time
        self.last_stamp = msg.header.stamp

        self.update()

    def odomCB(self, msg):
        '''
        Store the latest body-frame velocity (vx) and yaw rate (wz) from the odom
        twist. The motion is integrated over the scan interval in lidarCB, so this
        callback no longer computes position deltas or triggers an MCL update.
        '''
        self.current_speed = msg.twist.twist.linear.x
        if self.imu_sub is None:
            self.current_angular = msg.twist.twist.angular.z
        self.odom_initialized = True

    def imuCB(self, msg):
        '''
        Take the yaw rate from the gyro instead of the odometry twist.

        /vesc/odom's angular.z is not measured. vesc_to_odom takes the servo
        COMMAND and puts it through a bicycle model, so it reports how fast the
        car would be turning if the tyres never slipped. The error is
        systematic, not noise, and the motion model injects it on every scan.

        That is what runs the filter down over a lap. Each step starts the
        cloud off in the wrong direction, the sensor model has to drag it back,
        and when it cannot keep up the cloud spreads. A spread cloud makes
        every ray cast slower - the whole point of `possible` - which lengthens
        the interval, which makes the next motion step wronger still. Resetting
        the pose from RViz breaks the loop and `possible` jumps straight back
        up, which is how this was found.

        The gyro measures the rate the car ACTUALLY turns. Its z axis is the
        car's yaw axis whatever the mounting yaw is - a rotation about z cannot
        move z - so no transform is needed here.
        '''
        self.current_angular = msg.angular_velocity.z

    def clicked_pose(self, msg):
        '''
        Receive pose messages from RViz and initialize the particle distribution in response.
        '''
        if isinstance(msg, PointStamped):
            self.initialize_global()
        elif isinstance(msg, PoseWithCovarianceStamped):
            self.initialize_particles_pose(msg.pose.pose)

    def initialize_particles_pose(self, pose):
        '''
        Initialize particles in the general region of the provided pose.
        '''
        self.get_logger().info('SETTING POSE')
        self.get_logger().info(str([pose.position.x, pose.position.y]))
        self.state_lock.acquire()
        self.weights = np.ones(self.MAX_PARTICLES) / float(self.MAX_PARTICLES)
        self.particles[:,0] = pose.position.x + np.random.normal(loc=0.0,scale=0.5,size=self.MAX_PARTICLES)
        self.particles[:,1] = pose.position.y + np.random.normal(loc=0.0,scale=0.5,size=self.MAX_PARTICLES)
        self.particles[:,2] = Utils.quaternion_to_angle(pose.orientation) + np.random.normal(loc=0.0,scale=0.4,size=self.MAX_PARTICLES)
        self.state_lock.release()

    def initialize_global(self):
        '''
        Spread the particle distribution over the permissible region of the state space.
        '''
        self.get_logger().info('GLOBAL INITIALIZATION')
        # randomize over grid coordinate space
        self.state_lock.acquire()
        permissible_x, permissible_y = np.where(self.permissible_region == 1)
        indices = np.random.randint(0, len(permissible_x), size=self.MAX_PARTICLES)

        permissible_states = np.zeros((self.MAX_PARTICLES,3))
        permissible_states[:,0] = permissible_y[indices]
        permissible_states[:,1] = permissible_x[indices]
        permissible_states[:,2] = np.random.random(self.MAX_PARTICLES) * np.pi * 2.0

        Utils.map_to_world(permissible_states, self.map_info)
        self.particles = permissible_states
        self.weights[:] = 1.0 / self.MAX_PARTICLES
        self.state_lock.release()

    def precompute_sensor_model(self):
        '''
        Generate and store a table which represents the sensor model. For each discrete computed
        range value, this provides the probability of measuring any (discrete) range.

        This table is indexed by the sensor model at runtime by discretizing the measurements
        and computed ranges from RangeLibc.
        '''
        self.get_logger().info('Precomputing sensor model')
        # sensor model constants
        z_short = self.Z_SHORT
        z_max   = self.Z_MAX
        z_rand  = self.Z_RAND
        z_hit   = self.Z_HIT
        sigma_hit = self.SIGMA_HIT
        
        table_width = int(self.MAX_RANGE_PX) + 1
        self.sensor_model_table = np.zeros((table_width,table_width))

        t = time.time()
        # d is the computed range from RangeLibc
        for d in range(table_width):
            norm = 0.0
            sum_unkown = 0.0
            # r is the observed range from the lidar unit
            for r in range(table_width):
                prob = 0.0
                z = float(r-d)
                # reflects from the intended object
                prob += z_hit * np.exp(-(z*z)/(2.0*sigma_hit*sigma_hit)) / (sigma_hit * np.sqrt(2.0*np.pi))

                # observed range is less than the predicted range - short reading
                if r < d:
                    prob += 2.0 * z_short * (d - r) / float(d)

                # erroneous max range measurement
                if int(r) == int(self.MAX_RANGE_PX):
                    prob += z_max

                # random measurement
                if r < int(self.MAX_RANGE_PX):
                    prob += z_rand * 1.0/float(self.MAX_RANGE_PX)

                norm += prob
                self.sensor_model_table[int(r),int(d)] = prob

            # normalize
            self.sensor_model_table[:,int(d)] /= norm

        # upload the sensor model to RangeLib for ultra fast resolution
        if self.RANGELIB_VAR > 0:
            self.range_method.set_sensor_model(self.sensor_model_table)

    def motion_model(self, proposal_dist, action):
        '''
        The motion model applies the odometry to the particle distribution. Since there the odometry
        data is inaccurate, the motion model mixes in gaussian noise to spread out the distribution.

        Vectorized motion model. Computing the motion model over all particles is thousands of times
        faster than doing it for each particle individually due to vectorization and reduction in
        function call overhead
        
        TODO this could be better, but it works for now
            - fixed random noise is not very realistic
            - ackermann model provides bad estimates at high speed
        '''
        # rotate the action into the coordinate space of each particle
        # t1 = time.time()
        cosines = np.cos(proposal_dist[:,2])
        sines = np.sin(proposal_dist[:,2])

        self.local_deltas[:,0] = cosines*action[0] - sines*action[1]
        self.local_deltas[:,1] = sines*action[0] + cosines*action[1]
        self.local_deltas[:,2] = action[2]

        proposal_dist[:,:] += self.local_deltas
        proposal_dist[:,0] += np.random.normal(loc=0.0,scale=self.MOTION_DISPERSION_X,size=self.MAX_PARTICLES)
        proposal_dist[:,1] += np.random.normal(loc=0.0,scale=self.MOTION_DISPERSION_Y,size=self.MAX_PARTICLES)
        proposal_dist[:,2] += np.random.normal(loc=0.0,scale=self.MOTION_DISPERSION_THETA,size=self.MAX_PARTICLES)

    def sensor_model(self, proposal_dist, obs, weights):
        '''
        This function computes a probablistic weight for each particle in the proposal distribution.
        These weights represent how probable each proposed (x,y,theta) pose is given the measured
        ranges from the lidar scanner.

        There are 4 different variants using various features of RangeLibc for demonstration purposes.
        - VAR_REPEAT_ANGLES_EVAL_SENSOR is the most stable, and is very fast.
        - VAR_NO_EVAL_SENSOR_MODEL directly indexes the precomputed sensor model. This is slow
                                   but it demonstrates what self.range_method.eval_sensor_model does
        - VAR_RADIAL_CDDT_OPTIMIZATIONS is only compatible with CDDT or PCDDT, it implments the radial
                                        optimizations to CDDT which simultaneously performs ray casting
                                        in two directions, reducing the amount of work by roughly a third
        '''
        
        num_rays = self.downsampled_angles.shape[0]
        # only allocate buffers once to avoid slowness
        if self.first_sensor_update:
            if self.RANGELIB_VAR <= 1:
                self.queries = np.zeros((num_rays*self.MAX_PARTICLES,3), dtype=np.float32)
            else:
                self.queries = np.zeros((self.MAX_PARTICLES,3), dtype=np.float32)

            self.ranges = np.zeros(num_rays*self.MAX_PARTICLES, dtype=np.float32)
            self.tiled_angles = np.tile(self.downsampled_angles, self.MAX_PARTICLES)
            self.first_sensor_update = False

        if self.RANGELIB_VAR == VAR_RADIAL_CDDT_OPTIMIZATIONS:
            if 'cddt' in self.WHICH_RM:
                self.queries[:,:] = proposal_dist[:,:]
                self.range_method.calc_range_many_radial_optimized(num_rays, self.downsampled_angles[0], self.downsampled_angles[-1], self.queries, self.ranges)

                # evaluate the sensor model
                self.range_method.eval_sensor_model(obs, self.ranges, self.weights, num_rays, self.MAX_PARTICLES)
                # apply the squash factor. In place, like the other variants
                # below: rebinding self.weights allocated a fresh array every
                # scan for no reason - nothing else holds the old one, and the
                # `weights` argument above is only read by variant 0.
                np.power(self.weights, self.INV_SQUASH_FACTOR, self.weights)
            else:
                self.get_logger().info('Cannot use radial optimizations with non-CDDT based methods, use rangelib_variant 2')
        elif self.RANGELIB_VAR == VAR_REPEAT_ANGLES_EVAL_SENSOR_ONE_SHOT:
            self.queries[:,:] = proposal_dist[:,:]
            self.range_method.calc_range_repeat_angles_eval_sensor_model(self.queries, self.downsampled_angles, obs, self.weights)
            np.power(self.weights, self.INV_SQUASH_FACTOR, self.weights)
        elif self.RANGELIB_VAR == VAR_REPEAT_ANGLES_EVAL_SENSOR:
            if self.SHOW_FINE_TIMING:
                t_start = time.time()
            # this version demonstrates what this would look like with coordinate space conversion pushed to rangelib
            self.queries[:,:] = proposal_dist[:,:]
            if self.SHOW_FINE_TIMING:
                t_init = time.time()
            self.range_method.calc_range_repeat_angles(self.queries, self.downsampled_angles, self.ranges)
            if self.SHOW_FINE_TIMING:
                t_range = time.time()
            # evaluate the sensor model on the GPU
            self.range_method.eval_sensor_model(obs, self.ranges, self.weights, num_rays, self.MAX_PARTICLES)
            if self.SHOW_FINE_TIMING:
                t_eval = time.time()
            np.power(self.weights, self.INV_SQUASH_FACTOR, self.weights)
            if self.SHOW_FINE_TIMING:
                t_squash = time.time()
                t_total = (t_squash - t_start) / 100.0

            if self.SHOW_FINE_TIMING and self.iters % 10 == 0:
                self.get_logger().info(str(['sensor_model: init: ', np.round((t_init-t_start)/t_total, 2), 'range:', np.round((t_range-t_init)/t_total, 2), \
                      'eval:', np.round((t_eval-t_range)/t_total, 2), 'squash:', np.round((t_squash-t_eval)/t_total, 2)]))
        elif self.RANGELIB_VAR == VAR_CALC_RANGE_MANY_EVAL_SENSOR:
            # this version demonstrates what this would look like with coordinate space conversion pushed to rangelib
            # this part is inefficient since it requires a lot of effort to construct this redundant array
            self.queries[:,0] = np.repeat(proposal_dist[:,0], num_rays)
            self.queries[:,1] = np.repeat(proposal_dist[:,1], num_rays)
            self.queries[:,2] = np.repeat(proposal_dist[:,2], num_rays)
            self.queries[:,2] += self.tiled_angles

            self.range_method.calc_range_many(self.queries, self.ranges)

            # evaluate the sensor model on the GPU
            self.range_method.eval_sensor_model(obs, self.ranges, self.weights, num_rays, self.MAX_PARTICLES)
            np.power(self.weights, self.INV_SQUASH_FACTOR, self.weights)
        elif self.RANGELIB_VAR == VAR_NO_EVAL_SENSOR_MODEL:
            # this version directly uses the sensor model in Python, at a significant computational cost
            self.queries[:,0] = np.repeat(proposal_dist[:,0], num_rays)
            self.queries[:,1] = np.repeat(proposal_dist[:,1], num_rays)
            self.queries[:,2] = np.repeat(proposal_dist[:,2], num_rays)
            self.queries[:,2] += self.tiled_angles

            # compute the ranges for all the particles in a single functon call
            self.range_method.calc_range_many(self.queries, self.ranges)

            # resolve the sensor model by discretizing and indexing into the precomputed table
            obs /= float(self.map_info.resolution)
            ranges = self.ranges / float(self.map_info.resolution)
            obs[obs > self.MAX_RANGE_PX] = self.MAX_RANGE_PX
            ranges[ranges > self.MAX_RANGE_PX] = self.MAX_RANGE_PX

            intobs = np.rint(obs).astype(np.uint16)
            intrng = np.rint(ranges).astype(np.uint16)

            # compute the weight for each particle
            for i in range(self.MAX_PARTICLES):
                weight = np.product(self.sensor_model_table[intobs,intrng[i*num_rays:(i+1)*num_rays]])
                weight = np.power(weight, self.INV_SQUASH_FACTOR)
                weights[i] = weight
        else:
            self.get_logger().info('PLEASE SET rangelib_variant PARAM to 0-4')

    def MCL(self, a, o):
        '''
        Performs one step of Monte Carlo Localization.
            1. resample particle distribution to form the proposal distribution
            2. apply the motion model
            3. apply the sensor model
            4. normalize particle weights

        This is in the critical path of code execution, so it is optimized for speed.
        '''
        if self.SHOW_FINE_TIMING:
            t = time.time()
        # draw the proposal distribution from the old particles
        proposal_indices = np.random.choice(self.particle_indices, self.MAX_PARTICLES, p=self.weights)
        proposal_distribution = self.particles[proposal_indices,:]
        if self.SHOW_FINE_TIMING:
            t_propose = time.time()

        # compute the motion model to update the proposal distribution
        self.motion_model(proposal_distribution, a)
        if self.SHOW_FINE_TIMING:
            t_motion = time.time()

        # compute the sensor model
        self.sensor_model(proposal_distribution, o, self.weights)
        if self.SHOW_FINE_TIMING:
            t_sensor = time.time()

        # normalize importance weights
        self.weights /= np.sum(self.weights)
        if self.SHOW_FINE_TIMING:
            t_norm = time.time()
            t_total = (t_norm - t)/100.0

        if self.SHOW_FINE_TIMING and self.iters % 10 == 0:
            self.get_logger().info(str(['MCL: propose: ', np.round((t_propose-t)/t_total, 2), 'motion:', np.round((t_motion-t_propose)/t_total, 2), \
                  'sensor:', np.round((t_sensor-t_motion)/t_total, 2), 'norm:', np.round((t_norm-t_sensor)/t_total, 2)]))

        # save the particles
        self.particles = proposal_distribution
    
    def expected_pose(self):
        # returns the expected value of the pose given the particle distribution
        return np.dot(self.particles.transpose(), self.weights)

    def update(self):
        '''
        Apply the MCL function to update particle filter state. 

        Ensures the state is correctly initialized, and acquires the state lock before proceeding.
        '''
        if self.lidar_initialized and self.odom_initialized and self.map_initialized:
            if self.state_lock.locked():
                self.get_logger().info('Concurrency error avoided')
            else:
                self.state_lock.acquire()
                self.timer.tick()
                self.iters += 1

                t1 = time.time()
                observation = np.copy(self.downsampled_ranges).astype(np.float32)
                action = np.copy(self.odometry_data)
                self.odometry_data = np.zeros(3)

                # run the MCL update algorithm
                self.MCL(action, observation)

                # compute the expected value of the robot pose
                self.inferred_pose = self.expected_pose()
                self.state_lock.release()
                t2 = time.time()

                # publish transformation frame based on inferred pose
                self.publish_tf(self.inferred_pose, self.last_stamp)

                # this is for tracking particle filter speed
                ips = 1.0 / (t2 - t1)
                self.smoothing.append(ips)
                if self.iters % 10 == 0:
                    self.get_logger().info(str(['iters per sec:', int(self.timer.fps()), ' possible:', int(self.smoothing.mean())]))

                self.visualize()

# import argparse
# import sys
# parser = argparse.ArgumentParser(description='Particle filter.')
# parser.add_argument('--config', help='Path to yaml file containing config parameters. Helpful for calling node directly with Python for profiling.')

# def load_params_from_yaml(fp):
#     from yaml import load
#     with open(fp, 'r') as infile:
#         yaml_data = load(infile)
#         for param in yaml_data:
#             print 'param:', param, ':', yaml_data[param]
#             rospy.set_param('~'+param, yaml_data[param])

# # this function can be used to generate flame graphs easily
# def make_flamegraph(filterx=None):
#     import flamegraph, os
#     perf_log_path = os.path.join(os.path.dirname(__file__), '../tmp/perf.log')
#     flamegraph.start_profile_thread(fd=open(perf_log_path, 'w'),
#                                     filter=filterx,
#                                     interval=0.001)

def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFiler()
    rclpy.spin(pf)

if __name__ == '__main__':
    main()

# if __name__=='__main__':
#     rospy.init_node('particle_filter')

#     args,_ = parser.parse_known_args()
#     if args.config:
#         load_params_from_yaml(args.config)

#     # make_flamegraph(r'update')

#     pf = ParticleFiler()
#     rospy.spin()
