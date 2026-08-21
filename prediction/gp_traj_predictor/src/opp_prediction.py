#!/usr/bin/env python3

import rclpy
import numpy as np
from prediction_node import PredictionNode
from std_msgs.msg import Bool, Float32, String, Header
from nav_msgs.msg import Odometry
from f110_msgs.msg import ObstacleArray, WpntArray, Wpnt, Obstacle, ObstacleArray, OpponentTrajectory, OppWpnt, Prediction, PredictionArray
from visualization_msgs.msg import Marker, MarkerArray
from frenet_conversion_msgs.srv import Glob2FrenetArr, Frenet2GlobArr
import time
import copy

from std_srvs.srv import SetBool
from rcl_interfaces.msg import SetParametersResult
from tf_transformations import euler_from_quaternion

class OppTrajPredictor(PredictionNode):
    def __init__(self):
        super().__init__('opponent_propagation_predictor')

        # ROS Parameters
        self.opponent_traj_topic = '/opponent_trajectory'
        self.glob2frenet_client = self.create_client(Glob2FrenetArr, 'convert_glob2frenetarr_service')
        self.frenet2glob_client = self.create_client(Frenet2GlobArr, 'convert_frenet2globarr_service')
        self.loop_rate = 10 #Hz
        
        # Publisher
        self.marker_pub_beginn = self.create_publisher(Marker, '/opponent_predict/beginn', 10)
        self.marker_pub_end = self.create_publisher(Marker, '/opponent_predict/end', 10)
        self.prediction_obs_pub = self.create_publisher(ObstacleArray, '/opponent_prediction/obstacles', 10)
        self.prediction_obs_pred_pub = self.create_publisher(PredictionArray, '/opponent_prediction/obstacles_pred', 10)
        self.force_trailing_pub = self.create_publisher(Bool, '/opponent_prediction/force_trailing', 10)
        
        self.opp_traj_gp_pub = self.create_publisher(OpponentTrajectory, '/opponent_trajectory', 10)
        self.opp_traj_marker_pub = self.create_publisher(MarkerArray, '/opponent_traj_markerarray', 10)
        self.opp_marker_pub = self.create_publisher(MarkerArray, '/opponent_prediction_markerarray', 10)

        # Subscriber
        self.create_subscription(ObstacleArray, '/tracking/obstacles', self.opponent_state_cb, 10)
        self.create_subscription(Odometry, '/car_state/odom_frenet', self.odom_cb, 10)
        self.create_subscription(OpponentTrajectory, self.opponent_traj_topic, self.opponent_trajectory_cb, 10)
        self.create_subscription(WpntArray, '/global_waypoints_updated', self.wpnts_updated_cb, 10)
        self.create_subscription(WpntArray, '/centerline_waypoints', self.center_wpnts_cb, 10)
        self.create_subscription(String, '/state_machine', self.state_cb, 10)
        # Service server
        self.create_service(SetBool, '/init_opp_trajectory', self._init_opp_bool_callback)

        # Callback data
        self.opponent_pos = ObstacleArray()
        self.car_odom = Odometry()
        self.wpnts_opponent = list()
        self.wpnts_updated = list()
        self.state = String()

        self.speed_offset = 0 # m/s

        # Simulation parameters
        self.time_steps = 200
        self.dt = 0.02 # s
        self.save_distance_front = 6.0 # m
        self.save_distance_back = 0.4 # m
        self.max_v = 10 # m/s
        self.min_v = 0 # m/s
        self.max_a = 5.5 # m/s^2
        self.min_a = 5 # m/s^2
        self.max_expire_counter = 10

        # --- learned-trajectory authorization (force_trailing) ---
        # The old test was a bare, memoryless comparison:
        #     abs(opp_d - raceline_d) > 0.25 or lap_count < 1
        # One threshold, evaluated fresh every 10 Hz frame, so an opponent
        # hovering near 0.25 m off the learned line toggled force_trailing on and
        # off frame by frame -- and each toggle flipped the planner between
        # "publish an evasion path" and "publish nothing", which the state machine
        # sees as the avoidance path appearing and vanishing.
        #
        # Replaced with a two-threshold, frame-confirmed hysteresis. Same purpose,
        # same meaning of the output: True == "the prediction is only a
        # constant-velocity fallback, do not authorize a new overtake".
        self.min_training_laps = 0.5
        self.learned_deviation_enter_threshold = 0.35   # [m] enter learned (authorize)
        self.learned_deviation_exit_threshold = 0.55    # [m] leave learned (veto)
        self.learned_ready_confirm_frames = 3
        self.learned_reject_confirm_frames = 5
        # Start vetoing: a predictor that has just come up has not seen the
        # opponent run a lap yet, so it cannot have a learned trajectory.
        self.force_trailing = True
        self._learned_ready_frames = 0
        self._learned_reject_frames = 0

        # ROS 2 parameters replace ROS 1 dynamic_reconfigure.  They can be
        # changed at runtime with `ros2 param set /opponent_propagation_predictor ...`.
        for name, default in {
                'n_time_steps': self.time_steps,
                'dt': self.dt,
                'save_distance_front': self.save_distance_front,
                'max_expire_counter': self.max_expire_counter,
                'speed_offset': self.speed_offset,
                # --- learned-trajectory authorization hysteresis (see _update_learned_authorization) ---
                'min_training_laps': self.min_training_laps,
                'learned_deviation_enter_threshold': self.learned_deviation_enter_threshold,
                'learned_deviation_exit_threshold': self.learned_deviation_exit_threshold,
                'learned_ready_confirm_frames': self.learned_ready_confirm_frames,
                'learned_reject_confirm_frames': self.learned_reject_confirm_frames}.items():
            self.declare_parameter(name, default)
        self.dyn_param_cb(None)
        self.add_on_set_parameters_callback(self._on_parameter_update)

        # Number of time steps before prediction expires. Set when prediction is published.
        self.expire_counter = 0

        # Visualization
        self.marker_beginn = self.marker_init(a = 0.5, r = 0.63, g = 0.13, b = 0.94, id = 0)
        self.marker_end = self.marker_init(a = 0.5, r = 0.63, g = 0.13, b = 0.94, id = 1)

        # Opponent
        self.opponent_lap_count = None
        
        # Stanley params
        self.k = 0.5  # control gain
        self.Kp = 1.0  # speed proportional gain
        # self.dt = 0.1  # [s] time difference
        self.L = 0.33  # [m] Wheel base of vehicle
        self.max_steer = np.radians(30.0)  # [rad] max steering angle
        
        self.init_fixed_wpnts = False
        self.fixed_wpnts = []

    # Service function
    def init_opp_bool(self, req):
        if req.data:
            self.get_logger().info('Received request: ON')
            success = True
            message = "Feature turned ON"
            self.global_to_opptraj(self.wpnts_updated)
        else:
            self.get_logger().info('Received request: OFF')
            success = True
            message = "Feature turned OFF"
        response = SetBool.Response()
        response.success = success
        response.message = message
        return response

    def _init_opp_bool_callback(self, request, _response):
        return self.init_opp_bool(request)

    def frenet2glob(self, s, d):
        request = Frenet2GlobArr.Request()
        request.s, request.d = s, d
        while rclpy.ok() and not self.frenet2glob_client.wait_for_service(timeout_sec=0.25):
            self.get_logger().info('Waiting for convert_frenet2globarr_service...')
        future = self.frenet2glob_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()
    
    def global_to_opptraj(self, wptlist: list):
        wpnts_opponent = [OppWpnt(s_m=wp.s_m, d_m=wp.d_m, x_m=wp.x_m, y_m=wp.y_m, proj_vs_mps=wp.vx_mps) for wp in wptlist]
        
        opp_traj_gp_msg = self.make_opponent_trajectory_msg(wpnts_opponent)
        opp_traj_marker_array = self.visualize_opponent_wpnts(wpnts_opponent)
        
        self.opp_traj_gp_pub.publish(opp_traj_gp_msg)
        self.opp_traj_marker_pub.publish(opp_traj_marker_array)
        
    def make_opponent_trajectory_msg(self, oppwpnts_list: list):
        opponent_trajectory_msg = OpponentTrajectory()
        opponent_trajectory_msg.header.stamp = self.get_clock().now().to_msg()
        opponent_trajectory_msg.oppwpnts = oppwpnts_list
        return opponent_trajectory_msg
        
    def visualize_opponent_wpnts(self, oppwpnts_list: list):
        opp_traj_marker_array = MarkerArray()
    
        i=0
        for i in range(len(oppwpnts_list)):
            marker_height = oppwpnts_list[i].proj_vs_mps/10.0

            marker = Marker(header=Header(frame_id='map'), id=i, type=Marker.CYLINDER)
            marker.pose.position.x = oppwpnts_list[i].x_m
            marker.pose.position.y = oppwpnts_list[i].y_m
            marker.pose.position.z = marker_height/2
            marker.pose.orientation.w = 1.0
            marker.scale.x = min(max(5 * oppwpnts_list[i].d_var, 0.07),0.7)
            marker.scale.y = min(max(5 * oppwpnts_list[i].d_var, 0.07),0.7)
            marker.scale.z = marker_height
            if oppwpnts_list[i].vs_var == 69:
                marker.color.a = 0.8
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
            else:
                marker.color.a = 1.0
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            opp_traj_marker_array.markers.append(marker)
        return opp_traj_marker_array

    
    ### CALLBACKS ###
    # TODO: Only sees the first dynamic obstacle as opponent...
    def opponent_state_cb(self, data: ObstacleArray):
        self.opponent_pos.header = data.header
        is_dynamic = False
        if len(data.obstacles) > 0:
            for obs in data.obstacles:
                if obs.is_static == False: # and obs.is_opponent == True: # Changed by Tino and Nicolas for compatibility with new obstacle message
                    self.opponent_pos.obstacles = [obs]
                    is_dynamic = True
                    break
        if is_dynamic == False:
            self.opponent_pos.obstacles = []

    def odom_cb(self, data: Odometry): self.car_odom = data

    def opponent_trajectory_cb(self, data: OpponentTrajectory):
        self.wpnts_opponent = data.oppwpnts  # exclude last point (because last point == first point) <- Hopefully this is still the case?
        self.max_s_opponent = self.wpnts_opponent[-1].s_m
        self.opponent_lap_count = data.lap_count

    def wpnts_updated_cb(self, data: WpntArray):
        self.wpnts_updated = data.wpnts[:-1]
        self.max_s_updated = self.wpnts_updated[-1].s_m # Should be the same as self.max_s but just in case. Only used for wrap around

    def center_wpnts_cb(self, data: WpntArray):
        self.center_wpnts_msg = data
        self.center_wpnts_max_s = data.wpnts[-1].s_m
        self.center_wpnts_max_idx = data.wpnts[-1].id
        
        if not self.init_fixed_wpnts:
            self.fixed_wpnts = [[wp.x_m, wp.y_m] for wp in data.wpnts]
            self.cx, self.cy = zip(*self.fixed_wpnts)
            self.cyaw = [wp.psi_rad for wp in data.wpnts]
            self.init_fixed_wpnts = True

    def state_cb(self, data: String):
        self.state = data.data

    def _build_opponent_markers(self, obstacle_list):
        """RViz cylinders for the predicted opponent poses -- ONE service call.

        This used to be built inside the n_time_steps loop, and every iteration
        called frenet2glob() for a single point. frenet2glob() is a *synchronous*
        service round-trip (call_async + spin_until_future_complete), so a
        200-step prediction paid 200 blocking round-trips per cycle, purely for
        visualisation. Measured on the car (state_machine_20260822_044714.log):
        pred_age sawtoothed 0.1 -> 1.5 s with a median of 1.00 s, i.e. the
        predictor emitted at well under 1 Hz against its nominal 10 Hz, and the
        lane_change planner -- whose pred_timeout is 0.5 s -- therefore refused
        to plan on 76 % of the samples where a prediction existed at all.

        Two changes, neither of which touches what is predicted:
          * Frenet2GlobArr is an ARRAY service. Convert all the poses in one
            call instead of one call per pose.
          * Skip the work entirely when nothing is subscribed to the marker
            topic, which is the normal case during a race.
        """
        if self.opp_marker_pub.get_subscription_count() == 0 or not obstacle_list:
            return MarkerArray()

        s_list = [obs.s_center % self.max_s_updated for obs in obstacle_list]
        d_list = [obs.d_center for obs in obstacle_list]
        pos = self.frenet2glob(s_list, d_list)
        if pos is None or len(pos.x) != len(obstacle_list):
            return MarkerArray()

        stamp = self.get_clock().now().to_msg()
        markers = MarkerArray()
        for i, obs in enumerate(obstacle_list):
            marker = Marker()
            marker.header.stamp = stamp
            marker.header.frame_id = "map"
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = pos.x[i]
            marker.pose.position.y = pos.y[i]
            marker.pose.position.z = 0.1
            marker.scale.x = 0.15
            marker.scale.y = 0.15
            marker.scale.z = 0.15  # height
            marker.color.a = 0.8
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            markers.markers.append(marker)
        return markers

    def _update_learned_authorization(self, deviation_m):
        """Decide whether the opponent is on its learned trajectory, with hysteresis.

        `deviation_m` is |opponent_d - learned_trajectory_d| at the opponent's s.

        Returns True when the prediction must be treated as a constant-velocity
        fallback (i.e. publish force_trailing=True and do not authorize a new
        overtake), False when the learned GP trajectory is trusted.

        Two thresholds and two frame counters, so the answer cannot chatter:

          authorize (force_trailing -> False)
              training >= min_training_laps AND deviation <= enter_threshold,
              held for learned_ready_confirm_frames consecutive frames
          veto (force_trailing -> True)
              deviation >= exit_threshold,
              held for learned_reject_confirm_frames consecutive frames

        Between the two thresholds nothing changes -- whichever state is current
        is held. That band (0.35 to 0.55 m) is where the old single 0.25 m test
        did all of its flapping.

        Not enough training laps is an immediate, unconfirmed veto: there is no
        learned trajectory to be near, so waiting frames to say so would be
        pretending to a confidence we do not have.
        """
        laps = self.opponent_lap_count
        # opponent_lap_count is None until the first /opponent_trajectory message.
        # The old expression evaluated `None < 1` on any frame where the deviation
        # test short-circuited to False -- a TypeError that killed the loop.
        if laps is None or float(laps) < float(self.min_training_laps):
            self.force_trailing = True
            self._learned_ready_frames = 0
            self._learned_reject_frames = 0
            return self.force_trailing

        if deviation_m <= float(self.learned_deviation_enter_threshold):
            self._learned_ready_frames += 1
            self._learned_reject_frames = 0
            if self._learned_ready_frames >= int(self.learned_ready_confirm_frames):
                self.force_trailing = False
        elif deviation_m >= float(self.learned_deviation_exit_threshold):
            self._learned_reject_frames += 1
            self._learned_ready_frames = 0
            if self._learned_reject_frames >= int(self.learned_reject_confirm_frames):
                self.force_trailing = True
        else:
            # Inside the hysteresis band: hold the current decision, and reset both
            # counters so a run that drifts in and out of the band does not
            # accumulate credit toward a flip it never actually earned.
            self._learned_ready_frames = 0
            self._learned_reject_frames = 0

        return self.force_trailing

        # Callback triggered by dynamic spline reconf
    def dyn_param_cb(self, _params):
        """
        Notices the change in the parameters and changes spline params
        """
        self.time_steps = self.get_parameter('n_time_steps').value
        self.dt = self.get_parameter('dt').value
        self.save_distance_front = self.get_parameter('save_distance_front').value
        self.max_expire_counter = self.get_parameter('max_expire_counter').value
        self.speed_offset = self.get_parameter('speed_offset').value
        self.min_training_laps = self.get_parameter('min_training_laps').value
        self.learned_deviation_enter_threshold = self.get_parameter(
            'learned_deviation_enter_threshold').value
        self.learned_deviation_exit_threshold = self.get_parameter(
            'learned_deviation_exit_threshold').value
        self.learned_ready_confirm_frames = self.get_parameter(
            'learned_ready_confirm_frames').value
        self.learned_reject_confirm_frames = self.get_parameter(
            'learned_reject_confirm_frames').value

        print(
            f"[Opp. Pred.] Dynamic reconf triggered new params:\n"
            f" N time stepts: {self.time_steps}, \n"
            f" dt: {self.dt} [s], \n"
            f" save_distance_front: {self.save_distance_front} [m], \n"
            f" max_expire_counter: {self.max_expire_counter}"
        )

    def _on_parameter_update(self, parameters):
        for parameter in parameters:
            if parameter.name == 'n_time_steps':
                self.time_steps = parameter.value
            elif parameter.name == 'dt':
                self.dt = parameter.value
            elif parameter.name == 'save_distance_front':
                self.save_distance_front = parameter.value
            elif parameter.name == 'max_expire_counter':
                self.max_expire_counter = parameter.value
            elif parameter.name == 'speed_offset':
                self.speed_offset = parameter.value
            elif parameter.name == 'min_training_laps':
                self.min_training_laps = parameter.value
            elif parameter.name == 'learned_deviation_enter_threshold':
                self.learned_deviation_enter_threshold = parameter.value
            elif parameter.name == 'learned_deviation_exit_threshold':
                self.learned_deviation_exit_threshold = parameter.value
            elif parameter.name == 'learned_ready_confirm_frames':
                self.learned_ready_confirm_frames = parameter.value
            elif parameter.name == 'learned_reject_confirm_frames':
                self.learned_reject_confirm_frames = parameter.value
        return SetParametersResult(successful=True)


    ### HELPER FUNCTIONS ###
    def marker_init(self, a = 1, r = 1, g = 0, b = 0, id = 0):
        marker = Marker(header=Header(stamp=self.get_clock().now().to_msg(), frame_id='map'), id=id, type=Marker.SPHERE)
        # Set the pose of the marker.  This is a full 6DOF pose relative to the frame/time specified in the header
        marker.pose.orientation.w = 1.0
        # Set the marker's scale
        marker.scale.x = 0.4
        marker.scale.y = 0.4
        marker.scale.z = 0.4
        # Set the marker's color
        marker.color.a = a  # Alpha (transparency)
        marker.color.r = r  # Red
        marker.color.g = g  # Green
        marker.color.b = b  # Blue
        return marker

    def publish_begin_end_markers(self, beginn_s, beginn_d, end_s, end_d):
        # Visualize the prediction endpoints. Watch out for wrap around.
        #
        # Two synchronous frenet2glob round-trips, on the prediction hot path,
        # for two RViz spheres. Skipped when nobody is looking, and batched into
        # one call when they are -- same reasoning as _build_opponent_markers.
        if (self.marker_pub_beginn.get_subscription_count() == 0
                and self.marker_pub_end.get_subscription_count() == 0):
            return

        stamp = self.get_clock().now().to_msg()
        pos = self.frenet2glob(
            [beginn_s % self.max_s_updated, end_s % self.max_s_updated],
            [beginn_d, end_d])
        if pos is None or len(pos.x) != 2:
            return

        self.marker_beginn.header.stamp = stamp
        self.marker_beginn.action = Marker.ADD
        self.marker_beginn.pose.position.x = pos.x[0]
        self.marker_beginn.pose.position.y = pos.y[0]
        self.marker_pub_beginn.publish(self.marker_beginn)

        self.marker_end.header.stamp = stamp
        self.marker_end.action = Marker.ADD
        self.marker_end.pose.position.x = pos.x[1]
        self.marker_end.pose.position.y = pos.y[1]
        self.marker_pub_end.publish(self.marker_end)

    def delete_all(self) -> None:
        empty_marker = Marker(header=Header(stamp=self.get_clock().now().to_msg(), frame_id='map'), id=0)
        empty_marker.action = Marker.DELETE
        self.marker_pub_beginn.publish(empty_marker)
        empty_marker.id = 1
        self.marker_pub_end.publish(empty_marker)

        empty_obs_arr = ObstacleArray(header=Header(stamp=self.get_clock().now().to_msg(), frame_id='map'))
        self.prediction_obs_pub.publish(empty_obs_arr)

    ### MAIN LOOP ###
    def loop(self):
        self.get_logger().info('[Opp. Pred.] Opponent Predictor waiting...')
        self.wait_for_message('/global_waypoints', WpntArray)
        self.wait_for_message('/global_waypoints_scaled', WpntArray)
        self.wpnts_updated_cb(self.wait_for_message('/global_waypoints_updated', WpntArray))
        self.get_logger().info('[Opp. Pred.] Updated waypoints received!')
        self.opponent_trajectory_cb(self.wait_for_message(self.opponent_traj_topic, OpponentTrajectory))
        self.get_logger().info('[Opp. Pred.] Opponent waypoints received!')
        self.opponent_state_cb(self.wait_for_message('/tracking/obstacles', ObstacleArray))
        self.get_logger().info('[Opp. Pred.] Obstacles received!')
        self.get_logger().info('[Opp. Pred.] Opponent Predictor ready!')

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.0)

            opponent_pos_copy = copy.deepcopy(self.opponent_pos)

            prediction_obs_pred_arr = PredictionArray()


            if len(opponent_pos_copy.obstacles) != 0:
                current_ego_s = self.car_odom.pose.pose.position.x
                current_opponent_s = opponent_pos_copy.obstacles[0].s_center
                
                # Handle wrap around
                if current_ego_s > self.max_s_updated * (2/3) and current_opponent_s < self.max_s_updated * (1/3):
                    current_opponent_s += self.max_s_updated
                    
                current_opponent_d = opponent_pos_copy.obstacles[0].d_center
                
                s_points_center_array = np.array([wpnt.s_m for wpnt in self.center_wpnts_msg.wpnts])
                approx_opponent_center_indx = np.abs(s_points_center_array - current_opponent_s).argmin()
                opponent_approx_center_d = self.center_wpnts_msg.wpnts[approx_opponent_center_indx].d_m
                
                current_opponent_v = opponent_pos_copy.obstacles[0].vs
                
                approx_s_points_global_array = np.array([wpnt.s_m for wpnt in self.wpnts_opponent])
                opponent_approx_indx = np.abs(approx_s_points_global_array - current_opponent_s).argmin()
                opponent_approx_raceline_d = self.wpnts_opponent[opponent_approx_indx].d_m
                
                start = time.process_time()

                # Hysteresis, not a bare threshold: see _update_learned_authorization.
                # The published value keeps its original meaning -- True means the
                # prediction below is the constant-velocity fallback.
                learned_deviation = abs(current_opponent_d - opponent_approx_raceline_d)
                if self._update_learned_authorization(learned_deviation):
                    self.force_trailing_pub.publish(Bool(data=True))
                    
                    obstacle_list = []
                    prediction_list = []

                    opp_marker_array = MarkerArray()
                    
                    for i in range(self.time_steps):
                        w = i / (self.time_steps - 1)
                        
                        interpolated_d = (1 - w) * current_opponent_d + w * opponent_approx_center_d
                        
                        obs = Obstacle()
                        obs.id = i
                        obs.s_start = current_opponent_s
                        obs.s_end = current_opponent_s
                        obs.s_center = current_opponent_s + i * current_opponent_v * self.dt
                        obs.d_center = interpolated_d
                        obs.d_left = obs.d_center + 0.25
                        obs.d_right = obs.d_center - 0.25
                        obs.size = opponent_pos_copy.obstacles[0].size
                        obs.vs = current_opponent_v
                        obs.vd = 0
                        obs.is_actually_a_gap = False
                        obs.is_static = False
                        obstacle_list.append(obs)

                        pds = Prediction()
                        pds.id = i
                        pds.pred_s = obs.s_center
                        pds.pred_d = obs.d_center
                        prediction_list.append(pds)

                        
                    opp_marker_array = self._build_opponent_markers(obstacle_list)

                    prediction_obs_arr = ObstacleArray(header=Header(stamp=self.get_clock().now().to_msg(), frame_id='map'), obstacles=obstacle_list)
                    self.prediction_obs_pub.publish(prediction_obs_arr)

                    prediction_obs_pred_arr = PredictionArray(header=Header(stamp=self.get_clock().now().to_msg(), frame_id='map'), id=opponent_pos_copy.obstacles[0].id, dt=self.dt, predictions=prediction_list)
                    self.prediction_obs_pred_pub.publish(prediction_obs_pred_arr)

                    self.opp_marker_pub.publish(opp_marker_array)
                    if obstacle_list:
                        self.publish_begin_end_markers(
                            current_opponent_s,
                            current_opponent_d,
                            obstacle_list[-1].s_center,
                            obstacle_list[-1].d_center
                        )
                    
                    self.expire_counter = 0
                else:
                    self.force_trailing_pub.publish(Bool(data=False))
                    
                    # Temp params
                    beginn = False
                    end = False
                    beginn_s = 0
                    end_s = 0
                    beginn_d = 0
                    end_d = 0
                    obstacle_list = []
                    prediction_list = []
                    
                    # Find begin of the prediction
                    if (beginn == False and ((current_opponent_s - current_ego_s)%self.max_s_updated < self.save_distance_front or abs(current_opponent_s - current_ego_s) < self.save_distance_front)):
                        beginn_s = current_opponent_s
                        beginn_d = current_opponent_d
                        beginn = True
                        
                    opp_marker_array = MarkerArray()
                    
                    for i in range(self.time_steps):
                        # Get the speed at position i + 1
                        opponent_approx_indx = np.abs(approx_s_points_global_array - current_opponent_s % self.max_s_opponent).argmin()
                        opponent_speed = self.wpnts_opponent[opponent_approx_indx].proj_vs_mps
                        current_opponent_s = (current_opponent_s + opponent_speed * self.dt)
                        opponent_d = self.wpnts_opponent[opponent_approx_indx].d_m

                        obs = Obstacle()
                        obs.id = i
                        obs.s_start = current_opponent_s
                        obs.s_end = current_opponent_s + opponent_speed * self.dt
                        obs.s_center = (obs.s_start + obs.s_end) / 2
                        obs.d_center = opponent_d
                        obs.d_left = opponent_d + 0.25
                        obs.d_right = opponent_d - 0.25
                        obs.size = opponent_pos_copy.obstacles[0].size
                        obs.vs = opponent_speed
                        obs.vd = 0
                        obs.is_actually_a_gap = False
                        obs.is_static = False
                        obstacle_list.append(obs)

                        pds = Prediction()
                        pds.id = i
                        pds.pred_s = (obs.s_start + obs.s_end) / 2
                        pds.pred_d = opponent_d
                        prediction_list.append(pds)


                        
                    opp_marker_array = self._build_opponent_markers(obstacle_list)

                    # Find the end of the prediction
                    if (beginn == True and end == False):
                        end_s = current_opponent_s
                        end_d = opponent_d
                        end = True
                        
                        prediction_obs_arr = ObstacleArray(header=Header(stamp=self.get_clock().now().to_msg(), frame_id='map'), obstacles=obstacle_list)
                        self.prediction_obs_pub.publish(prediction_obs_arr)

                        prediction_obs_pred_arr = PredictionArray(header=Header(stamp=self.get_clock().now().to_msg(), frame_id='map'), id=opponent_pos_copy.obstacles[0].id, dt=self.dt, predictions=prediction_list)
                        self.prediction_obs_pred_pub.publish(prediction_obs_pred_arr)

                        self.opp_marker_pub.publish(opp_marker_array)
                        
                        self.expire_counter = 0
                        
                        self.publish_begin_end_markers(beginn_s, beginn_d, end_s, end_d)

            self.prediction_obs_pred_pub.publish(prediction_obs_pred_arr)

            self.expire_counter += 1
            if self.expire_counter >= self.max_expire_counter:
                self.expire_counter = self.max_expire_counter
                self.delete_all()

            # print("Time: {}".format(time.process_time() - start))

            time.sleep(1.0 / self.loop_rate)

if __name__ == '__main__':
    rclpy.init()
    node = OppTrajPredictor()
    try:
        node.loop()
    finally:
        node.destroy_node()
        rclpy.shutdown()
