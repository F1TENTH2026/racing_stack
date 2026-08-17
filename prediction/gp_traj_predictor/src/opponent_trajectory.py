#!/usr/bin/env python3

import rclpy
from rclpy.executors import MultiThreadedExecutor
import numpy as np
from prediction_node import PredictionNode
from f110_msgs.msg import ObstacleArray, OpponentTrajectory, Obstacle, OppWpnt, WpntArray,ProjOppTraj, ProjOppPoint
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
import time
from std_msgs.msg import String
from frenet_conversion.frenet_converter import FrenetConverter


class Opponent_Trajectory(PredictionNode):
    def __init__(self):
        super().__init__('opponent_trajectory')

        self.opponent = [0,0,0,0,False,True,0] #s, d, vs, vd, is_static, is_visible, timestamp
        self.position_in_map_frenet = [] # current position in frenet coordinates
        self.opponent_trajectory_list = [] # add all the points where the opponent was to this list
        self.opponent_as_obstacle = Obstacle()
        self.opponent_trajectory_list_of_obstacles = []
        self.oppwpnts = OppWpnt()
        self.oppwpnts_list = []
        self.converter = None
        self.opponent_positions = []
        self.track_length = 70 #initialize track_length with a feasible value
        self.opp_traj_count = 0
        self.overtake = False
        self.obstacle_array = ObstacleArray()

        

        # Time
        self.ros_time = self.get_clock().now()

        # Publisher
        self.proj_opponent_trajectory_pub = self.create_publisher(ProjOppTraj, '/proj_opponent_trajectory', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/opponent_marker', 10)
        # Subscriber
        self.create_subscription(ObstacleArray, '/tracking/obstacles', self.obstacle_cb, 10)
        self.create_subscription(Odometry, '/car_state/odom_frenet', self.car_state_frenet_cb, 10)
        self.create_subscription(WpntArray, '/global_waypoints', self.glb_wpnts_cb, 10)
        self.create_subscription(OpponentTrajectory, '/opponent_trajectory', self.opp_traj_cb, 10)
        #subscribe to state machine
        self.create_subscription(String, '/state_machine', self.state_machine_cb, 10)

        # Retrieve ROS parameters
        self.declare_parameter('loop_rate', 25)
        self.loop_rate = self.get_parameter('loop_rate').value
        self.declare_parameter('max_opponent_distance', 30.0)
        self.max_opponent_distance = self.get_parameter('max_opponent_distance').value

        self.glb_wpnts_cb(self.wait_for_message('/global_waypoints', WpntArray))
        self.converter = self.initialize_converter()

    #callbacks
    def state_machine_cb(self, data: String):
        if data.data == "OVERTAKE":
            self.overtake = True
        else:
            self.overtake = False

    def obstacle_cb(self, data: ObstacleArray):
        self.obstacle_array = data
        
    def car_state_frenet_cb(self, data: Odometry):
        s = data.pose.pose.position.x
        d = data.pose.pose.position.y
        vs = data.twist.twist.linear.x
        vd = data.twist.twist.linear.y
        self.position_in_map_frenet = np.array([s,d,vs,vd]) 
    
    def glb_wpnts_cb(self, data):
        self.waypoints = np.array([[wpnt.x_m, wpnt.y_m] for wpnt in data.wpnts])
        self.glb_wpnts = data
        self.track_length = data.wpnts[-1].s_m
    
    def opp_traj_cb(self, data: OpponentTrajectory):
        if not self.overtake:
            self.opponent_trajectory = data
        self.opp_traj_count += 1

    def initialize_converter(self) -> bool:
        """
        Initialize the FrenetConverter object"""
        # Initialize the FrenetConverter object
        converter = FrenetConverter(self.waypoints[:, 0], self.waypoints[:, 1])
        self.get_logger().info('[Tracking] initialized FrenetConverter object')

        return converter    

    
    #Main Loop
    def make_opponent_trajectory(self):

        """One-time setup, then hand control to a loop_rate-Hz timer serviced by a
        multithreaded executor. Replaces the old create_rate()+while-loop: that
        combo ran a second (busy-ish) wait loop in the main thread on top of the
        executor thread already spinning this node's subscriptions -- the timer
        lets the SAME executor drive both, like every other node in this repo
        (see recovery_spliner_node's create_timer()). State that used to live as
        locals inside the while-loop now lives on self so it survives between
        separate loop() calls -- see the block below for what moved."""

        self.lap_count = 0
        self.first_point = True
        # The constructor already waits for and caches the first global path.
        # Do not wait again: the ROS 2 waypoint publisher may publish it only once.
        self.global_wpnts = self.glb_wpnts
        self.ego_s_sorted = [wnpt.s_m for wnpt in self.global_wpnts.wpnts]
        self.bound_right = [wnpt.d_right for wnpt in self.global_wpnts.wpnts]#all positive
        self.bound_left = [wnpt.d_left for wnpt in self.global_wpnts.wpnts]#all positive
        self.treshold = self.track_length/2 #treshold for checking if we are going "backwards" in the s direction
        self.number_of_wpnts = len(self.ego_s_sorted)
        self.raw_opponent_traj = np.zeros((self.number_of_wpnts,11))
        self.projected_opponent_traj = ProjOppTraj()
        self.traversed_distance = 0
        self.index = None
        self.points_in_lap = 0
        self.marker_count = 0
        self.marker_max = 5
        self.mrk_array_msg = MarkerArray()
        self.initial_lap = True
        self.consecutive_points_off_opptraj = 0
        # Set by project_opponent_trajectory(); read back a loop() call later
        # (see the "if points_in_lap > 5" branch below), so it has to persist too.
        self.trav_dist = 0.0

        self.obstacle_cb(self.wait_for_message('/tracking/obstacles', ObstacleArray))
        # One executor now drives both loop() (via the timer) and the
        # subscriptions -- no separate rate/sleep loop needed.
        self._executor = MultiThreadedExecutor(num_threads=2)
        self._executor.add_node(self)
        self.create_timer(1.0 / self.loop_rate, self.loop)
        self._executor.spin()

    def loop(self):
        """Runs at loop_rate Hz (registered in make_opponent_trajectory). One
        pass of what used to be the while-loop body; unchanged logic, just
        reading/writing self.xxx instead of locals so it persists correctly
        between separate timer-triggered calls."""
        #sample data
        obstacle_array = self.obstacle_array
        position_in_map_frenet = self.position_in_map_frenet.copy()

        if self.index is not None and self.lap_count >= 1:
            time_diff = self.now_seconds() - self.raw_opponent_traj[self.index][6]
            #time_diff = 0 #for ROSBAG
            if time_diff > 0.5: #if the timestamp is older than 0.5 seconds we discard the measurement (variable)
                if  self.points_in_lap > 5: #publish some extra information about the opponent trajectory (variable)
                    self.projected_opponent_traj.nrofpoints = self.points_in_lap
                    self.lap_count += self.trav_dist/self.track_length
                    self.projected_opponent_traj.lapcount = self.lap_count
                    self.projected_opponent_traj.opp_is_on_trajectory = True
                    self.proj_opponent_trajectory_pub.publish(self.projected_opponent_traj)
                    self.projected_opponent_traj = ProjOppTraj()
                    self.raw_opponent_traj = np.zeros((self.number_of_wpnts,11))

                    self.traversed_distance = 0
                    self.points_in_lap = 1
                    self.first_point = True
                    self.index = None
                else:
                    self.projected_opponent_traj = ProjOppTraj()
                    self.raw_opponent_traj = np.zeros((self.number_of_wpnts,11))

                    self.traversed_distance = 0
                    self.points_in_lap = 1
                    self.first_point = True
                    self.index = None

        opponent = self.find_nearest_dyn_obstacle(obstacle_array=obstacle_array, position_in_map_frenet=position_in_map_frenet, track_length=self.track_length)
        if opponent is not None:
            old_index = self.index
            self.index = self.get_index(self.ego_s_sorted, opponent[0])

            if self.index != old_index: #if the index is the same as the old index we dont update the opponent trajectory (this also gets rid of the delta time = 0 problem in simulation)
                self.raw_opponent_traj[self.index] = opponent #s, d, vs, vd, is_static, is_visible, timestamp, s_var, d_var, vs_var, vd_var


                self.trav_dist, proj_opponent, ignore_datapoint_flag, self.first_point, publish_flag = self.project_opponent_trajectory(old_index=old_index, index=self.index,
                                                                            raw_opponent_traj=self.raw_opponent_traj,
                                                                            proj_opponent_traj=self.projected_opponent_traj,
                                                                            track_length=self.track_length,
                                                                            trav_dist=self.traversed_distance,
                                                                            bound_left=self.bound_left, bound_right=self.bound_right,
                                                                            first_point=self.first_point)

                if ignore_datapoint_flag:
                    self.index = old_index


                else:
                    if self.lap_count >= 1 and self.opp_traj_count > 1:
                        self.consecutive_points_off_opptraj = self.check_if_opponent_is_on_predicted_trajectory(proj_opponent=proj_opponent,
                                                                                                        consecutive_points_off_opptraj=self.consecutive_points_off_opptraj
                                                                                                        ,opponent_trajectory=self.opponent_trajectory)
                    self.projected_opponent_traj.detections.append(proj_opponent)
                    if self.first_point:
                        self.projected_opponent_traj.detections.pop(0)
                    self.traversed_distance += self.trav_dist
                    self.first_point = False
                    self.points_in_lap += 1


                    mrk_msg = self._opp_to_marker(proj_opp=proj_opponent, id=self.marker_count, delete_bool=False)
                    if len(self.mrk_array_msg.markers) > self.marker_max:
                        del_mrk = self.mrk_array_msg.markers[0]
                        self.mrk_array_msg.markers.pop(0)
                        self.mrk_array_msg.markers.pop(0)

                        del_mrk.action = Marker.DELETEALL
                        self.mrk_array_msg.markers.append(del_mrk)
                        self.mrk_array_msg.markers.append(mrk_msg)
                    else:
                        self.mrk_array_msg.markers.append(mrk_msg)
                    self.marker_pub.publish(self.mrk_array_msg)
                    self.marker_count += 1
                if self.consecutive_points_off_opptraj > 5 and self.traversed_distance < self.track_length/2: #if the opponent is off the predicted trajectory for more than 5 points we discard the measurement (variable)
                    self.projected_opponent_traj.nrofpoints = self.points_in_lap
                    self.projected_opponent_traj.lapcount = self.lap_count
                    self.projected_opponent_traj.opp_is_on_trajectory = False
                    opp_traj_count_old = self.opp_traj_count
                    self.proj_opponent_trajectory_pub.publish(self.projected_opponent_traj)
                    start_time = self.now_seconds()
                    while self.opp_traj_count == opp_traj_count_old and self.now_seconds() - start_time < 3:
                        #Wait for 0.02 seconds to get new opponent trajectory
                        time.sleep(0.02)
                        obstacle_array = self.obstacle_array
                        position_in_map_frenet = self.position_in_map_frenet.copy()
                        opponent = self.find_nearest_dyn_obstacle(obstacle_array=obstacle_array, position_in_map_frenet=position_in_map_frenet, track_length=self.track_length)
                        if opponent is not None:
                            old_index = self.index
                            self.index = self.get_index(self.ego_s_sorted, opponent[0])

                            if self.index != old_index:
                                self.raw_opponent_traj[self.index] = opponent
                                self.trav_dist, proj_opponent, ignore_datapoint_flag, self.first_point, publish_flag = self.project_opponent_trajectory(old_index=old_index, index=self.index,
                                                                                                            raw_opponent_traj=self.raw_opponent_traj,
                                                                                                            proj_opponent_traj=self.projected_opponent_traj,
                                                                                                            track_length=self.track_length,
                                                                                                            trav_dist=self.traversed_distance,
                                                                                                            bound_left=self.bound_left, bound_right=self.bound_right,
                                                                                                            first_point=self.first_point)
                                if ignore_datapoint_flag:
                                    self.index = old_index
                                else:
                                    self.projected_opponent_traj.detections.append(proj_opponent)
                                    if self.first_point:
                                        self.projected_opponent_traj.detections.pop(0)
                                    self.traversed_distance += self.trav_dist
                                    self.first_point = False
                                    self.points_in_lap += 1

                                    mrk_msg = self._opp_to_marker(proj_opp=proj_opponent, id=self.marker_count, delete_bool=False)
                                    if len(self.mrk_array_msg.markers) > self.marker_max:
                                        del_mrk = self.mrk_array_msg.markers[0]
                                        self.mrk_array_msg.markers.pop(0)
                                        self.mrk_array_msg.markers.pop(0)

                                        del_mrk.action = Marker.DELETEALL
                                        self.mrk_array_msg.markers.append(del_mrk)
                                        self.mrk_array_msg.markers.append(mrk_msg)
                                    else:
                                        self.mrk_array_msg.markers.append(mrk_msg)
                                    self.marker_pub.publish(self.mrk_array_msg)
                                    self.marker_count += 1
                    self.consecutive_points_off_opptraj = 0




                #lap complete
                if self.traversed_distance > (self.track_length/2):
                    self.lap_count += 0.5

                    #delete first point of the projected opponent trajectory since the velocity is not known
                    if self.initial_lap:
                        self.initial_lap = False
                     #checke for a large s gap between consecutive points
                        for i in range(len(self.projected_opponent_traj.detections)-1):
                            if (self.projected_opponent_traj.detections[i+1].s - self.projected_opponent_traj.detections[i].s)%self.track_length > 2.5:
                                self.initial_lap = True
                                start_index = i+1
                                break
                        if self.initial_lap: #start over from start_index
                            #remove points before start_index
                            for i in range(start_index):
                                self.projected_opponent_traj.detections.pop(0)

                            if len(self.projected_opponent_traj.detections) > 0:
                                self.index = self.get_index(self.ego_s_sorted, self.projected_opponent_traj.detections[-1].s)
                                self.points_in_lap = self.points_in_lap - start_index
                                self.lap_count = 0
                                #reset traversed distance
                                self.traversed_distance = (self.projected_opponent_traj.detections[-1].s-self.projected_opponent_traj.detections[0].s)%self.track_length
                            else:
                                #reset everything
                                self.projected_opponent_traj = ProjOppTraj()
                                self.raw_opponent_traj = np.zeros((self.number_of_wpnts,11))
                                self.traversed_distance = 0
                                self.points_in_lap = 0
                                self.lap_count = 0
                                self.index = None
                                self.first_point = True

                    if not self.initial_lap:
                        self.projected_opponent_traj.nrofpoints = self.points_in_lap
                        self.projected_opponent_traj.lapcount = self.lap_count
                        self.projected_opponent_traj.opp_is_on_trajectory = True
                        opp_traj_count_old = self.opp_traj_count

                        # publish projected opponent trajectory
                        self.proj_opponent_trajectory_pub.publish(self.projected_opponent_traj)

                        #reset (make last point of old half lap the first point of the new half lap)
                        self.projected_opponent_traj = ProjOppTraj()
                        self.projected_opponent_traj.detections.append(proj_opponent)
                        self.raw_opponent_traj = np.zeros((self.number_of_wpnts,11))
                        self.raw_opponent_traj[self.index] = opponent
                        self.points_in_lap = 1
                        self.traversed_distance = 0

#########################HELPER FUNCTIONS######################################

    def find_nearest_dyn_obstacle(self, obstacle_array: ObstacleArray, position_in_map_frenet: np.ndarray, track_length: float) -> (np.ndarray, Obstacle):

        """Find the nearest dynamic obstacle and return the obstacle and the obstacle message"""

        if len(obstacle_array.obstacles) > 0 and len(position_in_map_frenet):
            allowed_distance = self.max_opponent_distance
            timestanp = obstacle_array.header.stamp
            closest_opp = track_length
            opponent = None
            for obstacle in obstacle_array.obstacles: 
                opponent_dist = (obstacle.s_start - position_in_map_frenet[0]) % track_length
                if opponent_dist <= closest_opp and not obstacle.is_static: 
                    closest_opp = opponent_dist
                    opponent_static = obstacle.is_static
                    opponent_s = obstacle.s_center
                    opponent_d = obstacle.d_center
                    opponent_vs = obstacle.vs
                    opponent_vd = obstacle.vd
                    opponent_visible = obstacle.is_visible
                    s_var = obstacle.s_var
                    d_var = obstacle.d_var
                    vs_var = obstacle.vs_var
                    vd_var = obstacle.vd_var
                    opponent = [opponent_s, opponent_d, opponent_vs, opponent_vd, opponent_static, opponent_visible, 
                                timestanp, s_var, d_var, vs_var, vd_var] #s, d, vs, vd, is_static, is_visible, timestamp, s_var, d_var, vs_var, vd_var
                    opponent[6] = timestanp.sec + timestanp.nanosec * 1e-9
                
            if closest_opp > allowed_distance: #if the closest opponent is more than xm away we do not want to start trailing
                opponent = None
        else:
            opponent = None
        

        return opponent
    
    def check_if_opponent_is_on_predicted_trajectory(self, proj_opponent: ProjOppPoint, consecutive_points_off_opptraj: int, opponent_trajectory: OpponentTrajectory):
            
        """Check if the opponent is on the predicted trajectory"""
         #find corresponding oppwpnt to s value of opponent
        closest_oppwpnt = None
        for oppwpnt in self.opponent_trajectory.oppwpnts:
            if oppwpnt.s_m - proj_opponent.s > 0 and closest_oppwpnt is None:
                closest_oppwpnt = oppwpnt
        if closest_oppwpnt is not None:
            if abs(closest_oppwpnt.d_m - proj_opponent.d) < 0.3: #if the opponent is within 30cm of the predicted trajectory we assume that the prediction is correct (variable)
                consecutive_points_off_opptraj = 0
            else:
                consecutive_points_off_opptraj += 1
        return consecutive_points_off_opptraj
    

    def get_index(self, ego_s_sorted: list, opponent_s: float) -> int:

        """Get the index of the ego vehicle's s position that is closest to the opponent's s position """

        index = 0 
        for i in range(len(ego_s_sorted)):
            if ego_s_sorted[i] > opponent_s:
                index = i
                break
        return index

    def project_opponent_trajectory(self, old_index:int, index:int, raw_opponent_traj: np.ndarray, proj_opponent_traj: ProjOppTraj, trav_dist: float, track_length: float, first_point: bool = False, bound_left: list = None, bound_right: list = None) -> (float, np.ndarray, bool):

         
        """Make list of all opponent positions and velocities and return the list """

        delta_time = 0
        opp_vs = 0
        opp_vd = 0
        trav_dist = 0
        publish_flag = False
        #For first point only do this
        projected_opponent = ProjOppPoint()
        projected_opponent.s = raw_opponent_traj[index][0]
        projected_opponent.d = raw_opponent_traj[index][1]
        projected_opponent.vs = 0
        projected_opponent.vd = 0
        projected_opponent.is_static = int(raw_opponent_traj[index][4])
        projected_opponent.is_visible = int(raw_opponent_traj[index][5])
        projected_opponent.time = raw_opponent_traj[index][6]
        projected_opponent.s_var = raw_opponent_traj[index][7]
        projected_opponent.d_var = raw_opponent_traj[index][8]
        projected_opponent.vs_var = raw_opponent_traj[index][9]
        projected_opponent.vd_var = raw_opponent_traj[index][10]   

        discard_flag = False
        if(not first_point):
            diff_s = raw_opponent_traj[index][0] - raw_opponent_traj[old_index][0] 
            diff_d = raw_opponent_traj[index][1] - raw_opponent_traj[old_index][1]
            delta_time, discard_flag = self._get_delta_time(old_index=old_index, index=index,opponent=raw_opponent_traj)

            #check if we are going "backwards" in the s direction we dicard this point to avoid duplicate points (no negativ s velicity possible for opponent)
            if diff_s < 0 and abs(diff_s) < self.treshold:
                discard_flag = True

            # around the origin
            elif diff_s < 0 and abs(diff_s) > self.treshold:
                diff_s = diff_s % track_length
                if len(proj_opponent_traj.detections) > 0: 
                    opp_vs = proj_opponent_traj.detections[-1].vs #do this since track_length is not very accurate (could also ignore this point)
                else:
                    discard_flag = True
                opp_vd = diff_d / delta_time
                trav_dist += diff_s
        
            #normal case were s is increasing with time    
            else:
                opp_vs = diff_s / delta_time
                opp_vd = diff_d / delta_time
                trav_dist += diff_s

            #Add projected velocity to the projected opponent
            projected_opponent.vs = opp_vs
            projected_opponent.vd = opp_vd

            #check if opp is on the track
            if projected_opponent.d<0:
                if abs(projected_opponent.d) > bound_right[index]:
                    discard_flag = True
            else:
                if projected_opponent.d > bound_left[index]:
                    discard_flag = True
                    

            #leave out points where velocity or acceleration is over the threshold
            curr_vel = np.linalg.norm([raw_opponent_traj[index][2],raw_opponent_traj[index][3]])
            old_vel = np.linalg.norm([raw_opponent_traj[old_index][2],raw_opponent_traj[old_index][3]])
            delta_velocity= abs(curr_vel-old_vel)/delta_time
            if not discard_flag and (np.sqrt(opp_vs**2+opp_vd**2) > 15 or delta_velocity > 20):
                discard_flag = True
              
            if diff_s > 1 and self.lap_count>=1: #if detections are more than 1m apart (variable)
                discard_flag = True
                first_point = True
                publish_flag = True

        return trav_dist, projected_opponent, discard_flag, first_point, publish_flag
    



    def _get_delta_time(self, old_index:int, index:int, opponent: np.ndarray) -> float:

        """Get the delta time between two points in the opponent trajectory"""

        delta_time = opponent[index][6] - opponent[old_index][6]
        discard = False
        if delta_time < 0.01:
            discard = True
            delta_time = 0.04
        return delta_time, discard
       
    
    def _opp_to_marker(self, proj_opp: ProjOppPoint, id: int, delete_bool: bool = False) -> Marker:

        """Convert the opponent position to a marker"""
        cartesian_position = self.converter.get_cartesian(proj_opp.s, proj_opp.d)
        # Create a cylindrical marker to represent the opponent in RViz
        marker = Marker()
        marker.header.frame_id = "map"  # Set the frame ID
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD if not delete_bool else Marker.DELETE
        marker.pose.position.x = cartesian_position[0]
        marker.pose.position.y = cartesian_position[1]
        marker.pose.position.z = 0.0
        marker.scale.x = 0.3  # Adjust the size of the cylinder as needed
        marker.scale.y = 0.3
        marker.scale.z = 1.0 
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.id = id  # Give each marker an ID
        return marker


if __name__ == '__main__':
    rclpy.init()
    node = Opponent_Trajectory()
    try:
        node.make_opponent_trajectory()
    finally:
        if hasattr(node, '_executor'):
            node._executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
