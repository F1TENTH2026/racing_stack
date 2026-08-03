cmake_minimum_required(VERSION 3.8)
project(gp_traj_predictor)

find_package(ament_cmake REQUIRED)

# These Python nodes intentionally remain as executable scripts.  Keeping the
# original algorithm files together also makes it possible to run them with
# `ros2 run gp_traj_predictor <node>` without a ROS 1 catkin workspace.
install(PROGRAMS src/opponent_trajectory.py DESTINATION lib/${PROJECT_NAME} RENAME opponent_trajectory)
install(PROGRAMS src/gaussian_process_opp_traj.py DESTINATION lib/${PROJECT_NAME} RENAME gaussian_process_opp_traj)
install(PROGRAMS src/predictor_opponent_trajectory.py DESTINATION lib/${PROJECT_NAME} RENAME predictor_opponent_trajectory)
install(PROGRAMS src/opp_prediction.py DESTINATION lib/${PROJECT_NAME} RENAME opp_prediction)
install(PROGRAMS src/dynamic_prediction_server.py DESTINATION lib/${PROJECT_NAME} RENAME dynamic_prediction_server)

# Install the shared rclpy Node helper beside the scripts.
install(FILES src/prediction_node.py DESTINATION lib/${PROJECT_NAME})

ament_package()
