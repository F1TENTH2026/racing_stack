cmake_minimum_required(VERSION 3.8)
project(perception)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Find build tools and dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(f110_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(grid_filter REQUIRED)
find_package(frenet_conversion REQUIRED)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

# C++ detect node (ROS2 port of the ROS1 perception/src/detect.cpp)
add_executable(detect src/detect.cpp)
ament_target_dependencies(detect
  rclcpp
  tf2
  tf2_ros
  tf2_geometry_msgs
  sensor_msgs
  nav_msgs
  std_msgs
  geometry_msgs
  visualization_msgs
  f110_msgs
  ament_index_cpp
  grid_filter
  frenet_conversion
)
target_link_libraries(detect
  grid_filter::grid_filter_lib
  frenet_conversion::frenet_conversion_lib
  ${OpenCV_LIBS}
  yaml-cpp
)

install(TARGETS detect
  DESTINATION lib/${PROJECT_NAME}
)

# Python tracker (standalone executable, like the ROS1 scripts/).
install(PROGRAMS
  scripts/multi_tracking.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME tracking_node
)

ament_package()
