cmake_minimum_required(VERSION 3.8)
project(frenet_conversion_msgs)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(f110_msgs REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/Frenet2Glob.srv" "srv/Frenet2GlobArr.srv" "srv/Glob2Frenet.srv" "srv/Glob2FrenetArr.srv"
  DEPENDENCIES f110_msgs)
ament_package()
