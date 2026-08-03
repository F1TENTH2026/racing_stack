cmake_minimum_required(VERSION 3.5)
project(udp_msgs)

find_package(ros_environment REQUIRED QUIET)
set(ROS_VERSION $ENV{ROS_VERSION})

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# === ROS2 (AMENT) =============================================================
if(${ROS_VERSION} EQUAL 2)

  # find dependencies
  find_package(ament_cmake REQUIRED)
  find_package(std_msgs REQUIRED)
  find_package(rosidl_default_generators REQUIRED)

  rosidl_generate_interfaces(${PROJECT_NAME}
    "msg/UdpPacket.msg"
    "srv/UdpSend.srv"
    "srv/UdpSocket.srv"
    DEPENDENCIES std_msgs
  )

  if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    # the following line skips the linter which checks for copyrights
    # uncomment the line when a copyright and license is not present in all source files
    #set(ament_cmake_copyright_FOUND TRUE)
    # the following line skips cpplint (only works in a git repo)
    # uncomment the line when this package is not in a git repo
    #set(ament_cmake_cpplint_FOUND TRUE)
    ament_lint_auto_find_test_dependencies()
  endif()

  ament_export_dependencies(rosidl_default_runtime)
  ament_package()

# === ROS1 (CATKIN) ============================================================
elseif(${ROS_VERSION} EQUAL 1)

  # find dependencies
  find_package(catkin REQUIRED COMPONENTS
    message_generation
    std_msgs
  )

  add_message_files(
  FILES
    UdpPacket.msg
  )

  add_service_files(
  FILES
    UdpSend.srv
    UdpSocket.srv
  )

  generate_messages(
  DEPENDENCIES
    std_msgs
  )

  catkin_package(
  CATKIN_DEPENDS
    message_runtime
    std_msgs
  )

endif()