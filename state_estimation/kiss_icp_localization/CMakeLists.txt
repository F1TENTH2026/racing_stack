cmake_minimum_required(VERSION 3.16)
project(kiss_icp_localization)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-O3 -Wall -Wextra)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)
find_package(OpenMP COMPONENTS CXX)

# Pure-Eigen core (no ROS) so it can be unit-tested standalone
add_library(kiss_loc_core
  src/voxel_hash_map.cpp
  src/registration.cpp
  src/bev_detector.cpp
)
target_include_directories(kiss_loc_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(kiss_loc_core Eigen3::Eigen)
if(OpenMP_CXX_FOUND)
  target_link_libraries(kiss_loc_core OpenMP::OpenMP_CXX)
endif()

add_executable(localization_node src/localization_node.cpp)
target_link_libraries(localization_node kiss_loc_core ${PCL_LIBRARIES})
target_include_directories(localization_node PRIVATE ${PCL_INCLUDE_DIRS})
ament_target_dependencies(localization_node
  rclcpp sensor_msgs nav_msgs geometry_msgs tf2_ros pcl_conversions visualization_msgs
)

install(TARGETS localization_node kiss_loc_core
  RUNTIME DESTINATION lib/${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_core test/test_core.cpp)
  target_link_libraries(test_core kiss_loc_core)
endif()

ament_package()
