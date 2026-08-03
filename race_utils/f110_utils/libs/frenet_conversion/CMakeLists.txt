cmake_minimum_required(VERSION 3.8)
project(frenet_conversion)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Find build tools and dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(f110_msgs REQUIRED)

# Create the C++ library target (ROS2 port of the ROS1 frenet_conversion lib)
add_library(frenet_conversion_lib src/frenet_conversion.cc)
target_include_directories(frenet_conversion_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(frenet_conversion_lib rclcpp f110_msgs)

# Install the public headers
install(DIRECTORY include/
  DESTINATION include
)

# Install the library and register the export set
install(TARGETS frenet_conversion_lib
  EXPORT frenet_conversion_export
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Export the include dirs, targets and dependencies to downstream packages
ament_export_include_directories(include)
ament_export_targets(frenet_conversion_export HAS_LIBRARY_TARGET)
ament_export_dependencies(rclcpp f110_msgs)

# Install the Python module (frenet_conversion.frenet_converter) -- preserved
ament_python_install_package(${PROJECT_NAME})

ament_package()
