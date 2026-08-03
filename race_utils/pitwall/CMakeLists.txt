cmake_minimum_required(VERSION 3.8)
project(pitwall)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rviz_common REQUIRED)
find_package(pluginlib REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Core Widgets)

set(CMAKE_AUTOMOC ON)

# --- C++ producer library -------------------------------------------------
add_library(pitwall src/pitwall.cpp)
target_include_directories(pitwall PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(pitwall rclcpp std_msgs)

# --- RViz panel: Sim Control + telemetry viewer ---------------------------
add_library(pitwall_panel SHARED
  src/pitwall_panel.cpp
  include/pitwall/pitwall_panel.hpp)
target_include_directories(pitwall_panel PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
# QT_NO_KEYWORDS avoids clashes between Qt's signals/slots/emit and other libs.
target_compile_definitions(pitwall_panel PRIVATE QT_NO_KEYWORDS)
target_link_libraries(pitwall_panel Qt5::Widgets Qt5::Core)
ament_target_dependencies(pitwall_panel rclcpp rviz_common pluginlib std_msgs)
# Register the panel with RViz via pluginlib (base package: rviz_common).
pluginlib_export_plugin_description_file(rviz_common plugin_description.xml)

# --- C++ demo producer ----------------------------------------------------
add_executable(pitwall_demo src/demo_producer.cpp)
target_link_libraries(pitwall_demo pitwall)
ament_target_dependencies(pitwall_demo rclcpp)

install(DIRECTORY include/ DESTINATION include)
install(TARGETS pitwall
  EXPORT export_pitwall
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(TARGETS pitwall_demo DESTINATION lib/${PROJECT_NAME})
install(TARGETS pitwall_panel
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(FILES plugin_description.xml DESTINATION share/${PROJECT_NAME})

# --- Python producer module + nodes --------------------------------------
ament_python_install_package(pitwall)
install(PROGRAMS
  scripts/monitor_node.py
  scripts/demo_producer.py
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

ament_export_include_directories(include)
ament_export_libraries(pitwall)
ament_export_dependencies(rclcpp std_msgs rviz_common pluginlib)
ament_package()
