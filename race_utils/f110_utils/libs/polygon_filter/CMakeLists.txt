cmake_minimum_required(VERSION 3.8)
project(polygon_filter)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Find build tools and dependencies
find_package(ament_cmake REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

# Create the C++ library target
add_library(polygon_filter_lib src/polygon_filter.cpp)
target_include_directories(polygon_filter_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIRS}
)
target_link_libraries(polygon_filter_lib ${OpenCV_LIBS} yaml-cpp)

# Install the public headers
install(DIRECTORY include/
  DESTINATION include
)

# Install the library and register the export set
install(TARGETS polygon_filter_lib
  EXPORT polygon_filter_export
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Export the include dirs, targets and dependencies to downstream packages
ament_export_include_directories(include)
ament_export_targets(polygon_filter_export HAS_LIBRARY_TARGET)
ament_export_dependencies(OpenCV yaml-cpp)

ament_package()
