cmake_minimum_required(VERSION 3.8)
project(camera_lidar_calibration)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(pcl_conversions REQUIRED)

find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED COMPONENTS common)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

add_definitions(${PCL_DEFINITIONS})

add_executable(camera_lidar_projection_node
  src/calibration.cpp
  src/projection.cpp
  src/visualization.cpp
  src/camera_lidar_projection_node.cpp
)

ament_target_dependencies(camera_lidar_projection_node
  rclcpp
  sensor_msgs
  std_msgs
  cv_bridge
  tf2
  tf2_ros
  pcl_conversions
)

target_link_libraries(camera_lidar_projection_node
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
)

install(
  TARGETS camera_lidar_projection_node
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY config launch rviz docs
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY include/
  DESTINATION include
)

ament_package()
