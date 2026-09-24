cmake_minimum_required(VERSION 3.8)
project(pointcloud_registration)

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
find_package(pcl_conversions REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)

find_package(PCL REQUIRED COMPONENTS common filters registration search)
find_package(Eigen3 REQUIRED)

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
)

add_definitions(${PCL_DEFINITIONS})

add_executable(registration_node
  src/registration_node.cpp
  src/icp.cpp
)

target_include_directories(registration_node PRIVATE
  include
  ${PCL_INCLUDE_DIRS}
)

ament_target_dependencies(registration_node
  rclcpp
  sensor_msgs
  std_msgs
  pcl_conversions
  tf2
  tf2_ros
)

target_link_libraries(registration_node
  ${PCL_LIBRARIES}
  Eigen3::Eigen
)

install(
  TARGETS registration_node
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY config launch rviz
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY include/
  DESTINATION include
)

ament_package()
