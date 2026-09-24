cmake_minimum_required(VERSION 3.8)
project(world_frame_reconstruction)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(PCL REQUIRED COMPONENTS common filters)
find_package(pcl_conversions REQUIRED)

include_directories(include ${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(world_frame_reconstruction_node
  src/world_frame_reconstruction_node.cpp
)

target_link_libraries(world_frame_reconstruction_node ${PCL_LIBRARIES})
ament_target_dependencies(world_frame_reconstruction_node
  rclcpp sensor_msgs geometry_msgs tf2 tf2_ros tf2_geometry_msgs std_msgs pcl_conversions
)

install(TARGETS world_frame_reconstruction_node
  DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config launch rviz docs DESTINATION share/${PROJECT_NAME})

ament_package()
