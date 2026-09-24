cmake_minimum_required(VERSION 3.8)
project(semantic_perception)

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

find_package(OpenCV REQUIRED)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(semantic_perception_node
  src/inference.cpp
  src/postprocess.cpp
  src/visualization.cpp
  src/semantic_perception_node.cpp
)

ament_target_dependencies(semantic_perception_node
  rclcpp
  sensor_msgs
  std_msgs
  cv_bridge
)

target_link_libraries(semantic_perception_node
  ${OpenCV_LIBRARIES}
)

install(
  TARGETS semantic_perception_node
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
