cmake_minimum_required(VERSION 3.8)
project(semantic_3d_fusion)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(include)

add_executable(semantic_fusion_node
  src/projection.cpp
  src/semantic_fusion_node.cpp
)

ament_target_dependencies(semantic_fusion_node
  rclcpp sensor_msgs std_msgs tf2 tf2_ros tf2_sensor_msgs
)

target_link_libraries(semantic_fusion_node Eigen3::Eigen)

target_include_directories(semantic_fusion_node PRIVATE ${EIGEN3_INCLUDE_DIRS})

install(TARGETS semantic_fusion_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config launch rviz docs
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

ament_package()
