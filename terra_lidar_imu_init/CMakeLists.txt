cmake_minimum_required(VERSION 3.5)
project(terra_lidar_imu_init)

# Add policy settings to handle warnings
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)  # Find package uses PackageName_ROOT variables
endif()
if(POLICY CMP0002)
  cmake_policy(SET CMP0002 NEW)  # Target names must be unique
endif()

SET(CMAKE_BUILD_TYPE "Release")

ADD_COMPILE_OPTIONS(-std=c++17)
set(CMAKE_CXX_FLAGS "-std=c++17 -O3")

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread -fexceptions")

message("Current CPU architecture: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
  include(ProcessorCount)
  ProcessorCount(N)
  message("Processer number:  ${N}")
  if(N GREATER 4)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=3)
    message("core for MP: 3")
  elseif(N GREATER 3)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=2)
    message("core for MP: 2")
  else()
    add_definitions(-DMP_PROC_NUM=1)
  endif()
else()
  add_definitions(-DMP_PROC_NUM=1)
endif()

find_package(OpenMP QUIET)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")

find_package(PythonLibs REQUIRED)
find_path(MATPLOTLIB_CPP_INCLUDE_DIRS "matplotlibcpp.h")

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(livox_ros_driver2 REQUIRED)
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(rosidl_default_generators REQUIRED)

message(STATUS "Eigen: ${EIGEN3_INCLUDE_DIR}")

# Generate ROS2 message interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Pose6D.msg"
  "msg/States.msg"
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

include_directories(
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  include
)

set(dependencies 
  rclcpp
  std_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
  tf2_ros
  tf2_geometry_msgs
  tf2_eigen
  pcl_conversions
  livox_ros_driver2
)

add_executable(terra_lidar_imu_init_node
  src/laser_mapping.cpp
  src/ikd_tree.cpp
  src/lidar_imu_init.cpp
  src/preprocess.cpp
)

target_link_libraries(terra_lidar_imu_init_node 
  ${PCL_LIBRARIES} 
  ${PYTHON_LIBRARIES} 
  ${CERES_LIBRARIES}
)

target_include_directories(terra_lidar_imu_init_node PRIVATE 
  ${PYTHON_INCLUDE_DIRS}
)

ament_target_dependencies(terra_lidar_imu_init_node 
  ${dependencies}
)

# Properly link the message interfaces
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(terra_lidar_imu_init_node "${cpp_typesupport_target}")

install(TARGETS 
  terra_lidar_imu_init_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()




