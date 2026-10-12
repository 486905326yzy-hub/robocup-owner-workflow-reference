cmake_minimum_required(VERSION 3.0.2)
project(yoloworld_perception)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  perception_msgs
  rospy
  sensor_msgs
  std_msgs
  tf2_geometry_msgs
  tf2_ros
)

catkin_package(
  CATKIN_DEPENDS cv_bridge geometry_msgs perception_msgs rospy sensor_msgs std_msgs tf2_geometry_msgs tf2_ros
)

catkin_install_python(PROGRAMS
  scripts/person_3d_projector.py
  scripts/yoloworld_debug_viewer.py
  scripts/yoloworld_node.py
  scripts/yoloworld_smoke_test.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch models
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
