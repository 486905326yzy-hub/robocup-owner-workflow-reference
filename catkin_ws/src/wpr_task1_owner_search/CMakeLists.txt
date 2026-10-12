cmake_minimum_required(VERSION 3.0.2)
project(wpr_task1_owner_search)

find_package(catkin REQUIRED COMPONENTS
  actionlib
  actionlib_msgs
  cv_bridge
  gazebo_msgs
  geometry_msgs
  move_base_msgs
  nav_msgs
  perception_msgs
  rospy
  sensor_msgs
  std_msgs
  tf2_geometry_msgs
  tf2_ros
  waterplus_map_tools
)

catkin_package(
  CATKIN_DEPENDS actionlib actionlib_msgs cv_bridge gazebo_msgs geometry_msgs move_base_msgs nav_msgs perception_msgs rospy sensor_msgs std_msgs tf2_geometry_msgs tf2_ros waterplus_map_tools
)

catkin_install_python(PROGRAMS
  scripts/task1_find_owner.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config data launch tools
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
