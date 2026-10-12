cmake_minimum_required(VERSION 3.0.2)
project(offline_voice_bridge)

find_package(catkin REQUIRED COMPONENTS
  actionlib
  rospy
  std_msgs
  sound_play
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/offline_asr_node.py
  scripts/offline_tts_node.py
  scripts/natural_tts_node.py
  scripts/say_once.py
  scripts/qwen_action_recognition.py
  scripts/yolo_pose_action_recognition.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config tools models third_party
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
