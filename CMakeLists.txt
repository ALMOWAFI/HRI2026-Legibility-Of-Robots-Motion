cmake_minimum_required(VERSION 3.0.2)
project(hri_legible_motion)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  trajectory_msgs
  message_generation
)

catkin_python_setup()

add_message_files(
  FILES
  TrialInfo.msg
  ChoiceEvent.msg
)

generate_messages(DEPENDENCIES std_msgs geometry_msgs)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs geometry_msgs sensor_msgs trajectory_msgs message_runtime
)

catkin_install_python(PROGRAMS
  scripts/motion_node.py
  scripts/experiment_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
