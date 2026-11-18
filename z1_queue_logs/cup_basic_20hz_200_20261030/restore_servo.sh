#!/usr/bin/env bash
set -e
source /home/lp531/ros2_ws/src/z1_ros2/scripts/configure_environment.sh
set +u
source /opt/ros/humble/setup.bash
source /home/lp531/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=118
export Z1_SERVO_PREFIX=/home/lp531/.local/opt/ros-humble-moveit-servo/opt/ros/humble
exec /home/lp531/ros2_ws/install/z1_isaac_bridge/lib/z1_isaac_bridge/moveit_servo_vendor --ros-args -r __node:=servo_node --params-file /tmp/launch_params_601dqlql --params-file /tmp/launch_params_ar1tpor6 --params-file /tmp/launch_params___2q0epz --params-file /tmp/launch_params_22ufqs_m --params-file /tmp/launch_params_itsy9j4l
