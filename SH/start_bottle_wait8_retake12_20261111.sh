#!/usr/bin/env bash
set -eo pipefail
source /home/lp531/ros2_ws/src/z1_ros2/scripts/configure_environment.sh
set +u
source /opt/ros/humble/setup.bash
source /home/lp531/ros2_ws/install/setup.bash
set -u
export ROS_DOMAIN_ID=142 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export Z1_COLLECTION_GRID_ROWS=4 Z1_COLLECTION_GRID_COLUMNS=5
export Z1_SERVO_PREFIX=/home/lp531/.local/opt/ros-humble-moveit-servo/opt/ros/humble
export FASTDDS_DEFAULT_PROFILES_FILE="$FASTRTPS_DEFAULT_PROFILES_FILE"
export Z1_ISAAC_GRASP_CONFIG=/home/lp531/robot/bottle_wait8_retake12_20261111/grasp_physics.json
exec /usr/bin/python3 -u /home/lp531/robot/bottle_wait8_retake12_20261111/run.py "$@"
