#!/usr/bin/env bash
set -eo pipefail
source /home/lp531/ros2_ws/src/z1_ros2/scripts/configure_environment.sh
export Z1_SERVO_PREFIX=/home/lp531/.local/opt/ros-humble-moveit-servo/opt/ros/humble
set +u
source /opt/ros/humble/setup.bash
source /home/lp531/ros2_ws/install/setup.bash
set -u
export ROS_DOMAIN_ID=82
export FASTDDS_DEFAULT_PROFILES_FILE="$FASTRTPS_DEFAULT_PROFILES_FILE"
/usr/bin/python3 /home/lp531/ros2_ws/src/z1_ros2/scripts/correction_fragments_20261012/supervisor.py --output /home/lp531/smolvla_data/z1_bottle_correction_fragments_20261012 --port 8771 --headless --until-complete "$@"

for fragment_arg in "$@"; do
  if [[ "$fragment_arg" == --status || "$fragment_arg" == --dry-run ]]; then exit 0; fi
done
exec /home/lp531/pi05_lora_workspace/.venv/bin/python /home/lp531/pi05_lora_workspace/correction_fragments_20261012/prepare.py
