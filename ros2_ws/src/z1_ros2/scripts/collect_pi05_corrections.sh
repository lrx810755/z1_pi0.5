#!/usr/bin/env bash
set -euo pipefail
correction_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$correction_script_dir/configure_environment.sh"
if [[ ! -x "$Z1_SERVO_PREFIX/lib/moveit_servo/servo_node_main" ]] && \
   [[ -x /home/lp531/.local/opt/ros-humble-moveit-servo/opt/ros/humble/lib/moveit_servo/servo_node_main ]]; then
  export Z1_SERVO_PREFIX=/home/lp531/.local/opt/ros-humble-moveit-servo/opt/ros/humble
fi
set +u
source /opt/ros/humble/setup.bash
source "$Z1_REPO_ROOT/../../install/setup.bash"
set -u
# A dedicated simulation DDS domain avoids stale clocks from other sessions.
export ROS_DOMAIN_ID="${Z1_CORRECTION_DOMAIN_ID:-81}"
export FASTDDS_DEFAULT_PROFILES_FILE="$FASTRTPS_DEFAULT_PROFILES_FILE"
exec /usr/bin/python3 "$correction_script_dir/correction_recording/supervisor.py" "$@"
