#!/usr/bin/env bash
set -eo pipefail
source /home/lp531/ros2_ws/src/z1_ros2/scripts/configure_environment.sh
set +u
source /opt/ros/humble/setup.bash
source /home/lp531/ros2_ws/install/setup.bash
set -u
export ROS_DOMAIN_ID=133
export Z1_SERVO_PREFIX=/home/lp531/.local/opt/ros-humble-moveit-servo/opt/ros/humble
export Z1_ISAAC_GRASP_CONFIG=/home/lp531/robot/peach_scale120_correction60_20261106/grasp_physics.json
export FASTDDS_DEFAULT_PROFILES_FILE="$FASTRTPS_DEFAULT_PROFILES_FILE"
root=/home/lp531/robot/peach_scale120_correction60_20261106
data=/home/lp531/smolvla_data/peach_scale120_correction60_20261106
mkdir -p "$data"
if [[ ! -e "$data/correction_plan.json" ]];then cp "$root/correction_plan.template.json" "$data/correction_plan.json";fi
mode=(--until-complete)
if [[ -n "${Z1_PEACH_CORRECTION_ROUNDS:-}" ]];then
 [[ "$Z1_PEACH_CORRECTION_ROUNDS" =~ ^[1-9][0-9]*$ ]] || { echo '纠偏轮数必须是正整数';exit 2; }
 mode=(--rounds "$Z1_PEACH_CORRECTION_ROUNDS" --max-attempts 180)
fi
exec /usr/bin/python3 "$root/supervisor.py" --output "$data" --port 8836 "${mode[@]}" "$@"
