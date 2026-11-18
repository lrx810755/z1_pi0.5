#!/bin/bash
set -eo pipefail
source /opt/ros/humble/setup.bash
source /home/lp531/ros2_ws/install/setup.bash
exec /home/lp531/miniconda3/envs/unifolm-vla/bin/python /home/lp531/smolvla_eval/execution_integration_20261004/check_native_executor.py
