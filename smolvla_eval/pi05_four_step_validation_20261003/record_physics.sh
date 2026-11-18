#!/bin/bash
set -eo pipefail
source /opt/ros/humble/setup.bash
source /home/lp531/ros2_ws/install/setup.bash
exec /usr/bin/python3 /home/lp531/smolvla_eval/pi05_four_step_validation_20261003/record_physics.py
