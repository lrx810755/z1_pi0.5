#!/bin/bash
set -eo pipefail
source /opt/ros/humble/setup.bash
source /home/lp531/ros2_ws/install/setup.bash
exec /home/lp531/miniconda3/envs/unifolm-vla/bin/python -u /home/lp531/smolvla_eval/pi05_four_step_validation_20261003/collect_verified_layouts_round2.py
