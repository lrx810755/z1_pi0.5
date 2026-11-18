#!/usr/bin/env bash

# Source this file before building or running the organized repository.
z1_smolvla_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export Z1_SMOLVLA_ROOT="${Z1_SMOLVLA_ROOT:-${z1_smolvla_root}}"
export Z1_ASSET_ROOT="${Z1_ASSET_ROOT:-${Z1_SMOLVLA_ROOT}/isaac_assets/z1}"
export Z1_ROS_WS="${Z1_ROS_WS:-${Z1_SMOLVLA_ROOT}/ros2_ws}"
export SMOLVLA_DATA_ROOT="${SMOLVLA_DATA_ROOT:-${Z1_SMOLVLA_ROOT}/data/runtime}"
export ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-${HOME}/isaac-sim}"
export ROS_DISTRO="${ROS_DISTRO:-humble}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

unset z1_smolvla_root

