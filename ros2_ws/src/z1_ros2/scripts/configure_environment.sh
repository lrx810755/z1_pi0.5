#!/usr/bin/env bash
set -euo pipefail

z1_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export Z1_REPO_ROOT="$(cd "$z1_script_dir/.." && pwd -P)"
export Z1_ASSET_ROOT="${Z1_ASSET_ROOT:-$Z1_REPO_ROOT/isaac_assets/z1}"
export Z1_DATASET_ROOT="${Z1_DATASET_ROOT:-$HOME/smolvla_data}"
export ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-$HOME/isaac-sim}"
export Z1_SERVO_PREFIX="${Z1_SERVO_PREFIX:-/opt/ros/humble}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-$Z1_REPO_ROOT/z1_isaac_bridge/config/fastdds_camera.xml}"

# Keep ROS Humble's tested system Python packages ahead of incompatible
# packages that may have been installed into ~/.local.
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
