#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/setup_env.sh"

ros_setup="/opt/ros/${ROS_DISTRO}/setup.bash"
if [[ ! -f "${ros_setup}" ]]; then
  printf 'ROS 2 setup not found: %s\n' "${ros_setup}" >&2
  exit 1
fi
set +u
source "${ros_setup}"
set -u

# ROS Humble uses the Ubuntu Python toolchain. Ignore unrelated packages from
# ~/.local, which can otherwise override setuptools/packaging during colcon.
export PYTHONNOUSERSITE=1

rosdep install \
  --from-paths "${Z1_ROS_WS}/src" \
  --ignore-src \
  --rosdistro "${ROS_DISTRO}" \
  -r -y

cd "${Z1_ROS_WS}"
colcon build \
  --symlink-install \
  --packages-ignore z1_hardware_interface z1_bringup \
  --packages-select \
  z1_description \
  z1_examples \
  z1_moveit \
  z1_isaac_bridge

printf 'ROS 2 workspace built at %s\n' "${Z1_ROS_WS}"
