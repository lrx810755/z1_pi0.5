#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/setup_env.sh"

required=(
  "${Z1_ROS_WS}/src/z1_ros2/z1_description/urdf/z1.urdf.xacro"
  "${Z1_ROS_WS}/src/z1_ros2/z1_isaac_bridge/config/grasp_physics.json"
  "${Z1_ASSET_ROOT}/scripts/run_z1_ros2_scene.py"
  "${Z1_ASSET_ROOT}/scanned_objects/manifest.json"
  "${Z1_ASSET_ROOT}/hand_eye_calibration.yaml"
  "${Z1_ASSET_ROOT}/generated_dual_camera/smolvla_dual_wrist_camera.json"
  "${Z1_SMOLVLA_ROOT}/lerobot/pyproject.toml"
)

for path in "${required[@]}"; do
  if [[ ! -e "${path}" ]]; then
    printf 'Missing required path: %s\n' "${path}" >&2
    exit 1
  fi
done

if rg -n '/home/lp531|/opt/ros/humble' \
  "${Z1_ROS_WS}/src/z1_ros2" \
  "${Z1_ASSET_ROOT}/scripts" \
  "${Z1_ASSET_ROOT}/hand_eye_calibration.yaml" \
  --glob '*.py' --glob '*.yaml' --glob '*.json' --glob '*.xacro'; then
  printf 'Machine-specific hardcoded paths remain in active sources.\n' >&2
  exit 1
fi

if rg -n '"(robot|scene)_asset"\s*:\s*"/' \
  "${Z1_ASSET_ROOT}/generated_dual_camera/smolvla_dual_wrist_camera.json"; then
  printf 'Generated scene metadata contains an absolute asset path.\n' >&2
  exit 1
fi

bash -n "${script_dir}"/*.sh

printf 'Portable source checks passed.\n'
printf 'Repository: %s\n' "${Z1_SMOLVLA_ROOT}"
printf 'ROS workspace: %s\n' "${Z1_ROS_WS}"
printf 'Isaac assets: %s\n' "${Z1_ASSET_ROOT}"
printf 'Dataset root: %s\n' "${SMOLVLA_DATA_ROOT}"
printf 'Isaac Sim: %s\n' "${ISAAC_SIM_ROOT}"
