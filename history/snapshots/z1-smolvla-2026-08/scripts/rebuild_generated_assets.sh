#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/setup_env.sh"

ros_setup="/opt/ros/${ROS_DISTRO}/setup.bash"
if [[ ! -f "${ros_setup}" ]]; then
  printf 'ROS 2 setup not found: %s\n' "${ros_setup}" >&2
  exit 1
fi
if [[ ! -x "${ISAAC_SIM_ROOT}/python.sh" ]]; then
  printf 'Isaac Sim Python not found: %s\n' "${ISAAC_SIM_ROOT}/python.sh" >&2
  exit 1
fi

if [[ "${Z1_ALLOW_NO_GPU:-0}" != "1" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
    printf '%s\n' \
      'No usable NVIDIA GPU was detected. Isaac scene generation and smoke tests require a working driver.' \
      'Fix GPU access, or set Z1_ALLOW_NO_GPU=1 only for a best-effort USD rebuild without GPU validation.' >&2
    exit 1
  fi
fi

inotify_instances_file="/proc/sys/fs/inotify/max_user_instances"
if [[ -r "${inotify_instances_file}" ]]; then
  inotify_instances="$(<"${inotify_instances_file}")"
  if (( inotify_instances < 256 )); then
    printf 'Warning: fs.inotify.max_user_instances=%s is low; Isaac may report "No space left on device" for change watches.\n' \
      "${inotify_instances}" >&2
  fi
fi

set +u
source "${ros_setup}"
if [[ -f "${Z1_ROS_WS}/install/setup.bash" ]]; then
  source "${Z1_ROS_WS}/install/setup.bash"
fi
set -u

generated="${Z1_ASSET_ROOT}/generated_dual_camera"
mkdir -p "${generated}"

xacro \
  "${Z1_ROS_WS}/src/z1_ros2/z1_description/urdf/z1.urdf.xacro" \
  sim_ignition:=false \
  with_gripper:=true \
  -o "${generated}/z1_with_dual_d435.urdf"

"${ISAAC_SIM_ROOT}/python.sh" \
  "${Z1_ASSET_ROOT}/scripts/prepare_scanned_assets.py"

"${ISAAC_SIM_ROOT}/python.sh" \
  "${Z1_ASSET_ROOT}/scripts/build_smolvla_red_ball_scene.py" \
  --urdf "${generated}/z1_with_dual_d435.urdf" \
  --resolved-urdf "${generated}/z1_with_dual_d435_resolved.urdf" \
  --z1-description-root "${Z1_ROS_WS}/src/z1_ros2/z1_description" \
  --robot-usd "${generated}/z1_with_dual_d435.usd" \
  --scene-usd "${generated}/smolvla_dual_wrist_camera.usd" \
  --metadata "${generated}/smolvla_dual_wrist_camera.json"

Z1_SMOKE_SCENE="${generated}/smolvla_dual_wrist_camera.usd" \
Z1_SMOKE_PREVIEW_DIR="${generated}" \
  "${ISAAC_SIM_ROOT}/python.sh" \
  "${Z1_ASSET_ROOT}/scripts/smoke_test_smolvla_scene.py"

printf 'Generated assets rebuilt under %s\n' "${generated}"
