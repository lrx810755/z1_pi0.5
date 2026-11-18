#!/usr/bin/env bash
set -euo pipefail

z1_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=configure_environment.sh
source "$z1_script_dir/configure_environment.sh"

if [[ ! -f /opt/ros/humble/setup.bash ]]; then
  echo "ROS 2 Humble is required at /opt/ros/humble." >&2
  echo "Install it from the official ROS 2 Ubuntu 22.04 packages first." >&2
  exit 2
fi

sudo apt-get update
sudo apt-get install -y \
  git-lfs \
  python3-colcon-common-extensions \
  python3-pip \
  python3-rosdep

git -C "$Z1_REPO_ROOT" lfs install --local
git -C "$Z1_REPO_ROOT" lfs pull

# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
rosdep update
rosdep install \
  --from-paths "$Z1_REPO_ROOT" \
  --ignore-src \
  --rosdistro humble \
  --skip-keys ament_python \
  -r -y

z1_workspace_root="$(cd "$Z1_REPO_ROOT/../.." && pwd -P)"
if [[ "$(basename "$(dirname "$Z1_REPO_ROOT")")" != "src" ]]; then
  echo "Clone this repository under a ROS workspace src/ directory." >&2
  exit 2
fi

cd "$z1_workspace_root"
PYTHONNOUSERSITE=1 colcon build --symlink-install

echo "Bootstrap complete. Source $z1_workspace_root/install/setup.bash"
