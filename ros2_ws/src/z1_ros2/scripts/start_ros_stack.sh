#!/usr/bin/env bash
set -euo pipefail

z1_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=configure_environment.sh
source "$z1_script_dir/configure_environment.sh"

# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
z1_workspace_root="$(cd "$Z1_REPO_ROOT/../.." && pwd -P)"
if [[ ! -f "$z1_workspace_root/install/setup.bash" ]]; then
  echo "Workspace is not built: $z1_workspace_root/install/setup.bash is missing." >&2
  exit 2
fi
# shellcheck disable=SC1091
source "$z1_workspace_root/install/setup.bash"

export FASTRTPS_DEFAULT_PROFILES_FILE="$Z1_REPO_ROOT/z1_isaac_bridge/config/fastdds_wsl2.xml"
exec ros2 launch z1_isaac_bridge z1_isaac.launch.py \
  start_isaac:=false "$@"
