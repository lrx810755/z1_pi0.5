"""Portable path discovery shared by the Z1 SmolVLA utility scripts."""

from __future__ import annotations

import os
from pathlib import Path


def _path_from_env(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else None


def discover_repo_root(start: Path | None = None) -> Path:
    """Find the organized repository without depending on a user home path."""
    configured = _path_from_env("Z1_SMOLVLA_ROOT")
    if configured is not None:
        return configured

    origin = (start or Path(__file__)).resolve()
    for parent in (origin.parent, *origin.parents):
        if (parent / "isaac_assets" / "z1").is_dir() and (
            parent / "ros2_ws" / "src" / "z1_ros2"
        ).is_dir():
            return parent

    raise RuntimeError(
        "Unable to discover the Z1 SmolVLA repository. "
        "Set Z1_SMOLVLA_ROOT to the repository root."
    )


REPO_ROOT = discover_repo_root()
ASSET_ROOT = _path_from_env("Z1_ASSET_ROOT") or REPO_ROOT / "isaac_assets" / "z1"
ROS_WS = _path_from_env("Z1_ROS_WS") or REPO_ROOT / "ros2_ws"
Z1_DESCRIPTION_ROOT = ROS_WS / "src" / "z1_ros2" / "z1_description"
ISAAC_SIM_ROOT = _path_from_env("ISAAC_SIM_ROOT") or Path.home() / "isaac-sim"
DATA_ROOT = _path_from_env("SMOLVLA_DATA_ROOT") or REPO_ROOT / "data" / "runtime"

