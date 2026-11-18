"""Portable filesystem locations shared by Z1 ROS-side tools."""

from __future__ import annotations

import os
from pathlib import Path


def _resolved(value: os.PathLike[str] | str) -> Path:
    return Path(value).expanduser().resolve(strict=False)


def _looks_like_repository(path: Path) -> bool:
    return (
        (path / "z1_isaac_bridge").is_dir()
        and (path / "isaac_assets" / "z1").is_dir()
    )


def repository_root() -> Path:
    override = os.environ.get("Z1_REPO_ROOT")
    if override:
        return _resolved(override)

    module_path = _resolved(__file__)
    source_candidate = module_path.parents[2]
    if _looks_like_repository(source_candidate):
        return source_candidate

    for ancestor in module_path.parents:
        if ancestor.name not in {"build", "install"}:
            continue
        source_space = ancestor.parent / "src"
        if source_space.is_dir():
            for candidate in sorted(source_space.iterdir()):
                if candidate.is_dir() and _looks_like_repository(candidate):
                    return _resolved(candidate)

    working_directory = _resolved(Path.cwd())
    for candidate in (working_directory, *working_directory.parents):
        if _looks_like_repository(candidate):
            return candidate

    return source_candidate


def asset_root() -> Path:
    override = os.environ.get("Z1_ASSET_ROOT")
    if override:
        return _resolved(override)
    return _resolved(repository_root() / "isaac_assets" / "z1")


def isaac_sim_root() -> Path:
    override = os.environ.get("ISAAC_SIM_ROOT")
    return _resolved(override or Path.home() / "isaac-sim")


def dataset_root(name: str = "") -> Path:
    base = os.environ.get("Z1_DATASET_ROOT")
    root = _resolved(base or Path.home() / "smolvla_data")
    relative_name = str(name).lstrip("/\\")
    return _resolved(root / relative_name) if relative_name else root


def servo_prefix() -> Path:
    override = os.environ.get("Z1_SERVO_PREFIX")
    if override:
        return _resolved(override)
    system = Path('/opt/ros/humble')
    local = Path.home() / '.local/opt/ros-humble-moveit-servo/opt/ros/humble'
    binary = 'lib/moveit_servo/servo_node_main'
    if not (system / binary).is_file() and (local / binary).is_file():
        return _resolved(local)
    return _resolved(system)
