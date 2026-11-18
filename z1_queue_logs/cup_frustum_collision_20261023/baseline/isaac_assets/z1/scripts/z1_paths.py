"""Portable filesystem locations for standalone Isaac-side scripts."""

from __future__ import annotations

import os
from pathlib import Path


def _resolved(value: os.PathLike[str] | str) -> Path:
    return Path(value).expanduser().resolve(strict=False)


def asset_root() -> Path:
    override = os.environ.get("Z1_ASSET_ROOT")
    if override:
        return _resolved(override)
    return _resolved(Path(__file__).parents[1])


def repository_root() -> Path:
    override = os.environ.get("Z1_REPO_ROOT")
    if override:
        return _resolved(override)
    return _resolved(asset_root().parents[1])


def isaac_sim_root() -> Path:
    override = os.environ.get("ISAAC_SIM_ROOT")
    return _resolved(override or Path.home() / "isaac-sim")


def dataset_root(name: str = "") -> Path:
    override = os.environ.get("Z1_DATASET_ROOT")
    root = _resolved(override or Path.home() / "smolvla_data")
    relative_name = str(name).lstrip("/\\")
    return _resolved(root / relative_name) if relative_name else root


def unitree_root() -> Path:
    override = os.environ.get("Z1_UNITREE_ROOT")
    return _resolved(override or Path.home() / "unitree_z1")
