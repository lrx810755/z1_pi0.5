"""Shared physical-grasp configuration used by ROS nodes and Isaac Sim."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


def _candidate_paths(explicit_path: Optional[str] = None):
    if explicit_path:
        yield Path(explicit_path)
    environment_path = os.environ.get("Z1_ISAAC_GRASP_CONFIG")
    if environment_path:
        yield Path(environment_path)
    yield Path(__file__).resolve().parents[1] / "config" / "grasp_physics.json"
    try:
        from ament_index_python.packages import get_package_share_path

        yield (
            get_package_share_path("z1_isaac_bridge")
            / "config"
            / "grasp_physics.json"
        )
    except Exception:
        return


def load_grasp_config(explicit_path: Optional[str] = None) -> Dict[str, Any]:
    checked = []
    for path in _candidate_paths(explicit_path):
        checked.append(str(path))
        if path.is_file():
            with path.open("r", encoding="utf-8") as stream:
                return json.load(stream)
    raise FileNotFoundError(
        "grasp_physics.json was not found; checked " + ", ".join(checked)
    )


GRASP_CONFIG = load_grasp_config()
GRASP_ENVELOPE = GRASP_CONFIG["grasp_envelope"]
OBJECT_GRASP_PROFILES = GRASP_CONFIG.get("object_grasp_profiles", {})


def inside_grasp_envelope(
    relative_position: Sequence[float],
    envelope: Optional[Dict[str, float]] = None,
) -> bool:
    limits = GRASP_ENVELOPE if envelope is None else envelope
    if len(relative_position) != 3:
        return False
    local_x, local_y, local_z = (float(value) for value in relative_position)
    return (
        float(limits["x_min_m"]) <= local_x <= float(limits["x_max_m"])
        and abs(local_y) <= float(limits["y_abs_max_m"])
        and float(limits["z_min_m"]) <= local_z <= float(limits["z_max_m"])
    )


def target_bilateral_contact(
    status: Dict[str, Any],
    target_object: str,
    *,
    require_sustained: bool = False,
) -> bool:
    """Derive real contact from the atomic left/right target pair."""
    pair_matches = (
        bool(status.get("contact_report_available", False))
        and status.get("left_finger_contact_object") == target_object
        and status.get("right_finger_contact_object") == target_object
    )
    if not pair_matches or not require_sustained:
        return pair_matches
    required_frames = max(
        1,
        int(status.get("bilateral_contact_required_frames", 1)),
    )
    return (
        int(status.get("bilateral_contact_consecutive_frames", 0))
        >= required_frames
    )


def cylinder_side_contact_is_valid(
    status: Dict[str, Any], target_object: str, profile: Dict[str, Any]
) -> bool:
    """Validate that both fingers load the configured cylinder body band."""
    if not target_bilateral_contact(status, target_object):
        return False
    if not cylinder_target_contact_heights_are_valid(
        status, target_object, profile, require_bilateral=True
    ):
        return False
    impulses = [
        float(status.get("left_finger_contact_impulse_ns", 0.0)),
        float(status.get("right_finger_contact_impulse_ns", 0.0)),
    ]
    if min(impulses) <= 0.0:
        return False
    impulse_ratio = min(impulses) / max(impulses)
    return impulse_ratio >= float(
        profile.get("minimum_contact_impulse_ratio", 0.0)
    )


def cylinder_target_contact_heights_are_valid(
    status: Dict[str, Any],
    target_object: str,
    profile: Dict[str, Any],
    *,
    require_bilateral: bool = False,
) -> bool:
    """Reject cap/base contact as soon as either cylinder finger touches."""
    height_range = profile.get("grasp_height_range_m", ())
    if len(height_range) != 2:
        return False
    lower, upper = (float(value) for value in height_range)
    target_contacts = 0
    for side in ("left", "right"):
        if status.get(f"{side}_finger_contact_object") != target_object:
            if require_bilateral:
                return False
            continue
        target_contacts += 1
        point = status.get(f"{side}_finger_contact_point_object_m")
        if not isinstance(point, (list, tuple)) or len(point) != 3:
            return False
        if not lower <= float(point[2]) <= upper:
            return False
    return target_contacts == 2 if require_bilateral else target_contacts > 0


def object_loaded_grasp_is_valid(
    status: Dict[str, Any], target_object: str
) -> bool:
    """Dispatch physical contact validation by object geometry."""
    profile = OBJECT_GRASP_PROFILES.get(target_object, {})
    if profile.get("strategy") == "cylinder_side_pinch":
        return cylinder_side_contact_is_valid(status, target_object, profile)
    return target_bilateral_contact(status, target_object)
