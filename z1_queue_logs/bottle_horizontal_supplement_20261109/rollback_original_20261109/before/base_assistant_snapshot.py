"""Plan and execute collision-aware poses for automatic demonstrations."""

from __future__ import annotations

import copy
import itertools
import json
import math
import random
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose, PoseStamped, TwistStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    DisplayTrajectory,
    MoveItErrorCodes,
    OrientationConstraint,
    PositionConstraint,
    RobotTrajectory,
)
from moveit_msgs.srv import GetPositionIK
import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger
from tf2_msgs.msg import TFMessage
from trajectory_msgs.msg import JointTrajectoryPoint
import tf2_ros

from .grasp_evaluator import (
    OBJECT_FRAME_NAMES,
    canonical_object_name,
    quaternion_conjugate,
    quaternion_multiply,
    rotate_vector,
)
from .grasp_config import (
    GRASP_CONFIG,
    GRASP_ENVELOPE,
    OBJECT_GRASP_PROFILES,
    cylinder_contact_quality_metrics,
    cylinder_target_contact_heights_are_valid,
    inside_grasp_envelope,
    object_loaded_grasp_is_valid,
    target_bilateral_contact,
)


Vector3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]

BOTTLE_RECOVERY_VARIANTS = ("baseline", "left", "right", "high")


def _bottle_recovery_pregrasp_position(
    position: Vector3, azimuth_deg: float, variant: str
) -> Vector3:
    """Perturb only the waypoint before the calibrated bottle grasp pose."""
    if variant not in BOTTLE_RECOVERY_VARIANTS:
        raise ValueError(f"未知瓶子纠偏示范模式：{variant}")
    if variant == "baseline":
        return tuple(position)
    if variant == "high":
        return (position[0], position[1], position[2] + 0.006)
    azimuth = math.radians(azimuth_deg)
    lateral = 0.008 if variant == "left" else -0.008
    return (
        position[0] - lateral * math.sin(azimuth),
        position[1] + lateral * math.cos(azimuth),
        position[2],
    )

BOX_RELEASE_XY_TOLERANCE_M = {
    "apple": 0.025,
    "peach": 0.025,
    "cup": 0.020,
    "beverage_bottle": 0.020,
}

BOX_TRANSPORT_XY_TOLERANCE_M = {
    "apple": 0.055,
    "peach": 0.055,
    "cup": 0.055,
}

# Bounded lateral alignment above the box walls.  Every correction is a short
# straight-line step with a tight orientation tolerance, the measured error is
# refreshed after each step, and the loop stops as soon as the error enters the
# final release acceptance band.  A free-space plan with loose orientation
# freedom reaches the same waypoint from an unrelated wrist branch, which looks
# like the arm flailing and adds a fresh planning failure point per retry.
BOX_ALIGNMENT_STEP_M = 0.008
BOX_ALIGNMENT_MAX_STEPS = 3
BOX_ALIGNMENT_ORIENTATION_TOLERANCE_RAD = 0.025
BOX_ALIGNMENT_MIN_PROGRESS_M = 0.001
# Above this object-centre height, a failed descent must not turn into a
# long uncontrolled drop.  The 225 mm ceiling includes the apple lift target.
BOX_FALLBACK_RELEASE_MAX_OBJECT_Z_M = 0.225
CUP_DIRECT_RELEASE_MIN_ROOT_Z_M = 0.115


def _validate_cup_direct_box_release(position, box_center, xy_tolerance):
    """Cup direct release uses a fresh world-root pose, strictly above115mm."""
    if len(position) != 3 or not all(math.isfinite(float(v)) for v in position):
        raise RuntimeError("杯子直接释放前世界位置无效")
    xy_error = math.hypot(position[0] - box_center[0], position[1] - box_center[1])
    if xy_error > xy_tolerance:
        raise RuntimeError(f"杯子直接释放前水平对中超限：{xy_error:.4f} m > {xy_tolerance:.4f} m")
    if position[2] <= CUP_DIRECT_RELEASE_MIN_ROOT_Z_M:
        raise RuntimeError(f"杯子直接释放前高度不足：杯底 z={position[2]:.6f} m，必须严格大于0.115 m")




def _ee_position_for_object_in_stator(
    object_position: Vector3,
    ee_orientation: Quaternion,
    ee_to_stator_translation: Vector3,
    ee_to_stator_orientation: Quaternion,
    object_in_stator: Vector3,
) -> Vector3:
    """Solve the link pose that realizes an object-in-stator calibration."""
    stator_orientation = quaternion_multiply(
        ee_orientation, ee_to_stator_orientation
    )
    ee_to_object = tuple(
        a + b
        for a, b in zip(
            rotate_vector(ee_orientation, ee_to_stator_translation),
            rotate_vector(stator_orientation, object_in_stator),
        )
    )
    return tuple(
        float(object_position[index]) - ee_to_object[index]
        for index in range(3)
    )


def _object_in_stator_after_world_tool_offset(
    object_in_stator: Vector3,
    stator_orientation: Quaternion,
    world_tool_offset: Vector3,
) -> Vector3:
    """Update the local object reference after translating the tool in world."""
    local_tool_offset = rotate_vector(
        quaternion_conjugate(stator_orientation), world_tool_offset
    )
    return tuple(
        float(object_in_stator[index]) - local_tool_offset[index]
        for index in range(3)
    )


def _bottle_reference_for_orientation(
    captured_local: Vector3,
    orientation: Quaternion,
    captured_world_vertical_offset_m: float,
) -> Vector3:
    """Preserve the taught bottle-to-stator height across approach pitches."""
    fixed_world_z = rotate_vector(
        orientation,
        (float(captured_local[0]), 0.0, float(captured_local[2])),
    )[2]
    local_y_world_z = rotate_vector(orientation, (0.0, 1.0, 0.0))[2]
    if abs(local_y_world_z) < 0.1:
        raise ValueError("Bottle tool local Y is unsuitable for height correction")
    adjusted_y = (
        float(captured_world_vertical_offset_m) - fixed_world_z
    ) / local_y_world_z
    return (
        float(captured_local[0]),
        float(adjusted_y),
        float(captured_local[2]),
    )


def _bilateral_obstruction_evidence(
    status: dict, target_object: str, minimum_impulse_ratio: float
) -> bool:
    """Accept balanced bilateral object contact, never table contact."""
    if not bool(status.get("contact_report_available", False)):
        return False
    if (
        status.get("left_finger_contact_object") != target_object
        or status.get("right_finger_contact_object") != target_object
    ):
        return False
    if bool(status.get("left_finger_touching_table", False)) or bool(
        status.get("right_finger_touching_table", False)
    ):
        return False
    left = float(status.get("left_finger_contact_impulse_ns", 0.0))
    right = float(status.get("right_finger_contact_impulse_ns", 0.0))
    if left <= 0.0 or right <= 0.0:
        return False
    return min(left, right) / max(left, right) >= float(minimum_impulse_ratio)


def _bilateral_preload_entry_is_valid(status: dict, target_object: str) -> bool:
    """Allow one bounded preload after both real fingers begin carrying load."""
    return (
        bool(status.get("contact_report_available", False))
        and status.get("left_finger_contact_object") == target_object
        and status.get("right_finger_contact_object") == target_object
        and not bool(status.get("left_finger_touching_table", False))
        and not bool(status.get("right_finger_touching_table", False))
        and float(status.get("left_finger_contact_impulse_ns", 0.0)) > 0.0
        and float(status.get("right_finger_contact_impulse_ns", 0.0)) > 0.0
    )


def _safe_bilateral_preload_retry(
    status: dict,
    target_object: str,
    envelope: dict,
    margin: float,
) -> bool:
    """Allow one tighter hold only for a real, table-clear loaded pinch."""
    local_position = tuple(
        float(value) for value in status.get("local_position_m", [])
    )
    inside_loaded_envelope = (
        len(local_position) == 3
        and float(envelope["x_min_m"]) - margin
        <= local_position[0]
        <= float(envelope["x_max_m"]) + margin
        and abs(local_position[1])
        <= float(envelope["y_abs_max_m"]) + margin
        and float(envelope["z_min_m"]) - margin
        <= local_position[2]
        <= float(envelope["z_max_m"]) + margin
    )
    colliders_are_real_meshes = all(
        "/collisions" in str(
            status.get(f"{side}_finger_contact_collider", "")
        )
        for side in ("left", "right")
    )
    return (
        bool(status.get("contact_report_available", False))
        and status.get("left_finger_contact_object") == target_object
        and status.get("right_finger_contact_object") == target_object
        and not bool(status.get("left_finger_touching_table", False))
        and not bool(status.get("right_finger_touching_table", False))
        and float(status.get("left_finger_contact_impulse_ns", 0.0)) > 0.0
        and float(status.get("right_finger_contact_impulse_ns", 0.0)) > 0.0
        and colliders_are_real_meshes
        and inside_loaded_envelope
    )


def _bottle_candidate_pitch_deg(
    index: int, count: int, maximum_deg: float, minimum_deg: float
) -> float:
    """Deterministically cover the configured bottle pitch interval."""
    upper = float(max(maximum_deg, minimum_deg))
    lower = float(min(maximum_deg, minimum_deg))
    if count <= 1 or math.isclose(upper, lower):
        return upper
    fractions = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0)
    if count == len(fractions):
        fraction = fractions[index]
    else:
        fraction = float(index) / float(count - 1)
    return upper + (lower - upper) * fraction


def _bottle_candidate_pose_variant(
    index: int,
    count: int,
    maximum_pitch_deg: float,
    minimum_pitch_deg: float,
    base_roll_deg: float,
    mirror_wrist_roll: bool,
) -> Tuple[float, float]:
    """Allocate bottle pitches across the primary and mirrored wrist IK branches."""
    if not mirror_wrist_roll or count < 2:
        return (
            _bottle_candidate_pitch_deg(
                index, count, maximum_pitch_deg, minimum_pitch_deg
            ),
            float(base_roll_deg),
        )
    candidates_per_branch = max(1, count // 2)
    mirrored = index >= candidates_per_branch
    branch_index = index % candidates_per_branch
    return (
        _bottle_candidate_pitch_deg(
            branch_index,
            candidates_per_branch,
            maximum_pitch_deg,
            minimum_pitch_deg,
        ),
        -float(base_roll_deg) if mirrored else float(base_roll_deg),
    )


def _cylinder_reference_at_grasp_height(physical_profile: dict) -> dict:
    """Translate a measured reference when an object's grasp height changes.

    The stored center-minus-stator world Z describes the reference height.
    Raising the tool decreases that signed offset by the same amount. Keep
    the original measurement intact so repeated configuration loads cannot
    accumulate an offset.
    """
    reference = copy.deepcopy(physical_profile["calibrated_grasp_reference"])
    reference_height = reference.get("reference_grasp_height_m")
    if reference_height is not None:
        delta = float(physical_profile["grasp_height_target_m"]) - float(reference_height)
        reference["captured_object_center_minus_stator_world_z_m"] -= delta
    return reference


def _cylinder_candidate_local_reference(
    reference: Sequence[float],
    candidate_index: int,
    offsets_yz: Sequence[Sequence[float]] = (),
) -> Vector3:
    """Apply a configured local Y/Z candidate offset."""
    if not offsets_yz:
        return tuple(float(value) for value in reference)
    y_offset, z_offset = offsets_yz[candidate_index % len(offsets_yz)]
    return (
        float(reference[0]),
        float(reference[1]) + y_offset,
        float(reference[2]) + z_offset,
    )


def _candidate_azimuth_deg(
    base_azimuth_deg: float,
    candidate_index: int,
    offsets_deg: Sequence[float] = (),
    extra_world_azimuths_deg: Sequence[float] = (),
) -> float:
    extra_index = candidate_index - len(offsets_deg)
    if 0 <= extra_index < len(extra_world_azimuths_deg):
        return float(extra_world_azimuths_deg[extra_index])
    if not offsets_deg:
        return float(base_azimuth_deg)
    return float(base_azimuth_deg) + float(
        offsets_deg[candidate_index % len(offsets_deg)]
    )

GRASP_PROFILES = {
    "cup": {"height": float(GRASP_CONFIG["object_center_z_offsets_m"]["cup"]), "world_z_offset_m": 0.0, "elevation_deg": (25.0, 25.0), "standoff": 0.190, "pregrasp_m": (0.27, 0.27), "alignment_standoff_m": 0.27, "azimuth_spread_deg": 0.0, "roll_deg": 120.0, "max_joint_delta_rad": 2.70, "outer_alignment_max_joint_delta_rad": 0.65, "outer_alignment_min_effective_y_step_m": 0.015, "insertion_orientation_tolerance_rad": 0.008, "alignment_tolerance_y_m": 0.002, "insertion_alignment_tolerance_y_m": 0.008, "alignment_tolerance_z_m": 0.004, "insertion_x_tolerance_m": 0.003, "minimum_insertion_distance_m": 0.005, "maximum_insertion_distance_m": 0.12, "insertion_center_x_m": 0.152, "dynamic_insertion_stop_x_m": 0.152, "dynamic_insertion_step_m": 0.005, "decouple_dynamic_alignment": True, "disable_preclose_live_centering": True, "allow_preclose_target_contact": True, "preclose_clearance_lift_m": 0.0, "dynamic_lateral_step_m": 0.015, "minimum_effective_y_step_m": 0.015, "dynamic_alignment_corridor_m": 0.018, "desired_center_x_m": 0.152, "desired_center_y_m": 0.0, "entry_center_z_m": 0.017, "desired_center_z_m": 0.017, "preclose_contact_center_z_m": 0.024, "minimum_object_rise_m": 0.200, "gripper_rad": 0.0},
    # Round fruit must enter the open jaws from the side.  The former steep
    # top-down approach contacted the palm/finger exterior and pushed fruit
    # across the table once real collisions were enabled.
    # link06 is 51 mm behind the gripperStator origin and the fingertip ends
    # at stator-local x=149 mm.  A 181 mm standoff requests the fruit center
    # near x=130 mm; the measured execution residual leaves it around 140 mm,
    # still behind the fingertip plane and inside both finger bodies.  The steep approach keeps the
    # fingers above the table; zero roll makes the moving finger open upward
    # rather than sweeping down into the tabletop.
    "apple": {"height": float(GRASP_CONFIG["object_center_z_offsets_m"]["apple"]), "world_z_offset_m": 0.0, "elevation_deg": (70.0, 90.0), "standoff": 0.181, "pregrasp_m": (0.27, 0.30), "azimuth_spread_deg": 0.0, "roll_deg": 0.0, "max_joint_delta_rad": 2.5, "insertion_center_x_m": 0.141, "dynamic_insertion_stop_x_m": 0.152, "dynamic_insertion_step_m": 0.002, "dynamic_lateral_step_m": 0.006, "minimum_effective_y_step_m": 0.005, "dynamic_alignment_corridor_m": 0.010, "insertion_x_tolerance_m": 0.003, "desired_center_x_m": 0.142, "desired_center_y_m": 0.0, "entry_center_z_m": 0.017, "desired_center_z_m": 0.016, "preclose_contact_center_z_m": 0.024, "gripper_rad": 0.0},
    "peach": {"height": float(GRASP_CONFIG["object_center_z_offsets_m"]["peach"]), "world_z_offset_m": 0.0, "elevation_deg": (70.0, 90.0), "standoff": 0.181, "pregrasp_m": (0.27, 0.30), "azimuth_spread_deg": 0.0, "roll_deg": 0.0, "max_joint_delta_rad": 2.5, "outer_alignment_iterations": 8, "decouple_outer_alignment": True, "decouple_dynamic_alignment": True, "outer_alignment_min_effective_y_step_m": 0.0, "alignment_tolerance_y_m": 0.003, "alignment_tolerance_z_m": 0.003, "insertion_alignment_tolerance_y_m": 0.004, "insertion_alignment_tolerance_z_m": 0.0045, "insertion_x_tolerance_m": 0.003, "minimum_insertion_distance_m": 0.005, "maximum_insertion_distance_m": 0.12, "grasp_envelope_overrides": {"x_max_m": 0.165}, "insertion_center_x_m": 0.150, "dynamic_insertion_stop_x_m": 0.155, "dynamic_insertion_step_m": 0.002, "dynamic_lateral_step_m": 0.005, "minimum_effective_y_step_m": 0.005, "dynamic_alignment_corridor_m": 0.010, "guard_dynamic_alignment_depth": True, "dynamic_alignment_centering_depth_guard_m": 0.006, "dynamic_alignment_depth_recovery_max_step_m": 0.008, "dynamic_alignment_depth_recovery_min_improvement_m": 0.001, "dynamic_alignment_safe_pose_recovery_attempts": 2, "dynamic_alignment_safe_pose_max_joint_delta_rad": 0.65, "dynamic_alignment_world_z_floor_margin_m": 0.003, "desired_center_x_m": 0.150, "desired_center_y_m": 0.0, "entry_center_z_m": 0.015, "desired_center_z_m": 0.015, "preclose_contact_center_z_m": 0.021, "allow_preclose_target_contact": False, "recover_unilateral_preclose_contact": True, "preclose_contact_recovery_retreat_m": 0.006, "preclose_contact_recovery_lift_m": 0.005, "preclose_contact_recovery_attempts": 2, "disable_preclose_live_centering": False, "preclose_clearance_lift_m": 0.005, "prelift_axial_seating_steps": 0, "transport_object_z_m": 0.180, "physical_test_lift_completion_tolerance_m": 0.010, "physical_test_lift_segment_m": 0.020, "physical_test_lift_orientation_tolerances_rad": (0.04, 0.08, 0.12), "physical_test_lift_bounded_fallback_max_joint_delta_rad": 0.35, "gripper_rad": 0.0},
    "mouse": {"height": 0.020, "world_z_offset_m": 0.0, "elevation_deg": (75.0, 75.0), "standoff": 0.181, "pregrasp_m": (0.27, 0.30), "alignment_standoff_m": 0.27, "azimuth_offset_deg": 0.0, "azimuth_spread_deg": 0.0, "roll_deg": 0.0, "max_joint_delta_rad": 2.70, "outer_alignment_max_joint_delta_rad": 0.65, "outer_alignment_iterations": 8, "decouple_outer_alignment": True, "pregrasp_orientation_tolerance_rad": 0.020, "alignment_tolerance_y_m": 0.004, "insertion_alignment_tolerance_y_m": 0.008, "alignment_tolerance_z_m": 0.003, "insertion_alignment_tolerance_z_m": 0.006, "insertion_x_tolerance_m": 0.003, "minimum_insertion_distance_m": 0.005, "maximum_insertion_distance_m": 0.12, "grasp_envelope_overrides": {"x_max_m": 0.175}, "insertion_center_x_m": 0.168, "dynamic_insertion_stop_x_m": 0.168, "dynamic_insertion_step_m": 0.002, "decouple_dynamic_alignment": True, "allow_preclose_target_contact": True, "disable_preclose_live_centering": True, "preclose_clearance_lift_m": 0.0, "dynamic_lateral_step_m": 0.008, "dynamic_alignment_corridor_m": 0.012, "desired_center_x_m": 0.168, "desired_center_y_m": 0.0, "entry_center_z_m": 0.009, "desired_center_z_m": 0.009, "preclose_contact_center_z_m": 0.018, "prelift_axial_seating_steps": 3, "minimum_object_rise_m": 0.200, "physical_test_lift_segment_m": 0.005, "gripper_rad": 0.0},
    "beverage_bottle": {"height": float(OBJECT_GRASP_PROFILES["beverage_bottle"]["grasp_height_target_m"]), "world_z_offset_m": 0.0, "pitch_deg": float(OBJECT_GRASP_PROFILES["beverage_bottle"]["approach_pitch_deg"]), "elevation_deg": (float(OBJECT_GRASP_PROFILES["beverage_bottle"]["approach_elevation_deg"]), float(OBJECT_GRASP_PROFILES["beverage_bottle"]["approach_elevation_deg"])), "standoff": 0.195, "pregrasp_m": (0.27, 0.30), "alignment_standoff_m": 0.27, "azimuth_spread_deg": 0.0, "roll_deg": -90.0, "insertion_center_x_m": 0.150, "dynamic_insertion_stop_x_m": 0.154, "dynamic_insertion_step_m": 0.002, "dynamic_lateral_step_m": 0.004, "dynamic_alignment_corridor_m": 0.012, "desired_center_x_m": 0.150, "desired_center_y_m": 0.0, "entry_center_z_m": 0.022, "desired_center_z_m": 0.020, "preclose_contact_center_z_m": 0.024, "allow_preclose_target_contact": True, "disable_preclose_live_centering": True, "preclose_clearance_lift_m": 0.0, "minimum_object_rise_m": float(OBJECT_GRASP_PROFILES["beverage_bottle"]["minimum_proof_rise_m"]), "transport_object_z_m": float(OBJECT_GRASP_PROFILES["beverage_bottle"]["transport_object_z_m"]), "physical_test_lift_safe_completion_z_m": float(OBJECT_GRASP_PROFILES["beverage_bottle"]["physical_test_lift_safe_completion_z_m"]), "physical_test_lift_completion_tolerance_m": float(OBJECT_GRASP_PROFILES["beverage_bottle"]["physical_test_lift_completion_tolerance_m"]), "physical_test_lift_segment_m": float(OBJECT_GRASP_PROFILES["beverage_bottle"]["physical_test_lift_segment_m"]), "physical_test_lift_near_table_segment_m": float(OBJECT_GRASP_PROFILES["beverage_bottle"]["physical_test_lift_near_table_segment_m"]), "physical_test_lift_near_table_until_z_m": float(OBJECT_GRASP_PROFILES["beverage_bottle"]["physical_test_lift_near_table_until_z_m"]), "physical_test_lift_min_retry_segment_m": float(OBJECT_GRASP_PROFILES["beverage_bottle"]["physical_test_lift_min_retry_segment_m"]), "physical_test_lift_min_progress_m": float(OBJECT_GRASP_PROFILES["beverage_bottle"]["physical_test_lift_min_progress_m"]), "physical_test_lift_max_stalled_iterations": int(OBJECT_GRASP_PROFILES["beverage_bottle"]["physical_test_lift_max_stalled_iterations"]), "physical_test_lift_orientation_tolerances_rad": tuple(float(value) for value in OBJECT_GRASP_PROFILES["beverage_bottle"]["physical_test_lift_orientation_tolerances_rad"]), "physical_test_lift_bounded_fallback_max_joint_delta_rad": float(OBJECT_GRASP_PROFILES["beverage_bottle"]["physical_test_lift_bounded_fallback_max_joint_delta_rad"]), "gripper_contact_closed_limit_rad": float(OBJECT_GRASP_PROFILES["beverage_bottle"]["gripper_contact_angle_range_rad"][0]), "gripper_contact_open_limit_rad": float(OBJECT_GRASP_PROFILES["beverage_bottle"]["gripper_contact_angle_range_rad"][1]), "maximum_close_displacement_m": float(OBJECT_GRASP_PROFILES["beverage_bottle"]["maximum_close_displacement_m"]), "loaded_samples_required": int(OBJECT_GRASP_PROFILES["beverage_bottle"]["loaded_samples_required"]), "gripper_rad": 0.0},
}

# MoveIt remains responsible for collision-checked motion, while Isaac's
# measured object pose closes the final Cartesian loop.  These controller
# settings are shared; only geometry and acceptable residuals vary by object.
_CLOSED_LOOP_ALIGNMENT_DEFAULTS = {
    "outer_alignment_iterations": 8,
    "outer_alignment_gain_y": 0.85,
    "outer_alignment_gain_z": 0.85,
    "outer_alignment_max_y_step_m": 0.008,
    "outer_alignment_max_z_step_m": 0.012,
    "outer_alignment_moveit_position_tolerance_m": 0.001,
    "dynamic_alignment_gain_y": 0.75,
    "dynamic_alignment_gain_z": 0.75,
    "dynamic_alignment_abort_corridor_m": 0.030,
    "dynamic_alignment_correction_budget": 12,
    "align_before_axial_advance": True,
    "final_alignment_iterations": 6,
    "final_alignment_gain": 0.80,
    "final_alignment_max_step_m": 0.003,
    "final_alignment_moveit_position_tolerance_m": 0.001,
}
_OBJECT_ALIGNMENT_GEOMETRY = {
    "cup": {
        "outer_alignment_acceptance_y_m": 0.006,
        "outer_alignment_acceptance_z_m": 0.006,
        "insertion_alignment_tolerance_y_m": 0.008,
        "insertion_alignment_tolerance_z_m": 0.006,
    },
    "apple": {
        "single_position_grasp": True,
        "candidate_azimuth_offsets_deg": (0.0, -15.0, 15.0, -30.0, 30.0),
        "candidate_count": 16,
        "fallback_roll_start_index": 8,
        "fallback_roll_deg": 90.0,
        "accept_first_valid_candidate": True,
        # Preserve the full physical X envelope despite the measured
        # millimetre-scale one-shot trajectory residual near its lower edge.
        # The separate [142, 164] mm pre-close envelope remains authoritative.
        # The 15 mm world-Z clearance lift projects 1.1--2.7 mm into
        # stator-local +Y over the 80--85 degree full-workspace candidates.
        # Keep the physical pre-close envelope at |Y| <= 7 mm, while making
        # this target-relative check wide enough not to remove its negative
        # half solely because of that expected coordinate projection.
        "single_alignment_tolerance_m": (0.015, 0.010, 0.007),
        # A closed fruit can settle just outside the pre-close Y envelope
        # while real loaded mesh contact persists.  Let the physical lift
        # prove only this narrow post-close boundary case.
        "post_close_y_margin_m": 0.001,
        # The final Cartesian target is derived from an exact stator-local
        # object offset.  A loose angular solution rotates that 141 mm offset
        # into centimetres of lateral error even when link06 reaches its
        # requested world position.
        "single_position_orientation_tolerance_rad": 0.015,
        "single_position_world_clearance_lift_m": 0.015,
        "grasp_envelope_overrides": {
            "x_min_m": 0.142,
            "x_max_m": 0.164,
            "y_abs_max_m": 0.007,
            "z_min_m": 0.0,
            "z_max_m": 0.021,
        },
        # Make the approved candidate end at the collision-checked outer
        # alignment pose.  The old 270-300 mm waypoint required a second
        # fixed-orientation Pilz LIN move to 245 mm after execution; that
        # redundant segment repeatedly crossed the wrist IK boundary.
        "plan_directly_to_alignment_pose": True,
        "alignment_standoff_m": 0.245,
        "alignment_tolerance_y_m": 0.002,
        "alignment_tolerance_z_m": 0.002,
        "outer_alignment_acceptance_y_m": 0.0035,
        "outer_alignment_acceptance_z_m": 0.0035,
        "insertion_alignment_tolerance_y_m": 0.004,
        "insertion_alignment_tolerance_z_m": 0.004,
        # Merge free-space progress, retaining fine steps near contact and
        # bounded subdivision if the wrist approaches an IK branch boundary.
        "coarse_insertion_step_m": 0.010,
        "fine_insertion_distance_m": 0.015,
        "fine_insertion_step_m": 0.002,
        # A 50 mm apple with its centre at 130 mm already leaves about
        # 10 mm below the fruit above the 100 mm box walls.  Treat 140 mm as
        # the preferred transfer height, not a reason to chase the last few
        # millimetres at the wrist IK boundary.
        "physical_test_lift_safe_completion_z_m": 0.125,
        "physical_test_lift_min_progress_m": 0.001,
        "physical_test_lift_max_stalled_iterations": 2,
        # Stop finite preload once real bilateral contact is sustained.  The
        # 150 mrad value remains a hard upper bound for difficult grasps.
        "contact_hold_minimum_preload_rad": 0.050,
        "physical_test_lift_segment_m": 0.060,
        "physical_test_lift_near_table_segment_m": 0.020,
        "physical_test_lift_near_table_until_z_m": 0.040,
        "physical_test_lift_min_retry_segment_m": 0.005,
        "physical_test_lift_orientation_tolerances_rad": (
            0.04,
            0.08,
            0.12,
        ),
        "physical_test_lift_bounded_fallback_max_joint_delta_rad": 0.35,
    },
    "peach": {
        "single_position_grasp": True,
        "candidate_azimuth_offsets_deg": (0.0, -15.0, 15.0, -30.0, 30.0),
        "candidate_count": 16,
        "fallback_roll_start_index": 8,
        "fallback_roll_deg": 90.0,
        "accept_first_valid_candidate": True,
        "single_alignment_tolerance_m": (0.007, 0.007, 0.004),
        "single_position_orientation_tolerance_rad": 0.015,
        "single_position_world_clearance_lift_m": 0.005,
        "plan_directly_to_alignment_pose": True,
        "alignment_standoff_m": 0.245,
        "outer_alignment_acceptance_y_m": 0.005,
        "outer_alignment_acceptance_z_m": 0.005,
        # Use the same staged progress and useful-clearance completion as
        # apple without changing peach's contact recovery, depth guard or
        # final alignment geometry.  Both fruits clear the 100 mm box walls
        # once their centre reaches 130 mm; continuing toward the preferred
        # transport pose only adds motion near the wrist IK boundary.
        "coarse_insertion_step_m": 0.010,
        "fine_insertion_distance_m": 0.015,
        "fine_insertion_step_m": 0.002,
        "physical_test_lift_safe_completion_z_m": 0.130,
        "physical_test_lift_min_progress_m": 0.001,
        "physical_test_lift_max_stalled_iterations": 2,
        # Stop adding preload once sustained real bilateral contact has
        # reached the same useful minimum used for apple.
        "contact_hold_minimum_preload_rad": 0.050,
        "physical_test_lift_segment_m": 0.060,
        "physical_test_lift_near_table_segment_m": 0.020,
        "physical_test_lift_near_table_until_z_m": 0.040,
        "physical_test_lift_min_retry_segment_m": 0.005,
    },
    "mouse": {
        # The scan is widest about 20 mm above the table.  Aim the steep
        # approach at that slice, then finish stator-local Z before allowing
        # lateral centering or axial insertion.  This keeps the thin mouse
        # body between the finger faces instead of pinching a tapered edge.
        "height": 0.020,
        # User-selected baseline after the 2026-09-10 angle trials:
        # 75 degrees downward from horizontal (15 degrees from vertical).
        # Keep both bounds equal; 65/85 were temporary trial overrides only.
        # This fixes the approach angle, not the observed finger/table contact.
        "elevation_deg": (75.0, 75.0),
        "roll_deg": 0.0,
        # The one-pose manual calibration remains available as a reference,
        # but production demonstrations must continue through automatic
        # alignment, insertion, closing, lift, and placement.
        "manual_tuning_handoff": False,
        "manual_tuning_object_in_stator_m": (0.1416, 0.00715, 0.021),
        "manual_tuning_orientation_tolerance_rad": 0.006,
        "use_object_axis_approach": True,
        # Enter along the mouse's long axis, while the parallel jaws close
        # across its short axis.  Both axes are rotated by the live object
        # yaw, and the long-axis sign is chosen toward free workspace.
        "object_approach_local_axis": (1.0, 0.0, 0.0),
        "object_pinch_local_axis": (0.0, 1.0, 0.0),
        "reverse_object_axis_approach": False,
        "alignment_axis_order": ("z", "y", "x"),
        "decouple_outer_alignment": True,
        "decouple_dynamic_alignment": True,
        "alignment_tolerance_y_m": 0.002,
        "alignment_tolerance_z_m": 0.002,
        "grasp_envelope_overrides": {
            "x_max_m": 0.172,
            "y_abs_max_m": 0.010,
            "z_min_m": 0.000,
            "z_max_m": 0.046,
        },
        "insertion_center_x_m": 0.155,
        "dynamic_insertion_stop_x_m": 0.158,
        "desired_center_x_m": 0.155,
        "insertion_x_tolerance_m": 0.002,
        # At 75 degrees the leading finger reaches the mouse's widest shell
        # near x=0.168-0.176 m. Treat that physical stop as the pre-close point;
        # demanding the free-space x target would only push the mouse.
        "contact_preclose_x_tolerance_m": 0.019,
        "contact_alignment_tolerance_y_m": 0.010,
        "contact_alignment_tolerance_z_m": 0.005,
        "contact_preclose_center_z_m": 0.035,
        "preclose_contact_center_z_m": 0.035,
        "disable_preclose_bilateral_height_correction": True,
        "preclose_partial_gripper_rad": -0.55,
        "preclose_axial_seating_step_m": 0.001,
        "preclose_axial_seating_max_translation_m": 0.0,
        "preclose_axial_seating_min_x_m": 0.158,
        "preclose_axial_seating_y_limit_m": 0.008,
        "preclose_axial_seating_z_limit_m": 0.006,
        # A plateau while only the fixed finger touches this thin shell is
        # not a grasp.  Keep the slow close active until PhysX reports both
        # real finger meshes, then use only a very small verified preload.
        "require_bilateral_close_obstruction": False,
        "contact_hold_preload_rad": 0.030,
        "contact_hold_preload_step_rad": 0.002,
        "require_preclose_target_contact": False,
        "contact_search_min_x_m": 0.152,
        "dynamic_alignment_correction_budget": 24,
        "approach_beyond_target_m": 0.060,
        "minimum_object_rise_m": 0.100,
        "transport_object_z_m": 0.145,
        "physical_test_lift_segment_m": 0.010,
        "physical_test_lift_near_table_segment_m": 0.005,
        "physical_test_lift_near_table_until_z_m": 0.055,
        "physical_test_lift_near_target_segment_m": 0.005,
        "physical_test_lift_near_target_threshold_m": 0.020,
        "physical_test_lift_position_tolerance_m": 0.001,
        "physical_test_lift_near_table_position_tolerance_m": 0.002,
        "physical_test_lift_max_joint_delta_rad": 0.15,
        "physical_test_lift_orientation_tolerances_rad": (
            0.04,
            0.08,
            0.12,
        ),
        "physical_test_lift_near_table_orientation_tolerances_rad": (
            0.005,
            0.010,
            0.020,
        ),
        "physical_test_lift_near_target_orientation_tolerances_rad": (
            0.005,
            0.010,
            0.020,
        ),
        "physical_test_lift_bounded_fallback_max_joint_delta_rad": 0.15,
        "dynamic_insertion_step_m": 0.001,
        "entry_center_z_m": 0.030,
        "dynamic_entry_center_z_until_x_m": 0.150,
        "insertion_alignment_tolerance_y_m": 0.003,
        "desired_center_z_m": 0.014,
        "insertion_alignment_tolerance_z_m": 0.002,
        "dynamic_axial_feedforward_z_m": 0.0,
        "lock_outer_alignment_orientation": True,
        "outer_alignment_orientation_tolerance_rad": 0.006,
        "lock_dynamic_insertion_orientation": True,
        "insertion_orientation_tolerance_rad": 0.006,
        "outer_alignment_acceptance_y_m": 0.003,
        "outer_alignment_acceptance_z_m": 0.004,
    },
    "beverage_bottle": {
        "hardware_pitch_deg": float(
            OBJECT_GRASP_PROFILES["beverage_bottle"][
                "hardware_approach_pitch_deg"
            ]
        ),
        "calibrated_grasp_reference": dict(
            OBJECT_GRASP_PROFILES["beverage_bottle"][
                "calibrated_grasp_reference"
            ]
        ),
        # Use only the positive-roll bottle branch selected by the operator.
        "roll_deg": -90.0,
        # The single home-to-grasp trajectory legitimately traverses almost
        # 2.4 rad on joint2.  Retain margin without relaxing URDF limits.
        "max_joint_delta_rad": 2.70,
        # Keep the long home-to-cylinder segment within the measured Isaac
        # position-loop tracking margin without weakening the safety guard.
        "pregrasp_velocity_scale": 0.18,
        # Cover alternate shoulder/wrist IK branches in edge workspace cells.
        # Pitch/roll-only variants repeat nearly identical radial approaches.
        "candidate_azimuth_offsets_deg": (
            0.0, -45.0, 45.0, -30.0,
            0.0, -45.0, 45.0, 30.0,
        ),
        "precheck_final_ik": True,
        "final_pose_position_tolerance_m": 0.003,
        "accept_first_valid_candidate": True,
        # Keep the bottle dynamic, with real collisions and gravity, during
        # the entire approach as well as closure and physical transport.
        "pregrasp_target_command": "unfreeze",
        # Try the direct loaded carry first.  Only a planning failure may use
        # the existing diagonal midpoint before retrying the box approach.
        "single_pass_box_transport": False,
        "single_pass_box_alignment": True,
        "single_pass_preclose": bool(
            OBJECT_GRASP_PROFILES["beverage_bottle"][
                "single_pass_preclose"
            ]
        ),
        "single_alignment_tolerance_m": tuple(
            float(value)
            for value in OBJECT_GRASP_PROFILES["beverage_bottle"][
                "calibrated_grasp_reference"
            ][
                "position_tolerance_m"
            ]
        ),
        "single_contact_preload_rad": float(
            OBJECT_GRASP_PROFILES["beverage_bottle"][
                "single_contact_preload_rad"
            ]
        ),
        "single_contact_retry_preload_rad": float(
            OBJECT_GRASP_PROFILES["beverage_bottle"][
                "single_contact_retry_preload_rad"
            ]
        ),
        "single_contact_retry_attempts": int(
            OBJECT_GRASP_PROFILES["beverage_bottle"][
                "single_contact_retry_attempts"
            ]
        ),
        # Reach the collision-free outer waypoint on a repeatable IK branch,
        # then rotate near the bottle before measuring stator-local alignment.
        # Keep a single orientation through approach and closure.  The first
        # physical trial at 25 degrees contacted the cylinder too high and
        # produced a 31.5 mm left/right height split, so use the conservatively
        # flatter angle stored with the transformed calibration below.
        "elevation_deg": (
            float(OBJECT_GRASP_PROFILES["beverage_bottle"][
                "approach_elevation_deg"
            ]),
            float(OBJECT_GRASP_PROFILES["beverage_bottle"][
                "approach_elevation_deg"
            ]),
        ),
        "near_grasp_elevation_steps_deg": (),
        "near_grasp_orientation_max_joint_delta_rad": 0.65,
        "near_grasp_orientation_tolerance_rad": 0.020,
        # 2026-09-17: 抓取点与接近点整体相对瓶身下移 2 mm（世界 Z）。
        "world_z_offset_m": -0.002,
        "near_grasp_transition_x_m": 0.145,
        # Keep the long approach axial.  Starting Y/Z fine alignment around
        # local X=0.188 m produced the measured 0.188 <-> 0.180 m X/Y loop.
        "dynamic_fine_alignment_start_x_m": 0.155,
        # The taught XYZ relation is a final grasp relation, not an outer
        # waypoint.  Stay axial until the bottle is near the open jaws, then
        # apply the complete calibrated Y/Z relation.
        "defer_calibrated_alignment_until_x_m": 0.155,
        "axial_only_before_calibrated_alignment": True,
        "unlimited_fine_adjustment": True,
        # Keep coarse insertion moving forward while containing drift.  The
        # taught relative pose is now the controller target, so do not force
        # the old synthetic XY-coupling fallback near the bottle.
        "coarse_alignment_y_guard_m": 0.010,
        "assume_dynamic_xy_coupling": False,
        "dynamic_alignment_abort_corridor_m": 0.050,
        # Retain the 50 mm hard stop, but slow X at 30 mm and pause it at
        # 40 mm. Y-only recovery stays latched until the error is below 30 mm.
        "far_alignment_soft_corridor_m": 0.030,
        "far_alignment_recovery_trigger_m": 0.040,
        "far_alignment_recovery_release_m": 0.030,
        # The measured trajectory bridge has a roughly 2 mm Cartesian dead
        # zone on this branch. Keep a trackable 5 mm command; the 40/30 mm
        # hysteresis, not an untrackable tiny command, provides the slowdown.
        "far_alignment_reduced_axial_step_m": 0.005,
        "near_grasp_transition_alignment_y_m": 0.015,
        "near_grasp_transition_alignment_z_m": 0.012,
        "near_grasp_transition_envelope": {
            "x_min_m": 0.135,
            "x_max_m": 0.180,
            "y_abs_max_m": 0.018,
            "z_min_m": 0.005,
            "z_max_m": 0.045,
        },
        # A stopped approach may be safely resumed anywhere in this open-jaw
        # corridor.  This envelope only skips the duplicate outer approach;
        # it does not relax the pre-close or loaded-grasp envelopes.
        "open_jaw_resume_envelope": {
            "x_min_m": 0.110,
            "x_max_m": 0.220,
            "y_abs_max_m": 0.030,
            "z_min_m": 0.000,
            "z_max_m": 0.050,
        },
        "cylinder_high_contact_recovery_attempts": 3,
        "cylinder_high_contact_recovery_lower_m": 0.025,
        "cylinder_high_contact_recovery_retreat_m": 0.010,
        "insertion_center_x_m": 0.12583045,
        "dynamic_insertion_stop_x_m": 0.12583045,
        "maximum_insertion_distance_m": 0.110,
        "dynamic_insertion_step_m": 0.005,
        # Start conservatively, then grow above the controller's measured
        # handover error only when closed-loop feedback shows under-tracking.
        "maximum_dynamic_insertion_step_m": 0.020,
        "dynamic_feedback_min_progress_ratio": 0.35,
        "dynamic_feedback_step_growth": 1.8,
        "dynamic_feedback_min_progress_m": 0.001,
        "dynamic_feedback_joint_space_after_stalls": 2,
        # Measured bottle insertion on this IK branch couples local X into Y.
        # Once observed, request one bounded two-axis pose instead of issuing
        # a pure-Y LIN correction that undoes the gained insertion depth.
        "dynamic_feedback_coupling_ratio": 0.70,
        "dynamic_feedback_coupling_min_drift_m": 0.004,
        "dynamic_feedback_coupled_xy_correction": False,
        # Keep the horizontal pinch intent, but do not force the coupled
        # correction through the same 0.006 rad orientation tube that made
        # the final XY pose effectively unreachable on the observed branch.
        "dynamic_feedback_coupled_orientation_tolerance_rad": 0.10,
        "dynamic_feedback_coupled_max_joint_delta_rad": 0.65,
        "desired_center_x_m": 0.12583045,
        # Medium randomization can start with about 54 mm of lateral error.
        # Two passes are consumed by Z-first alignment and the measured Y
        # convergence is 2.5--3 mm/pass, so 20 passes cover that envelope.
        "outer_alignment_iterations": 20,
        # On the far-side medium-layout IK branch an 8 mm command only
        # produced 1.7--2.4 mm of measured lateral motion.  The larger bounded
        # command remains at the frozen, collision-checked outer waypoint.
        "outer_alignment_max_y_step_m": 0.015,
        "alignment_axis_order": ("z", "y", "x"),
        # Live-captured operator-confirmed pre-close center.  The dedicated
        # +/-6 mm band below remains subject to the table-clearance envelope.
        # 2026-09-17: 对齐目标同样下移 2 mm（世界 Z）。该姿态 local +Y 的
        # 世界 Z 分量为 -0.9457765（标定四元数；运行时 TF 实测 -0.950），
        # 故 -0.00284520 + 0.002 * (-0.9457765) = -0.00473675。
        "desired_center_y_m": -0.00473675,
        "alignment_tolerance_y_m": 0.006,
        # Retain a 6 mm band around the taught local coordinates.
        "alignment_tolerance_z_m": 0.006,
        "outer_alignment_acceptance_y_m": 0.008,
        "outer_alignment_acceptance_z_m": 0.008,
        "insertion_alignment_tolerance_y_m": 0.006,
        "insertion_alignment_tolerance_z_m": 0.006,
        # On the -90-degree IK branch, pure lateral commands converge but a
        # simultaneous 2 mm axial advance reverses the measured Y response.
        # Finish Y/Z centering before feeding the bottle into the jaws.
        "align_before_axial_advance": True,
        "dynamic_lateral_step_m": 0.015,
        "dynamic_vertical_step_m": 0.010,
        # Keep near-threshold corrections above the observed ~10 mm tracking
        # dead zone; the command remains capped by dynamic_lateral_step_m.
        "dynamic_alignment_gain_y": 1.5,
        # Contact acceptance remains inside the dedicated capture and loaded
        # envelopes and still requires real bilateral physics.
        "contact_preclose_x_tolerance_m": 0.005,
        "contact_alignment_tolerance_y_m": 0.012,
        "contact_alignment_tolerance_z_m": 0.010,
        # The capture envelope already bounds a safe body-height interval.
        # Do not leave it for another Cartesian Z correction before closing;
        # that motion previously reintroduced the same X/Y coupling.
        "disable_preclose_bilateral_height_correction": True,
        "entry_center_z_m": 0.01877970,
        "desired_center_z_m": 0.01877970,
        "contact_preclose_center_z_m": 0.01877970,
        "preclose_contact_center_z_m": 0.01877970,
        "accept_calibrated_high_contact_as_ready": True,
        # After the staged rotation, accepting a fresh orientation after every
        # Cartesian correction lets small MoveIt residuals rotate the
        # stator-local X/Z basis.  The resulting coupled correction was
        # observed to increase the Z error while consuming insertion depth.
        # Hold the candidate orientation through both alignment phases and
        # guard against any remaining depth loss while centering.
        "lock_outer_alignment_orientation": True,
        "outer_alignment_orientation_tolerance_rad": 0.006,
        "lock_dynamic_insertion_orientation": True,
        "insertion_orientation_tolerance_rad": 0.006,
        "decouple_outer_alignment": True,
        "decouple_dynamic_alignment": True,
        # With the jaws rolled 90 degrees, a lateral centering move remains
        # clear of the table and bottle cap.  The old axial retreat couples
        # into local Y through execution residuals and creates a repeatable
        # advance/retreat limit cycle before either finger can make contact.
        "guard_dynamic_alignment_depth": False,
        "dynamic_alignment_centering_depth_guard_m": 0.006,
        "dynamic_alignment_depth_recovery_max_step_m": 0.006,
        "dynamic_alignment_depth_recovery_min_improvement_m": 0.001,
        "dynamic_alignment_safe_pose_recovery_attempts": 2,
        # Clamp every dynamic command above the collision-free entry height.
        # The 5 mm target guard absorbs the observed 5--6 mm execution error.
        "dynamic_alignment_world_z_floor_margin_m": 0.005,
        # This measured open-jaw volume is safe to attempt a slow physical
        # close. It does not relax bilateral contact, body-height, or loaded
        # lift validation.
        "preclose_capture_envelope": dict(
            OBJECT_GRASP_PROFILES["beverage_bottle"][
                "preclose_capture_envelope"
            ]
        ),
        # The wider capture volume prevents another coupled X/Y centering
        # loop, but it must not end axial insertion while both fingers are
        # still clear of the bottle.  Require the measured centre to reach
        # the leading-finger depth before a no-contact close may begin.
        "preclose_no_contact_x_max_m": float(
            OBJECT_GRASP_PROFILES["beverage_bottle"][
                "preclose_no_contact_x_max_m"
            ]
        ),
        "grasp_envelope_overrides": dict(
            OBJECT_GRASP_PROFILES["beverage_bottle"]["preclose_envelope"]
        ),
    },
}
for _object_name, _profile in GRASP_PROFILES.items():
    _profile.update(_CLOSED_LOOP_ALIGNMENT_DEFAULTS)
    _profile.update(_OBJECT_ALIGNMENT_GEOMETRY[_object_name])
    # The old controller enlarged small residuals to 5–15 mm.  That defeated
    # feedback near the goal and made Y/Z oscillate, so those fields are no
    # longer part of any object profile.
    _profile.pop("minimum_effective_y_step_m", None)
    _profile.pop("outer_alignment_min_effective_y_step_m", None)

# The short cup uses the bottle trajectory/closure pipeline, with its own
# upper-body height and strict acceptance. Bottle debug switches cannot disable it.
_cup_geometry = OBJECT_GRASP_PROFILES["cup"]
GRASP_PROFILES["cup"] = copy.deepcopy(GRASP_PROFILES["beverage_bottle"])
GRASP_PROFILES["cup"].update({
    "roll_deg": -90.0,
    "freeze_before_approach": True,
    "pregrasp_target_command": "unfreeze",
    "height": float(_cup_geometry["grasp_height_target_m"]),
    "world_z_offset_m": 0.0,
    "desired_center_y_m": -0.00284520,
    "calibrated_grasp_reference": _cylinder_reference_at_grasp_height(_cup_geometry),
    "grasp_envelope_overrides": dict(_cup_geometry["preclose_envelope"]),
    "preclose_capture_envelope": dict(_cup_geometry["preclose_capture_envelope"]),
    "open_jaw_resume_envelope": dict(_cup_geometry["preclose_envelope"]),
    "single_alignment_tolerance_m": tuple(_cup_geometry["calibrated_grasp_reference"]["position_tolerance_m"]),
    "single_contact_retry_attempts": 0,
    "single_pass_box_transport": False,
    # The cup's shorter body means the calibrated reference is taken from the
    # cup's own upper-body height while the tool keeps the bottle orientation
    # and the tool-frame roll, so the geometry point is offset by the cup
    # centre height rather than by the bottle's grasp height.
    "alignment_reference_uses_tool_rotation": True,
    "alignment_geometry_z_offset_m": float(
        GRASP_CONFIG["object_center_z_offsets_m"]["cup"]
    ),
    # A cup that leaves its 6 mm band after the target is released to physics
    # would be pinched on the base or rim, so its ordered pre-close guard is
    # part of its acceptance specification and never a runtime debug switch.
    "preclose_alignment_guard": True,
    "alignment_tolerance_z_m": 0.004,
    "pregrasp_velocity_scale": 0.08,
    "final_pose_position_tolerance_m": 0.003,
    "precheck_final_ik": True,
    "candidate_local_offsets_yz_m": (
        (0.0, 0.0),
        (0.002, 0.0),
        (-0.002, 0.0),
        (0.0, 0.002),
        (0.0, -0.002),
        (0.004, 0.0),
        (-0.004, 0.0),
        (0.002, 0.002),
    ),
    "candidate_azimuth_offsets_deg": (
        0.0, -45.0, 45.0, -30.0,
        0.0, -45.0, 45.0, 30.0,
    ),
    "accept_first_valid_candidate": True,
    "accept_calibrated_high_contact_as_ready": False,
})
# Reuse the bottle state machine, but resolve every cup geometry/force/lift
# setting from the cup record. Copying the bottle profile alone silently kept
# its 180 mm lift, 160 mm clearance and 5 degree preload on the shorter cup.
for _key in (
    "single_pass_preclose", "single_contact_preload_rad",
    "single_contact_retry_preload_rad", "single_contact_retry_attempts",
    "maximum_close_displacement_m", "loaded_samples_required",
    "transport_object_z_m", "physical_test_lift_safe_completion_z_m",
    "physical_test_lift_completion_tolerance_m", "physical_test_lift_segment_m",
    "physical_test_lift_near_table_segment_m", "physical_test_lift_near_table_until_z_m",
    "physical_test_lift_min_retry_segment_m", "physical_test_lift_min_progress_m",
    "physical_test_lift_max_stalled_iterations",
    "physical_test_lift_orientation_tolerances_rad",
    "physical_test_lift_bounded_fallback_max_joint_delta_rad",
    "unfreeze_settle_duration_s",
):
    GRASP_PROFILES["cup"][_key] = copy.deepcopy(_cup_geometry[_key])
_cup_reference = _cup_geometry["calibrated_grasp_reference"]["object_in_gripper_local_m"]
for _axis, _value in zip("xyz", _cup_reference):
    GRASP_PROFILES["cup"][f"desired_center_{_axis}_m"] = float(_value)
GRASP_PROFILES["cup"].update({
    "pitch_deg": float(_cup_geometry["approach_pitch_deg"]),
    "hardware_pitch_deg": float(_cup_geometry["hardware_approach_pitch_deg"]),
    "elevation_deg": (float(_cup_geometry["approach_elevation_deg"]),) * 2,
    "minimum_object_rise_m": float(_cup_geometry["minimum_proof_rise_m"]),
    "gripper_contact_closed_limit_rad": float(_cup_geometry["gripper_contact_angle_range_rad"][0]),
    "gripper_contact_open_limit_rad": float(_cup_geometry["gripper_contact_angle_range_rad"][1]),
    "insertion_center_x_m": float(_cup_reference[0]),
    "dynamic_insertion_stop_x_m": float(_cup_reference[0]),
    "entry_center_z_m": float(_cup_reference[2]),
})

for _name in ("cup", "beverage_bottle"):
    _extra = tuple(GRASP_CONFIG["scene_randomization"][
        "target_sweep_clearance"
    ]["extra_world_azimuths_deg_by_object"][_name])
    GRASP_PROFILES[_name]["extra_candidate_azimuths_deg"] = _extra
    GRASP_PROFILES[_name]["candidate_count"] = len(
        GRASP_PROFILES[_name]["candidate_azimuth_offsets_deg"]
    ) + len(_extra)


RANDOMIZATION_CONFIG = GRASP_CONFIG["scene_randomization"]
OBJECT_FOOTPRINT_RADII = {
    name: float(radius)
    for name, radius in RANDOMIZATION_CONFIG[
        "object_footprint_radii_m"
    ].items()
}
TARGET_SWEEP_CONFIG = RANDOMIZATION_CONFIG["target_sweep_clearance"]
BOX_EXCLUSION_X = tuple(
    float(value) for value in RANDOMIZATION_CONFIG["box_exclusion_x_m"]
)
BOX_EXCLUSION_Y = tuple(
    float(value) for value in RANDOMIZATION_CONFIG["box_exclusion_y_m"]
)


def _normalize(value: Sequence[float]):
    norm = math.sqrt(sum(component * component for component in value))
    if norm < 1.0e-9:
        raise ValueError("cannot normalize a zero vector")
    return tuple(component / norm for component in value)


def _bounded_axis_correction(
    error: float,
    tolerance: float,
    gain: float,
    maximum_step: float,
) -> float:
    """Return a deadbanded proportional correction without minimum-step overshoot."""
    if abs(error) <= tolerance:
        return 0.0
    return max(-maximum_step, min(maximum_step, gain * error))


def _single_pass_alignment_correction(
    measured: Sequence[float],
    desired: Sequence[float],
    tolerance: Sequence[float],
    maximum_step: float,
    gain: float = 1.0,
) -> Vector3:
    """Return a bounded tool-local correction for a frozen open-jaw target."""
    if len(measured) != 3 or len(desired) != 3 or len(tolerance) != 3:
        raise ValueError("single-pass alignment inputs must be 3D")
    return tuple(
        _bounded_axis_correction(
            float(measured[index]) - float(desired[index]),
            float(tolerance[index]),
            float(gain),
            float(maximum_step),
        )
        for index in range(3)
    )


def _pregrasp_velocity_scale(profile: dict, default: float) -> float:
    """Return an object-specific global approach speed when configured."""
    return float(profile.get("pregrasp_velocity_scale", default))


def _pregrasp_target_command(profile: dict) -> str:
    """Choose the scene mode used only during open-finger arm approach."""
    command = str(profile.get("pregrasp_target_command", "freeze"))
    if command not in ("freeze", "unfreeze"):
        raise ValueError(f"invalid pregrasp target command: {command!r}")
    return command


def _envelope_validation_enabled(target_object: str) -> bool:
    """Return whether geometric envelope acceptance still gates this object.

    ``envelope_validation_enabled: false`` pauses every envelope-based
    rejection for that object (pre-close envelope, pre-close positioning
    residual, loaded envelope and transport guard).  Real contact, force,
    physics and contact-height checks are unaffected.
    """
    return bool(
        OBJECT_GRASP_PROFILES.get(target_object, {}).get(
            "envelope_validation_enabled", True
        )
    )


def _trajectory_tolerance_enforcement_enabled(target_object: str) -> bool:
    """Return whether an unfinished trajectory goal still aborts the task.

    ``trajectory_tolerance_enforcement_enabled: false`` pauses the
    ``GOAL_TOLERANCE_VIOLATED`` abort for that object: the residual joint
    error is only logged, so a sub-millimetre controller tolerance miss no
    longer discards the episode.  ``PATH_TOLERANCE_VIOLATED`` (real path
    deviation protection) and hard execution errors are always kept.
    """
    return bool(
        OBJECT_GRASP_PROFILES.get(target_object, {}).get(
            "trajectory_tolerance_enforcement_enabled", True
        )
    )


def _adaptive_dynamic_step_limit(
    current_limit: float,
    commanded_step: float,
    measured_progress: float,
    nominal_limit: float,
    maximum_limit: float,
    minimum_progress_ratio: float = 0.35,
    growth: float = 1.8,
) -> float:
    """Adapt a Cartesian step from measured motion, not planner intent."""
    nominal_limit = max(1.0e-4, float(nominal_limit))
    maximum_limit = max(nominal_limit, float(maximum_limit))
    current_limit = min(
        maximum_limit, max(nominal_limit, float(current_limit))
    )
    commanded_magnitude = abs(float(commanded_step))
    if commanded_magnitude <= 1.0e-6:
        return current_limit
    progress_ratio = max(0.0, float(measured_progress)) / commanded_magnitude
    minimum_progress_ratio = max(0.05, float(minimum_progress_ratio))
    if progress_ratio < minimum_progress_ratio:
        compensated = nominal_limit / max(
            progress_ratio, 0.5 * minimum_progress_ratio
        )
        return min(
            maximum_limit,
            max(current_limit * max(1.05, float(growth)), compensated),
        )
    if progress_ratio >= 0.65 and current_limit > nominal_limit:
        return max(nominal_limit, current_limit * 0.8)
    return current_limit


def _outer_alignment_is_inside_safety_corridor(
    y_error: float,
    z_error: float,
    safety_corridor: float,
) -> bool:
    """Reject only geometry outside the following collision loop's corridor."""
    safety_corridor = max(0.0, float(safety_corridor))
    return (
        abs(float(y_error)) <= safety_corridor
        and abs(float(z_error)) <= safety_corridor
    )


def _far_alignment_guard(
    y_error: float,
    axial_step: float,
    soft_corridor: float,
    recovery_trigger: float,
    recovery_release: float,
    reduced_axial_step: float,
    recovery_active: bool,
) -> Tuple[float, bool, bool]:
    """Reduce or pause far insertion before Y drift reaches the hard limit."""
    absolute_error = abs(float(y_error))
    soft_corridor = max(0.0, float(soft_corridor))
    recovery_trigger = max(soft_corridor, float(recovery_trigger))
    recovery_release = min(
        recovery_trigger, max(0.0, float(recovery_release))
    )
    if recovery_active:
        recovery_active = absolute_error > recovery_release
    elif absolute_error >= recovery_trigger:
        recovery_active = True
    if recovery_active:
        return 0.0, True, True
    if absolute_error >= soft_corridor:
        reduced = min(
            abs(float(axial_step)),
            max(0.0, float(reduced_axial_step)),
        )
        return math.copysign(reduced, float(axial_step)), False, True
    return float(axial_step), False, False


def _is_recoverable_lift_planning_error(error: Exception) -> bool:
    """Return true only for planning/IK failures safe to retry more locally."""
    message = str(error)
    return (
        "MoveIt error_code=-31" in message
        or "MoveIt error_code=-2" in message
        or "MoveIt error_code=99999" in message
        or "拒绝关节跳变" in message
    )


def _reduced_lift_segment_distance(
    segment_distance: float,
    minimum_distance: float,
) -> Optional[float]:
    """Halve a failed lift waypoint without going below the safe minimum."""
    segment_distance = float(segment_distance)
    minimum_distance = max(1.0e-4, float(minimum_distance))
    if segment_distance <= minimum_distance + 1.0e-6:
        return None
    return max(minimum_distance, 0.5 * segment_distance)


def _loaded_transport_height_is_acceptable(
    object_z: float,
    *,
    safe_completion_z: float,
    tolerance_m: float,
) -> bool:
    """Accept measured loaded height within the configured completion band."""
    minimum_z = float(safe_completion_z) - max(0.0, float(tolerance_m))
    return float(object_z) >= minimum_z


def _guard_dynamic_alignment_depth(
    local_x: float,
    stop_x: float,
    y_outside: bool,
    z_outside: bool,
    step_distance: float,
    lateral_step: float,
    vertical_step: float,
    centering_guard: float,
    maximum_retreat: float,
) -> Tuple[float, float, float, bool]:
    """Retreat before correcting Y/Z at or inside the insertion depth."""
    guard = max(0.0, float(centering_guard))
    safe_centering_x = float(stop_x) + guard
    if not (y_outside or z_outside) or float(local_x) > safe_centering_x:
        return (
            float(step_distance),
            float(lateral_step),
            float(vertical_step),
            False,
        )
    required_retreat = max(
        guard,
        safe_centering_x - float(local_x),
    )
    retreat = min(
        max(1.0e-4, float(maximum_retreat)),
        required_retreat,
    )
    return (-retreat, 0.0, 0.0, True)


def _local_axis_centering_step(
    measured_local_value: float, step_size: float
) -> float:
    """Move the gripper with the local error so the measured value approaches zero."""
    if abs(measured_local_value) <= 1.0e-9:
        return 0.0
    return math.copysign(abs(step_size), measured_local_value)


def _planar_object_axis_toward_workspace(
    object_orientation: Quaternion,
    local_axis: Vector3,
    object_position: Vector3,
) -> Tuple[float, float]:
    """Choose the object-axis sign whose approach points outward from the base."""
    world_axis = rotate_vector(object_orientation, local_axis)
    axis_length = math.hypot(world_axis[0], world_axis[1])
    if axis_length <= 1.0e-6:
        raise ValueError("object grasp axis has no horizontal projection")
    axis_xy = (
        float(world_axis[0]) / axis_length,
        float(world_axis[1]) / axis_length,
    )
    if (
        axis_xy[0] * float(object_position[0])
        + axis_xy[1] * float(object_position[1])
        < 0.0
    ):
        axis_xy = (-axis_xy[0], -axis_xy[1])
    return axis_xy


def _point_to_aabb_distance(
    point: Sequence[float],
    x_bounds: Sequence[float],
    y_bounds: Sequence[float],
) -> float:
    x = float(point[0])
    y = float(point[1])
    dx = max(float(x_bounds[0]) - x, 0.0, x - float(x_bounds[1]))
    dy = max(float(y_bounds[0]) - y, 0.0, y - float(y_bounds[1]))
    return math.hypot(dx, dy)


def _segment_intersects_expanded_aabb(
    start: Sequence[float],
    end: Sequence[float],
    x_bounds: Sequence[float],
    y_bounds: Sequence[float],
    expansion: float,
) -> bool:
    """Return whether a 2D swept centre line enters an expanded rectangle."""
    minimums = (
        float(x_bounds[0]) - float(expansion),
        float(y_bounds[0]) - float(expansion),
    )
    maximums = (
        float(x_bounds[1]) + float(expansion),
        float(y_bounds[1]) + float(expansion),
    )
    start_xy = (float(start[0]), float(start[1]))
    end_xy = (float(end[0]), float(end[1]))
    t_min = 0.0
    t_max = 1.0
    for axis in range(2):
        delta = end_xy[axis] - start_xy[axis]
        if abs(delta) <= 1.0e-12:
            if not minimums[axis] <= start_xy[axis] <= maximums[axis]:
                return False
            continue
        enter = (minimums[axis] - start_xy[axis]) / delta
        leave = (maximums[axis] - start_xy[axis]) / delta
        if enter > leave:
            enter, leave = leave, enter
        t_min = max(t_min, enter)
        t_max = min(t_max, leave)
        if t_min > t_max:
            return False
    return True


def _candidate_sweep_blockers(
    target_object: str,
    target_position: Sequence[float],
    pregrasp_position: Sequence[float],
    object_positions: Dict[str, Sequence[float]],
    approach_beyond_target_m: Optional[float] = None,
) -> Dict[str, str]:
    direction_x = float(target_position[0]) - float(pregrasp_position[0])
    direction_y = float(target_position[1]) - float(pregrasp_position[1])
    direction_length = math.hypot(direction_x, direction_y)
    if direction_length <= 1.0e-9:
        direction_length = math.hypot(
            float(target_position[0]), float(target_position[1])
        )
        if direction_length <= 1.0e-9:
            return {"workspace": "目标与接近方向均无法定义"}
        direction_x = float(target_position[0])
        direction_y = float(target_position[1])
    direction = (
        direction_x / direction_length,
        direction_y / direction_length,
    )
    before = float(TARGET_SWEEP_CONFIG["approach_before_target_m"])
    beyond = (
        float(TARGET_SWEEP_CONFIG["approach_beyond_target_m"])
        if approach_beyond_target_m is None
        else float(approach_beyond_target_m)
    )
    blockers: Dict[str, str] = {}

    # Dynamic-object clearance alone is insufficient near the placement box.
    # The initial MoveIt pregrasp can be collision-free while the subsequent
    # open-jaw alignment/insertion sweeps gripperMover into a wall.  Reject
    # that geometry before recording begins by treating the complete box
    # footprint as a static obstacle expanded by the end-effector footprint.
    target_box_distance = _point_to_aabb_distance(
        target_position,
        BOX_EXCLUSION_X,
        BOX_EXCLUSION_Y,
    )
    closing_radius = float(TARGET_SWEEP_CONFIG["closing_radius_m"])
    lift_radius = float(TARGET_SWEEP_CONFIG["lift_radius_m"])
    if target_box_distance < closing_radius:
        blockers["smolvla_box"] = "蓝盒闭爪空间"
    elif target_box_distance < lift_radius:
        blockers["smolvla_box"] = "蓝盒初始抬升空间"
    else:
        robot_start = (
            float(target_position[0]) - before * direction[0],
            float(target_position[1]) - before * direction[1],
        )
        target_xy = (
            float(target_position[0]),
            float(target_position[1]),
        )
        target_end = (
            float(target_position[0]) + beyond * direction[0],
            float(target_position[1]) + beyond * direction[1],
        )
        robot_side_blocked = _segment_intersects_expanded_aabb(
            robot_start,
            target_xy,
            BOX_EXCLUSION_X,
            BOX_EXCLUSION_Y,
            float(TARGET_SWEEP_CONFIG["robot_side_half_width_m"]),
        )
        target_side_blocked = _segment_intersects_expanded_aabb(
            target_xy,
            target_end,
            BOX_EXCLUSION_X,
            BOX_EXCLUSION_Y,
            float(TARGET_SWEEP_CONFIG["target_side_half_width_m"]),
        )
        if robot_side_blocked or target_side_blocked:
            blockers["smolvla_box"] = "蓝盒夹爪接近扫掠通道"

    for name, position in object_positions.items():
        if name == target_object or name not in OBJECT_FOOTPRINT_RADII:
            continue
        # The simulator isolates non-target training objects at z=-1 m with
        # collisions disabled.  They remain in the truth TF stream so dataset
        # consumers keep stable frame names, but must not block an XY sweep.
        if len(position) >= 3 and float(position[2]) < -0.10:
            continue
        radius = OBJECT_FOOTPRINT_RADII[name]
        relative_x = float(position[0]) - float(target_position[0])
        relative_y = float(position[1]) - float(target_position[1])
        longitudinal = (
            relative_x * direction[0] + relative_y * direction[1]
        )
        lateral = abs(
            relative_x * (-direction[1]) + relative_y * direction[0]
        )
        center_distance = math.hypot(
            float(position[0]) - float(target_position[0]),
            float(position[1]) - float(target_position[1]),
        )
        closing_limit = (
            float(TARGET_SWEEP_CONFIG["closing_radius_m"]) + radius
        )
        if center_distance < closing_limit:
            blockers[name] = "闭爪空间"
            continue
        lift_limit = float(TARGET_SWEEP_CONFIG["lift_radius_m"]) + radius
        if center_distance < lift_limit:
            blockers[name] = "初始抬升空间"
            continue
        robot_limit = (
            float(TARGET_SWEEP_CONFIG["robot_side_half_width_m"]) + radius
        )
        target_limit = (
            float(TARGET_SWEEP_CONFIG["target_side_half_width_m"]) + radius
        )
        if longitudinal < -before:
            approach_distance = math.hypot(
                longitudinal + before, lateral
            )
            approach_limit = robot_limit
        elif longitudinal <= 0.0:
            approach_distance = lateral
            approach_limit = robot_limit
        elif longitudinal <= beyond:
            approach_distance = lateral
            approach_limit = target_limit
        else:
            approach_distance = math.hypot(
                longitudinal - beyond, lateral
            )
            approach_limit = target_limit
        if approach_distance < approach_limit:
            blockers[name] = "夹爪接近扫掠通道"
    return blockers


def _quaternion_angular_distance(
    first: Sequence[float], second: Sequence[float]
) -> float:
    """Return the shortest rotation angle between two quaternions."""
    first_norm = math.sqrt(sum(component * component for component in first))
    second_norm = math.sqrt(sum(component * component for component in second))
    if first_norm < 1.0e-9 or second_norm < 1.0e-9:
        return math.inf
    dot = sum(
        a * b for a, b in zip(first, second)
    ) / (first_norm * second_norm)
    return 2.0 * math.acos(max(-1.0, min(1.0, abs(dot))))


def _inside_calibrated_grasp_reference(
    local_position: Sequence[float], profile: dict
) -> bool:
    """Check the measured bottle-in-gripper pose against its taught pose."""
    calibration = profile.get("calibrated_grasp_reference", {})
    reference = calibration.get("object_in_gripper_local_m", ())
    tolerance = calibration.get("position_tolerance_m", ())
    if len(local_position) != 3 or len(reference) != 3 or len(tolerance) != 3:
        return False
    return all(
        abs(float(measured) - float(target)) <= float(limit)
        for measured, target, limit in zip(
            local_position, reference, tolerance
        )
    )


def _world_gripper_origin_from_object_reference(
    object_world_position: Sequence[float],
    gripper_world_orientation: Sequence[float],
    object_in_gripper_local: Sequence[float],
) -> Vector3:
    """Solve p_WG = p_WO - R_WG p_GO for a taught relative pose."""
    if (
        len(object_world_position) != 3
        or len(gripper_world_orientation) != 4
        or len(object_in_gripper_local) != 3
    ):
        raise ValueError("calibrated grasp transform dimensions are invalid")
    world_offset = rotate_vector(
        tuple(float(value) for value in gripper_world_orientation),
        tuple(float(value) for value in object_in_gripper_local),
    )
    return tuple(
        float(object_world_position[index]) - world_offset[index]
        for index in range(3)
    )


def _calibrated_high_cylinder_contact_is_ready(
    status: dict, target_object: str, profile: dict, cylinder_profile: dict
) -> bool:
    """Treat only the taught high contact as a proximity cue, never success."""
    if not bool(profile.get("accept_calibrated_high_contact_as_ready", False)):
        return False
    if not _inside_calibrated_grasp_reference(
        status.get("local_position_m", ()), profile
    ):
        return False
    height_range = cylinder_profile.get("grasp_height_range_m", ())
    if len(height_range) != 2:
        return False
    heights = []
    for side in ("left", "right"):
        if status.get(f"{side}_finger_contact_object") != target_object:
            continue
        point = status.get(f"{side}_finger_contact_point_object_m")
        if not isinstance(point, (list, tuple)) or len(point) != 3:
            return False
        heights.append(float(point[2]))
    return bool(heights) and min(heights) > float(height_range[1])


def _inside_preclose_corridor(
    local_position: Sequence[float],
    profile: dict,
    envelope: dict,
    contact_present: bool = False,
) -> bool:
    """Accept the collision-safe interval, not one millimetre-perfect point."""
    if len(local_position) != 3:
        return False
    # A tall cylinder can already be safely enclosed by the open jaws before
    # it reaches the ideal deep-insertion point. Once inside this dedicated
    # capture volume, stop Cartesian X/Y chasing and let real bilateral
    # contact during the slow close decide whether the grasp is valid. This
    # remains separate from the stricter preload/lift success envelopes.
    capture_envelope = profile.get("preclose_capture_envelope")
    if (
        not contact_present
        and isinstance(capture_envelope, dict)
        and inside_grasp_envelope(local_position, capture_envelope)
    ):
        return True
    desired_x = float(
        profile.get(
            "desired_center_x_m",
            profile.get("dynamic_insertion_stop_x_m", 0.0),
        )
    )
    entry_x = float(
        profile.get("dynamic_insertion_stop_x_m", desired_x)
    )
    x_tolerance = float(profile.get("insertion_x_tolerance_m", 0.001))
    if contact_present:
        x_tolerance = max(
            x_tolerance,
            float(profile.get("contact_preclose_x_tolerance_m", 0.0)),
        )
    desired_y = float(profile.get("desired_center_y_m", 0.0))
    desired_z = float(profile.get("desired_center_z_m", 0.0))
    if contact_present:
        desired_z = float(
            profile.get("contact_preclose_center_z_m", desired_z)
        )
    y_tolerance = float(
        profile.get(
            "insertion_alignment_tolerance_y_m",
            profile.get("alignment_tolerance_y_m", 0.002),
        )
    )
    z_tolerance = float(
        profile.get(
            "insertion_alignment_tolerance_z_m",
            profile.get("alignment_tolerance_z_m", 0.002),
        )
    )
    if contact_present:
        y_tolerance = max(
            y_tolerance,
            float(
                profile.get(
                    "contact_alignment_tolerance_y_m", y_tolerance
                )
            ),
        )
        z_tolerance = max(
            z_tolerance,
            float(
                profile.get(
                    "contact_alignment_tolerance_z_m", z_tolerance
                )
            ),
        )
    effective_envelope = envelope
    if contact_present:
        effective_envelope = dict(envelope)
        effective_envelope["z_min_m"] = min(
            float(envelope["z_min_m"]), desired_z - z_tolerance
        )
        effective_envelope["z_max_m"] = max(
            float(envelope["z_max_m"]), desired_z + z_tolerance
        )
    return (
        inside_grasp_envelope(local_position, effective_envelope)
        and min(desired_x, entry_x) - x_tolerance
        <= float(local_position[0])
        <= max(desired_x, entry_x) + x_tolerance
        and abs(float(local_position[1]) - desired_y) <= y_tolerance
        and abs(float(local_position[2]) - desired_z) <= z_tolerance
    )


def _dynamic_insertion_depth_allows_close(
    local_position: Sequence[float],
    profile: dict,
    contact_present: bool = False,
) -> bool:
    """Prevent a wide capture envelope from ending insertion too shallowly."""
    if len(local_position) != 3:
        return False
    no_contact_x_max = profile.get("preclose_no_contact_x_max_m")
    if contact_present or no_contact_x_max is None:
        return True
    return float(local_position[0]) <= float(no_contact_x_max)


def _ordered_axis_correction(
    correction: Sequence[float],
    outside_tolerance: Sequence[bool],
    axis_order: Sequence[str],
) -> Vector3:
    """Return one correction axis, respecting an object-specific priority."""
    if len(correction) != 3 or len(outside_tolerance) != 3:
        raise ValueError("alignment correction and threshold state must be 3D")
    axis_indices = {"x": 0, "y": 1, "z": 2}
    for axis_name in axis_order:
        if axis_name not in axis_indices:
            raise ValueError(f"unknown alignment axis: {axis_name}")
        axis_index = axis_indices[axis_name]
        if bool(outside_tolerance[axis_index]):
            result = [0.0, 0.0, 0.0]
            result[axis_index] = float(correction[axis_index])
            return tuple(result)
    return tuple(float(value) for value in correction)


def _dynamic_axis_coupling_detected(
    axial_progress: float,
    lateral_drift: float,
    minimum_lateral_drift: float = 0.004,
    coupling_ratio: float = 0.70,
) -> bool:
    """Return whether an axial move produced material lateral cross-talk."""
    useful_progress = max(abs(float(axial_progress)), 1.0e-6)
    drift = abs(float(lateral_drift))
    return (
        float(axial_progress) > 0.0
        and drift >= float(minimum_lateral_drift)
        and drift / useful_progress >= float(coupling_ratio)
    )


def _staged_insertion_step(
    planned_step: float,
    remaining: float,
    coarse_step: float,
    fine_distance: float,
    fine_step: float,
    coarse_allowed: bool,
) -> float:
    """Merge aligned free-space advances without crossing the fine zone.

    Never turn a hold/retraction or recovery step into a forward advance.
    """
    if planned_step <= 0.0 or remaining <= 0.0:
        return planned_step
    if remaining <= fine_distance:
        return min(planned_step, fine_step, remaining)
    if not coarse_allowed:
        return planned_step
    return min(coarse_step, remaining - fine_distance)


def _candidate_at_alignment_standoff(
    candidate: dict, requested_standoff: float
) -> dict:
    """Move a sampled outer candidate to its collision-safe alignment depth."""
    adjusted = copy.deepcopy(candidate)
    direction = _normalize(
        tuple(
            adjusted["target_point"][axis] - adjusted["position"][axis]
            for axis in range(3)
        )
    )
    alignment_standoff = min(
        float(requested_standoff), float(adjusted["standoff_m"])
    )
    adjusted["position"] = tuple(
        adjusted["target_point"][axis]
        - alignment_standoff * direction[axis]
        for axis in range(3)
    )
    adjusted["standoff_m"] = alignment_standoff
    adjusted["direct_alignment_target"] = True
    return adjusted


def _loaded_bilateral_grasp_is_valid(
    probe: dict,
    target_object: str,
    envelope: dict,
    margin: float,
    *,
    require_envelope: bool = True,
) -> bool:
    """Validate a loaded real-contact grasp without requiring pre-close pose."""
    local_position = tuple(
        float(value) for value in probe.get("local_position_m", [])
    )
    inside_loaded_envelope = (not require_envelope) or (
        len(local_position) == 3
        and float(envelope["x_min_m"]) - margin
        <= local_position[0]
        <= float(envelope["x_max_m"]) + margin
        and abs(local_position[1])
        <= float(envelope["y_abs_max_m"]) + margin
        and float(envelope["z_min_m"]) - margin
        <= local_position[2]
        <= float(envelope["z_max_m"]) + margin
    )
    return (
        inside_loaded_envelope
        and bool(probe.get("physics_enabled", False))
        and bool(probe.get("gravity_enabled", False))
        and object_loaded_grasp_is_valid(probe, target_object)
    )


def _post_close_geometry_is_valid(
    probe: dict,
    target_object: str,
    envelope: dict,
    y_margin: float,
) -> bool:
    """Allow a narrow apple Y-edge case only with sustained real contact."""
    local = tuple(float(value) for value in probe.get("local_position_m", ()))
    if inside_grasp_envelope(local, envelope):
        return True
    if target_object != "apple" or y_margin <= 0.0 or len(local) != 3:
        return False
    x, y, z = local
    impulses = (
        float(probe.get("left_finger_contact_impulse_ns", 0.0)),
        float(probe.get("right_finger_contact_impulse_ns", 0.0)),
    )
    return (
        all(math.isfinite(value) for value in local)
        and float(envelope["x_min_m"]) <= x <= float(envelope["x_max_m"])
        and abs(y) <= float(envelope["y_abs_max_m"]) + y_margin
        and float(envelope["z_min_m"]) <= z <= float(envelope["z_max_m"])
        and bool(probe.get("contact_report_available", False))
        and bool(probe.get("physics_enabled", False))
        and bool(probe.get("gravity_enabled", False))
        and not bool(probe.get("frozen", False))
        and not bool(probe.get("left_finger_touching_table", False))
        and not bool(probe.get("right_finger_touching_table", False))
        and all(math.isfinite(value) and value > 0.0 for value in impulses)
        and all(
            "/collisions" in str(probe.get(f"{side}_finger_contact_collider", ""))
            for side in ("left", "right")
        )
        and int(probe.get("bilateral_contact_consecutive_frames", 0)) >= 10
        and target_bilateral_contact(
            probe, target_object, require_sustained=True
        )
        and object_loaded_grasp_is_valid(probe, target_object)
    )


def _transport_loaded_grasp_is_valid(
    probe: dict,
    target_object: str,
    envelope: dict,
    guard_margin: float,
    reference_local: tuple,
    relative_position_tolerance: float,
    *,
    require_envelope: bool = True,
) -> bool:
    """Validate transport using physical contact plus the verified lift pose.

    The strict pre-lift envelope proves that the object started between the
    fingers.  During transport it may settle a few millimetres under gravity,
    so continued bilateral contact and drift from the successful top-lift pose
    are authoritative.  The expanded envelope remains only as an absolute
    anti-penetration / gross-pose guard.
    """
    local_position = tuple(
        float(value) for value in probe.get("local_position_m", [])
    )
    reference = tuple(float(value) for value in reference_local)
    if len(local_position) != 3 or len(reference) != 3:
        return False
    inside_absolute_guard = (not require_envelope) or (
        float(envelope["x_min_m"]) - guard_margin
        <= local_position[0]
        <= float(envelope["x_max_m"]) + guard_margin
        and abs(local_position[1])
        <= float(envelope["y_abs_max_m"]) + guard_margin
        and float(envelope["z_min_m"]) - guard_margin
        <= local_position[2]
        <= float(envelope["z_max_m"]) + guard_margin
    )
    reference_error = math.sqrt(
        sum(
            (current - baseline) ** 2
            for current, baseline in zip(local_position, reference)
        )
    )
    return (
        bool(probe.get("contact_report_available", False))
        and bool(probe.get("physics_enabled", False))
        and bool(probe.get("gravity_enabled", False))
        and not bool(probe.get("object_touching_table", False))
        and object_loaded_grasp_is_valid(probe, target_object)
        and inside_absolute_guard
        and reference_error <= relative_position_tolerance
    )


def _preclose_envelope_correction(
    local_position: Sequence[float],
    envelope: dict,
    target_inset: float,
    max_step: float,
) -> Tuple[float, float, float]:
    """Return a bounded stator-local motion that re-centres the open jaws."""
    if len(local_position) != 3:
        raise ValueError("pre-close correction requires a 3D local position")
    inset = max(0.0, float(target_inset))
    step_limit = max(1.0e-6, float(max_step))
    lower = (
        float(envelope["x_min_m"]) + inset,
        -float(envelope["y_abs_max_m"]) + inset,
        float(envelope["z_min_m"]) + inset,
    )
    upper = (
        float(envelope["x_max_m"]) - inset,
        float(envelope["y_abs_max_m"]) - inset,
        float(envelope["z_max_m"]) - inset,
    )
    target = tuple(
        min(max(float(value), minimum), maximum)
        for value, minimum, maximum in zip(local_position, lower, upper)
    )
    # Moving the stator by +delta in its local frame changes the measured
    # object coordinate by -delta.
    correction = tuple(
        float(value) - desired
        for value, desired in zip(local_position, target)
    )
    correction_norm = math.sqrt(sum(value * value for value in correction))
    if correction_norm <= step_limit:
        return correction
    scale = step_limit / correction_norm
    return tuple(value * scale for value in correction)


def _partial_box_release_is_valid(
    start_angle: float,
    actual_angle: float,
    probe: dict,
    target_object: str,
    minimum_opening_delta: float,
) -> bool:
    """Accept a wall-limited first opening only after the clamp is released."""
    contact_pair = (
        probe.get("left_finger_contact_object"),
        probe.get("right_finger_contact_object"),
    )
    return (
        start_angle - actual_angle >= minimum_opening_delta
        and bool(probe.get("physics_enabled", False))
        and bool(probe.get("gravity_enabled", False))
        and bool(probe.get("contact_report_available", False))
        and contact_pair != (target_object, target_object)
    )


def _cross(a: Vector3, b: Vector3) -> Vector3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _rotation_matrix_to_quaternion(columns: Sequence[Vector3]) -> Quaternion:
    x_axis, y_axis, z_axis = columns
    m00, m10, m20 = x_axis
    m01, m11, m21 = y_axis
    m02, m12, m22 = z_axis
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        quaternion = (
            (m21 - m12) / s,
            (m02 - m20) / s,
            (m10 - m01) / s,
            0.25 * s,
        )
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        quaternion = (
            0.25 * s,
            (m01 + m10) / s,
            (m02 + m20) / s,
            (m21 - m12) / s,
        )
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        quaternion = (
            (m01 + m10) / s,
            0.25 * s,
            (m12 + m21) / s,
            (m02 - m20) / s,
        )
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        quaternion = (
            (m02 + m20) / s,
            (m12 + m21) / s,
            0.25 * s,
            (m10 - m01) / s,
        )
    return _normalize(quaternion)


def orientation_for_approach(
    direction: Vector3, roll_deg: float = 120.0
) -> Quaternion:
    """Point +X at the object and keep the wrist camera above the gripper."""
    x_axis = _normalize(direction)
    reference = (0.0, 0.0, 1.0)
    if abs(sum(a * b for a, b in zip(x_axis, reference))) > 0.94:
        reference = (0.0, 1.0, 0.0)
    projection = sum(a * b for a, b in zip(reference, x_axis))
    z_axis = _normalize(
        tuple(
            reference[index] - projection * x_axis[index]
            for index in range(3)
        )
    )
    y_axis = _normalize(_cross(z_axis, x_axis))
    # The calibrated D435 sits on local -Z. A 120 degree roll moves it above
    # and to the side of the fingers, while remaining inside joint6's range.
    roll = math.radians(roll_deg)
    cosine = math.cos(roll)
    sine = math.sin(roll)
    rolled_y = tuple(
        cosine * y_axis[index] + sine * z_axis[index]
        for index in range(3)
    )
    rolled_z = tuple(
        -sine * y_axis[index] + cosine * z_axis[index]
        for index in range(3)
    )
    return _rotation_matrix_to_quaternion((x_axis, rolled_y, rolled_z))


def orientation_for_approach_and_pinch_axis(
    direction: Vector3, pinch_axis: Vector3
) -> Quaternion:
    """Point +X at the object and align the finger-closing plane with +Z."""
    x_axis = _normalize(direction)
    projection = sum(a * b for a, b in zip(pinch_axis, x_axis))
    z_axis = _normalize(
        tuple(
            pinch_axis[index] - projection * x_axis[index]
            for index in range(3)
        )
    )
    y_axis = _normalize(_cross(z_axis, x_axis))
    return _rotation_matrix_to_quaternion((x_axis, y_axis, z_axis))


class TransportHeightError(RuntimeError):
    """The object is still held, but transport clearance was not achieved."""


class GraspAssistant(Node):
    def __init__(self) -> None:
        super().__init__("smolvla_grasp_assistant")
        self.declare_parameter("target_object", "apple")
        self.declare_parameter("freeze_target_keep_collisions", False)
        self.declare_parameter("bottle_recovery_variant", "baseline")
        self.declare_parameter("base_frame", "link00")
        self.declare_parameter("end_effector_link", "link06")
        self.declare_parameter("group_name", "z1_arm")
        self.declare_parameter("candidate_count", 8)
        self.declare_parameter("planning_time_s", 4.0)
        self.declare_parameter("speed_scale", 0.35)
        self.declare_parameter("standoff_min_m", 0.20)
        self.declare_parameter("standoff_max_m", 0.23)
        self.declare_parameter("minimum_pregrasp_z_m", 0.12)
        self.declare_parameter("azimuth_spread_deg", 28.0)
        self.declare_parameter("position_tolerance_m", 0.010)
        self.declare_parameter("orientation_tolerance_rad", 0.30)
        self.declare_parameter("object_timeout_s", 5.0)
        self.declare_parameter("max_joint_delta_rad", 2.4)
        self.declare_parameter("jog_max_joint_delta_rad", 0.35)
        # The URDF mover rotates about +Y: 0 rad is mechanically closed and
        # increasingly negative angles open the gap.  Earlier code had these
        # semantics reversed and could only press fruit against the tips.
        # Match the Isaac reset pose.  This remains 0.15 rad wider than the
        # bottle profile's widest accepted contact angle, without demanding
        # the unnecessary 69-degree extreme opening during every approach.
        self.declare_parameter("gripper_open_rad", -0.90)
        self.declare_parameter("gripper_closed_rad", 0.0)
        self.declare_parameter("gripper_tolerance_rad", 0.03)
        self.declare_parameter(
            "gripper_grasp_threshold_rad",
            float(
                GRASP_CONFIG["gripper"][
                    "obstruction_min_position_rad"
                ]
            ),
        )
        self.declare_parameter("gripper_release_threshold_rad", -0.75)
        self.declare_parameter("gripper_contact_open_limit_rad", -0.05)
        self.declare_parameter(
            "gripper_contact_closed_limit_rad",
            float(
                GRASP_CONFIG["gripper"][
                    "obstruction_min_position_rad"
                ]
            ),
        )
        self.declare_parameter("manual_minimum_z_m", 0.04)
        self.declare_parameter("guided_lift_distance_m", 0.07)
        unfreeze_config = GRASP_CONFIG["unfreeze"]
        lift_config = GRASP_CONFIG["physical_test_lift"]
        gripper_config = GRASP_CONFIG["gripper"]
        self.declare_parameter(
            "unfreeze_settle_duration_s",
            float(unfreeze_config["settle_duration_s"]),
        )
        self.declare_parameter(
            "grasp_alignment_velocity_scale",
            float(unfreeze_config["alignment_velocity_scale"]),
        )
        self.declare_parameter(
            "unfreeze_settle_timeout_s",
            float(unfreeze_config["settle_timeout_s"]),
        )
        self.declare_parameter(
            "unfreeze_poll_period_s",
            float(unfreeze_config["poll_period_s"]),
        )
        self.declare_parameter(
            "unfreeze_stable_samples_required",
            int(unfreeze_config["stable_samples_required"]),
        )
        self.declare_parameter(
            "unfreeze_preclose_envelope_margin_m",
            float(unfreeze_config["preclose_envelope_margin_m"]),
        )
        self.declare_parameter(
            "preclose_clearance_correction_max_attempts",
            int(unfreeze_config["clearance_correction_max_attempts"]),
        )
        self.declare_parameter(
            "preclose_clearance_correction_step_m",
            float(unfreeze_config["clearance_correction_step_m"]),
        )
        self.declare_parameter(
            "preclose_clearance_correction_target_inset_m",
            float(unfreeze_config["clearance_correction_target_inset_m"]),
        )
        self.declare_parameter(
            "unfreeze_max_position_drift_m",
            float(unfreeze_config["max_position_drift_m"]),
        )
        self.declare_parameter(
            "unfreeze_max_linear_speed_m_s",
            float(unfreeze_config["max_linear_speed_m_s"]),
        )
        self.declare_parameter(
            "unfreeze_max_angular_speed_rad_s",
            float(unfreeze_config["max_angular_speed_rad_s"]),
        )
        self.declare_parameter(
            "physical_test_lift_relative_position_tolerance_m",
            float(lift_config["relative_position_tolerance_m"]),
        )
        self.declare_parameter(
            "physical_test_lift_loaded_envelope_margin_m",
            float(lift_config["loaded_envelope_margin_m"]),
        )
        self.declare_parameter(
            "transport_loaded_grasp_guard_margin_m",
            float(lift_config["transport_guard_margin_m"]),
        )
        self.declare_parameter(
            "physical_test_lift_velocity_scale",
            float(lift_config["velocity_scale"]),
        )
        self.declare_parameter(
            "physical_test_lift_verification_poll_s",
            float(lift_config["verification_poll_period_s"]),
        )
        self.declare_parameter(
            "loaded_grasp_samples_required",
            int(lift_config["loaded_grasp_samples_required"]),
        )
        self.declare_parameter(
            "loaded_grasp_max_attempts",
            int(lift_config["loaded_grasp_max_attempts"]),
        )
        self.declare_parameter(
            "gripper_close_duration_s",
            float(gripper_config["close_duration_s"]),
        )
        self.declare_parameter(
            "gripper_open_duration_s",
            float(gripper_config["open_duration_s"]),
        )
        self.declare_parameter(
            "gripper_obstruction_window_s",
            float(gripper_config["obstruction_window_s"]),
        )
        self.declare_parameter(
            "gripper_obstruction_max_motion_rad",
            float(gripper_config["obstruction_max_motion_rad"]),
        )
        self.declare_parameter(
            "gripper_stator_support_z_max_m",
            float(gripper_config["stator_support_z_max_m"]),
        )
        self.declare_parameter(
            "gripper_preclose_seating_z_target_m",
            float(gripper_config["preclose_seating_z_target_m"]),
        )
        self.declare_parameter(
            "gripper_preclose_seating_max_translation_m",
            float(gripper_config["preclose_seating_max_translation_m"]),
        )
        self.declare_parameter(
            "gripper_preclose_seating_step_m",
            float(gripper_config["preclose_seating_step_m"]),
        )
        self.declare_parameter(
            "gripper_contact_hold_preload_rad",
            float(gripper_config["contact_hold_preload_rad"]),
        )
        self.declare_parameter(
            "gripper_contact_hold_preload_step_rad",
            float(gripper_config["contact_hold_preload_step_rad"]),
        )
        self.declare_parameter(
            "gripper_contact_hold_duration_s",
            float(gripper_config["contact_hold_duration_s"]),
        )
        self.declare_parameter(
            "gripper_contact_unjam_lift_m",
            float(gripper_config["contact_unjam_lift_m"]),
        )
        self.declare_parameter(
            "gripper_contact_stator_seating_m",
            float(gripper_config["contact_stator_seating_m"]),
        )
        self.declare_parameter("box_center_x_m", 0.32)
        self.declare_parameter("box_center_y_m", 0.14)
        self.declare_parameter("box_approach_object_z_m", 0.14)
        # Release just above the 0.10 m walls and let gravity settle the item.
        self.declare_parameter("box_release_object_z_m", 0.115)
        self.declare_parameter("box_release_min_opening_delta_rad", 0.10)
        self.declare_parameter("bottle_servo_max_error_m", 0.025)
        self.declare_parameter("bottle_servo_max_correction_m", 0.025)
        self.declare_parameter("bottle_servo_max_speed_m_s", 0.020)
        self.declare_parameter("bottle_servo_gain", 2.0)
        self.declare_parameter("bottle_servo_timeout_s", 2.5)
        self.declare_parameter("bottle_servo_settle_s", 0.25)
        self.declare_parameter("bottle_servo_enabled", False)
        self.declare_parameter(
            "bottle_preclose_alignment_guard_enabled", False
        )

        self.target_object = str(self.get_parameter("target_object").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.ee_link = str(self.get_parameter("end_effector_link").value)
        self.group_name = str(self.get_parameter("group_name").value)
        self.candidate_count = int(self.get_parameter("candidate_count").value)
        self.planning_time = float(self.get_parameter("planning_time_s").value)
        self.speed_scale = float(self.get_parameter("speed_scale").value)
        self.standoff_min = float(self.get_parameter("standoff_min_m").value)
        self.standoff_max = float(self.get_parameter("standoff_max_m").value)
        self.minimum_pregrasp_z = float(
            self.get_parameter("minimum_pregrasp_z_m").value
        )
        self.azimuth_spread = math.radians(
            float(self.get_parameter("azimuth_spread_deg").value)
        )
        self.position_tolerance = float(
            self.get_parameter("position_tolerance_m").value
        )
        self.orientation_tolerance = float(
            self.get_parameter("orientation_tolerance_rad").value
        )
        self.object_timeout = float(
            self.get_parameter("object_timeout_s").value
        )
        self.max_joint_delta = float(
            self.get_parameter("max_joint_delta_rad").value
        )
        self.jog_max_joint_delta = float(
            self.get_parameter("jog_max_joint_delta_rad").value
        )
        self.gripper_open = float(
            self.get_parameter("gripper_open_rad").value
        )
        self.gripper_closed = float(
            self.get_parameter("gripper_closed_rad").value
        )
        self.gripper_tolerance = float(
            self.get_parameter("gripper_tolerance_rad").value
        )
        self.gripper_grasp_threshold = float(
            self.get_parameter("gripper_grasp_threshold_rad").value
        )
        self.gripper_release_threshold = float(
            self.get_parameter("gripper_release_threshold_rad").value
        )
        self.gripper_contact_open_limit = float(
            self.get_parameter("gripper_contact_open_limit_rad").value
        )
        self.gripper_contact_closed_limit = float(
            self.get_parameter("gripper_contact_closed_limit_rad").value
        )
        self.manual_minimum_z = float(
            self.get_parameter("manual_minimum_z_m").value
        )
        self.guided_lift_distance = float(
            self.get_parameter("guided_lift_distance_m").value
        )
        self.unfreeze_settle_duration = float(
            self.get_parameter("unfreeze_settle_duration_s").value
        )
        self.grasp_alignment_velocity_scale = float(
            self.get_parameter("grasp_alignment_velocity_scale").value
        )
        self.bottle_recovery_variant = str(
            self.get_parameter("bottle_recovery_variant").value
        )
        if self.bottle_recovery_variant not in BOTTLE_RECOVERY_VARIANTS:
            raise ValueError(
                f"bottle_recovery_variant 必须是 {BOTTLE_RECOVERY_VARIANTS}"
            )
        self.unfreeze_settle_timeout = float(
            self.get_parameter("unfreeze_settle_timeout_s").value
        )
        self.unfreeze_poll_period = float(
            self.get_parameter("unfreeze_poll_period_s").value
        )
        self.unfreeze_stable_samples_required = max(
            1,
            int(
                self.get_parameter(
                    "unfreeze_stable_samples_required"
                ).value
            ),
        )
        self.unfreeze_preclose_envelope_margin = max(
            0.0,
            float(
                self.get_parameter(
                    "unfreeze_preclose_envelope_margin_m"
                ).value
            ),
        )
        self.preclose_clearance_correction_max_attempts = max(
            1,
            int(
                self.get_parameter(
                    "preclose_clearance_correction_max_attempts"
                ).value
            ),
        )
        self.preclose_clearance_correction_step = max(
            0.0002,
            float(
                self.get_parameter(
                    "preclose_clearance_correction_step_m"
                ).value
            ),
        )
        self.preclose_clearance_correction_target_inset = max(
            0.0,
            float(
                self.get_parameter(
                    "preclose_clearance_correction_target_inset_m"
                ).value
            ),
        )
        self.unfreeze_max_position_drift = float(
            self.get_parameter("unfreeze_max_position_drift_m").value
        )
        self.unfreeze_max_linear_speed = float(
            self.get_parameter("unfreeze_max_linear_speed_m_s").value
        )
        self.unfreeze_max_angular_speed = float(
            self.get_parameter("unfreeze_max_angular_speed_rad_s").value
        )
        self.physical_test_lift_relative_position_tolerance = float(
            self.get_parameter(
                "physical_test_lift_relative_position_tolerance_m"
            ).value
        )
        self.physical_test_lift_loaded_envelope_margin = float(
            self.get_parameter(
                "physical_test_lift_loaded_envelope_margin_m"
            ).value
        )
        self.transport_loaded_grasp_guard_margin = max(
            self.physical_test_lift_loaded_envelope_margin,
            float(
                self.get_parameter(
                    "transport_loaded_grasp_guard_margin_m"
                ).value
            ),
        )
        self.physical_test_lift_velocity_scale = float(
            self.get_parameter("physical_test_lift_velocity_scale").value
        )
        self.physical_test_lift_verification_poll = max(
            0.05,
            float(
                self.get_parameter(
                    "physical_test_lift_verification_poll_s"
                ).value
            ),
        )
        self.loaded_grasp_samples_required = max(
            1,
            int(self.get_parameter("loaded_grasp_samples_required").value),
        )
        self.loaded_grasp_max_attempts = max(
            self.loaded_grasp_samples_required,
            int(self.get_parameter("loaded_grasp_max_attempts").value),
        )
        self.gripper_close_duration = float(
            self.get_parameter("gripper_close_duration_s").value
        )
        self.gripper_open_duration = float(
            self.get_parameter("gripper_open_duration_s").value
        )
        self.gripper_obstruction_window = float(
            self.get_parameter("gripper_obstruction_window_s").value
        )
        self.gripper_obstruction_max_motion = float(
            self.get_parameter("gripper_obstruction_max_motion_rad").value
        )
        self.gripper_stator_support_z_max = float(
            self.get_parameter("gripper_stator_support_z_max_m").value
        )
        self.gripper_preclose_seating_z_target = float(
            self.get_parameter(
                "gripper_preclose_seating_z_target_m"
            ).value
        )
        self.gripper_preclose_seating_max_translation = float(
            self.get_parameter(
                "gripper_preclose_seating_max_translation_m"
            ).value
        )
        self.gripper_preclose_seating_step = float(
            self.get_parameter("gripper_preclose_seating_step_m").value
        )
        self.gripper_contact_hold_preload = max(
            0.0,
            float(
                self.get_parameter(
                    "gripper_contact_hold_preload_rad"
                ).value
            ),
        )
        self.gripper_contact_hold_preload_step = max(
            0.001,
            min(
                self.gripper_contact_hold_preload,
                float(
                    self.get_parameter(
                        "gripper_contact_hold_preload_step_rad"
                    ).value
                ),
            ),
        )
        self.gripper_contact_hold_duration = max(
            5.0,
            float(
                self.get_parameter(
                    "gripper_contact_hold_duration_s"
                ).value
            ),
        )
        self.gripper_contact_unjam_lift = max(
            0.0,
            float(
                self.get_parameter(
                    "gripper_contact_unjam_lift_m"
                ).value
            ),
        )
        self.gripper_contact_stator_seating = max(
            0.0,
            float(
                self.get_parameter(
                    "gripper_contact_stator_seating_m"
                ).value
            ),
        )
        self.box_center = (
            float(self.get_parameter("box_center_x_m").value),
            float(self.get_parameter("box_center_y_m").value),
        )
        self.box_approach_object_z = float(
            self.get_parameter("box_approach_object_z_m").value
        )
        self.box_release_object_z = float(
            self.get_parameter("box_release_object_z_m").value
        )
        self.box_release_min_opening_delta = float(
            self.get_parameter(
                "box_release_min_opening_delta_rad"
            ).value
        )
        self.bottle_servo_max_error = max(
            0.0, float(self.get_parameter("bottle_servo_max_error_m").value)
        )
        self.bottle_servo_max_correction = max(
            0.0,
            float(self.get_parameter("bottle_servo_max_correction_m").value),
        )
        self.bottle_servo_max_speed = max(
            0.001,
            float(self.get_parameter("bottle_servo_max_speed_m_s").value),
        )
        self.bottle_servo_gain = max(
            0.1, float(self.get_parameter("bottle_servo_gain").value)
        )
        self.bottle_servo_timeout = max(
            0.2, float(self.get_parameter("bottle_servo_timeout_s").value)
        )
        self.bottle_servo_settle = max(
            0.0, float(self.get_parameter("bottle_servo_settle_s").value)
        )
        self.bottle_servo_enabled = bool(
            self.get_parameter("bottle_servo_enabled").value
        )
        self.bottle_preclose_alignment_guard_enabled = bool(
            self.get_parameter(
                "bottle_preclose_alignment_guard_enabled"
            ).value
        )

        self._lock = threading.RLock()
        self._operation_lock = threading.Lock()
        self._rng = random.SystemRandom()
        self._object_poses: Dict[str, Tuple[float, Vector3]] = {}
        self._object_orientations: Dict[
            str, Tuple[float, Quaternion]
        ] = {}
        self._joint_positions: Dict[str, float] = {}
        self._gripper_feedback = None
        self._approved_trajectory: Optional[RobotTrajectory] = None
        self._approved_target = ""
        self._grasp_candidate: Optional[dict] = None
        self._task_status: Dict[str, object] = {}
        self._scene_grasp_status: Dict[str, object] = {}
        self._scene_grasp_contact_status: Dict[str, object] = {}
        self._verified_grasp_sequence = False
        self._loaded_grasp_reference_local: Optional[
            Tuple[float, float, float]
        ] = None
        self._grasp_control_state = {
            "gripper_commanded_closed": False,
            "gripper_obstructed": False,
            "physical_test_lift_active": False,
            "physical_grasp_ok": False,
            "release_started": False,
            "release_sequence_complete": False,
            "transport_failure_reason": "",
        }
        self._active_goal_handle = None
        self._persistent_gripper_goal_handle = None
        self._stop_requested = False
        self._status = {
            "phase": "IDLE",
            "message": "等待一键自动示教",
            "plan_ready": False,
            "target_object": self.target_object,
        }

        latched_qos = QoSProfile(depth=1)
        latched_qos.reliability = ReliabilityPolicy.RELIABLE
        latched_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        callback_group = ReentrantCallbackGroup()
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(
            self._tf_buffer, self, spin_thread=False
        )
        self._status_pub = self.create_publisher(
            String, "/smolvla/pregrasp_status", latched_qos
        )
        self._display_pub = self.create_publisher(
            DisplayTrajectory, "/display_planned_path", 10
        )
        self._scene_grasp_pub = self.create_publisher(
            String, "/smolvla/scene_grasp_command", 10
        )
        self._grasp_control_pub = self.create_publisher(
            String, "/smolvla/grasp_control_state", latched_qos
        )
        self._servo_twist_pub = self.create_publisher(
            TwistStamped, "/servo_node/delta_twist_cmds", 10
        )
        self.create_subscription(
            String,
            "/smolvla/target_object",
            self._target_callback,
            10,
            callback_group=callback_group,
        )
        self.create_subscription(
            TFMessage,
            "/isaac/object_transforms",
            self._object_callback,
            20,
            callback_group=callback_group,
        )
        self.create_subscription(
            JointState,
            "/joint_states",
            self._joint_callback,
            20,
            callback_group=callback_group,
        )
        self.create_subscription(
            String,
            "/smolvla/status",
            self._task_status_callback,
            10,
            callback_group=callback_group,
        )
        self.create_subscription(
            String,
            "/smolvla/scene_grasp_status",
            self._scene_grasp_status_callback,
            latched_qos,
            callback_group=callback_group,
        )
        self._move_group = ActionClient(
            self, MoveGroup, "/move_action", callback_group=callback_group
        )
        self._bottle_collision_client = self.create_client(
            SetBool, "/smolvla/set_bottle_finger_collision",
            callback_group=callback_group,
        )
        self._compute_ik_client = self.create_client(
            GetPositionIK,
            "/compute_ik",
            callback_group=callback_group,
        )
        self._trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/joint_trajectory_controller/follow_joint_trajectory",
            callback_group=callback_group,
        )
        self._gripper_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/gripper_trajectory_controller/follow_joint_trajectory",
            callback_group=callback_group,
        )
        self._servo_start_client = self.create_client(
            Trigger,
            "/servo_node/start_servo",
            callback_group=callback_group,
        )
        self._servo_stop_client = self.create_client(
            Trigger,
            "/servo_node/stop_servo",
            callback_group=callback_group,
        )
        self._servo_stream_client = self.create_client(
            SetBool,
            "/smolvla/enable_servo_stream",
            callback_group=callback_group,
        )
        self.create_service(
            Trigger,
            "/smolvla/auto_generate_pregrasp",
            self._generate_for_auto_service,
            callback_group=callback_group,
        )
        self.create_service(
            Trigger,
            "/smolvla/auto_execute_pregrasp",
            self._execute_for_auto_service,
            callback_group=callback_group,
        )
        self.create_service(
            Trigger,
            "/smolvla/stop_assistant",
            self._stop_assistant_service,
            callback_group=callback_group,
        )
        self.create_service(
            Trigger,
            "/smolvla/reset_assistant",
            self._reset_assistant_service,
            callback_group=callback_group,
        )
        self.create_service(
            Trigger,
            "/smolvla/auto_guided_grasp",
            self._guided_grasp_service,
            callback_group=callback_group,
        )
        self.create_service(
            Trigger,
            "/smolvla/auto_guided_place",
            self._guided_place_service,
            callback_group=callback_group,
        )
        self._publish_status()
        self._publish_grasp_control_state()
        self.get_logger().info("SmolVLA automatic grasp assistant ready")

    def _target_callback(self, message: String) -> None:
        target = message.data.strip().lower()
        if target not in OBJECT_FRAME_NAMES:
            return
        with self._lock:
            self.target_object = target
            self._approved_trajectory = None
            self._approved_target = ""
            self._grasp_candidate = None
            self._verified_grasp_sequence = False
            self._loaded_grasp_reference_local = None
            self._reset_grasp_control_state_locked()
        self._publish_grasp_control_state()
        self._set_status("IDLE", "目标已改变，可以开始自动示教", False)

    def _object_callback(self, message: TFMessage) -> None:
        for transform in message.transforms:
            name = canonical_object_name(transform.child_frame_id)
            if name is None:
                continue
            position = transform.transform.translation
            orientation = transform.transform.rotation
            sample_time = time.monotonic()
            with self._lock:
                self._object_poses[name] = (
                    sample_time,
                    (float(position.x), float(position.y), float(position.z)),
                )
                self._object_orientations[name] = (
                    sample_time,
                    (
                        float(orientation.x),
                        float(orientation.y),
                        float(orientation.z),
                        float(orientation.w),
                    ),
                )

    def _joint_callback(self, message: JointState) -> None:
        with self._lock:
            for name, position in zip(message.name, message.position):
                self._joint_positions[name] = float(position)
                if name == "jointGripper":
                    self._gripper_feedback = (
                        time.monotonic(),
                        message.header.stamp.sec + message.header.stamp.nanosec * 1e-9,
                    )

    def _task_status_callback(self, message: String) -> None:
        try:
            status = json.loads(message.data)
        except json.JSONDecodeError:
            return
        with self._lock:
            self._task_status = status

    def _scene_grasp_status_callback(self, message: String) -> None:
        try:
            status = json.loads(message.data)
        except json.JSONDecodeError:
            return
        # Periodic contact telemetry must not overwrite a command response.
        if status.get("phase") == "CONTACT_STATE":
            with self._lock:
                self._scene_grasp_contact_status = status
            return
        with self._lock:
            self._scene_grasp_status = status

    def _stop_assistant_service(self, _request, response):
        with self._lock:
            self._stop_requested = True
            self._approved_trajectory = None
            self._approved_target = ""
            self._grasp_candidate = None
            active_handle = self._active_goal_handle
            persistent_handle = self._persistent_gripper_goal_handle
            self._persistent_gripper_goal_handle = None
        for goal_handle in (active_handle, persistent_handle):
            if goal_handle is not None:
                goal_handle.cancel_goal_async()
        self._set_grasp_control_state(
            gripper_commanded_closed=False,
            gripper_obstructed=False,
            physical_test_lift_active=False,
            physical_grasp_ok=False,
        )
        # Never leave a target kinematic/frozen after an emergency stop.
        self._command_scene_grasp("unfreeze")
        self._set_status(
            "STOPPED",
            "用户已强制停止抓取；运动取消，目标已恢复动态物理",
            False,
        )
        response.success = True
        response.message = "抓取停止请求已执行，正在保持当前关节位置"
        return response

    def _raise_if_stop_requested(self) -> None:
        with self._lock:
            stopped = self._stop_requested
        if stopped:
            raise RuntimeError("用户强制停止抓取")

    def _reset_assistant_service(self, _request, response):
        with self._lock:
            self._approved_trajectory = None
            self._approved_target = ""
            self._grasp_candidate = None
            self._verified_grasp_sequence = False
            self._loaded_grasp_reference_local = None
            self._stop_requested = False
            self._active_goal_handle = None
            self._reset_grasp_control_state_locked()
        self._publish_grasp_control_state()
        self._set_status("IDLE", "Episode 已复位，可以开始自动示教", False)
        response.success = True
        response.message = "自动抓取状态已清除"
        return response

    def _guided_grasp_service(self, _request, response):
        return self._run_operation_service(
            self._guided_grasp, "引导抓取", response
        )

    def _guided_place_service(self, _request, response):
        return self._run_operation_service(
            self._guided_place, "引导放置", response
        )

    def _run_operation_service(self, operation, label, response):
        if not self._operation_lock.acquire(blocking=False):
            response.success = False
            response.message = "预抓取助手正在执行另一项操作"
            return response
        try:
            # Reset once per operation.  A stop must remain sticky across all
            # trajectories in a multi-segment approach.
            with self._lock:
                self._stop_requested = False
            response.message = operation()
            response.success = True
        except Exception as exc:
            self._set_status("ERROR", f"{label}失败：{exc}", False)
            response.success = False
            response.message = str(exc)
        finally:
            self._operation_lock.release()
        return response

    def _set_status(
        self,
        phase: str,
        message: str,
        plan_ready: bool,
        **details,
    ) -> None:
        with self._lock:
            self._status = {
                "phase": phase,
                "message": message,
                "plan_ready": plan_ready,
                "target_object": self.target_object,
                **details,
            }
        self._publish_status()

    def _publish_status(self) -> None:
        with self._lock:
            payload = dict(self._status)
        message = String()
        message.data = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        self._status_pub.publish(message)

    def _reset_grasp_control_state_locked(self) -> None:
        self._grasp_control_state = {
            "gripper_commanded_closed": False,
            "gripper_obstructed": False,
            "physical_test_lift_active": False,
            "physical_grasp_ok": False,
            "release_started": False,
            "release_sequence_complete": False,
        }

    def _set_grasp_control_state(self, **updates) -> None:
        with self._lock:
            self._grasp_control_state.update(updates)
        self._publish_grasp_control_state()

    def _publish_grasp_control_state(self) -> None:
        with self._lock:
            payload = {
                "target_object": self.target_object,
                **self._grasp_control_state,
            }
        message = String()
        message.data = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        self._grasp_control_pub.publish(message)

    def _generate_for_auto_service(self, _request, response):
        if not self._operation_lock.acquire(blocking=False):
            response.success = False
            response.message = "预抓取助手正在执行另一项操作"
            return response
        try:
            result = self._generate_plan()
            response.success = True
            response.message = result
        except Exception as exc:
            self._set_status("ERROR", str(exc), False)
            response.success = False
            response.message = str(exc)
        finally:
            self._operation_lock.release()
        return response

    def _execute_for_auto_service(self, _request, response):
        if not self._operation_lock.acquire(blocking=False):
            response.success = False
            response.message = "预抓取助手正在执行另一项操作"
            return response
        try:
            result = self._execute_plan()
            response.success = True
            response.message = result
        except Exception as exc:
            self._set_status("ERROR", str(exc), False)
            response.success = False
            response.message = str(exc)
        finally:
            self._operation_lock.release()
        return response

    def _current_object_position(self, target_object: str) -> Vector3:
        with self._lock:
            sample = self._object_poses.get(target_object)
        if sample is None:
            raise RuntimeError(f"尚未收到 {target_object} 的 Isaac 位姿")
        stamp, position = sample
        age = time.monotonic() - stamp
        if age > self.object_timeout:
            raise RuntimeError(f"{target_object} 位姿已过期 ({age:.2f}s)")
        return position

    def _current_object_orientation(
        self, target_object: str
    ) -> Quaternion:
        with self._lock:
            sample = self._object_orientations.get(target_object)
        if sample is None:
            raise RuntimeError(f"尚未收到 {target_object} 的 Isaac 姿态")
        stamp, orientation = sample
        age = time.monotonic() - stamp
        if age > self.object_timeout:
            raise RuntimeError(f"{target_object} 姿态已过期 ({age:.2f}s)")
        return orientation

    def _sample_candidate(
        self,
        object_position: Vector3,
        target_object: str,
        candidate_index: int = 0,
        avoid_box: bool = False,
        elevation_range_override: Optional[Tuple[float, float]] = None,
    ) -> dict:
        profile = GRASP_PROFILES[target_object]
        roll_deg = float(profile.get("roll_deg", 120.0))
        fallback_start = profile.get("fallback_roll_start_index")
        if fallback_start is not None and candidate_index >= fallback_start:
            candidate_index -= int(fallback_start)
            roll_deg = float(profile["fallback_roll_deg"])
        target_point = (
            object_position[0],
            object_position[1],
            object_position[2]
            + float(profile["height"])
            + float(profile.get("world_z_offset_m", 0.0)),
        )
        base_azimuth = math.atan2(target_point[1], target_point[0])
        pinch_axis = None
        if bool(
            profile.get(
                "use_object_axis_approach",
                profile.get("use_object_short_axis_approach", False),
            )
        ):
            object_orientation = self._current_object_orientation(
                target_object
            )
            object_approach_axis = tuple(
                float(value)
                for value in profile.get(
                    "object_approach_local_axis", (0.0, 1.0, 0.0)
                )
            )
            approach_axis = _planar_object_axis_toward_workspace(
                object_orientation,
                object_approach_axis,
                object_position,
            )
            if bool(
                profile.get("reverse_object_axis_approach", False)
            ):
                approach_axis = (-approach_axis[0], -approach_axis[1])
            base_azimuth = math.atan2(
                approach_axis[1], approach_axis[0]
            )
            if "object_pinch_local_axis" in profile:
                object_pinch_axis = tuple(
                    float(value)
                    for value in profile["object_pinch_local_axis"]
                )
                rotated_pinch_axis = rotate_vector(
                    object_orientation, object_pinch_axis
                )
                pinch_axis = _normalize(
                    (
                        rotated_pinch_axis[0],
                        rotated_pinch_axis[1],
                        0.0,
                    )
                )
                # A parallel gripper closes along an undirected axis: +Y and
                # -Y grasp the same two faces.  Alternate the equivalent wrist
                # rolls so MoveIt can use whichever IK branch is reachable.
                if candidate_index % 2:
                    pinch_axis = tuple(-value for value in pinch_axis)
        else:
            base_azimuth += math.radians(
                float(profile.get("azimuth_offset_deg", 0.0))
            )
        if avoid_box:
            # Below the box, a positive azimuth puts the open jaws' approach
            # waypoint farther from its near wall. Keep the original heading
            # as the final fallback.
            base_azimuth += math.radians(
                (45.0, 30.0, 15.0, 0.0)[candidate_index % 4]
            )
        elif profile.get("single_position_grasp", False):
            base_azimuth = math.radians(_candidate_azimuth_deg(
                math.degrees(base_azimuth), candidate_index,
                profile.get("candidate_azimuth_offsets_deg", ()),
            ))
        azimuth_spread = math.radians(
            float(
                profile.get(
                    "azimuth_spread_deg", math.degrees(self.azimuth_spread)
                )
            )
        )
        azimuth = base_azimuth + self._rng.uniform(
            -azimuth_spread, azimuth_spread
        )
        pitch_deg = profile.get("pitch_deg")
        elevation_range = profile["elevation_deg"] if elevation_range_override is None else elevation_range_override
        elevation = math.radians(
            self._rng.uniform(
                float(elevation_range[0]), float(elevation_range[1])
            )
        )
        pregrasp_range = profile.get(
            "pregrasp_m", (self.standoff_min, self.standoff_max)
        )
        distance = self._rng.uniform(
            float(pregrasp_range[0]), float(pregrasp_range[1])
        )
        if pitch_deg is not None:
            # Hardware convention: positive pitch raises the tool's local +X
            # approach axis above the world-horizontal plane.  The legacy
            # ``elevation_deg`` convention below is downward-positive.
            pitch = math.radians(float(pitch_deg))
            sampled_direction = (
                math.cos(pitch) * math.cos(azimuth),
                math.cos(pitch) * math.sin(azimuth),
                math.sin(pitch),
            )
        else:
            sampled_direction = (
                math.cos(elevation) * math.cos(azimuth),
                math.cos(elevation) * math.sin(azimuth),
                -math.sin(elevation),
            )
        pregrasp_values = [
            target_point[index] - distance * sampled_direction[index]
            for index in range(3)
        ]
        minimum_pregrasp_z = float(
            profile.get("minimum_pregrasp_z_m", self.minimum_pregrasp_z)
        )
        pregrasp_values[2] = max(pregrasp_values[2], minimum_pregrasp_z)
        pregrasp = tuple(pregrasp_values)
        approach_vector = tuple(
            target_point[index] - pregrasp[index] for index in range(3)
        )
        direction = _normalize(approach_vector)
        distance = math.sqrt(sum(value * value for value in approach_vector))
        elevation = math.asin(-direction[2])
        # Clamping the pregrasp position above the table must not silently
        # rewrite a hardware-calibrated tool pitch. It changes only where
        # the collision-free outer waypoint is placed; the tool's local +X
        # axis remains at the requested pitch for alignment and closure.
        orientation_direction = (
            sampled_direction if pitch_deg is not None else direction
        )
        return {
            "position": pregrasp,
            "orientation": (
                orientation_for_approach_and_pinch_axis(
                    direction, pinch_axis
                )
                if pinch_axis is not None
                else orientation_for_approach(
                    orientation_direction,
                    roll_deg,
                )
            ),
            "target_point": target_point,
            "azimuth_deg": math.degrees(azimuth),
            "elevation_deg": math.degrees(elevation),
            "roll_deg": roll_deg,
            "standoff_m": distance,
        }

    @staticmethod
    def _state_after_trajectory(result):
        state = copy.deepcopy(result.trajectory_start)
        trajectory = result.planned_trajectory.joint_trajectory
        if not trajectory.points:
            return state
        positions = dict(
            zip(state.joint_state.name, state.joint_state.position)
        )
        positions.update(
            zip(trajectory.joint_names, trajectory.points[-1].positions)
        )
        state.joint_state.name = list(positions)
        state.joint_state.position = list(positions.values())
        state.is_diff = True
        return state

    @staticmethod
    def _concatenate_plans(first_result, second_result):
        combined = copy.deepcopy(first_result)
        first = combined.planned_trajectory.joint_trajectory
        second = second_result.planned_trajectory.joint_trajectory
        if set(first.joint_names) != set(second.joint_names):
            raise RuntimeError("无法拼接关节集合不同的接近和抓取轨迹")
        if not first.points or not second.points:
            raise RuntimeError("无法拼接空的接近或抓取轨迹")
        offset_ns = (
            int(first.points[-1].time_from_start.sec) * 1_000_000_000
            + int(first.points[-1].time_from_start.nanosec)
        )
        previous_ns = offset_ns
        second_indices = [
            list(second.joint_names).index(name) for name in first.joint_names
        ]
        for source_point in second.points:
            point = copy.deepcopy(source_point)
            for field_name in (
                "positions", "velocities", "accelerations", "effort"
            ):
                values = list(getattr(point, field_name))
                if len(values) == len(second.joint_names):
                    setattr(
                        point,
                        field_name,
                        [values[position] for position in second_indices],
                    )
            relative_ns = (
                int(point.time_from_start.sec) * 1_000_000_000
                + int(point.time_from_start.nanosec)
            )
            combined_ns = offset_ns + relative_ns
            if combined_ns <= previous_ns:
                if all(
                    abs(float(a) - float(b)) <= 1.0e-9
                    for a, b in zip(
                        point.positions, first.points[-1].positions
                    )
                ):
                    continue
                combined_ns = previous_ns + 1_000_000
            point.time_from_start.sec = combined_ns // 1_000_000_000
            point.time_from_start.nanosec = combined_ns % 1_000_000_000
            first.points.append(point)
            previous_ns = combined_ns
        first.joint_names = list(first.joint_names)
        return combined

    def _generate_plan(self) -> str:
        if not self._move_group.wait_for_server(timeout_sec=3.0):
            raise RuntimeError("MoveIt /move_action 不可用")
        with self._lock:
            planning_target = self.target_object
            object_positions = {
                name: sample[1]
                for name, sample in self._object_poses.items()
                if time.monotonic() - sample[0] <= self.object_timeout
            }
        if GRASP_PROFILES[planning_target].get("single_pass_preclose", False):
            self._set_bottle_finger_collision(False)
        object_position = self._current_object_position(planning_target)
        profile = GRASP_PROFILES[planning_target]
        recovery_variant = (
            self.bottle_recovery_variant
            if planning_target == "beverage_bottle"
            else "baseline"
        )
        final_approach_velocity_scale = self.grasp_alignment_velocity_scale * (
            0.5 if recovery_variant != "baseline" else 1.0
        )
        avoid_box = (
            planning_target == "peach"
            and object_position[1] < BOX_EXCLUSION_Y[0]
            and _point_to_aabb_distance(
                object_position, BOX_EXCLUSION_X, BOX_EXCLUSION_Y
            ) < 0.10
        )
        fruit_tool_transform = None
        if (
            profile.get("single_position_grasp", False)
            or profile.get("manual_tuning_handoff", False)
        ):
            try:
                fruit_tool_transform = self._tf_buffer.lookup_transform(
                    self.ee_link, "gripperStator", Time()
                ).transform
            except tf2_ros.TransformException as exc:
                raise RuntimeError(f"无法读取末端到固定指变换：{exc}") from exc
        bottle_tool_transform = None
        if bool(profile.get("single_pass_preclose", False)):
            self._command_scene_grasp("unfreeze", planning_target)
            if not self._wait_scene_grasp(
                "UNFROZEN", "", 2.0
            ):
                raise RuntimeError("Isaac 未确认圆柱目标在规划前保持动态物理")
            object_position = self._current_object_position(planning_target)
            object_positions[planning_target] = object_position
            try:
                bottle_tool_transform = self._tf_buffer.lookup_transform(
                    self.ee_link, "gripperStator", Time()
                ).transform
            except tf2_ros.TransformException as exc:
                raise RuntimeError(
                    "无法读取 link06->gripperStator 标定变换："
                    f"{exc}"
                ) from exc
        self._set_status(
            "PLANNING",
            (
                "正在采样并规划候选最终抓取位"
                if bottle_tool_transform is not None
                else "正在采样并规划候选预抓取位"
            ),
            False,
        )
        best = None
        errors: List[str] = []
        deferred_ik_candidates = []
        candidate_count = int(profile.get("candidate_count", self.candidate_count))
        base_candidate_count = len(profile.get("candidate_azimuth_offsets_deg", ()))
        for index in range(candidate_count):
            candidate = self._sample_candidate(
                object_position, planning_target, candidate_index=index,
                avoid_box=avoid_box,
            )
            if (profile.get("plan_directly_to_alignment_pose", False)
                    and not profile.get("single_position_grasp", False)):
                candidate = _candidate_at_alignment_standoff(
                    candidate,
                    float(profile["alignment_standoff_m"]),
                )
            if bottle_tool_transform is not None:
                pitch_deg, roll_deg = _bottle_candidate_pose_variant(
                    index % base_candidate_count,
                    base_candidate_count,
                    float(
                        OBJECT_GRASP_PROFILES[planning_target].get(
                            "approach_pitch_max_deg", profile["pitch_deg"]
                        )
                    ),
                    float(
                        OBJECT_GRASP_PROFILES[planning_target].get(
                            "approach_pitch_min_deg", profile["pitch_deg"]
                        )
                    ),
                    float(profile["roll_deg"]),
                    bool(
                        OBJECT_GRASP_PROFILES[planning_target].get(
                            "mirror_wrist_roll_candidates", False
                        )
                    ),
                )
                candidate["azimuth_deg"] = _candidate_azimuth_deg(
                    candidate["azimuth_deg"],
                    index,
                    profile.get("candidate_azimuth_offsets_deg", ()),
                    profile.get("extra_candidate_azimuths_deg", ()),
                )
                azimuth = math.radians(candidate["azimuth_deg"])
                pitch = math.radians(pitch_deg)
                approach_direction = (
                    math.cos(pitch) * math.cos(azimuth),
                    math.cos(pitch) * math.sin(azimuth),
                    math.sin(pitch),
                )
                candidate["orientation"] = orientation_for_approach(
                    approach_direction, roll_deg
                )
                distance = float(profile["standoff"])
                position = [
                    float(candidate["target_point"][axis])
                    - distance * approach_direction[axis]
                    for axis in range(3)
                ]
                position[2] = max(
                    position[2],
                    float(
                        profile.get(
                            "minimum_pregrasp_z_m", self.minimum_pregrasp_z
                        )
                    ),
                )
                candidate["position"] = tuple(position)
                candidate["pitch_deg"] = pitch_deg
                candidate["roll_deg"] = roll_deg
                if recovery_variant != "baseline":
                    candidate["position"] = _bottle_recovery_pregrasp_position(
                        candidate["position"], candidate["azimuth_deg"],
                        recovery_variant,
                    )
                    candidate["standoff_m"] = math.dist(
                        candidate["target_point"], candidate["position"]
                    )
            if fruit_tool_transform is not None:
                translation = fruit_tool_transform.translation
                rotation = fruit_tool_transform.rotation
                manual_tuning_target = bool(
                    profile.get("manual_tuning_handoff", False)
                )
                if manual_tuning_target:
                    desired_local = tuple(
                        float(value)
                        for value in profile[
                            "manual_tuning_object_in_stator_m"
                        ]
                    )
                else:
                    desired_local = tuple(
                        float(profile[f"desired_center_{axis}_m"])
                        for axis in ("x", "y", "z")
                    )
                ee_to_stator_orientation = (
                    rotation.x, rotation.y, rotation.z, rotation.w
                )
                stator_orientation = quaternion_multiply(
                    candidate["orientation"], ee_to_stator_orientation
                )
                clearance_lift = (
                    0.0
                    if manual_tuning_target
                    else float(
                        profile.get(
                            "single_position_world_clearance_lift_m", 0.0
                        )
                    )
                )
                base_position = _ee_position_for_object_in_stator(
                    candidate["target_point"], candidate["orientation"],
                    (translation.x, translation.y, translation.z),
                    ee_to_stator_orientation,
                    desired_local,
                )
                candidate["position"] = (
                    base_position[0],
                    base_position[1],
                    base_position[2] + clearance_lift,
                )
                desired_local = _object_in_stator_after_world_tool_offset(
                    desired_local,
                    stator_orientation,
                    (0.0, 0.0, clearance_lift),
                )
                candidate["desired_object_in_stator_m"] = desired_local
                candidate["final_grasp_target"] = not manual_tuning_target
                candidate["manual_tuning_target"] = manual_tuning_target
                candidate["world_clearance_lift_m"] = clearance_lift
                candidate["standoff_m"] = math.dist(
                    candidate["target_point"], candidate["position"])
            sweep_position = candidate["position"]
            insertion_timed_out = False
            sweep_blockers = _candidate_sweep_blockers(
                planning_target,
                candidate["target_point"],
                sweep_position,
                object_positions,
                approach_beyond_target_m=profile.get(
                    "approach_beyond_target_m"
                ),
            )
            # Manual calibration requests one exact, collision-checked pose.
            # The legacy analytic sweep assumes an automatic lift/transport
            # sequence and can reject this stationary handoff because another
            # object occupies space that the calibration move never uses.
            # MoveIt still checks the complete arm path against every current
            # planning-scene collision object.
            if sweep_blockers and not candidate.get(
                "manual_tuning_target", False
            ):
                blocker_summary = ", ".join(
                    f"{name}({reason})"
                    for name, reason in sorted(sweep_blockers.items())
                )
                errors.append(
                    f"candidate {index + 1}: 夹爪扫掠通道被占用："
                    f"{blocker_summary}"
                )
                continue
            final_candidate = None
            if bottle_tool_transform is not None:
                calibrated = profile["calibrated_grasp_reference"]
                translation = bottle_tool_transform.translation
                rotation = bottle_tool_transform.rotation
                final_candidate = copy.deepcopy(candidate)
                captured_local = tuple(
                    float(value)
                    for value in calibrated["object_in_gripper_local_m"]
                )
                additional_insertion_depth = float(
                    OBJECT_GRASP_PROFILES[planning_target].get(
                        "additional_insertion_depth_m", 0.0
                    )
                )
                captured_local = (
                    captured_local[0] - additional_insertion_depth,
                    captured_local[1],
                    captured_local[2],
                )
                reference_orientation = candidate["orientation"]
                if bool(
                    profile.get(
                        "alignment_reference_uses_tool_rotation", False
                    )
                ):
                    reference_orientation = quaternion_multiply(
                        reference_orientation,
                        (rotation.x, rotation.y, rotation.z, rotation.w),
                    )
                desired_local = _bottle_reference_for_orientation(
                    captured_local,
                    reference_orientation,
                    float(
                        calibrated[
                            "captured_object_center_minus_stator_world_z_m"
                        ]
                    ),
                )
                desired_local = _cylinder_candidate_local_reference(
                    desired_local,
                    index,
                    profile.get("candidate_local_offsets_yz_m", ()),
                )
                geometry_point = candidate["target_point"]
                alignment_z_offset = profile.get(
                    "alignment_geometry_z_offset_m"
                )
                if alignment_z_offset is not None:
                    geometry_point = (
                        object_position[0],
                        object_position[1],
                        object_position[2] + float(alignment_z_offset),
                    )
                final_candidate["position"] = _ee_position_for_object_in_stator(
                    geometry_point,
                    candidate["orientation"],
                    (
                        float(translation.x),
                        float(translation.y),
                        float(translation.z),
                    ),
                    (
                        float(rotation.x),
                        float(rotation.y),
                        float(rotation.z),
                        float(rotation.w),
                    ),
                    desired_local,
                )
                # Isaac probes the object center, while geometry_point may
                # be a lower grasp point. Convert the acceptance reference
                # after solving the pose; keep the planned motion unchanged.
                stator_orientation = quaternion_multiply(
                    candidate["orientation"],
                    (rotation.x, rotation.y, rotation.z, rotation.w),
                )
                center_z = object_position[2] + float(
                    GRASP_CONFIG["object_center_z_offsets_m"][planning_target]
                )
                final_candidate["desired_grasp_point_in_stator_m"] = desired_local
                final_candidate["planning_geometry_point_m"] = geometry_point
                final_candidate["desired_object_in_stator_m"] = (
                    _object_in_stator_after_world_tool_offset(
                        desired_local,
                        stator_orientation,
                        (0.0, 0.0, geometry_point[2] - center_z),
                    )
                )
                final_candidate["standoff_m"] = math.dist(
                    candidate["target_point"], final_candidate["position"]
                )
                final_candidate["final_grasp_target"] = True
                if (
                    profile.get("precheck_final_ik", False)
                    and not self._candidate_has_valid_ik(
                        final_candidate, open_gripper=avoid_box
                    )
                ):
                    candidate_error = (
                        f"candidate {index + 1}: 精确最终位姿 IK 预筛选未通过，"
                        "延后执行容差规划"
                    )
                    errors.append(candidate_error)
                    self.get_logger().warning(candidate_error)
                    deferred_ik_candidates.append(
                        (index, candidate, final_candidate)
                    )
                    continue
            try:
                result = self._plan_candidate(
                    candidate,
                    open_gripper=avoid_box,
                    position_tolerance=(0.001 if fruit_tool_transform is not None else None),
                    velocity_scale=(
                        self.grasp_alignment_velocity_scale
                        if fruit_tool_transform is not None
                        else _pregrasp_velocity_scale(
                            profile, self.speed_scale
                        )
                    ),
                    orientation_tolerance=float(
                        profile.get(
                            "manual_tuning_orientation_tolerance_rad",
                            profile.get(
                                "single_position_orientation_tolerance_rad",
                                profile.get(
                                    "pregrasp_orientation_tolerance_rad",
                                    self.orientation_tolerance,
                                ),
                            ),
                        )
                        if fruit_tool_transform is not None
                        else profile.get(
                            "pregrasp_orientation_tolerance_rad",
                            self.orientation_tolerance,
                        )
                    ),
                )
                if final_candidate is not None:
                    final_result = self._plan_candidate(
                        final_candidate,
                        linear=(planning_target != "beverage_bottle" and bool(
                            self.get_parameter("freeze_target_keep_collisions").value
                        )),
                        open_gripper=avoid_box,
                        position_tolerance=float(
                            profile.get(
                                "final_pose_position_tolerance_m", 0.001
                            )
                        ),
                        orientation_tolerance=0.020,
                        velocity_scale=final_approach_velocity_scale,
                        start_state=self._state_after_trajectory(result),
                    )
                    result = self._concatenate_plans(result, final_result)
                    candidate = final_candidate
                if planning_target == "apple":
                    self._plan_apple_transfer(
                        object_position, candidate["position"], candidate["orientation"],
                        max(self.box_approach_object_z, 0.14),
                        start_state=self._state_after_trajectory(result),
                    )
            except RuntimeError as exc:
                candidate_error = f"candidate {index + 1}: {exc}"
                errors.append(candidate_error)
                self.get_logger().warning(candidate_error)
                continue
            trajectory = result.planned_trajectory.joint_trajectory
            score, max_delta = self._trajectory_score(trajectory)
            allowed_max_delta = float(
                GRASP_PROFILES[planning_target].get(
                    "max_joint_delta_rad", self.max_joint_delta
                )
            )
            if max_delta > allowed_max_delta:
                candidate_error = (
                    f"candidate {index + 1}: joint delta {max_delta:.3f} rad "
                    f"> {allowed_max_delta:.3f} rad"
                )
                errors.append(candidate_error)
                self.get_logger().warning(candidate_error)
                continue
            if best is None or score < best[0]:
                best = (score, max_delta, candidate, result)
                if avoid_box or profile.get("accept_first_valid_candidate", False):
                    break
        if best is None:
            for deferred_index, candidate, final_candidate in (
                deferred_ik_candidates
            ):
                try:
                    result = self._plan_candidate(
                        candidate,
                        open_gripper=avoid_box,
                        velocity_scale=_pregrasp_velocity_scale(
                            profile, self.speed_scale
                        ),
                        orientation_tolerance=float(
                            profile.get(
                                "pregrasp_orientation_tolerance_rad",
                                self.orientation_tolerance,
                            )
                        ),
                    )
                    final_result = self._plan_candidate(
                        final_candidate,
                        linear=(planning_target != "beverage_bottle" and bool(
                            self.get_parameter("freeze_target_keep_collisions").value
                        )),
                        open_gripper=avoid_box,
                        position_tolerance=float(
                            profile["final_pose_position_tolerance_m"]
                        ),
                        orientation_tolerance=0.020,
                        velocity_scale=final_approach_velocity_scale,
                        start_state=self._state_after_trajectory(result),
                    )
                    result = self._concatenate_plans(result, final_result)
                    candidate = final_candidate
                    if planning_target == "apple":
                        self._plan_apple_transfer(
                            object_position, candidate["position"], candidate["orientation"],
                            max(self.box_approach_object_z, 0.14),
                            start_state=self._state_after_trajectory(result),
                        )
                except RuntimeError as exc:
                    candidate_error = (
                        f"candidate {deferred_index + 1}: {exc}"
                    )
                    errors.append(candidate_error)
                    self.get_logger().warning(candidate_error)
                    continue
                trajectory = result.planned_trajectory.joint_trajectory
                score, max_delta = self._trajectory_score(trajectory)
                allowed_max_delta = float(
                    profile.get("max_joint_delta_rad", self.max_joint_delta)
                )
                if max_delta > allowed_max_delta:
                    candidate_error = (
                        f"candidate {deferred_index + 1}: joint delta "
                        f"{max_delta:.3f} rad > {allowed_max_delta:.3f} rad"
                    )
                    errors.append(candidate_error)
                    self.get_logger().warning(candidate_error)
                    continue
                best = (score, max_delta, candidate, result)
                if avoid_box or profile.get("accept_first_valid_candidate", False):
                    break
        if best is None:
            summary = errors[-1] if errors else "MoveIt 未返回可执行轨迹"
            raise RuntimeError(f"{candidate_count} 个候选均失败：{summary}")

        score, max_delta, candidate, result = best
        with self._lock:
            if self.target_object != planning_target:
                raise RuntimeError("规划期间目标已改变，请重新生成")
            self._approved_trajectory = copy.deepcopy(
                result.planned_trajectory
            )
            self._approved_target = planning_target
            self._grasp_candidate = copy.deepcopy(candidate)
        display = DisplayTrajectory()
        display.trajectory_start = result.trajectory_start
        display.trajectory.append(result.planned_trajectory)
        self._display_pub.publish(display)
        position = candidate["position"]
        trajectory_label = (
            "最终抓取" if candidate.get("final_grasp_target") else "预抓取"
        )
        if candidate.get("direct_alignment_target"):
            trajectory_label = "安全对中"
        message = (
            f"{trajectory_label}轨迹已生成：位置 "
            f"[{position[0]:.3f}, {position[1]:.3f}, "
            f"{position[2]:.3f}] m，最大关节变化 {max_delta:.3f} rad"
        )
        self._set_status(
            "PLAN_READY",
            message,
            True,
            candidate_index="best",
            candidate_count=candidate_count,
            pregrasp_position_m=[round(value, 5) for value in position],
            orientation_xyzw=[
                round(value, 6) for value in candidate["orientation"]
            ],
            target_point_m=[
                round(value, 5) for value in candidate["target_point"]
            ],
            azimuth_deg=round(candidate["azimuth_deg"], 2),
            elevation_deg=round(candidate["elevation_deg"], 2),
            roll_deg=round(candidate["roll_deg"], 2),
            standoff_m=round(candidate["standoff_m"], 4),
            trajectory_score=round(score, 4),
            max_joint_delta_rad=round(max_delta, 4),
            trajectory_points=len(
                result.planned_trajectory.joint_trajectory.points
            ),
            target_sweep_clear=True,
            final_joint_names=list(
                result.planned_trajectory.joint_trajectory.joint_names
            ),
            final_joint_positions=[
                round(value, 6)
                for value in result.planned_trajectory.joint_trajectory.points[
                    -1
                ].positions
            ],
        )
        return message

    def _candidate_has_valid_ik(
        self, candidate: dict, open_gripper: bool = False
    ) -> bool:
        """Reject unreachable or colliding final goals before OMPL runs."""
        if not self._compute_ik_client.wait_for_service(timeout_sec=0.2):
            self.get_logger().warning(
                "MoveIt /compute_ik 不可用，跳过最终位姿预筛选"
            )
            return True
        request = GetPositionIK.Request()
        request.ik_request.group_name = self.group_name
        request.ik_request.ik_link_name = self.ee_link
        request.ik_request.avoid_collisions = True
        request.ik_request.robot_state.is_diff = True
        if open_gripper:
            request.ik_request.robot_state.joint_state.name = ["jointGripper"]
            request.ik_request.robot_state.joint_state.position = [
                self.gripper_open
            ]
        request.ik_request.timeout = Duration(seconds=1.0).to_msg()
        pose = PoseStamped()
        pose.header.frame_id = self.base_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        (
            pose.pose.position.x,
            pose.pose.position.y,
            pose.pose.position.z,
        ) = candidate["position"]
        quaternion = candidate["orientation"]
        pose.pose.orientation.x = quaternion[0]
        pose.pose.orientation.y = quaternion[1]
        pose.pose.orientation.z = quaternion[2]
        pose.pose.orientation.w = quaternion[3]
        request.ik_request.pose_stamped = pose
        response = self._wait_future(
            self._compute_ik_client.call_async(request), 1.5
        )
        if response is None:
            self.get_logger().warning(
                "MoveIt /compute_ik 超时，保留候选交由完整规划验证"
            )
            return True
        return response.error_code.val == MoveItErrorCodes.SUCCESS

    def _plan_candidate(
        self,
        candidate: dict,
        position_tolerance: Optional[float] = None,
        orientation_tolerance: Optional[float] = None,
        linear: bool = False,
        velocity_scale: Optional[float] = None,
        acceleration_scale: Optional[float] = None,
        start_state=None,
        open_gripper: bool = False,
    ):
        goal = MoveGroup.Goal()
        request = goal.request
        request.group_name = self.group_name
        if linear:
            request.pipeline_id = "pilz_industrial_motion_planner"
            request.planner_id = "LIN"
        else:
            # Native horizontal sequences use Pilz explicitly. All ordinary
            # free-space/loaded-carry requests must retain the OMPL pipeline.
            request.pipeline_id = "ompl"
        request.num_planning_attempts = 3
        request.allowed_planning_time = self.planning_time
        requested_scale = (
            self.speed_scale if velocity_scale is None else float(velocity_scale)
        )
        request.max_velocity_scaling_factor = requested_scale
        request.max_acceleration_scaling_factor = (
            requested_scale
            if acceleration_scale is None
            else float(acceleration_scale)
        )
        if start_state is None:
            request.start_state.is_diff = True
        else:
            request.start_state = copy.deepcopy(start_state)
        if open_gripper:
            # The arm trajectory is executed with jaws open. Make MoveIt's
            # collision geometry match that state even if the last grasp left
            # the current joint state closed during this planning request.
            joints = request.start_state.joint_state
            if "jointGripper" in joints.name:
                joints.position[joints.name.index("jointGripper")] = (
                    self.gripper_open
                )
            else:
                joints.name.append("jointGripper")
                joints.position.append(self.gripper_open)
                joints.velocity = []
                joints.effort = []
        request.goal_constraints = [
            self._pose_constraints(
                candidate,
                position_tolerance=position_tolerance,
                orientation_tolerance=orientation_tolerance,
            )
        ]
        goal.planning_options.plan_only = True
        goal.planning_options.look_around = False
        goal.planning_options.replan = False
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True

        send_future = self._move_group.send_goal_async(goal)
        goal_handle = self._wait_future(send_future, self.planning_time + 2.0)
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError("MoveGroup 拒绝候选")
        result_wrapper = self._wait_future(
            goal_handle.get_result_async(), self.planning_time + 4.0
        )
        if result_wrapper is None:
            raise RuntimeError("MoveGroup 未返回结果")
        result = result_wrapper.result
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(f"MoveIt error_code={result.error_code.val}")
        if not result.planned_trajectory.joint_trajectory.points:
            raise RuntimeError("MoveIt 返回空轨迹")
        return result

    def _pose_constraints(
        self,
        candidate: dict,
        position_tolerance: Optional[float] = None,
        orientation_tolerance: Optional[float] = None,
    ) -> Constraints:
        constraints = Constraints()
        constraints.name = "smolvla_pregrasp"
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [
            self.position_tolerance
            if position_tolerance is None
            else position_tolerance
        ]
        pose = Pose()
        (
            pose.position.x,
            pose.position.y,
            pose.position.z,
        ) = candidate["position"]
        pose.orientation.w = 1.0
        if "position_z_range" in candidate:
            lower, upper = map(float, candidate["position_z_range"])
            if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
                raise ValueError("Invalid target height range")
            radius = self.position_tolerance if position_tolerance is None else position_tolerance
            sphere.type = SolidPrimitive.BOX
            sphere.dimensions = [2.0 * radius, 2.0 * radius, upper - lower]
            pose.position.z = (lower + upper) / 2.0
        position = PositionConstraint()
        position.header.frame_id = self.base_frame
        position.link_name = self.ee_link
        offset = candidate.get("target_point_offset", (0.0, 0.0, 0.0))
        if len(offset) != 3 or not all(math.isfinite(float(v)) for v in offset):
            raise ValueError("Invalid link-frame target-point offset")
        position.target_point_offset.x = float(offset[0])
        position.target_point_offset.y = float(offset[1])
        position.target_point_offset.z = float(offset[2])
        position.constraint_region.primitives.append(sphere)
        position.constraint_region.primitive_poses.append(pose)
        position.weight = 1.0
        constraints.position_constraints.append(position)

        orientation = OrientationConstraint()
        orientation.header.frame_id = self.base_frame
        orientation.link_name = self.ee_link
        quaternion = candidate["orientation"]
        orientation.orientation.x = quaternion[0]
        orientation.orientation.y = quaternion[1]
        orientation.orientation.z = quaternion[2]
        orientation.orientation.w = quaternion[3]
        tolerance = (
            self.orientation_tolerance
            if orientation_tolerance is None
            else orientation_tolerance
        )
        orientation.absolute_x_axis_tolerance = tolerance
        orientation.absolute_y_axis_tolerance = tolerance
        orientation.absolute_z_axis_tolerance = tolerance
        orientation.weight = 1.0
        constraints.orientation_constraints.append(orientation)
        return constraints

    def _current_ee_pose(self) -> Tuple[Vector3, Quaternion]:
        try:
            transform = self._tf_buffer.lookup_transform(
                self.base_frame, self.ee_link, Time()
            ).transform
        except tf2_ros.TransformException as exc:
            raise RuntimeError(
                f"无法读取 {self.base_frame}->{self.ee_link}: {exc}"
            ) from exc
        return (
            (
                float(transform.translation.x),
                float(transform.translation.y),
                float(transform.translation.z),
            ),
            (
                float(transform.rotation.x),
                float(transform.rotation.y),
                float(transform.rotation.z),
                float(transform.rotation.w),
            ),
        )

    def _current_stator_orientation(self) -> Quaternion:
        """Return the frame used by Isaac's local grasp-position probes."""
        try:
            transform = self._tf_buffer.lookup_transform(
                self.base_frame, "gripperStator", Time()
            ).transform
        except tf2_ros.TransformException as exc:
            raise RuntimeError(
                f"无法读取 {self.base_frame}->gripperStator: {exc}"
            ) from exc
        return (
            float(transform.rotation.x),
            float(transform.rotation.y),
            float(transform.rotation.z),
            float(transform.rotation.w),
        )

    def _execute_manual_pose(
        self,
        position: Vector3,
        orientation: Quaternion,
        description: str,
        linear: bool = False,
        max_joint_delta: Optional[float] = None,
        position_tolerance: Optional[float] = None,
        orientation_tolerance: Optional[float] = None,
        velocity_scale: Optional[float] = None,
        acceleration_scale: Optional[float] = None,
    ) -> None:
        if position[2] < self.manual_minimum_z:
            raise RuntimeError(
                f"{description} 的末端目标 z={position[2]:.3f} m，"
                f"低于安全下限 {self.manual_minimum_z:.3f} m"
            )
        if math.hypot(position[0], position[1]) > 0.78 or position[2] > 0.75:
            raise RuntimeError("运动目标超出允许工作空间")
        if not self._move_group.wait_for_server(timeout_sec=2.0):
            raise RuntimeError("MoveIt /move_action 不可用")
        self._set_status("JOG_PLANNING", f"正在规划：{description}", False)
        result = self._plan_candidate(
            {"position": position, "orientation": orientation},
            position_tolerance=(
                0.002
                if position_tolerance is None
                else float(position_tolerance)
            ),
            orientation_tolerance=(
                0.04
                if orientation_tolerance is None
                else orientation_tolerance
            ),
            linear=linear,
            velocity_scale=velocity_scale,
            acceleration_scale=acceleration_scale,
        )
        _, max_delta = self._trajectory_score(
            result.planned_trajectory.joint_trajectory
        )
        allowed_delta = (
            self.jog_max_joint_delta
            if max_joint_delta is None
            else max_joint_delta
        )
        if max_delta > allowed_delta:
            raise RuntimeError(
                f"拒绝关节跳变 {max_delta:.3f} rad 的微调轨迹"
            )
        self._set_status("JOGGING", f"正在执行：{description}", False)
        stopped = self._send_arm_trajectory(result.planned_trajectory)
        if stopped:
            self._set_status("HANDED_OVER", "自动运动已停止", False)
            raise RuntimeError("自动运动已停止")
        self._set_status(
            "HANDED_OVER",
            f"运动完成：{description}",
            False,
            ee_target_position_m=[round(value, 5) for value in position],
        )

    def _call_servo_service(self, client, request, label: str) -> None:
        if not client.wait_for_service(timeout_sec=2.0):
            raise RuntimeError(f"{label}服务不可用")
        result = self._wait_future(client.call_async(request), 2.0)
        if result is None or not result.success:
            message = "无响应" if result is None else result.message
            raise RuntimeError(f"{label}失败：{message}")

    def _publish_servo_y_twist(self, velocity_m_s: float) -> None:
        command = TwistStamped()
        command.header.stamp = self.get_clock().now().to_msg()
        command.header.frame_id = self.ee_link
        command.twist.linear.y = float(velocity_m_s)
        self._servo_twist_pub.publish(command)

    def _servo_correct_bottle_y(
        self,
        desired_local: Sequence[float],
        tolerance: Sequence[float],
        initial_local: Sequence[float],
        target_object: str,
    ) -> bool:
        """Run one bounded Servo-only Y correction with live XYZ guards."""
        initial_error_y = float(initial_local[1]) - float(desired_local[1])
        if abs(initial_error_y) > self.bottle_servo_max_error:
            return False

        enable = SetBool.Request()
        enable.data = True
        disable = SetBool.Request()
        disable.data = False
        self._call_servo_service(
            self._servo_stream_client,
            enable,
            "Servo 独占关节流启用",
        )
        started = False
        success = False
        start_time = time.monotonic()
        last_time = start_time
        commanded_distance = 0.0
        best_abs_error_y = abs(initial_error_y)
        try:
            self._call_servo_service(
                self._servo_start_client,
                Trigger.Request(),
                "MoveIt Servo 启动",
            )
            started = True
            self._set_status(
                "SERVO_ALIGN",
                "圆柱抓取：执行一次受限 MoveIt Servo Y 向校正",
                False,
                initial_y_error_m=initial_error_y,
                global_replanning=False,
            )
            while time.monotonic() - start_time < self.bottle_servo_timeout:
                self._raise_if_stop_requested()
                with self._lock:
                    live_status = dict(self._task_status)
                relative = live_status.get("relative_position_gripper_m", ())
                if (
                    live_status.get("target_object") == target_object
                    and isinstance(relative, (list, tuple))
                    and len(relative) == 3
                ):
                    errors = tuple(
                        float(relative[index]) - float(desired_local[index])
                        for index in range(3)
                    )
                    if (
                        abs(errors[0]) > float(tolerance[0])
                        or abs(errors[2]) > float(tolerance[2])
                    ):
                        raise RuntimeError(
                            "Servo Y 向校正期间 X/Z 越出定位区间"
                        )
                    if abs(errors[1]) <= float(tolerance[1]):
                        success = True
                        break
                    error_y = errors[1]
                    absolute_error_y = abs(error_y)
                    if absolute_error_y < best_abs_error_y:
                        best_abs_error_y = absolute_error_y
                    elif (
                        commanded_distance >= 0.002
                        and absolute_error_y
                        > abs(initial_error_y) + 0.001
                    ):
                        raise RuntimeError(
                            "Servo Y 向校正发散，已在安全小步后停止"
                        )
                    elif (
                        commanded_distance >= 0.004
                        and best_abs_error_y
                        > abs(initial_error_y) - 0.0005
                    ):
                        raise RuntimeError(
                            "Servo Y 向校正无有效进展，已提前停止"
                        )
                else:
                    error_y = initial_error_y

                now = time.monotonic()
                elapsed = max(0.0, now - last_time)
                last_time = now
                velocity = max(
                    -self.bottle_servo_max_speed,
                    min(
                        self.bottle_servo_max_speed,
                        -self.bottle_servo_gain * error_y,
                    ),
                )
                commanded_distance += abs(velocity) * elapsed
                if commanded_distance > self.bottle_servo_max_correction:
                    break
                self._publish_servo_y_twist(velocity)
                time.sleep(0.02)
        finally:
            for _ in range(5):
                self._publish_servo_y_twist(0.0)
                time.sleep(0.02)
            if started and self._servo_stop_client.service_is_ready():
                self._wait_future(
                    self._servo_stop_client.call_async(Trigger.Request()), 2.0
                )
            if self._servo_stream_client.service_is_ready():
                self._wait_future(
                    self._servo_stream_client.call_async(disable), 2.0
                )
        if self.bottle_servo_settle > 0.0:
            time.sleep(self.bottle_servo_settle)
        return success

    def _guided_single_pass_cylinder_grasp(
        self, candidate: dict, task_status: dict, target_object: str
    ) -> str:
        """Validate the selected final-pose trajectory, then grasp once."""
        profile = GRASP_PROFILES[target_object]
        # The ordered pre-close guard is a per-object acceptance parameter.
        # Objects whose configured band is mandatory always keep it; the rest
        # can opt in at runtime with the debug switch.
        alignment_guard = target_object == "beverage_bottle" or bool(
            profile.get("preclose_alignment_guard", False)
        ) or self.bottle_preclose_alignment_guard_enabled
        cylinder_profile = OBJECT_GRASP_PROFILES[target_object]
        # Pausing envelope validation is a per-object, config-only switch.
        envelope_validation = (target_object == "beverage_bottle"
                               or _envelope_validation_enabled(target_object))
        self._set_grasp_control_state(transport_failure_reason="")
        calibrated = profile["calibrated_grasp_reference"]
        desired_local = tuple(
            float(value)
            for value in candidate.get(
                "desired_object_in_stator_m",
                calibrated["object_in_gripper_local_m"],
            )
        )
        tolerance = tuple(
            float(value)
            for value in profile["single_alignment_tolerance_m"]
        )
        preclose_envelope = dict(profile["grasp_envelope_overrides"])
        loaded_envelope = dict(cylinder_profile["loaded_envelope"])

        current_position, _ = self._current_ee_pose()
        pregrasp_error = math.sqrt(
            sum(
                (actual - expected) ** 2
                for actual, expected in zip(
                    current_position, candidate["position"]
                )
            )
        )
        resume_relative = task_status.get("relative_position_gripper_m", ())
        resume_inside_envelope = (
            task_status.get("target_object") == target_object
            and isinstance(resume_relative, (list, tuple))
            and len(resume_relative) == 3
            and inside_grasp_envelope(
                resume_relative, profile["open_jaw_resume_envelope"]
            )
            and not bool(task_status.get("gripper_commanded_closed", False))
        )
        if (
            alignment_guard
            and pregrasp_error > 0.08
            and not resume_inside_envelope
        ):
            raise RuntimeError(
                f"机械臂尚未到达目标物体最终抓取位（偏差 {pregrasp_error:.3f} m）"
            )

        with self._lock:
            self._verified_grasp_sequence = False
            self._loaded_grasp_reference_local = None
        stage = "目标物体单次定位验收准备"
        try:
            self._set_status(
                "GUIDED_OPEN",
                (
                    "目标物体抓取：Servo 已禁用，三维验收已暂停"
                    if not alignment_guard
                    else "目标物体抓取：确认开爪并验收单次最终定位结果"
                ),
                False,
                grasp_strategy="single_trajectory_final_pose",
                hardware_pitch_deg=float(profile["hardware_pitch_deg"]),
                urdf_pitch_deg=float(
                    candidate.get("pitch_deg", profile["pitch_deg"])
                ),
            )
            self._execute_gripper("open")

            aligned_local = desired_local
            aligned_error = (0.0, 0.0, 0.0)
            # The frozen open-jaw correction is an opt-in profile parameter.
            # Objects that configure no bounded correction (bottle, cup) go
            # straight from the planned final pose to the dynamic acceptance
            # below, so no frozen residual is measured or compensated.
            frozen_alignment_attempts = int(
                profile.get("frozen_alignment_attempts", 0)
            )
            if alignment_guard and frozen_alignment_attempts > 0:
                stage = "目标物体冻结开爪定位校正"
                self._command_scene_grasp("freeze", target_object)
                if not self._wait_scene_grasp(
                    "FROZEN", target_object, 2.0
                ):
                    raise RuntimeError("Isaac 未确认目标物体冻结校正状态")
                maximum_step = float(
                    profile.get("frozen_alignment_max_step_m", 0.0)
                )
                maximum_attempts = frozen_alignment_attempts
                for correction_attempt in range(maximum_attempts + 1):
                    self._command_scene_grasp("probe", target_object)
                    if not self._wait_scene_grasp(
                        "PROBED", target_object, 2.0
                    ):
                        raise RuntimeError("Isaac 未返回目标物体冻结相对位置")
                    with self._lock:
                        aligned_probe = dict(self._scene_grasp_status)
                    aligned_local = tuple(
                        float(value)
                        for value in aligned_probe.get("local_position_m", [])
                    )
                    if len(aligned_local) != 3:
                        raise RuntimeError(
                            "Isaac 返回的目标物体冻结相对位置不是三维坐标"
                        )
                    local_correction = _single_pass_alignment_correction(
                        aligned_local,
                        desired_local,
                        tolerance,
                        maximum_step,
                        gain=float(
                            profile.get("frozen_alignment_gain", 1.0)
                        ),
                    )
                    if (
                        local_correction == (0.0, 0.0, 0.0)
                        and inside_grasp_envelope(
                            aligned_local, preclose_envelope
                        )
                    ):
                        break
                    if correction_attempt >= maximum_attempts:
                        raise RuntimeError(
                            "冻结开爪校正未收敛："
                            f"local={aligned_local}, target={desired_local}, "
                            f"tolerance={tolerance}"
                        )
                    ee_position, ee_orientation = self._current_ee_pose()
                    stator_orientation = self._current_stator_orientation()
                    world_correction = rotate_vector(
                        stator_orientation, local_correction
                    )
                    corrected_position = tuple(
                        ee_position[index] + world_correction[index]
                        for index in range(3)
                    )
                    self._set_status(
                        "FROZEN_ALIGNMENT",
                        "开爪状态下校正控制器最终到位残差",
                        False,
                        measured_local_m=[
                            round(value, 6) for value in aligned_local
                        ],
                        desired_local_m=[
                            round(value, 6) for value in desired_local
                        ],
                        local_correction_m=[
                            round(value, 6) for value in local_correction
                        ],
                    )
                    maximum_joint_delta = float(
                        profile.get(
                            "frozen_alignment_max_joint_delta_rad", 0.35
                        )
                    )
                    correction_description = (
                        f"冻结开爪定位校正 {correction_attempt + 1}"
                    )
                    try:
                        self._execute_manual_pose(
                            corrected_position,
                            ee_orientation,
                            correction_description,
                            linear=True,
                            max_joint_delta=maximum_joint_delta,
                            position_tolerance=0.002,
                            orientation_tolerance=0.04,
                            velocity_scale=self.grasp_alignment_velocity_scale,
                        )
                    except RuntimeError as linear_error:
                        if not _is_recoverable_lift_planning_error(linear_error):
                            raise
                        self.get_logger().warning(
                            "Frozen alignment LIN planning failed; retrying "
                            f"the same bounded pose with OMPL: {linear_error}"
                        )
                        self._execute_manual_pose(
                            corrected_position,
                            ee_orientation,
                            correction_description + " OMPL 备用规划",
                            linear=False,
                            max_joint_delta=maximum_joint_delta,
                            position_tolerance=0.002,
                            orientation_tolerance=0.04,
                            velocity_scale=self.grasp_alignment_velocity_scale,
                        )
                    settle_duration = float(
                        profile.get("frozen_alignment_settle_s", 0.0)
                    )
                    if settle_duration > 0.0:
                        time.sleep(settle_duration)

            stage = "目标物体恢复动态物理"
            self._command_scene_grasp("unfreeze", target_object)
            if not self._wait_scene_grasp("UNFROZEN", "", 2.0):
                raise RuntimeError("Isaac 未确认目标物体保持动态刚体和重力")
            # Objects already dynamic throughout approach do not need another
            # fixed wait at the final pose. Fresh probes still validate physics,
            # contacts and alignment before closing.
            settle_duration = float(profile.get(
                "unfreeze_settle_duration_s", self.unfreeze_settle_duration
            ))
            if settle_duration > 0.0:
                time.sleep(settle_duration)

            if alignment_guard:
                stage = "目标物体闭爪前最终位置验收"
                self._command_scene_grasp("probe", target_object)
                if not self._wait_scene_grasp(
                    "PROBED", target_object, 2.0
                ):
                    raise RuntimeError("Isaac 未返回目标物体最终相对位置")
                with self._lock:
                    aligned_probe = dict(self._scene_grasp_status)
                aligned_local = tuple(
                    float(value)
                    for value in aligned_probe.get("local_position_m", [])
                )
                aligned_error = tuple(
                    aligned_local[index] - desired_local[index]
                    for index in range(3)
                ) if len(aligned_local) == 3 else ()
            servo_eligible = (
                self.bottle_servo_enabled
                and alignment_guard
                and len(aligned_error) == 3
                and abs(aligned_error[0]) <= tolerance[0]
                and abs(aligned_error[2]) <= tolerance[2]
                and abs(aligned_error[1]) > tolerance[1]
                and abs(aligned_error[1]) <= self.bottle_servo_max_error
            )
            if servo_eligible:
                stage = "目标物体闭爪前 MoveIt Servo 单次 Y 向校正"
                self._servo_correct_bottle_y(
                    desired_local, tolerance, aligned_local, target_object
                )
                stage = "目标物体 Servo 校正后三维复验"
                self._command_scene_grasp("probe", target_object)
                if not self._wait_scene_grasp(
                    "PROBED", target_object, 2.0
                ):
                    raise RuntimeError("Isaac 未返回 Servo 校正后相对位置")
                with self._lock:
                    aligned_probe = dict(self._scene_grasp_status)
                aligned_local = tuple(
                    float(value)
                    for value in aligned_probe.get("local_position_m", [])
                )
                aligned_error = tuple(
                    aligned_local[index] - desired_local[index]
                    for index in range(3)
                ) if len(aligned_local) == 3 else ()
            if alignment_guard and envelope_validation and (
                len(aligned_error) != 3
                or any(
                    abs(aligned_error[index]) > tolerance[index]
                    for index in range(3)
                )
                or not inside_grasp_envelope(
                    aligned_local, preclose_envelope
                )
            ):
                raise RuntimeError(
                    "单次最终定位后未进入闭爪允许区间："
                    f"local={aligned_local}, target={desired_local}, "
                    f"tolerance={tolerance}"
                )

            stage = "目标物体闭爪前动态确认"
            self._command_scene_grasp("probe", target_object)
            if not self._wait_scene_grasp("PROBED", target_object, 2.0):
                raise RuntimeError("Isaac 未返回目标物体闭爪前动态状态")
            with self._lock:
                preclose_probe = dict(self._scene_grasp_status)
            self._raise_if_finger_touching_table(preclose_probe, "目标物体闭爪前")
            if alignment_guard:
                preclose_local = tuple(
                    float(value)
                    for value in preclose_probe.get("local_position_m", [])
                )
                if (
                    envelope_validation
                    and (
                        len(preclose_local) != 3
                        or not inside_grasp_envelope(
                            preclose_local, preclose_envelope
                        )
                    )
                ):
                    raise RuntimeError(
                        "目标物体解冻后离开闭爪允许区间："
                        f"local={preclose_local}, "
                        f"envelope={preclose_envelope}"
                    )
                # The frozen correction only proves the calibration was
                # reachable while Isaac held the target.  Re-accept the
                # freshly measured dynamic pose so the fingers never close on
                # a residual that only survived because the object was frozen.
                if (
                    envelope_validation
                    and (
                        not all(
                            math.isfinite(value)
                            for value in preclose_local
                        )
                        or any(
                            abs(
                                preclose_local[index]
                                - desired_local[index]
                            )
                            > tolerance[index]
                            for index in range(3)
                        )
                    )
                ):
                    raise RuntimeError(
                        "目标物体动态定位超差，停止闭爪："
                        f"local={preclose_local}, target={desired_local}"
                    )
                if not preclose_probe.get(
                    "physics_enabled"
                ) or not preclose_probe.get("gravity_enabled"):
                    raise RuntimeError(
                        "目标物体必须恢复动态刚体和重力后才能闭爪"
                    )
                if not preclose_probe.get("contact_report_available"):
                    raise RuntimeError("目标物体闭爪前缺少接触反馈")

            if profile.get("single_pass_preclose", False):
                self._set_bottle_finger_collision(True)
            stage = "目标物体一次闭爪"
            object_before_close = self._current_object_position(target_object)
            self._set_status(
                "GUIDED_CLOSE",
                (
                    "目标物体闭爪前保护已暂停，正在无保护闭爪尝试"
                    if not alignment_guard
                    else "目标物体已通过三维判定，正在一次闭爪"
                ),
                False,
            )
            self._set_grasp_control_state(
                gripper_commanded_closed=True,
                gripper_obstructed=False,
                physical_test_lift_active=False,
                physical_grasp_ok=False,
                release_started=False,
                release_sequence_complete=False,
            )
            contact_angle = self._execute_gripper_close_until_obstructed(
                float(profile["gripper_rad"]),
                target_object,
                require_bilateral=True,
            )
            self._set_grasp_control_state(gripper_obstructed=True)
            contact_limits = cylinder_profile[
                "gripper_contact_angle_range_rad"
            ]
            if not (
                float(contact_limits[0])
                <= contact_angle
                <= float(contact_limits[1])
            ):
                raise RuntimeError(
                    "目标物体闭爪停止角不在有效范围："
                    f"jointGripper={contact_angle:.3f} rad, "
                    f"range={contact_limits}"
                )
            hold_target = min(
                self.gripper_closed,
                contact_angle
                + float(profile["single_contact_preload_rad"]),
            )
            self._hold_gripper(hold_target, persistent=True)
            object_after_preload = self._current_object_position(target_object)
            preload_displacement = math.sqrt(
                sum(
                    (after - before) ** 2
                    for after, before in zip(
                        object_after_preload, object_before_close
                    )
                )
            )
            self.get_logger().info(
                "圆柱预紧质量："
                f"contact_angle_rad={contact_angle}, "
                f"hold_target_rad={hold_target}, "
                f"preload_angle_rad={hold_target - contact_angle}, "
                f"preload_displacement_m={preload_displacement}"
            )
            if preload_displacement > float(
                profile["maximum_close_displacement_m"]
            ):
                preload_deg = math.degrees(
                    float(profile["single_contact_preload_rad"])
                )
                raise RuntimeError(
                    f"目标物体一次 {preload_deg:.1f}° 预紧位移过大："
                    f"{preload_displacement:.4f} m"
                )
            try:
                closed_probe = self._require_loaded_physical_grasp(
                    target_object,
                    loaded_envelope,
                    "目标物体闭爪后",
                    use_verified_lift_reference=False,
                )
            except RuntimeError as initial_validation_error:
                with self._lock:
                    retry_probe = dict(self._scene_grasp_status)
                retry_preload = float(
                    profile.get("single_contact_retry_preload_rad", 0.0)
                )
                retry_attempts = int(
                    profile.get("single_contact_retry_attempts", 0)
                )
                if (
                    retry_attempts < 1
                    or retry_preload <= 0.0
                    or not _safe_bilateral_preload_retry(
                        retry_probe,
                        target_object,
                        loaded_envelope,
                        self.physical_test_lift_loaded_envelope_margin,
                    )
                ):
                    raise
                retry_target = min(
                    self.gripper_closed,
                    hold_target + retry_preload,
                )
                self.get_logger().warning(
                    "Initial bottle loaded-grasp validation was rejected "
                    "despite safe bilateral loading; applying the single "
                    "bounded retry preload: "
                    f"additional={retry_preload:.6f} rad, "
                    f"target={retry_target:.6f} rad, "
                    f"reason={initial_validation_error}"
                )
                self._set_status(
                    "BOTTLE_RETRY_PRELOAD",
                    "首次物理判定未通过，但双指安全受力有效；追加一次 3° 闭合后复验",
                    False,
                )
                self._hold_gripper(retry_target, persistent=True)
                closed_probe = self._require_loaded_physical_grasp(
                    target_object,
                    loaded_envelope,
                    "目标物体追加预紧后",
                    use_verified_lift_reference=False,
                )
            object_after_close = self._current_object_position(target_object)

            stage = "目标物体纯物理抬升"
            object_before_lift = object_after_close
            direct_lift_only = bool(
                cylinder_profile.get("physical_test_lift_direct_only", False)
            )
            if direct_lift_only:
                preferred_transport_z = float(profile["transport_object_z_m"])
                safe_completion_z = float(
                    profile["physical_test_lift_safe_completion_z_m"]
                )
                safe_acceptance_tolerance = float(
                    cylinder_profile.get(
                        "physical_test_lift_safe_acceptance_tolerance_m", 0.0
                    )
                )
            else:
                preferred_transport_z = max(
                    self.box_approach_object_z,
                    float(profile["transport_object_z_m"]),
                    object_before_lift[2]
                    + float(profile["minimum_object_rise_m"]),
                )
                safe_completion_z = min(
                    preferred_transport_z,
                    max(
                        self.box_approach_object_z,
                        float(profile["physical_test_lift_safe_completion_z_m"]),
                        object_before_lift[2]
                        + float(profile["minimum_object_rise_m"]),
                    ),
                )
                safe_acceptance_tolerance = 0.0
            lift_completion_tolerance = float(
                profile["physical_test_lift_completion_tolerance_m"]
            )
            safe_acceptance_tolerance = max(
                safe_acceptance_tolerance,
                lift_completion_tolerance,
            )
            minimum_accepted_z = (
                safe_completion_z - safe_acceptance_tolerance
            )
            # Per-object acceptance floor for the loaded transport.  It only
            # lowers the height the object must reach, without touching the
            # commanded transport target or the per-segment tolerances.
            minimum_accepted_z = float(
                OBJECT_GRASP_PROFILES.get(target_object, {}).get(
                    "physical_test_lift_minimum_accepted_z_m",
                    minimum_accepted_z,
                )
            )
            nominal_segment = float(
                profile["physical_test_lift_segment_m"]
            )
            near_table_segment = float(
                profile["physical_test_lift_near_table_segment_m"]
            )
            near_table_until_z = float(
                profile["physical_test_lift_near_table_until_z_m"]
            )
            minimum_retry_segment = float(
                profile["physical_test_lift_min_retry_segment_m"]
            )
            minimum_progress = float(
                profile["physical_test_lift_min_progress_m"]
            )
            maximum_stalled_iterations = int(
                profile["physical_test_lift_max_stalled_iterations"]
            )
            orientation_tolerances = tuple(
                float(value)
                for value in profile[
                    "physical_test_lift_orientation_tolerances_rad"
                ]
            )
            self._set_grasp_control_state(physical_test_lift_active=True)
            if target_object == "apple":
                self._lift_apple_preplanned(preferred_transport_z, loaded_envelope)
            else:
                self._set_status(
                    "PHYSICAL_TEST_LIFT",
                    "正在完整重力下自由规划到蓝盒上方安全位",
                    False,
                )
                lift_probe = self._carry_loaded_object_toward_box(
                    target_object,
                    loaded_envelope,
                    minimum_accepted_z,
                    "目标物体带载荷自由搬运后",
                    require_box_approach=True,
                    preferred_object_z=preferred_transport_z,
                )
            current_object_z = self._current_object_position(target_object)[2]
            # Keep the commanded target: a measured undershoot must never
            # redefine the goal and silently disable clearance correction.
            stalled_iterations = 0
            lift_iteration = 0
            maximum_lift_iterations = max(
                1,
                int(
                    math.ceil(
                        (preferred_transport_z - object_before_lift[2])
                        / max(minimum_retry_segment, 1.0e-4)
                    )
                )
                * 2,
            )
            while target_object != "apple" and not _loaded_transport_height_is_acceptable(
                current_object_z,
                safe_completion_z=safe_completion_z,
                tolerance_m=safe_acceptance_tolerance,
            ):
                lift_iteration += 1
                if lift_iteration > maximum_lift_iterations:
                    if _loaded_transport_height_is_acceptable(
                        current_object_z,
                        safe_completion_z=safe_completion_z,
                        tolerance_m=safe_acceptance_tolerance,
                    ):
                        break
                    raise RuntimeError(
                        "目标物体抬升未在安全迭代次数内取得足够高度："
                        f"z={current_object_z:.4f} m, "
                        f"safe={safe_completion_z:.4f} m"
                    )
                remaining_lift = max(
                    0.0, preferred_transport_z - current_object_z
                )
                local_segment_limit = (
                    near_table_segment
                    if current_object_z < near_table_until_z
                    else nominal_segment
                )
                # Preserve the bottle's fast successful path: first ask for
                # the original direct lift to the preferred height.  Only an
                # IK/planning rejection activates the shorter local waypoints.
                segment_distance = min(
                    remaining_lift
                    if lift_iteration == 1
                    else local_segment_limit,
                    remaining_lift,
                )
                while True:
                    lift_start, lift_orientation = self._current_ee_pose()
                    lift_target = (
                        lift_start[0],
                        lift_start[1],
                        lift_start[2] + segment_distance,
                    )
                    lift_error = None
                    for orientation_tolerance in orientation_tolerances:
                        try:
                            self._execute_manual_pose(
                                lift_target,
                                lift_orientation,
                                "目标物体竖直物理抬升",
                                linear=True,
                                max_joint_delta=0.65,
                                position_tolerance=0.002,
                                orientation_tolerance=orientation_tolerance,
                                velocity_scale=(
                                    self.physical_test_lift_velocity_scale
                                ),
                            )
                            lift_error = None
                            break
                        except RuntimeError as exc:
                            lift_error = exc
                            if not _is_recoverable_lift_planning_error(exc):
                                raise
                    if lift_error is None:
                        break
                    if direct_lift_only:
                        raise RuntimeError(
                            "目标物体单次直抬到搬运高度规划失败："
                            f"target_z={preferred_transport_z:.4f} m, "
                            f"safe={safe_completion_z:.4f} m, "
                            f"reason={lift_error}"
                        ) from lift_error
                    if _loaded_transport_height_is_acceptable(
                        current_object_z,
                        safe_completion_z=safe_completion_z,
                        tolerance_m=safe_acceptance_tolerance,
                    ):
                        self.get_logger().warning(
                            "Bottle reached safe transport clearance; "
                            "accepting the grasp after the preferred final "
                            f"waypoint was rejected: z={current_object_z:.4f} m, "
                            f"preferred={preferred_transport_z:.4f} m, "
                            f"reason={lift_error}"
                        )
                        segment_distance = 0.0
                        break
                    if segment_distance > local_segment_limit + 1.0e-6:
                        self.get_logger().warning(
                            "Direct bottle lift to the preferred height was "
                            "rejected; switching to local collision-checked "
                            f"segments: {segment_distance:.4f} -> "
                            f"{local_segment_limit:.4f} m, reason={lift_error}"
                        )
                        segment_distance = min(
                            local_segment_limit,
                            remaining_lift,
                        )
                        continue
                    try:
                        self._execute_manual_pose(
                            lift_target,
                            lift_orientation,
                            "目标物体抬升 IK 分支受限过渡",
                            linear=False,
                            max_joint_delta=float(
                                profile[
                                    "physical_test_lift_"
                                    "bounded_fallback_max_joint_delta_rad"
                                ]
                            ),
                            position_tolerance=0.002,
                            orientation_tolerance=max(
                                orientation_tolerances
                            ),
                            velocity_scale=(
                                self.physical_test_lift_velocity_scale
                            ),
                        )
                        break
                    except RuntimeError as exc:
                        if not _is_recoverable_lift_planning_error(exc):
                            raise
                        reduced_segment = _reduced_lift_segment_distance(
                            segment_distance,
                            minimum_retry_segment,
                        )
                        if reduced_segment is None:
                            raise RuntimeError(
                                "目标物体抬升已缩短到最小安全分段仍无法规划："
                                f"segment={segment_distance:.4f} m, "
                                f"reason={exc}"
                            ) from exc
                        self.get_logger().warning(
                            "Bottle lift waypoint failed; preserving the "
                            "verified grasp and shortening the segment: "
                            f"{segment_distance:.4f} -> "
                            f"{reduced_segment:.4f} m, reason={exc}"
                        )
                        segment_distance = reduced_segment
                if segment_distance <= 0.0:
                    break
                if direct_lift_only:
                    settle_duration = float(
                        cylinder_profile.get(
                            "physical_test_lift_settle_duration_s", 0.0
                        )
                    )
                    if settle_duration > 0.0:
                        self._set_status(
                            "PHYSICAL_TEST_LIFT_SETTLING",
                            "单次直抬已完成，保持机械臂静止并等待瓶子物理稳定",
                            False,
                        )
                        settle_deadline = time.monotonic() + settle_duration
                        while time.monotonic() < settle_deadline:
                            self._raise_if_stop_requested()
                            time.sleep(
                                max(
                                    0.0,
                                    min(
                                        0.05,
                                        settle_deadline - time.monotonic(),
                                    ),
                                )
                            )
                segment_probe = self._require_loaded_physical_grasp(
                    target_object,
                    loaded_envelope,
                    "目标物体分段抬升中",
                    use_verified_lift_reference=False,
                    loaded_envelope_margin_m=self.transport_loaded_grasp_guard_margin,
                )
                if bool(segment_probe.get("object_touching_table", True)):
                    raise RuntimeError("目标物体分段抬升后仍接触桌面")
                measured_object_z = self._current_object_position(
                    target_object
                )[2]
                measured_progress = measured_object_z - current_object_z
                stalled_iterations = (
                    stalled_iterations + 1
                    if measured_progress < minimum_progress
                    else 0
                )
                current_object_z = measured_object_z
                if direct_lift_only:
                    if _loaded_transport_height_is_acceptable(
                        current_object_z,
                        safe_completion_z=safe_completion_z,
                        tolerance_m=safe_acceptance_tolerance,
                    ):
                        break
                    raise RuntimeError(
                        "目标物体单次直抬后未达到安全搬运高度："
                        f"z={current_object_z:.4f} m, "
                        f"safe={safe_completion_z:.4f} m, "
                        f"accept_min={minimum_accepted_z:.4f} m, "
                        f"target={preferred_transport_z:.4f} m"
                    )
                if (
                    not _loaded_transport_height_is_acceptable(
                        current_object_z,
                        safe_completion_z=safe_completion_z,
                        tolerance_m=safe_acceptance_tolerance,
                    )
                    and stalled_iterations >= maximum_stalled_iterations
                ):
                    raise RuntimeError(
                        "目标物体抬升连续无有效进展："
                        f"z={current_object_z:.4f} m, "
                        f"safe={safe_completion_z:.4f} m, "
                        f"minimum_progress={minimum_progress:.4f} m"
                    )
            lift_probe = self._require_loaded_physical_grasp(
                target_object,
                loaded_envelope,
                "目标物体抬升后",
                use_verified_lift_reference=False,
                loaded_envelope_margin_m=self.transport_loaded_grasp_guard_margin,
            )
            if bool(lift_probe.get("object_touching_table", True)):
                raise RuntimeError("目标物体抬升后仍接触桌面")
            object_after_lift = self._current_object_position(target_object)
            if not _loaded_transport_height_is_acceptable(
                object_after_lift[2],
                safe_completion_z=safe_completion_z,
                tolerance_m=safe_acceptance_tolerance,
            ):
                raise TransportHeightError(
                    "目标物体未达到最低安全搬运高度："
                    f"z={object_after_lift[2]:.4f} m, "
                    f"safe={safe_completion_z:.4f} m, "
                    f"accept_min={minimum_accepted_z:.4f} m, "
                    f"preferred={preferred_transport_z:.4f} m"
                )
            lift_local = tuple(
                float(value)
                for value in lift_probe.get("local_position_m", [])
            )
            closed_local = tuple(closed_probe.get("local_position_m", []))
            if not _transport_loaded_grasp_is_valid(
                lift_probe,
                target_object,
                loaded_envelope,
                self.transport_loaded_grasp_guard_margin,
                closed_local,
                self.physical_test_lift_relative_position_tolerance,
                require_envelope=_envelope_validation_enabled(target_object),
            ):
                raise RuntimeError(
                    "目标物体试抬后夹持保持验证失败：检查相对滑移、双指受力及离桌状态；"
                    f"closed_local={closed_local}, lifted_local={lift_local}, "
                    f"slip_limit_m={self.physical_test_lift_relative_position_tolerance}"
                )
            with self._lock:
                self._loaded_grasp_reference_local = lift_local
                self._verified_grasp_sequence = True
            self._set_grasp_control_state(
                physical_test_lift_active=False,
                physical_grasp_ok=True,
            )
            self._set_status(
                "GRASP_CONFIRMED",
                (
                    "目标物体无保护闭爪尝试已通过真实双指接触和离桌验证"
                    if not alignment_guard
                    else "目标物体已通过三维对齐、真实双指接触和离桌验证"
                ),
                False,
                aligned_local_m=[round(value, 6) for value in aligned_local],
                closed_local_m=closed_probe.get("local_position_m"),
            )
            return "目标物体无保护闭爪及物理抬升验证已完成"
        except Exception as exc:
            if isinstance(exc, TransportHeightError):
                # Height failure does not open the fingers. Preserve the
                # commanded closure, but never mark this task as verified.
                self._set_grasp_control_state(
                    physical_test_lift_active=False,
                    physical_grasp_ok=False,
                    transport_failure_reason=str(exc),
                )
            else:
                self._set_grasp_control_state(
                    gripper_commanded_closed=False,
                    gripper_obstructed=False,
                    physical_test_lift_active=False,
                    physical_grasp_ok=False,
                )
            self._command_scene_grasp("unfreeze")
            raise RuntimeError(f"{stage}失败：{exc}") from exc

    def _validate_single_position_grasp(self, target_object, candidate, envelope):
        """Inspect the reached dynamic pose once; never correct or retry it."""
        profile = GRASP_PROFILES[target_object]
        desired = candidate.get("desired_object_in_stator_m")
        if not candidate.get("final_grasp_target") or desired is None:
            raise RuntimeError("候选来自旧抓取流程，请重新生成一次定位轨迹")
        self._command_scene_grasp("unfreeze", target_object)
        if not self._wait_scene_grasp("UNFROZEN", "", 2.0):
            raise RuntimeError("Isaac 未确认恢复动态物理")
        time.sleep(self.unfreeze_settle_duration)
        self._command_scene_grasp("probe", target_object)
        if not self._wait_scene_grasp("PROBED", target_object, 2.0):
            raise RuntimeError("Isaac 未返回一次定位验收状态")
        with self._lock:
            probe = dict(self._scene_grasp_status)
        self._raise_if_finger_touching_table(probe, "一次定位验收")
        local = probe.get("local_position_m", ())
        tolerance = profile["single_alignment_tolerance_m"]
        error = tuple(float(a) - float(b) for a, b in zip(local, desired))
        self._set_status(
            "GUIDED_ALIGNMENT_CHECK", "正在验收一次定位结果", False,
            grasp_strategy="single_position_grasp", measured_local_m=local,
            desired_local_m=desired, alignment_error_m=error,
            alignment_tolerance_m=tolerance,
        )
        if (len(local) != 3 or not all(math.isfinite(x) for x in error)
                or any(abs(error[i]) > tolerance[i] for i in range(3))
                or not inside_grasp_envelope(local, envelope)):
            raise RuntimeError(
                f"一次定位超差，已停止；local={local}, target={desired}, "
                f"error={error}, tolerance={tolerance}")
        if not probe.get("physics_enabled") or not probe.get("gravity_enabled"):
            raise RuntimeError("一次定位验收要求动态刚体及重力开启")
        if not probe.get("contact_report_available"):
            raise RuntimeError("一次定位验收缺少真实接触反馈")
        for side in ("left", "right"):
            contact = probe.get(f"{side}_finger_contact_object")
            if contact:
                raise RuntimeError(f"一次定位闭爪前夹指已接触物体：{side}={contact}")
        return probe

    def _guided_grasp(self) -> str:
        with self._lock:
            candidate = copy.deepcopy(self._grasp_candidate)
            target_object = self.target_object
            task_status = copy.deepcopy(self._task_status)
        if candidate is None:
            raise RuntimeError("请先生成并执行预抓取轨迹")
        profile = GRASP_PROFILES[target_object]
        if bool(profile.get("manual_tuning_handoff", False)):
            raise RuntimeError(
                "鼠标已启用一次定位手动调试模式：执行预抓取轨迹后，"
                "请在 RViz 中微调；自动外侧对中、插入和闭爪已禁用"
            )
        if bool(profile.get("single_pass_preclose", False)):
            return self._guided_single_pass_cylinder_grasp(
                candidate, task_status, target_object
            )
        with self._lock:
            # A new grasp must establish its own top-lift reference.  Never
            # reuse a relative pose captured by an earlier episode.
            self._verified_grasp_sequence = False
            self._loaded_grasp_reference_local = None
        profile_envelope = dict(GRASP_ENVELOPE)
        profile_envelope.update(
            profile.get("grasp_envelope_overrides", {})
        )
        current_position, _ = self._current_ee_pose()
        pregrasp_error = math.sqrt(
            sum(
                (a - b) ** 2
                for a, b in zip(current_position, candidate["position"])
            )
        )
        # The continuously published evaluator status owns the live relative
        # pose.  The scene-grasp status only gains ``local_position_m`` after
        # an explicit probe command and therefore cannot authorize a resume
        # immediately after a stop.
        resume_relative = task_status.get("relative_position_gripper_m", ())
        resume_envelope = profile.get(
            "open_jaw_resume_envelope", profile_envelope
        )
        resume_inside_envelope = (
            task_status.get("target_object") == target_object
            and isinstance(resume_relative, (list, tuple))
            and len(resume_relative) == 3
            and inside_grasp_envelope(resume_relative, resume_envelope)
            and not bool(task_status.get("gripper_commanded_closed", False))
        )
        if pregrasp_error > 0.08 and not resume_inside_envelope:
            raise RuntimeError(
                f"机械臂尚未到达预抓取位（偏差 {pregrasp_error:.3f} m）"
            )
        if pregrasp_error > 0.08:
            self.get_logger().info(
                "Resuming guided grasp from the current validated envelope: "
                f"local={tuple(round(float(value), 6) for value in resume_relative)}, "
                f"pregrasp_error={pregrasp_error:.3f} m"
            )
        object_position = self._current_object_position(target_object)
        target_point = (
            object_position[0],
            object_position[1],
            object_position[2]
            + float(profile["height"])
            + float(profile.get("world_z_offset_m", 0.0)),
        )
        direction = _normalize(
            tuple(
                candidate["target_point"][index]
                - candidate["position"][index]
                for index in range(3)
            )
        )
        # With the imported gripper collision changed from one solid convex
        # hull to convex decomposition, the open-jaw collision-free entry is
        # at the front/top side of the valid envelope.  Closing later draws
        # the fruit inward; forcing the centre depth while fully open contacts
        # the accurately decomposed fixed finger and stalls the arm.
        desired_center_x = float(
            profile.get(
                "desired_center_x_m",
                float(profile_envelope["x_max_m"]) - 0.001,
            )
        )
        insertion_center_x = float(
            profile.get("insertion_center_x_m", desired_center_x)
        )
        dynamic_insertion_stop_x = float(
            profile.get(
                "dynamic_insertion_stop_x_m",
                float(profile_envelope["x_max_m"]) - 0.002,
            )
        )
        dynamic_insertion_step = float(
            profile.get("dynamic_insertion_step_m", 0.005)
        )
        dynamic_lateral_step = float(
            profile.get("dynamic_lateral_step_m", 0.006)
        )
        dynamic_alignment_corridor = float(
            profile.get("dynamic_alignment_corridor_m", 0.010)
        )
        insertion_x_tolerance = float(
            profile.get("insertion_x_tolerance_m", 0.001)
        )
        if not (
            float(profile_envelope["x_min_m"])
            < dynamic_insertion_stop_x
            < float(profile_envelope["x_max_m"])
        ):
            raise RuntimeError(
                "动态插入停止深度必须位于共享夹持包络内部："
                f"x={dynamic_insertion_stop_x:.4f} m"
            )
        maximum_dynamic_insertion_step = float(
            profile.get("maximum_dynamic_insertion_step_m", 0.005)
        )
        if not (
            0.001
            <= dynamic_insertion_step
            <= maximum_dynamic_insertion_step
        ):
            raise RuntimeError(
                "动态插入步长超出物体策略允许范围："
                f"step={dynamic_insertion_step:.4f} m"
            )
        if not 0.003 <= dynamic_lateral_step <= 0.020:
            raise RuntimeError(
                "动态横向修正步长必须位于 3–20 mm："
                f"step={dynamic_lateral_step:.4f} m"
            )
        if not 0.006 <= dynamic_alignment_corridor <= 0.020:
            raise RuntimeError(
                "动态对中通道必须位于 6–12 mm："
                f"corridor={dynamic_alignment_corridor:.4f} m"
            )
        # At the fruit insertion depth the fixed finger's inner support face
        # is around local z=-9 mm.  Insert above that face first, then seat
        # downward only after the fruit center has passed the fingertip plane;
        # trying to enter directly at the support height hits the finger end.
        # Dynamic settling consistently raises the spherical proxy by roughly
        # 3 mm in stator-local Z.  Keep the collision-free entry near the top
        # of the slot, but finish the frozen alignment 2 mm lower so settling
        # and slow closure remain inside the unchanged strict envelope.
        entry_center_z = float(
            profile.get(
                "entry_center_z_m",
                float(GRASP_ENVELOPE["z_max_m"]) - 0.003,
            )
        )
        desired_center_z = float(
            profile.get(
                "desired_center_z_m",
                float(GRASP_ENVELOPE["z_max_m"]) - 0.004,
            )
        )
        desired_center_y = float(
            profile.get("desired_center_y_m", 0.0)
        )
        alignment_tolerance_y = float(
            profile.get("alignment_tolerance_y_m", 0.002)
        )
        insertion_alignment_tolerance_y = float(
            profile.get(
                "insertion_alignment_tolerance_y_m",
                alignment_tolerance_y,
            )
        )
        alignment_tolerance_z = float(
            profile.get("alignment_tolerance_z_m", 0.002)
        )
        insertion_alignment_tolerance_z = float(
            profile.get(
                "insertion_alignment_tolerance_z_m",
                alignment_tolerance_z,
            )
        )
        # Some side-grasp profiles start closer than the apple's 245 mm
        # outer-alignment waypoint.  Sending those profiles backwards to the
        # apple waypoint adds a visible pause and can make Pilz reject the
        # otherwise-valid straight segment.  Never retreat past the approved
        # collision-checked pregrasp; align in place when it is already the
        # safer outer waypoint.
        alignment_standoff = min(
            float(profile.get("alignment_standoff_m", 0.245)),
            float(candidate["standoff_m"]),
        )
        alignment_position = tuple(
            target_point[index] - alignment_standoff * direction[index]
            for index in range(3)
        )
        self._command_scene_grasp("freeze", target_object)
        if not self._wait_scene_grasp("FROZEN", target_object, 2.0):
            raise RuntimeError("Isaac 未确认物体已固定在接近起点")
        stage = "确认开爪"
        try:
            self._set_status(
                "GUIDED_OPEN", "已到达接近位，正在确认夹爪完全打开", False
            )
            self._execute_gripper("open")
            if profile.get("single_position_grasp", False):
                stage = "一次定位到位验收"
                grasp_probe = self._validate_single_position_grasp(
                    target_object, candidate, profile_envelope)
            else:
                stage = "安全深度直线接近"
                self._set_status(
                    "GUIDED_APPROACH",
                    "夹爪已打开，正在接近安全对中深度",
                    False,
                )
                pregrasp_is_alignment_pose = (
                    abs(
                        alignment_standoff
                        - float(candidate["standoff_m"])
                    )
                    <= 0.001
                )
                if not pregrasp_is_alignment_pose and not resume_inside_envelope:
                    self._execute_manual_pose(
                        alignment_position,
                        candidate["orientation"],
                        "接近闭爪前安全对中位",
                        linear=True,
                        max_joint_delta=0.85,
                    )
                else:
                    self.get_logger().info(
                        "Current pose is already valid for alignment or envelope "
                        "resume; skipping the duplicate outer-approach plan"
                    )
                stage = "安全深度横向对中"
                self._command_scene_grasp("probe", target_object)
                if not self._wait_scene_grasp("PROBED", target_object, 2.0):
                    raise RuntimeError("Isaac 未返回安全深度相对位置")
                with self._lock:
                    grasp_probe = dict(self._scene_grasp_status)
                self.get_logger().info(
                    "Pre-close probe: "
                    f"target={target_object} "
                    f"local={grasp_probe.get('local_position_m')} "
                    f"kinematic={not bool(grasp_probe.get('physics_enabled', False))}"
                )
                measured_local = tuple(
                    float(value)
                    for value in grasp_probe.get("local_position_m", [])
                )
                if len(measured_local) != 3:
                    raise RuntimeError("Isaac 返回的夹爪局部位置无效")
                calibrated_reference = profile.get(
                    "calibrated_grasp_reference", {}
                ).get("object_in_gripper_local_m", ())
                if len(calibrated_reference) == 3:
                    calibrated_stator_world = (
                        _world_gripper_origin_from_object_reference(
                            self._current_object_position(target_object),
                            self._current_stator_orientation(),
                            calibrated_reference,
                        )
                    )
                    calibrated_error = tuple(
                        round(measured_local[index] - float(calibrated_reference[index]), 6)
                        for index in range(3)
                    )
                    self.get_logger().info(
                        "Calibrated bottle geometry: "
                        f"object_in_gripper_target={tuple(calibrated_reference)}, "
                        f"local_error={calibrated_error}, "
                        "stator_world_target="
                        f"{tuple(round(value, 6) for value in calibrated_stator_world)}"
                    )
                # Center at a collision-free depth before inserting between the
                # fingers.  The former flow attempted this correction only after
                # insertion, so an 18 mm lateral error hit one fingertip first.
                outer_alignment_iterations = int(
                    profile.get("outer_alignment_iterations", 3)
                )
                outer_alignment_max_y_step = float(
                    profile.get("outer_alignment_max_y_step_m", 0.020)
                )
                outer_alignment_max_z_step = float(
                    profile.get("outer_alignment_max_z_step_m", 0.020)
                )
                outer_acceptance_y = float(
                    profile.get(
                        "outer_alignment_acceptance_y_m",
                        alignment_tolerance_y,
                    )
                )
                outer_acceptance_z = float(
                    profile.get(
                        "outer_alignment_acceptance_z_m",
                        alignment_tolerance_z,
                    )
                )
                outer_gain_y = float(
                    profile.get("outer_alignment_gain_y", 0.85)
                )
                outer_gain_z = float(
                    profile.get("outer_alignment_gain_z", 0.85)
                )
                moveit_alignment_tolerance = float(
                    profile.get(
                        "outer_alignment_moveit_position_tolerance_m",
                        0.0005,
                    )
                )
                locked_outer_orientation = None
                if bool(
                    profile.get("lock_outer_alignment_orientation", False)
                ):
                    locked_outer_orientation = self._current_ee_pose()[1]
                unlimited_fine_adjustment = bool(
                    profile.get("unlimited_fine_adjustment", False)
                )
                outer_alignment_indices = (
                    itertools.count()
                    if unlimited_fine_adjustment
                    else range(outer_alignment_iterations)
                )
                deferred_alignment_x = profile.get(
                    "defer_calibrated_alignment_until_x_m"
                )
                defer_outer_alignment = (
                    deferred_alignment_x is not None
                    and measured_local[0] > float(deferred_alignment_x)
                )
                if defer_outer_alignment:
                    self.get_logger().info(
                        "Deferring taught bottle Y/Z alignment until the near "
                        "region; far collision feedback will guard Y drift: "
                        f"local_x={measured_local[0]:.4f} m, "
                        f"threshold={float(deferred_alignment_x):.4f} m"
                    )
                    outer_alignment_indices = ()
                for alignment_iteration in outer_alignment_indices:
                    y_alignment_error = measured_local[1] - desired_center_y
                    z_alignment_error = measured_local[2] - entry_center_z
                    y_outside = (
                        abs(y_alignment_error) > alignment_tolerance_y
                    )
                    z_outside = (
                        abs(z_alignment_error) > alignment_tolerance_z
                    )
                    if not y_outside and not z_outside:
                        break
                    ee_position, current_ee_orientation = self._current_ee_pose()
                    ee_orientation = (
                        locked_outer_orientation
                        if locked_outer_orientation is not None
                        else current_ee_orientation
                    )
                    stator_orientation = self._current_stator_orientation()
                    y_correction = _bounded_axis_correction(
                        y_alignment_error,
                        alignment_tolerance_y,
                        outer_gain_y,
                        outer_alignment_max_y_step,
                    )
                    z_correction = _bounded_axis_correction(
                        z_alignment_error,
                        alignment_tolerance_z,
                        outer_gain_z,
                        outer_alignment_max_z_step,
                    )
                    local_correction = (
                        0.0,
                        y_correction,
                        z_correction,
                    )
                    alignment_axis_order = tuple(
                        profile.get("alignment_axis_order", ())
                    )
                    if alignment_axis_order:
                        local_correction = _ordered_axis_correction(
                            local_correction,
                            (False, y_outside, z_outside),
                            alignment_axis_order,
                        )
                    elif bool(profile.get("decouple_outer_alignment", False)):
                        normalized_y = (
                            abs(y_alignment_error)
                            / max(alignment_tolerance_y, 1.0e-6)
                        )
                        normalized_z = (
                            abs(z_alignment_error)
                            / max(alignment_tolerance_z, 1.0e-6)
                        )
                        if y_outside and (
                            not z_outside or normalized_y >= normalized_z
                        ):
                            local_correction = (
                                local_correction[0],
                                local_correction[1],
                                0.0,
                            )
                        elif z_outside:
                            local_correction = (
                                local_correction[0],
                                0.0,
                                local_correction[2],
                            )
                    world_correction = rotate_vector(
                        stator_orientation, local_correction
                    )
                    centered_position = tuple(
                        ee_position[index] + world_correction[index]
                        for index in range(3)
                    )
                    self.get_logger().info(
                        "Outer alignment command: "
                        f"ee_before={ee_position}, "
                        f"local_correction={local_correction}, "
                        f"world_correction={world_correction}, "
                        f"ee_target={centered_position}"
                    )
                    self._execute_manual_pose(
                        centered_position,
                        ee_orientation,
                        "安全深度迭代校正到双指中心 "
                        f"{alignment_iteration + 1}",
                        linear=True,
                        max_joint_delta=float(
                            profile.get(
                                "outer_alignment_max_joint_delta_rad", 0.45
                            )
                        ),
                        position_tolerance=moveit_alignment_tolerance,
                        orientation_tolerance=float(
                            profile.get(
                                "outer_alignment_orientation_tolerance_rad",
                                0.04,
                            )
                        ),
                        velocity_scale=self.grasp_alignment_velocity_scale,
                    )
                    ee_after, _ = self._current_ee_pose()
                    ee_tracking_error = math.sqrt(
                        sum(
                            (actual - requested) ** 2
                            for actual, requested in zip(
                                ee_after, centered_position
                            )
                        )
                    )
                    self.get_logger().info(
                        "Outer alignment executed: "
                        f"ee_after={ee_after}, "
                        f"tracking_error={ee_tracking_error:.4f} m"
                    )
                    self._command_scene_grasp("probe", target_object)
                    if not self._wait_scene_grasp("PROBED", target_object, 2.0):
                        raise RuntimeError("Isaac 未返回横向校正结果")
                    with self._lock:
                        grasp_probe = dict(self._scene_grasp_status)
                    measured_local = tuple(
                        float(value)
                        for value in grasp_probe.get("local_position_m", [])
                    )
                    if len(measured_local) != 3:
                        raise RuntimeError("Isaac 返回的外侧对中位置无效")
                    self.get_logger().info(
                        "Outer frozen alignment probe: "
                        f"iteration={alignment_iteration}, "
                        f"local={measured_local}"
                    )
                if len(measured_local) != 3:
                    raise RuntimeError("Isaac 返回的外侧对中位置无效")
                final_y_error = abs(measured_local[1] - desired_center_y)
                final_z_error = abs(measured_local[2] - entry_center_z)
                outer_recovery_corridor = float(
                    profile.get(
                        "dynamic_alignment_abort_corridor_m",
                        max(0.030, 2.0 * dynamic_alignment_corridor),
                    )
                )
                if not _outer_alignment_is_inside_safety_corridor(
                    final_y_error,
                    final_z_error,
                    outer_recovery_corridor,
                ):
                    raise RuntimeError(
                        "安全深度闭环对中已离开动态纠偏安全通道："
                        f"local={measured_local}, "
                        f"corridor={outer_recovery_corridor:.4f} m"
                    )
                if (
                    final_y_error > alignment_tolerance_y
                    or final_z_error > alignment_tolerance_z
                ):
                    self.get_logger().warning(
                        "Outer alignment stopped inside the object's recoverable "
                        "geometry rather than the strict target; real-collision "
                        f"feedback will finish centering: local={measured_local}"
                    )
                insertion_distance = measured_local[0] - insertion_center_x
                minimum_insertion_distance = float(
                    profile.get("minimum_insertion_distance_m", 0.02)
                )
                maximum_insertion_distance = float(
                    profile.get("maximum_insertion_distance_m", 0.10)
                )
                if not (
                    minimum_insertion_distance
                    <= insertion_distance
                    <= maximum_insertion_distance
                ):
                    raise RuntimeError(
                        "安全深度测量无法生成有效插入距离："
                        f"local={measured_local}, insertion={insertion_distance:.3f} m"
                    )
                # Freeze is only an outer alignment aid.  Restore real physics
                # before entering the jaws, while the apple is still far enough
                # away that enabling its collider cannot overlap either finger.
                stage = "外侧安全对中后恢复真实碰撞"
                object_before_outer_unfreeze = self._current_object_position(
                    target_object
                )
                self._command_scene_grasp("unfreeze", target_object)
                if not self._wait_scene_grasp("UNFROZEN", "", 2.0):
                    raise RuntimeError("Isaac 未确认目标已在外侧安全位恢复碰撞")
                time.sleep(self.unfreeze_settle_duration)
                self._command_scene_grasp("probe", target_object)
                if not self._wait_scene_grasp("PROBED", target_object, 2.0):
                    raise RuntimeError("Isaac 未返回外侧解冻状态")
                with self._lock:
                    outer_probe = dict(self._scene_grasp_status)
                measured_local = tuple(
                    float(value)
                    for value in outer_probe.get("local_position_m", [])
                )
                object_after_outer_unfreeze = self._current_object_position(
                    target_object
                )
                outer_unfreeze_drift = math.sqrt(
                    sum(
                        (after - before) ** 2
                        for after, before in zip(
                            object_after_outer_unfreeze,
                            object_before_outer_unfreeze,
                        )
                    )
                )
                outer_left = outer_probe.get("left_finger_contact_object")
                outer_right = outer_probe.get("right_finger_contact_object")
                if (
                    len(measured_local) != 3
                    or not bool(outer_probe.get("physics_enabled", False))
                    or outer_unfreeze_drift
                    > self.unfreeze_max_position_drift
                    or target_object in (outer_left, outer_right)
                    or not _outer_alignment_is_inside_safety_corridor(
                        measured_local[1] - desired_center_y,
                        measured_local[2] - entry_center_z,
                        outer_recovery_corridor,
                    )
                ):
                    raise RuntimeError(
                        "外侧安全位恢复碰撞失败："
                        f"local={measured_local}, "
                        f"drift={outer_unfreeze_drift:.4f} m, "
                        f"left={outer_left}, right={outer_right}"
                    )

                stage = "真实碰撞分段预抓取接近"
                maximum_segments = (
                    int(
                        math.ceil(
                            max(
                                0.0,
                                measured_local[0] - dynamic_insertion_stop_x,
                            )
                            / dynamic_insertion_step
                        )
                    )
                    * int(
                        profile.get(
                            "dynamic_alignment_segment_multiplier",
                            2
                            if bool(
                                profile.get(
                                    "decouple_dynamic_alignment", False
                                )
                            )
                            else 1,
                        )
                    )
                    + int(
                        profile.get(
                            "dynamic_alignment_correction_budget", 12
                        )
                    )
                )
                dynamic_abort_corridor = outer_recovery_corridor
                dynamic_gain_y = float(
                    profile.get("dynamic_alignment_gain_y", 0.75)
                )
                dynamic_gain_z = float(
                    profile.get("dynamic_alignment_gain_z", 0.75)
                )
                segmented_local = measured_local
                preclose_contact_observed = False
                unilateral_contact_recoveries = 0
                cylinder_high_contact_recoveries = 0
                adaptive_insertion_step = dynamic_insertion_step
                previous_command_local = None
                previous_commanded_axial_step = 0.0
                previous_commanded_lateral_step = 0.0
                axial_stall_count = 0
                axial_lateral_coupling = False
                best_dynamic_depth_x = float(measured_local[0])
                dynamic_safe_position, dynamic_safe_orientation = (
                    self._current_ee_pose()
                )
                depth_recovery_reference_x = None
                safe_pose_recoveries = 0
                locked_dynamic_orientation = None
                if bool(
                    profile.get("lock_dynamic_insertion_orientation", False)
                ):
                    locked_dynamic_orientation = self._current_ee_pose()[1]
                near_grasp_orientation_steps = tuple(
                    float(value)
                    for value in profile.get(
                        "near_grasp_elevation_steps_deg", ()
                    )
                )
                near_grasp_orientation_complete = not bool(
                    near_grasp_orientation_steps
                )
                near_grasp_transition_x = float(
                    profile.get(
                        "near_grasp_transition_x_m",
                        dynamic_insertion_stop_x,
                    )
                )
                fine_alignment_start_x_value = profile.get(
                    "dynamic_fine_alignment_start_x_m"
                )
                fine_alignment_start_x = (
                    float(fine_alignment_start_x_value)
                    if fine_alignment_start_x_value is not None
                    else math.inf
                )
                fine_alignment_started = (
                    fine_alignment_start_x_value is None
                    or float(measured_local[0]) <= fine_alignment_start_x
                )
                far_alignment_recovery_active = False
                insertion_segment_indices = (
                    itertools.count()
                    if unlimited_fine_adjustment
                    else range(maximum_segments)
                )
                for segment_index in insertion_segment_indices:
                    self._command_scene_grasp("probe", target_object)
                    if not self._wait_scene_grasp(
                        "PROBED", target_object, 2.0
                    ):
                        raise RuntimeError("Isaac 未返回真实碰撞分段接近状态")
                    with self._lock:
                        insertion_probe = dict(self._scene_grasp_status)
                    segmented_local = tuple(
                        float(value)
                        for value in insertion_probe.get("local_position_m", [])
                    )
                    if len(segmented_local) != 3:
                        raise RuntimeError("Isaac 返回的动态插入局部位姿无效")
                    if (
                        not fine_alignment_started
                        and segmented_local[0] <= fine_alignment_start_x
                    ):
                        fine_alignment_started = True
                        self.get_logger().info(
                            "Axial coarse approach reached the fine-alignment "
                            "entry boundary: "
                            f"local_x={segmented_local[0]:.4f} m, "
                            f"boundary={fine_alignment_start_x:.4f} m"
                        )
                    if (
                        previous_command_local is not None
                        and abs(previous_commanded_axial_step) > 1.0e-6
                    ):
                        measured_axial_progress = math.copysign(
                            1.0, previous_commanded_axial_step
                        ) * (
                            float(previous_command_local[0])
                            - float(segmented_local[0])
                        )
                        previous_step_limit = adaptive_insertion_step
                        adaptive_insertion_step = _adaptive_dynamic_step_limit(
                            adaptive_insertion_step,
                            previous_commanded_axial_step,
                            measured_axial_progress,
                            dynamic_insertion_step,
                            maximum_dynamic_insertion_step,
                            float(
                                profile.get(
                                    "dynamic_feedback_min_progress_ratio", 0.35
                                )
                            ),
                            float(
                                profile.get(
                                    "dynamic_feedback_step_growth", 1.8
                                )
                            ),
                        )
                        minimum_effective_progress = float(
                            profile.get(
                                "dynamic_feedback_min_progress_m", 0.001
                            )
                        )
                        if measured_axial_progress < minimum_effective_progress:
                            axial_stall_count += 1
                        else:
                            axial_stall_count = 0
                        # Learn X->Y cross-talk only from a pure axial sample.
                        # A combined correction cannot identify the two axes.
                        if abs(previous_commanded_lateral_step) <= 1.0e-6:
                            measured_lateral_drift = (
                                float(segmented_local[1])
                                - float(previous_command_local[1])
                            )
                            coupling_observed = (
                                _dynamic_axis_coupling_detected(
                                    measured_axial_progress,
                                    measured_lateral_drift,
                                    float(
                                        profile.get(
                                            "dynamic_feedback_coupling_min_drift_m",
                                            0.004,
                                        )
                                    ),
                                    float(
                                        profile.get(
                                            "dynamic_feedback_coupling_ratio", 0.70
                                        )
                                    ),
                                )
                            )
                            # Coupling belongs to the active IK branch.  Latch a
                            # positive observation so one lower-ratio sample does
                            # not send the controller back to alternating X/Y.
                            axial_lateral_coupling = (
                                axial_lateral_coupling or coupling_observed
                            )
                            if coupling_observed:
                                self.get_logger().warning(
                                    "Measured axial/lateral coupling: "
                                    f"axial_progress={measured_axial_progress:.4f} m, "
                                    f"lateral_drift={measured_lateral_drift:.4f} m"
                                )
                        if (
                            abs(adaptive_insertion_step - previous_step_limit)
                            > 1.0e-6
                            or axial_stall_count > 0
                        ):
                            self.get_logger().warning(
                                "Adaptive insertion feedback: "
                                f"commanded={previous_commanded_axial_step:.4f} m, "
                                f"measured_progress={measured_axial_progress:.4f} m, "
                                f"next_limit={adaptive_insertion_step:.4f} m, "
                                f"stall_count={axial_stall_count}"
                            )
                    best_dynamic_depth_x = min(
                        best_dynamic_depth_x, float(segmented_local[0])
                    )
                    if not bool(insertion_probe.get("physics_enabled", False)):
                        raise RuntimeError("真实碰撞接近期间苹果不再是动态刚体")
                    self._raise_if_finger_touching_table(
                        insertion_probe,
                        f"动态分段插入第 {segment_index + 1} 段前",
                    )
                    if depth_recovery_reference_x is not None:
                        minimum_depth_improvement = float(
                            profile.get(
                                "dynamic_alignment_depth_recovery_min_improvement_m",
                                0.001,
                            )
                        )
                        if (
                            segmented_local[0]
                            < depth_recovery_reference_x
                            + minimum_depth_improvement
                        ):
                            safe_pose_limit = int(
                                profile.get(
                                    "dynamic_alignment_safe_pose_recovery_attempts",
                                    0,
                                )
                            )
                            if safe_pose_recoveries >= safe_pose_limit:
                                raise RuntimeError(
                                    "动态对中深度回退没有改善，且已耗尽外侧"
                                    "安全位恢复次数："
                                    f"local={segmented_local}, "
                                    f"previous_x={depth_recovery_reference_x:.4f}"
                                )
                            safe_pose_recoveries += 1
                            safe_pose_margin = float(
                                profile.get(
                                    "dynamic_alignment_world_z_floor_margin_m",
                                    0.0,
                                )
                            )
                            restored_safe_position = (
                                dynamic_safe_position[0],
                                dynamic_safe_position[1],
                                dynamic_safe_position[2] + safe_pose_margin,
                            )
                            self.get_logger().warning(
                                "Local depth retreat did not improve measured X; "
                                "restoring the collision-free dynamic entry pose: "
                                f"attempt={safe_pose_recoveries}/"
                                f"{safe_pose_limit}, local={segmented_local}"
                            )
                            self._execute_manual_pose(
                                restored_safe_position,
                                dynamic_safe_orientation,
                                "动态深度漂移后恢复外侧安全位",
                                linear=True,
                                max_joint_delta=float(
                                    profile.get(
                                        "dynamic_alignment_safe_pose_max_joint_delta_rad",
                                        0.65,
                                    )
                                ),
                                velocity_scale=(
                                    self.grasp_alignment_velocity_scale
                                ),
                            )
                            depth_recovery_reference_x = None
                            continue
                        depth_recovery_reference_x = None
                    inside_segment_envelope = inside_grasp_envelope(
                        segmented_local, profile_envelope
                    )
                    dynamic_target_z = desired_center_z
                    dynamic_entry_until_x = profile.get(
                        "dynamic_entry_center_z_until_x_m"
                    )
                    entry_height_active = (
                        dynamic_entry_until_x is not None
                        and segmented_local[0]
                        > float(dynamic_entry_until_x)
                    )
                    if entry_height_active:
                        dynamic_target_z = entry_center_z
                    reached_safe_entry = (
                        inside_segment_envelope
                        and abs(
                            segmented_local[0] - dynamic_insertion_stop_x
                        )
                        <= insertion_x_tolerance
                        and abs(
                            segmented_local[1] - desired_center_y
                        )
                        <= insertion_alignment_tolerance_y
                        and abs(
                            segmented_local[2] - dynamic_target_z
                        )
                        <= insertion_alignment_tolerance_z
                    )
                    left_contact = insertion_probe.get(
                        "left_finger_contact_object"
                    )
                    right_contact = insertion_probe.get(
                        "right_finger_contact_object"
                    )
                    target_contact_allowed = bool(
                        profile.get("allow_preclose_target_contact", False)
                    )
                    target_contact_present = target_object in (
                        left_contact,
                        right_contact,
                    )
                    if (
                        not near_grasp_orientation_complete
                        and segmented_local[0] <= near_grasp_transition_x
                        and abs(segmented_local[1] - desired_center_y)
                        <= float(
                            profile.get(
                                "near_grasp_transition_alignment_y_m",
                                insertion_alignment_tolerance_y,
                            )
                        )
                        and abs(segmented_local[2] - dynamic_target_z)
                        <= float(
                            profile.get(
                                "near_grasp_transition_alignment_z_m",
                                insertion_alignment_tolerance_z,
                            )
                        )
                    ):
                        if target_contact_present:
                            raise RuntimeError(
                                "近水平姿态转换前夹指已接触目标："
                                f"local={segmented_local}, left={left_contact}, "
                                f"right={right_contact}"
                            )
                        stage = "瓶前安全位近水平姿态过渡"
                        self._set_status(
                            "GUIDED_ORIENTATION",
                            "已完成大部分轴向接近，正在瓶前分段转为"
                            "近水平侧夹姿态",
                            False,
                        )
                        azimuth = math.radians(
                            float(candidate["azimuth_deg"])
                        )
                        roll_deg = float(profile.get("roll_deg", 120.0))
                        previous_elevation = float(candidate["elevation_deg"])
                        for step_index, elevation_deg in enumerate(
                            near_grasp_orientation_steps, start=1
                        ):
                            if elevation_deg >= previous_elevation:
                                raise RuntimeError(
                                    "近水平姿态过渡角必须逐段减小："
                                    f"previous={previous_elevation:.1f}°, "
                                    f"next={elevation_deg:.1f}°"
                                )
                            elevation = math.radians(elevation_deg)
                            approach_direction = (
                                math.cos(elevation) * math.cos(azimuth),
                                math.cos(elevation) * math.sin(azimuth),
                                -math.sin(elevation),
                            )
                            transition_orientation = orientation_for_approach(
                                approach_direction, roll_deg
                            )
                            current_ee_position, _ = self._current_ee_pose()
                            current_object_position = (
                                self._current_object_position(target_object)
                            )
                            transition_target_point = (
                                current_object_position[0],
                                current_object_position[1],
                                current_object_position[2]
                                + float(profile["height"])
                                + float(
                                    profile.get("world_z_offset_m", 0.0)
                                ),
                            )
                            transition_standoff = math.sqrt(
                                sum(
                                    (target - current) ** 2
                                    for target, current in zip(
                                        transition_target_point,
                                        current_ee_position,
                                    )
                                )
                            )
                            transition_position = tuple(
                                transition_target_point[index]
                                - transition_standoff
                                * approach_direction[index]
                                for index in range(3)
                            )
                            self._execute_manual_pose(
                                transition_position,
                                transition_orientation,
                                "瓶前安全位近水平姿态过渡 "
                                f"{step_index}/"
                                f"{len(near_grasp_orientation_steps)} "
                                f"({elevation_deg:.1f}°)",
                                linear=False,
                                max_joint_delta=float(
                                    profile.get(
                                        "near_grasp_orientation_max_joint_delta_rad",
                                        0.65,
                                    )
                                ),
                                position_tolerance=0.003,
                                orientation_tolerance=float(
                                    profile.get(
                                        "near_grasp_orientation_tolerance_rad",
                                        0.020,
                                    )
                                ),
                                velocity_scale=(
                                    self.grasp_alignment_velocity_scale
                                ),
                            )
                            previous_elevation = elevation_deg
                        self._command_scene_grasp("probe", target_object)
                        if not self._wait_scene_grasp(
                            "PROBED", target_object, 2.0
                        ):
                            raise RuntimeError(
                                "Isaac 未返回近水平姿态转换后的局部位置"
                            )
                        with self._lock:
                            transition_probe = dict(
                                self._scene_grasp_status
                            )
                        transitioned_local = tuple(
                            float(value)
                            for value in transition_probe.get(
                                "local_position_m", []
                            )
                        )
                        transition_envelope = profile.get(
                            "near_grasp_transition_envelope"
                        )
                        if (
                            len(transitioned_local) != 3
                            or not bool(
                                transition_probe.get("physics_enabled", False)
                            )
                            or target_object
                            in (
                                transition_probe.get(
                                    "left_finger_contact_object"
                                ),
                                transition_probe.get(
                                    "right_finger_contact_object"
                                ),
                            )
                            or (
                                isinstance(transition_envelope, dict)
                                and not inside_grasp_envelope(
                                    transitioned_local, transition_envelope
                                )
                            )
                        ):
                            raise RuntimeError(
                                "近水平姿态转换后不在安全过渡包络："
                                f"local={transitioned_local}, "
                                f"envelope={transition_envelope}"
                            )
                        self._raise_if_finger_touching_table(
                            transition_probe, "近水平姿态转换后"
                        )
                        near_grasp_orientation_complete = True
                        locked_dynamic_orientation = (
                            self._current_ee_pose()[1]
                        )
                        dynamic_safe_position, dynamic_safe_orientation = (
                            self._current_ee_pose()
                        )
                        previous_command_local = None
                        previous_commanded_axial_step = 0.0
                        previous_commanded_lateral_step = 0.0
                        axial_stall_count = 0
                        axial_lateral_coupling = False
                        best_dynamic_depth_x = float(transitioned_local[0])
                        stage = "真实碰撞分段预抓取接近"
                        continue
                    cylinder_profile = OBJECT_GRASP_PROFILES.get(
                        target_object, {}
                    )
                    calibrated_high_contact_ready = (
                        _calibrated_high_cylinder_contact_is_ready(
                            insertion_probe,
                            target_object,
                            profile,
                            cylinder_profile,
                        )
                    )
                    if calibrated_high_contact_ready:
                        self.get_logger().warning(
                            "Taught bottle grasp pose reached by high-contact "
                            "proximity cue; preserving pose and proceeding to "
                            "slow close with bilateral-contact validation: "
                            f"local={segmented_local}"
                        )
                    if (
                        target_contact_present
                        and cylinder_profile.get("strategy")
                        == "cylinder_side_pinch"
                        and not calibrated_high_contact_ready
                        and not cylinder_target_contact_heights_are_valid(
                            insertion_probe,
                            target_object,
                            cylinder_profile,
                        )
                    ):
                        allowed_band = cylinder_profile.get(
                            "grasp_height_range_m", ()
                        )
                        contact_points = [
                            insertion_probe.get(
                                "left_finger_contact_point_object_m"
                            ),
                            insertion_probe.get(
                                "right_finger_contact_point_object_m"
                            ),
                        ]
                        contact_heights = [
                            float(point[2])
                            for point in contact_points
                            if isinstance(point, (list, tuple))
                            and len(point) == 3
                        ]
                        high_contact = (
                            len(allowed_band) == 2
                            and bool(contact_heights)
                            and min(contact_heights) > float(allowed_band[1])
                        )
                        recovery_limit = int(
                            profile.get(
                                "cylinder_high_contact_recovery_attempts", 0
                            )
                        )
                        if (
                            high_contact
                            and cylinder_high_contact_recoveries < recovery_limit
                        ):
                            cylinder_high_contact_recoveries += 1
                            recovery_position, recovery_orientation = (
                                self._current_ee_pose()
                            )
                            recovery_stator_orientation = (
                                self._current_stator_orientation()
                            )
                            recovery_retreat = float(
                                profile.get(
                                    "cylinder_high_contact_recovery_retreat_m",
                                    0.010,
                                )
                            )
                            recovery_lower = float(
                                profile.get(
                                    "cylinder_high_contact_recovery_lower_m",
                                    0.025,
                                )
                            )
                            retreat_world = rotate_vector(
                                recovery_stator_orientation,
                                (-recovery_retreat, 0.0, 0.0),
                            )
                            recovery_target = (
                                recovery_position[0] + retreat_world[0],
                                recovery_position[1] + retreat_world[1],
                                recovery_position[2]
                                + retreat_world[2]
                                - recovery_lower,
                            )
                            self.get_logger().warning(
                                "Recovering high cylinder contact: "
                                f"attempt={cylinder_high_contact_recoveries}/"
                                f"{recovery_limit}, heights={contact_heights}, "
                                f"retreat={recovery_retreat:.3f} m, "
                                f"lower={recovery_lower:.3f} m"
                            )
                            self._execute_manual_pose(
                                recovery_target,
                                recovery_orientation,
                                "瓶盖提前接触后退并降低至瓶身",
                                linear=False,
                                max_joint_delta=0.65,
                                position_tolerance=0.002,
                                orientation_tolerance=0.10,
                                velocity_scale=(
                                    self.grasp_alignment_velocity_scale
                                ),
                            )
                            continue
                        raise RuntimeError(
                            "柱体预抓取首次接触不在瓶身夹持高度带："
                            f"left={insertion_probe.get('left_finger_contact_point_object_m')}, "
                            f"right={insertion_probe.get('right_finger_contact_point_object_m')}, "
                            f"allowed_z={allowed_band}"
                        )
                    preclose_contact_observed = (
                        preclose_contact_observed or target_contact_present
                    )
                    contact_state_acceptable = (
                        target_contact_allowed or not target_contact_present
                    )
                    reached_preclose_corridor = (
                        (not entry_height_active or target_contact_present)
                        and
                        contact_state_acceptable
                        and _inside_preclose_corridor(
                            segmented_local,
                            profile,
                            profile_envelope,
                            target_contact_present,
                        )
                    )
                    insertion_depth_allows_close = (
                        _dynamic_insertion_depth_allows_close(
                            segmented_local,
                            profile,
                            target_contact_present,
                        )
                    )
                    self.get_logger().info(
                        "Real-collision insertion probe: "
                        f"segment={segment_index}, local={segmented_local}, "
                        f"safe_entry={reached_safe_entry}, "
                        f"preclose_corridor={reached_preclose_corridor}, "
                        f"depth_allows_close={insertion_depth_allows_close}, "
                        f"left={left_contact}, right={right_contact}"
                    )
                    if (
                        near_grasp_orientation_complete
                        and
                        contact_state_acceptable
                        and (reached_safe_entry or reached_preclose_corridor)
                        and insertion_depth_allows_close
                        and (
                            not bool(
                                profile.get(
                                    "require_preclose_target_contact", False
                                )
                            )
                            or preclose_contact_observed
                        )
                    ):
                        break
                    if (
                        not target_contact_allowed
                        and target_contact_present
                    ):
                        unilateral_target_contact = (
                            (left_contact == target_object)
                            != (right_contact == target_object)
                        )
                        recovery_limit = int(
                            profile.get(
                                "preclose_contact_recovery_attempts",
                                0,
                            )
                        )
                        if (
                            bool(
                                profile.get(
                                    "recover_unilateral_preclose_contact",
                                    False,
                                )
                            )
                            and unilateral_target_contact
                            and unilateral_contact_recoveries < recovery_limit
                        ):
                            unilateral_contact_recoveries += 1
                            recovery_start, recovery_orientation = (
                                self._current_ee_pose()
                            )
                            recovery_retreat = float(
                                profile.get(
                                    "preclose_contact_recovery_retreat_m",
                                    0.005,
                                )
                            )
                            recovery_lift = float(
                                profile.get(
                                    "preclose_contact_recovery_lift_m",
                                    0.005,
                                )
                            )
                            recovery_stator_orientation = (
                                self._current_stator_orientation()
                            )
                            recovery_world = rotate_vector(
                                recovery_stator_orientation,
                                (-recovery_retreat, 0.0, 0.0),
                            )
                            recovery_target = (
                                recovery_start[0] + recovery_world[0],
                                recovery_start[1] + recovery_world[1],
                                recovery_start[2]
                                + recovery_world[2]
                                + recovery_lift,
                            )
                            self.get_logger().warning(
                                "Recovering an early unilateral target contact "
                                "before close: "
                                f"attempt={unilateral_contact_recoveries}/"
                                f"{recovery_limit}, local={segmented_local}, "
                                f"retreat={recovery_retreat:.4f} m, "
                                f"lift={recovery_lift:.4f} m"
                            )
                            self._execute_manual_pose(
                                recovery_target,
                                recovery_orientation,
                                "单指提前接触后退并抬高重对中",
                                linear=True,
                                max_joint_delta=0.45,
                                velocity_scale=(
                                    self.grasp_alignment_velocity_scale
                                ),
                            )
                            self._command_scene_grasp(
                                "probe", target_object
                            )
                            if not self._wait_scene_grasp(
                                "PROBED", target_object, 2.0
                            ):
                                raise RuntimeError(
                                    "Isaac 未返回单指接触恢复状态"
                                )
                            with self._lock:
                                recovery_probe = dict(
                                    self._scene_grasp_status
                                )
                            self._raise_if_finger_touching_table(
                                recovery_probe,
                                "单指提前接触恢复后",
                            )
                            continue
                        raise RuntimeError(
                            "目标尚未进入夹持包络便接触夹指，已停止继续推进："
                            f"segment={segment_index}, local={segmented_local}, "
                            f"left={left_contact}, right={right_contact}"
                        )
                    if (
                        abs(segmented_local[1] - desired_center_y)
                        > dynamic_abort_corridor
                        or abs(segmented_local[2] - dynamic_target_z)
                        > dynamic_abort_corridor
                    ):
                        raise RuntimeError(
                            "真实碰撞接近时目标已偏离可恢复安全通道："
                            f"segment={segment_index}, local={segmented_local}, "
                            f"abort_corridor={dynamic_abort_corridor:.4f} m"
                        )
                    remaining_distance = (
                        segmented_local[0] - dynamic_insertion_stop_x
                    )
                    step_distance = max(
                        -adaptive_insertion_step,
                        min(adaptive_insertion_step, remaining_distance),
                    )
                    if (
                        bool(
                            profile.get(
                                "require_preclose_target_contact", False
                            )
                        )
                        and not preclose_contact_observed
                    ):
                        contact_search_remaining = (
                            segmented_local[0]
                            - float(
                                profile.get(
                                    "contact_search_min_x_m",
                                    dynamic_insertion_stop_x,
                                )
                            )
                        )
                        step_distance = max(
                            0.0,
                            min(
                                adaptive_insertion_step,
                                contact_search_remaining,
                            ),
                        )
                    ee_position, current_ee_orientation = self._current_ee_pose()
                    ee_orientation = (
                        locked_dynamic_orientation
                        if locked_dynamic_orientation is not None
                        else current_ee_orientation
                    )
                    stator_orientation = self._current_stator_orientation()
                    lateral_error = segmented_local[1] - desired_center_y
                    vertical_error = (
                        segmented_local[2] - dynamic_target_z
                    )
                    active_y_tolerance = insertion_alignment_tolerance_y
                    active_z_tolerance = insertion_alignment_tolerance_z
                    if target_contact_present:
                        active_y_tolerance = max(
                            active_y_tolerance,
                            float(
                                profile.get(
                                    "contact_alignment_tolerance_y_m",
                                    active_y_tolerance,
                                )
                            ),
                        )
                        active_z_tolerance = max(
                            active_z_tolerance,
                            float(
                                profile.get(
                                    "contact_alignment_tolerance_z_m",
                                    active_z_tolerance,
                                )
                            ),
                        )
                    y_outside = (
                        abs(lateral_error)
                        > active_y_tolerance
                    )
                    z_outside = (
                        abs(vertical_error)
                        > active_z_tolerance
                    )
                    lateral_step = _bounded_axis_correction(
                        lateral_error,
                        active_y_tolerance,
                        dynamic_gain_y,
                        dynamic_lateral_step,
                    )
                    lateral_step *= float(
                        profile.get("dynamic_lateral_correction_sign", 1.0)
                    )
                    requested_lateral_step = lateral_step
                    if bool(
                        profile.get(
                            "disable_dynamic_lateral_correction", False
                        )
                    ):
                        lateral_step = 0.0
                    dynamic_vertical_step = float(
                        profile.get(
                            "dynamic_vertical_step_m",
                            dynamic_lateral_step,
                        )
                    )
                    vertical_step = _bounded_axis_correction(
                        vertical_error,
                        active_z_tolerance,
                        dynamic_gain_z,
                        dynamic_vertical_step,
                    )
                    if bool(
                        profile.get(
                            "disable_dynamic_vertical_correction", False
                        )
                    ):
                        vertical_step = 0.0
                    if bool(
                        profile.get("align_before_axial_advance", True)
                    ) and fine_alignment_started and (y_outside or z_outside):
                        step_distance = 0.0
                    alignment_axis_order = tuple(
                        profile.get("alignment_axis_order", ())
                    )
                    if (
                        not alignment_axis_order
                        and bool(
                            profile.get("decouple_dynamic_alignment", False)
                        )
                        and y_outside
                        and z_outside
                    ):
                        normalized_y = abs(lateral_error) / max(
                            insertion_alignment_tolerance_y, 1.0e-6
                        )
                        normalized_z = abs(vertical_error) / max(
                            insertion_alignment_tolerance_z, 1.0e-6
                        )
                        if normalized_y >= normalized_z:
                            vertical_step = 0.0
                        else:
                            lateral_step = 0.0
                    elif not y_outside and not z_outside:
                        axial_feedforward = float(
                            profile.get(
                                "dynamic_axial_feedforward_y_m", 0.0
                            )
                        )
                        if axial_feedforward:
                            lateral_step = max(
                                -dynamic_lateral_step,
                                min(
                                    dynamic_lateral_step,
                                    axial_feedforward + lateral_error,
                                ),
                            )
                        vertical_feedforward = float(
                            profile.get(
                                "dynamic_axial_feedforward_z_m", 0.0
                            )
                        )
                        contact_search_active = (
                            bool(
                                profile.get(
                                    "require_preclose_target_contact", False
                                )
                            )
                            and not preclose_contact_observed
                            and segmented_local[0]
                            <= dynamic_insertion_stop_x
                            + insertion_x_tolerance
                        )
                        if contact_search_active:
                            vertical_feedforward = 0.0
                        if step_distance and vertical_feedforward:
                            vertical_step = max(
                                -dynamic_vertical_step,
                                min(
                                    dynamic_vertical_step,
                                    vertical_feedforward + vertical_error,
                                ),
                            )
                    if fine_alignment_started and alignment_axis_order:
                        (
                            step_distance,
                            lateral_step,
                            vertical_step,
                        ) = _ordered_axis_correction(
                            (step_distance, lateral_step, vertical_step),
                            (
                                abs(remaining_distance)
                                > insertion_x_tolerance,
                                y_outside,
                                z_outside,
                            ),
                            alignment_axis_order,
                        )
                    if not fine_alignment_started:
                        # Outside the near region, Y/Z correction has no grasping
                        # benefit.  Only contain material Y drift, and do so in
                        # the same pose as axial progress so X is never undone.
                        axial_only_before_calibration = bool(
                            profile.get(
                                "axial_only_before_calibrated_alignment", False
                            )
                        )
                        coarse_xy_correction = (
                            not axial_only_before_calibration
                            and abs(lateral_error)
                            > float(
                                profile.get(
                                    "coarse_alignment_y_guard_m", math.inf
                                )
                            )
                        )
                        lateral_step = (
                            requested_lateral_step
                            if coarse_xy_correction
                            else 0.0
                        )
                        vertical_step = 0.0
                        if axial_only_before_calibration:
                            (
                                step_distance,
                                far_alignment_recovery_active,
                                far_guard_active,
                            ) = _far_alignment_guard(
                                lateral_error,
                                step_distance,
                                float(
                                    profile.get(
                                        "far_alignment_soft_corridor_m",
                                        math.inf,
                                    )
                                ),
                                float(
                                    profile.get(
                                        "far_alignment_recovery_trigger_m",
                                        math.inf,
                                    )
                                ),
                                float(
                                    profile.get(
                                        "far_alignment_recovery_release_m",
                                        math.inf,
                                    )
                                ),
                                float(
                                    profile.get(
                                        "far_alignment_reduced_axial_step_m",
                                        dynamic_insertion_step,
                                    )
                                ),
                                far_alignment_recovery_active,
                            )
                            lateral_step = (
                                requested_lateral_step
                                if far_alignment_recovery_active
                                else 0.0
                            )
                            if far_guard_active:
                                self.get_logger().warning(
                                    "Far bottle Y guard active: "
                                    f"local={segmented_local}, "
                                    f"y_error={lateral_error:.4f} m, "
                                    f"axial_step={step_distance:.4f} m, "
                                    "recovery="
                                    f"{far_alignment_recovery_active}"
                                )
                    else:
                        coarse_xy_correction = False
                    coupled_xy_correction = (
                        fine_alignment_started
                        and bool(
                            profile.get(
                                "dynamic_feedback_coupled_xy_correction", False
                            )
                        )
                        and (
                            axial_lateral_coupling
                            or bool(
                                profile.get(
                                    "assume_dynamic_xy_coupling", False
                                )
                            )
                        )
                        and y_outside
                        and not z_outside
                        and abs(remaining_distance) > insertion_x_tolerance
                    )
                    if coupled_xy_correction:
                        # Preserve centering and axial advance in one final pose.
                        # MoveIt's joint-space plan avoids the observed alternating
                        # LIN X/Y commands whose physical effects cancel out.
                        step_distance = max(
                            -adaptive_insertion_step,
                            min(adaptive_insertion_step, remaining_distance),
                        )
                        lateral_step = requested_lateral_step
                        vertical_step = 0.0
                    depth_guard_active = False
                    if bool(
                        profile.get("guard_dynamic_alignment_depth", False)
                    ):
                        (
                            step_distance,
                            lateral_step,
                            vertical_step,
                            depth_guard_active,
                        ) = _guard_dynamic_alignment_depth(
                            segmented_local[0],
                            dynamic_insertion_stop_x,
                            y_outside,
                            z_outside,
                            step_distance,
                            lateral_step,
                            vertical_step,
                            float(
                                profile.get(
                                    "dynamic_alignment_centering_depth_guard_m",
                                    0.005,
                                )
                            ),
                            float(
                                profile.get(
                                    "dynamic_alignment_depth_recovery_max_step_m",
                                    0.008,
                                )
                            ),
                        )
                        if depth_guard_active:
                            depth_recovery_reference_x = segmented_local[0]
                    if "coarse_insertion_step_m" in profile:
                        step_distance = _staged_insertion_step(
                            step_distance,
                            remaining_distance,
                            float(profile["coarse_insertion_step_m"]),
                            float(profile["fine_insertion_distance_m"]),
                            float(profile["fine_insertion_step_m"]),
                            coarse_allowed=(
                                not y_outside and not z_outside
                                and not depth_guard_active
                                and axial_stall_count == 0
                                and abs(lateral_step) < 1.0e-6
                                and abs(vertical_step) < 1.0e-6
                                and insertion_probe.get("contact_report_available", False)
                                and insertion_probe.get("left_finger_contact_object") is None
                                and insertion_probe.get("right_finger_contact_object") is None
                            ),
                        )
                    insertion_world = rotate_vector(
                        stator_orientation,
                        (step_distance, lateral_step, vertical_step),
                    )
                    inserted_position = tuple(
                        ee_position[index] + insertion_world[index]
                        for index in range(3)
                    )
                    world_z_floor_margin = float(
                        profile.get(
                            "dynamic_alignment_world_z_floor_margin_m",
                            0.0,
                        )
                    )
                    if world_z_floor_margin > 0.0:
                        inserted_position = (
                            inserted_position[0],
                            inserted_position[1],
                            max(
                                inserted_position[2],
                                dynamic_safe_position[2]
                                + world_z_floor_margin,
                            ),
                        )
                    self.get_logger().info(
                        "Real-collision insertion command: "
                        f"segment={segment_index}, "
                        "stator_local_step=("
                        f"{step_distance:.4f}, {lateral_step:.4f}, "
                        f"{vertical_step:.4f}), "
                        f"depth_guard={depth_guard_active}, "
                        f"best_x={best_dynamic_depth_x:.4f}, "
                        f"adaptive_limit={adaptive_insertion_step:.4f}, "
                        f"world_z_target={inserted_position[2]:.4f}"
                    )
                    previous_command_local = segmented_local
                    previous_commanded_axial_step = step_distance
                    previous_commanded_lateral_step = lateral_step
                    # Keep coarse recovery on the more permissive joint-space
                    # planner, but execute a near-object coupled XY correction as
                    # one orientation-locked Cartesian move.  Allowing OMPL a
                    # 0.10-rad orientation tube here changed the stator-local
                    # basis and repeatedly produced lateral motion with almost no
                    # axial progress.
                    xy_joint_space_correction = coarse_xy_correction
                    coupled_cartesian_correction = coupled_xy_correction
                    joint_space_fallback = (
                        xy_joint_space_correction
                        or (
                            not coupled_cartesian_correction
                            and
                            abs(step_distance) > 1.0e-6
                            and axial_stall_count
                            >= int(
                                profile.get(
                                    "dynamic_feedback_joint_space_after_stalls", 3
                                )
                            )
                        )
                    )
                    if joint_space_fallback:
                        self.get_logger().warning(
                            "Switching this bounded insertion correction to "
                            "joint-space planning: "
                            f"segment={segment_index}, "
                            f"coarse_xy={coarse_xy_correction}, "
                            f"coupled_xy={coupled_xy_correction}, "
                            f"stall_count={axial_stall_count}"
                        )
                    try:
                        self._execute_manual_pose(
                            inserted_position,
                            ee_orientation,
                            "目标真实碰撞状态下分段进入预抓取位 "
                            f"{segment_index + 1}",
                            linear=not joint_space_fallback,
                            max_joint_delta=(
                                float(
                                    profile.get(
                                        "dynamic_feedback_coupled_max_joint_delta_rad",
                                        0.65,
                                    )
                                )
                                if xy_joint_space_correction
                                else 0.45
                            ),
                            position_tolerance=float(
                                profile.get(
                                    "dynamic_alignment_moveit_position_tolerance_m",
                                    0.001,
                                )
                            ),
                            orientation_tolerance=(
                                float(
                                    profile.get(
                                        "dynamic_feedback_coupled_orientation_tolerance_rad",
                                        0.10,
                                    )
                                )
                                if xy_joint_space_correction
                                else float(
                                    profile.get(
                                        "insertion_orientation_tolerance_rad",
                                        0.04,
                                    )
                                )
                            ),
                            velocity_scale=self.grasp_alignment_velocity_scale,
                        )
                    except RuntimeError as exc:
                        message = str(exc)
                        if (
                            "Isaac Sim did not reach the final point within tolerance"
                            not in message
                            or "：-5 " not in message
                        ):
                            raise
                        self.get_logger().warning(
                            "Real-collision insertion segment timed out; "
                            "probing actual geometry before deciding: "
                            f"segment={segment_index + 1}, {message}"
                        )
                        insertion_timed_out = True
                else:
                    # The old code failed immediately after issuing the final
                    # segment, so it never inspected the pose produced by that
                    # motion.  Always probe once more and let the physical
                    # pre-close corridor decide whether grasping may continue.
                    self._command_scene_grasp("probe", target_object)
                    if not self._wait_scene_grasp(
                        "PROBED", target_object, 2.0
                    ):
                        raise RuntimeError("Isaac 未返回最大段数后的复检状态")
                    with self._lock:
                        insertion_probe = dict(self._scene_grasp_status)
                    segmented_local = tuple(
                        float(value)
                        for value in insertion_probe.get("local_position_m", [])
                    )
                    final_left_contact = insertion_probe.get(
                        "left_finger_contact_object"
                    )
                    final_right_contact = insertion_probe.get(
                        "right_finger_contact_object"
                    )
                    recovered_at_limit = (
                        near_grasp_orientation_complete
                        and
                        (
                            target_contact_allowed
                            or target_object
                            not in (final_left_contact, final_right_contact)
                        )
                        and bool(insertion_probe.get("physics_enabled", False))
                        and _inside_preclose_corridor(
                            segmented_local,
                            profile,
                            profile_envelope,
                            target_object
                            in (final_left_contact, final_right_contact),
                        )
                        and _dynamic_insertion_depth_allows_close(
                            segmented_local,
                            profile,
                            target_object
                            in (final_left_contact, final_right_contact),
                        )
                        and (
                            not bool(
                                profile.get(
                                    "require_preclose_target_contact", False
                                )
                            )
                            or preclose_contact_observed
                            or target_object
                            in (final_left_contact, final_right_contact)
                        )
                    )
                    if not recovered_at_limit:
                        raise RuntimeError(
                            "真实碰撞分段接近达到最大段数，复检后仍不在"
                            "安全可夹取区间："
                            f"local={segmented_local}, "
                            f"left={final_left_contact}, "
                            f"right={final_right_contact}"
                        )
                    self.get_logger().warning(
                        "Maximum real-collision insertion segments reached, "
                        "but the final probe is inside the safe pre-close "
                        f"corridor; continuing physical grasp: {segmented_local}"
                    )

                stage = "动态状态最终三维验收"
                final_frozen_local = None
                final_contact_present = False
                max_dynamic_corrections = int(
                    profile.get("final_alignment_iterations", 6)
                )
                final_alignment_gain = float(
                    profile.get("final_alignment_gain", 0.80)
                )
                final_alignment_max_step = float(
                    profile.get("final_alignment_max_step_m", 0.003)
                )
                final_alignment_indices = (
                    itertools.count()
                    if unlimited_fine_adjustment
                    else range(max_dynamic_corrections + 1)
                )
                for correction_index in final_alignment_indices:
                    self._command_scene_grasp("probe", target_object)
                    if not self._wait_scene_grasp("PROBED", target_object, 2.0):
                        raise RuntimeError("Isaac 未返回动态状态最终预抓取位")
                    with self._lock:
                        frozen_probe = dict(self._scene_grasp_status)
                    final_frozen_local = tuple(
                        float(value)
                        for value in frozen_probe.get("local_position_m", [])
                    )
                    if len(final_frozen_local) != 3:
                        raise RuntimeError("Isaac 返回的动态状态局部位姿无效")
                    final_contact_present = target_object in (
                        frozen_probe.get("left_finger_contact_object"),
                        frozen_probe.get("right_finger_contact_object"),
                    )
                    corridor_x_min = min(
                        desired_center_x, dynamic_insertion_stop_x
                    ) - insertion_x_tolerance
                    corridor_x_max = max(
                        desired_center_x, dynamic_insertion_stop_x
                    ) + insertion_x_tolerance
                    correction_x_target = min(
                        corridor_x_max,
                        max(corridor_x_min, final_frozen_local[0]),
                    )
                    desired_local = (
                        correction_x_target,
                        desired_center_y,
                        desired_center_z,
                    )
                    local_error = tuple(
                        measured - desired
                        for measured, desired in zip(
                            final_frozen_local, desired_local
                        )
                    )
                    self.get_logger().info(
                        "Final real-collision pre-grasp probe: "
                        f"iteration={correction_index}, "
                        f"local={final_frozen_local}, error={local_error}"
                    )
                    if (
                        not bool(
                            profile.get(
                                "allow_preclose_target_contact", False
                            )
                        )
                        and target_object
                        in (
                            frozen_probe.get("left_finger_contact_object"),
                            frozen_probe.get("right_finger_contact_object"),
                        )
                    ):
                        raise RuntimeError(
                            "最终三维对中时夹指提前接触苹果："
                            f"local={final_frozen_local}, "
                            "left="
                            f"{frozen_probe.get('left_finger_contact_object')}, "
                            "right="
                            f"{frozen_probe.get('right_finger_contact_object')}"
                        )
                    if (
                        _inside_preclose_corridor(
                            final_frozen_local,
                            profile,
                            profile_envelope,
                            final_contact_present,
                        )
                    ):
                        break
                    if (
                        not unlimited_fine_adjustment
                        and correction_index >= max_dynamic_corrections
                    ):
                        break
                    ee_position, ee_orientation = self._current_ee_pose()
                    stator_orientation = self._current_stator_orientation()
                    bounded_correction = (
                        _bounded_axis_correction(
                            local_error[0],
                            insertion_x_tolerance,
                            final_alignment_gain,
                            final_alignment_max_step,
                        ),
                        _bounded_axis_correction(
                            local_error[1],
                            insertion_alignment_tolerance_y,
                            final_alignment_gain,
                            final_alignment_max_step,
                        ),
                        _bounded_axis_correction(
                            local_error[2],
                            insertion_alignment_tolerance_z,
                            final_alignment_gain,
                            final_alignment_max_step,
                        ),
                    )
                    alignment_axis_order = tuple(
                        profile.get("alignment_axis_order", ())
                    )
                    if alignment_axis_order:
                        bounded_correction = _ordered_axis_correction(
                            bounded_correction,
                            (
                                abs(local_error[0]) > insertion_x_tolerance,
                                abs(local_error[1])
                                > insertion_alignment_tolerance_y,
                                abs(local_error[2])
                                > insertion_alignment_tolerance_z,
                            ),
                            alignment_axis_order,
                        )
                    world_correction = rotate_vector(
                        stator_orientation, bounded_correction
                    )
                    corrected_position = tuple(
                        ee_position[index] + world_correction[index]
                        for index in range(3)
                    )
                    try:
                        self._execute_manual_pose(
                            corrected_position,
                            ee_orientation,
                            "真实碰撞状态最终三维闭环微调",
                            linear=True,
                            max_joint_delta=0.45,
                            position_tolerance=float(
                                profile.get(
                                    "final_alignment_moveit_position_tolerance_m",
                                    0.0005,
                                )
                            ),
                            velocity_scale=self.grasp_alignment_velocity_scale,
                        )
                    except RuntimeError as exc:
                        message = str(exc)
                        if (
                            "Isaac Sim did not reach the final point within tolerance"
                            not in message
                            or "：-5 " not in message
                        ):
                            raise
                        self.get_logger().warning(
                            "Final 3D alignment retry timed out; probing the "
                            f"resulting geometry: {message}"
                        )
                if (
                    final_frozen_local is None
                    or not _inside_preclose_corridor(
                        final_frozen_local,
                        profile,
                        profile_envelope,
                        final_contact_present,
                    )
                ):
                    raise RuntimeError(
                        "闭爪前目标未进入安全可夹取区间："
                        f"local={final_frozen_local}, "
                        "x_range=("
                        f"{min(desired_center_x, dynamic_insertion_stop_x) - insertion_x_tolerance:.4f}, "
                        f"{max(desired_center_x, dynamic_insertion_stop_x) + insertion_x_tolerance:.4f})"
                    )
                stage = "最终预抓取位动态稳定验收"
                object_before_unfreeze = self._current_object_position(target_object)
                self._set_status(
                    "GUIDED_UNFREEZE",
                    "已在真实碰撞下到达最终预抓取位，正在确认稳定",
                    False,
                )
                settle_started = time.monotonic()
                unfreeze_max_linear_speed = float(
                    profile.get(
                        "unfreeze_max_linear_speed_m_s",
                        self.unfreeze_max_linear_speed,
                    )
                )
                unfreeze_max_angular_speed = float(
                    profile.get(
                        "unfreeze_max_angular_speed_rad_s",
                        self.unfreeze_max_angular_speed,
                    )
                )
                time.sleep(self.unfreeze_settle_duration)
                stable_samples = 0
                grasp_probe = {}
                measured_local = ()
                previous_probe_position = None
                previous_probe_orientation = None
                previous_probe_time = None
                while True:
                    self._command_scene_grasp("probe", target_object)
                    if not self._wait_scene_grasp("PROBED", target_object, 2.0):
                        raise RuntimeError("Isaac 未返回最终预抓取位解冻状态")
                    with self._lock:
                        grasp_probe = dict(self._scene_grasp_status)
                    object_after_unfreeze = self._current_object_position(
                        target_object
                    )
                    unfreeze_drift = math.sqrt(
                        sum(
                            (a - b) ** 2
                            for a, b in zip(
                                object_after_unfreeze, object_before_unfreeze
                            )
                        )
                    )
                    measured_local = tuple(
                        float(value)
                        for value in grasp_probe.get("local_position_m", [])
                    )
                    local_settle_shift = (
                        math.sqrt(
                            sum(
                                (after - before) ** 2
                                for after, before in zip(
                                    measured_local, final_frozen_local
                                )
                            )
                        )
                        if len(measured_local) == 3
                        else math.inf
                    )
                    margin = self.unfreeze_preclose_envelope_margin
                    inside_preclose_envelope = (
                        len(measured_local) == 3
                        and float(profile_envelope["x_min_m"]) - margin
                        <= measured_local[0]
                        <= float(profile_envelope["x_max_m"]) + margin
                        and abs(measured_local[1])
                        <= float(profile_envelope["y_abs_max_m"]) + margin
                        and float(profile_envelope["z_min_m"]) - margin
                        <= measured_local[2]
                        <= float(profile_envelope["z_max_m"]) + margin
                    )
                    settle_contact_present = target_object in (
                        grasp_probe.get("left_finger_contact_object"),
                        grasp_probe.get("right_finger_contact_object"),
                    )
                    inside_preclose_envelope = (
                        inside_preclose_envelope
                        or _inside_preclose_corridor(
                            measured_local,
                            profile,
                            profile_envelope,
                            settle_contact_present,
                        )
                    )
                    if not inside_preclose_envelope:
                        raise RuntimeError(
                            "目标恢复动态物理后离开闭爪前安全区域："
                            f"before={final_frozen_local}, after={measured_local}, "
                            f"local_shift={local_settle_shift:.4f} m"
                        )
                    if not bool(grasp_probe.get("physics_enabled", False)):
                        raise RuntimeError("闭爪前目标不是动态刚体")
                    velocity_available = bool(
                        grasp_probe.get("velocity_available", False)
                    )
                    linear_speed = float(
                        grasp_probe.get("linear_speed_m_s", math.inf)
                    )
                    angular_speed = float(
                        grasp_probe.get("angular_speed_rad_s", math.inf)
                    )
                    probe_position = tuple(
                        float(value)
                        for value in grasp_probe.get("object_position_m", [])
                    )
                    probe_orientation = tuple(
                        float(value)
                        for value in grasp_probe.get(
                            "object_orientation_wxyz", []
                        )
                    )
                    probe_time = time.monotonic()
                    pose_velocity_available = (
                        previous_probe_time is not None
                        and len(probe_position) == 3
                        and len(probe_orientation) == 4
                        and previous_probe_position is not None
                        and previous_probe_orientation is not None
                    )
                    if pose_velocity_available:
                        sample_period = max(
                            1.0e-6, probe_time - previous_probe_time
                        )
                        pose_linear_speed = math.sqrt(
                            sum(
                                (after - before) ** 2
                                for after, before in zip(
                                    probe_position, previous_probe_position
                                )
                            )
                        ) / sample_period
                        pose_angular_speed = _quaternion_angular_distance(
                            probe_orientation, previous_probe_orientation
                        ) / sample_period
                    else:
                        pose_linear_speed = math.inf
                        pose_angular_speed = math.inf
                    previous_probe_position = (
                        probe_position if len(probe_position) == 3 else None
                    )
                    previous_probe_orientation = (
                        probe_orientation if len(probe_orientation) == 4 else None
                    )
                    previous_probe_time = probe_time
                    stable_now = (
                        pose_velocity_available
                        and unfreeze_drift
                        <= self.unfreeze_max_position_drift
                        and pose_linear_speed <= unfreeze_max_linear_speed
                        and pose_angular_speed <= unfreeze_max_angular_speed
                    )
                    stable_samples = stable_samples + 1 if stable_now else 0
                    self.get_logger().info(
                        "Unfreeze settle probe: "
                        f"before_local={final_frozen_local}, "
                        f"after_local={measured_local}, "
                        f"local_shift={local_settle_shift:.4f} m, "
                        f"drift={unfreeze_drift:.4f} m, "
                        f"reported_linear={linear_speed:.4f} m/s, "
                        f"reported_angular={angular_speed:.4f} rad/s, "
                        f"pose_linear={pose_linear_speed:.4f} m/s, "
                        f"pose_angular={pose_angular_speed:.4f} rad/s, "
                        f"strict_envelope={inside_grasp_envelope(measured_local, profile_envelope)}, "
                        f"stable_samples={stable_samples}/"
                        f"{self.unfreeze_stable_samples_required}"
                    )
                    if (
                        stable_samples
                        >= self.unfreeze_stable_samples_required
                    ):
                        break
                    if (
                        time.monotonic() - settle_started
                        >= self.unfreeze_settle_timeout
                    ):
                        raise RuntimeError(
                            "目标恢复动态物理后在限定时间内未稳定："
                            f"drift={unfreeze_drift:.4f} m, "
                            f"reported_linear={linear_speed:.4f} m/s, "
                            f"reported_angular={angular_speed:.4f} rad/s, "
                            f"pose_linear={pose_linear_speed:.4f} m/s, "
                            f"pose_angular={pose_angular_speed:.4f} rad/s, "
                            f"stable_samples={stable_samples}/"
                            f"{self.unfreeze_stable_samples_required}"
                        )
                    time.sleep(self.unfreeze_poll_period)
                require_clear_fingers = bool(
                    GRASP_CONFIG["gripper"].get(
                        "require_clear_fingers_before_close", True
                    )
                )
                if (
                    require_clear_fingers
                    and not bool(
                        profile.get(
                            "allow_preclose_target_contact", False
                        )
                    )
                    and bool(
                        grasp_probe.get("contact_report_available", False)
                    )
                    and target_object
                    in (
                        grasp_probe.get("left_finger_contact_object"),
                        grasp_probe.get("right_finger_contact_object"),
                    )
                ):
                    raise RuntimeError(
                        "恢复碰撞后、闭爪前已发生夹指接触；"
                        "已拒绝把提前碰撞误判为抓取："
                        f"local={measured_local}, "
                        "left="
                        f"{grasp_probe.get('left_finger_contact_object')}, "
                        "right="
                        f"{grasp_probe.get('right_finger_contact_object')}"
                    )
                preclose_clearance_lift = 0.0
                preclose_clearance_target = float(
                    profile.get("preclose_clearance_lift_m", 0.005)
                )
                while preclose_clearance_lift < preclose_clearance_target:
                    clearance_step = min(
                        0.005,
                        preclose_clearance_target - preclose_clearance_lift,
                    )
                    position, orientation = self._current_ee_pose()
                    clearance_position = (
                        position[0],
                        position[1],
                        position[2] + clearance_step,
                    )
                    self._set_status(
                        "GUIDED_PRECLOSE_CLEARANCE",
                        "闭爪前正在建立活动指旋转所需的 5 mm 离桌余量",
                        False,
                    )
                    self._execute_manual_pose(
                        clearance_position,
                        orientation,
                        "闭爪前解除夹指桌面接触",
                        linear=True,
                        max_joint_delta=0.45,
                        velocity_scale=self.grasp_alignment_velocity_scale,
                    )
                    preclose_clearance_lift += clearance_step
                    self._command_scene_grasp("probe", target_object)
                    if not self._wait_scene_grasp(
                        "PROBED", target_object, 2.0
                    ):
                        raise RuntimeError("Isaac 未返回闭爪前离桌状态")
                    with self._lock:
                        grasp_probe = dict(self._scene_grasp_status)
                    measured_local = tuple(
                        float(value)
                        for value in grasp_probe.get("local_position_m", [])
                    )
                    self.get_logger().info(
                        "Pre-close finger/table clearance: "
                        f"total={preclose_clearance_lift:.4f} m, "
                        f"local={measured_local}, "
                        "table_contact=("
                        f"{grasp_probe.get('left_finger_touching_table', False)}, "
                        f"{grasp_probe.get('right_finger_touching_table', False)})"
                    )
                    self._raise_if_finger_touching_table(
                        grasp_probe, "闭爪前离桌"
                    )
                    if not bool(grasp_probe.get("physics_enabled", False)):
                        raise RuntimeError("闭爪前离桌后目标不是动态刚体")
                    if not bool(grasp_probe.get("gravity_enabled", False)):
                        raise RuntimeError("闭爪前离桌后目标重力未启用")
                    if (
                        require_clear_fingers
                        and not bool(
                            profile.get(
                                "allow_preclose_target_contact", False
                            )
                        )
                        and bool(
                            grasp_probe.get(
                                "contact_report_available", False
                            )
                        )
                        and target_object
                        in (
                            grasp_probe.get("left_finger_contact_object"),
                            grasp_probe.get("right_finger_contact_object"),
                        )
                    ):
                        raise RuntimeError(
                            "闭爪前离桌动作已提前接触目标，拒绝继续纠偏："
                            f"local={measured_local}, "
                            "left="
                            f"{grasp_probe.get('left_finger_contact_object')}, "
                            "right="
                            f"{grasp_probe.get('right_finger_contact_object')}"
                        )
                    correction_object_origin = (
                        self._current_object_position(target_object)
                    )
                    correction_attempt = 0
                    while not inside_grasp_envelope(
                        measured_local, profile_envelope
                    ):
                        if len(measured_local) != 3:
                            raise RuntimeError(
                                "闭爪前离桌纠偏缺少三维局部位姿："
                                f"local={measured_local}"
                            )
                        margin = self.unfreeze_preclose_envelope_margin
                        inside_correction_guard = (
                            float(profile_envelope["x_min_m"]) - margin
                            <= measured_local[0]
                            <= float(profile_envelope["x_max_m"]) + margin
                            and abs(measured_local[1])
                            <= float(profile_envelope["y_abs_max_m"]) + margin
                            and float(profile_envelope["z_min_m"]) - margin
                            <= measured_local[2]
                            <= float(profile_envelope["z_max_m"]) + margin
                        )
                        if not inside_correction_guard:
                            raise RuntimeError(
                                "闭爪前离桌后目标明显离开可纠偏安全区："
                                f"local={measured_local}, margin={margin:.4f} m"
                            )
                        if (
                            correction_attempt
                            >= self.preclose_clearance_correction_max_attempts
                        ):
                            raise RuntimeError(
                                "闭爪前离桌闭环纠偏未收敛："
                                f"attempts={correction_attempt}, "
                                f"local={measured_local}"
                            )
                        correction_attempt += 1
                        local_correction = _preclose_envelope_correction(
                            measured_local,
                            profile_envelope,
                            self.preclose_clearance_correction_target_inset,
                            self.preclose_clearance_correction_step,
                        )
                        stator_orientation = (
                            self._current_stator_orientation()
                        )
                        world_correction = rotate_vector(
                            stator_orientation, local_correction
                        )
                        position, orientation = self._current_ee_pose()
                        correction_position = tuple(
                            position[index] + world_correction[index]
                            for index in range(3)
                        )
                        self._set_status(
                            "GUIDED_PRECLOSE_RECENTER",
                            "离桌后出现毫米级偏差，正在闭环恢复夹取中心",
                            False,
                        )
                        self._execute_manual_pose(
                            correction_position,
                            orientation,
                            f"闭爪前离桌局部纠偏 "
                            f"{correction_attempt}/"
                            f"{self.preclose_clearance_correction_max_attempts}",
                            linear=True,
                            max_joint_delta=0.45,
                            velocity_scale=self.grasp_alignment_velocity_scale,
                        )
                        self._command_scene_grasp("probe", target_object)
                        if not self._wait_scene_grasp(
                            "PROBED", target_object, 2.0
                        ):
                            raise RuntimeError(
                                "Isaac 未返回闭爪前离桌纠偏后的状态"
                            )
                        with self._lock:
                            grasp_probe = dict(self._scene_grasp_status)
                        measured_local = tuple(
                            float(value)
                            for value in grasp_probe.get(
                                "local_position_m", []
                            )
                        )
                        self._raise_if_finger_touching_table(
                            grasp_probe, "闭爪前离桌纠偏"
                        )
                        if not bool(
                            grasp_probe.get("physics_enabled", False)
                        ):
                            raise RuntimeError(
                                "闭爪前离桌纠偏后目标不是动态刚体"
                            )
                        if not bool(
                            grasp_probe.get("gravity_enabled", False)
                        ):
                            raise RuntimeError(
                                "闭爪前离桌纠偏后目标重力未启用"
                            )
                        left_contact = grasp_probe.get(
                            "left_finger_contact_object"
                        )
                        right_contact = grasp_probe.get(
                            "right_finger_contact_object"
                        )
                        if (
                            require_clear_fingers
                            and not bool(
                                profile.get(
                                    "allow_preclose_target_contact", False
                                )
                            )
                            and bool(
                                grasp_probe.get(
                                    "contact_report_available", False
                                )
                            )
                            and target_object
                            in (left_contact, right_contact)
                        ):
                            raise RuntimeError(
                                "闭爪前离桌纠偏已提前接触目标："
                                f"attempt={correction_attempt}, "
                                f"local={measured_local}, "
                                f"left={left_contact}, right={right_contact}"
                            )
                        correction_object_position = (
                            self._current_object_position(target_object)
                        )
                        correction_object_drift = math.sqrt(
                            sum(
                                (current - origin) ** 2
                                for current, origin in zip(
                                    correction_object_position,
                                    correction_object_origin,
                                )
                            )
                        )
                        if (
                            correction_object_drift
                            > self.unfreeze_max_position_drift
                        ):
                            raise RuntimeError(
                                "闭爪前离桌纠偏推动了目标，已停止："
                                f"attempt={correction_attempt}, "
                                f"drift={correction_object_drift:.4f} m, "
                                f"local={measured_local}"
                            )
                        self.get_logger().info(
                            "Pre-close clearance closed-loop correction: "
                            f"attempt={correction_attempt}/"
                            f"{self.preclose_clearance_correction_max_attempts}, "
                            f"local_step={local_correction}, "
                            f"local_after={measured_local}, "
                            f"object_drift={correction_object_drift:.4f} m, "
                            f"contacts=({left_contact}, {right_contact})"
                        )
                # Enter above the fixed finger's inner support face, then move
                # the still-open gripper around the fruit to the bilateral
                # contact height.  The collision-free insertion has already put
                # the apple inside the open jaws; all motion after unfreeze is
                # guarded so premature finger contact cannot count as a grasp.
                # Place the apple centre above the finger midline so first
                # contact is on its lower half and the closing force seats it
                # upward between the real meshes instead of rolling it down
                # toward the tabletop.
                preclose_contact_center_z = float(
                    profile.get("preclose_contact_center_z_m", 0.024)
                )
                if (
                    measured_local[2] > preclose_contact_center_z + 0.002
                    and not bool(
                        profile.get(
                            "disable_preclose_bilateral_height_correction",
                            False,
                        )
                    )
                ):
                    position, orientation = self._current_ee_pose()
                    local_z_correction = min(
                        measured_local[2] - preclose_contact_center_z,
                        float(
                            profile.get(
                                "preclose_z_correction_max_m", 0.010
                            )
                        ),
                    )
                    world_correction = rotate_vector(
                        orientation, (0.0, 0.0, local_z_correction)
                    )
                    centered_position = tuple(
                        position[index] + world_correction[index]
                        for index in range(3)
                    )
                    self._execute_manual_pose(
                        centered_position,
                        orientation,
                        "动态插入后移动开爪到双指接触高度",
                        linear=True,
                        max_joint_delta=0.45,
                        velocity_scale=self.grasp_alignment_velocity_scale,
                    )
                    self._command_scene_grasp("probe", target_object)
                    if not self._wait_scene_grasp(
                        "PROBED", target_object, 2.0
                    ):
                        raise RuntimeError("Isaac 未返回闭爪前接触高度精调状态")
                    with self._lock:
                        grasp_probe = dict(self._scene_grasp_status)
                    measured_local = tuple(
                        float(value)
                        for value in grasp_probe.get("local_position_m", [])
                    )
                    self.get_logger().info(
                        "Pre-close bilateral-height correction: "
                        f"local={measured_local}, "
                        "table_contact=("
                        f"{grasp_probe.get('left_finger_touching_table', False)}, "
                        f"{grasp_probe.get('right_finger_touching_table', False)})"
                    )
                if (
                    bool(grasp_probe.get("contact_report_available", False))
                    and not bool(
                        profile.get(
                            "disable_preclose_live_centering", False
                        )
                    )
                    and grasp_probe.get("left_finger_contact_object")
                    != target_object
                ):
                    try:
                        stator_transform = self._tf_buffer.lookup_transform(
                            self.base_frame, "gripperStator", Time()
                        ).transform
                    except tf2_ros.TransformException as exc:
                        raise RuntimeError(
                            "无法读取固定指坐标系完成闭爪前接触搜索："
                            f"{exc}"
                        ) from exc
                    stator_orientation = (
                        float(stator_transform.rotation.x),
                        float(stator_transform.rotation.y),
                        float(stator_transform.rotation.z),
                        float(stator_transform.rotation.w),
                    )
                    # Center the open gripper around the live fruit pose and move
                    # the fruit into the actual finger-mesh depth.  The
                    # former fixed-direction contact sweep accumulated lateral
                    # error when the fruit slid on the table and could hand a
                    # one-sided configuration to the close controller.
                    for center_iteration in range(2):
                        measured_local = tuple(
                            float(value)
                            for value in grasp_probe.get(
                                "local_position_m", []
                            )
                        )
                        if (
                            len(measured_local) != 3
                            or (
                                abs(measured_local[1]) <= 0.003
                                and measured_local[0]
                                >= desired_center_x - 0.003
                            )
                        ):
                            break
                        axial_distance = max(
                            -0.020,
                            min(
                                0.020,
                                measured_local[0] - desired_center_x,
                            ),
                        )
                        center_distance = max(
                            -0.020, min(0.020, measured_local[1])
                        )
                        center_origin, center_orientation = (
                            self._current_ee_pose()
                        )
                        center_world = rotate_vector(
                            stator_orientation,
                            (axial_distance, center_distance, 0.0),
                        )
                        center_position = tuple(
                            center_origin[index] + center_world[index]
                            for index in range(3)
                        )
                        self._execute_manual_pose(
                            center_position,
                            center_orientation,
                            "张爪状态按目标实时位姿横向居中",
                            linear=True,
                            max_joint_delta=0.45,
                            velocity_scale=self.grasp_alignment_velocity_scale,
                        )
                        self._command_scene_grasp("probe", target_object)
                        if not self._wait_scene_grasp(
                            "PROBED", target_object, 2.0
                        ):
                            raise RuntimeError(
                                "Isaac 未返回闭爪前横向居中状态"
                            )
                        with self._lock:
                            grasp_probe = dict(self._scene_grasp_status)
                        self._raise_if_finger_touching_table(
                            grasp_probe, "张爪状态横向居中时"
                        )
                        left_object = grasp_probe.get(
                            "left_finger_contact_object"
                        )
                        right_object = grasp_probe.get(
                            "right_finger_contact_object"
                        )
                        if any(
                            value not in (None, target_object)
                            for value in (left_object, right_object)
                        ):
                            raise RuntimeError(
                                "闭爪前横向居中碰到非目标物体："
                                f"left={left_object}, right={right_object}"
                            )
                        measured_local = tuple(
                            float(value)
                            for value in grasp_probe.get(
                                "local_position_m", []
                            )
                        )
                        self.get_logger().info(
                            "Pre-close live centering: "
                            f"iteration={center_iteration}, "
                            f"dx={axial_distance:.4f} m, "
                            f"dy={center_distance:.4f} m, "
                            f"left={left_object}, right={right_object}, "
                            f"local={measured_local}"
                        )
                        if (
                            len(measured_local) != 3
                            or measured_local[0]
                            < float(profile_envelope["x_min_m"]) - 0.002
                            or measured_local[0]
                            > float(profile_envelope["x_max_m"]) + 0.002
                            or measured_local[2]
                            < float(profile_envelope["z_min_m"]) - 0.002
                            or measured_local[2]
                            > float(profile_envelope["z_max_m"]) + 0.002
                        ):
                            raise RuntimeError(
                                "闭爪前横向居中使目标离开安全区域："
                                f"local={measured_local}"
                            )
                self._raise_if_finger_touching_table(
                    grasp_probe, "闭爪前离桌检查"
                )
                stage = "未附着固定指就位"
                seating_translation = 0.0
                contact_reports_available = bool(
                    grasp_probe.get("contact_report_available", False)
                )
                # Contact-report mode uses a calibrated frozen insertion followed
                # by dynamic settling.  Do not add a second tool-frame seating
                # motion here: link06 tool X is not the stator collision-mesh X
                # axis and the extra motion produces lateral fork drift.
                if contact_reports_available:
                    self.get_logger().info(
                        "Using calibrated insertion; no secondary seating motion"
                    )
                while (
                    not contact_reports_available
                    and measured_local[2]
                    > self.gripper_stator_support_z_max
                    and seating_translation
                    < self.gripper_preclose_seating_max_translation
                ):
                    seating_step = min(
                        self.gripper_preclose_seating_step,
                        measured_local[2]
                        - self.gripper_preclose_seating_z_target,
                        self.gripper_preclose_seating_max_translation
                        - seating_translation,
                    )
                    if seating_step <= 0.0005:
                        break
                    position, orientation = self._current_ee_pose()
                    seating_world = rotate_vector(
                        orientation, (0.0, 0.0, seating_step)
                    )
                    seating_position = tuple(
                        position[index] + seating_world[index]
                        for index in range(3)
                    )
                    self._set_status(
                        "GUIDED_SEATING",
                        "目标保持未附着动态状态，正在慢速靠近固定指",
                        False,
                    )
                    self._execute_manual_pose(
                        seating_position,
                        orientation,
                        "未附着固定指就位",
                        linear=True,
                        max_joint_delta=0.45,
                        velocity_scale=self.grasp_alignment_velocity_scale,
                    )
                    seating_translation += seating_step
                    time.sleep(0.2)
                    self._command_scene_grasp("probe", target_object)
                    if not self._wait_scene_grasp(
                        "PROBED", target_object, 2.0
                    ):
                        raise RuntimeError("Isaac 未返回固定指就位结果")
                    with self._lock:
                        seating_probe = dict(self._scene_grasp_status)
                    measured_local = tuple(
                        float(value)
                        for value in seating_probe.get("local_position_m", [])
                    )
                    self.get_logger().info(
                        "Physical stator seating probe: "
                        f"step={seating_step:.4f} m, "
                        f"total={seating_translation:.4f} m, "
                        f"local={measured_local}"
                    )
                    if (
                        len(measured_local) != 3
                        or not bool(
                            seating_probe.get("physics_enabled", False)
                        )
                        or measured_local[0]
                        < float(profile_envelope["x_min_m"]) - 0.002
                        or measured_local[0]
                        > float(profile_envelope["x_max_m"]) + 0.002
                        or abs(measured_local[1])
                        > float(profile_envelope["y_abs_max_m"]) + 0.002
                        or measured_local[2]
                        > float(profile_envelope["z_max_m"]) + 0.002
                    ):
                        raise RuntimeError(
                            "固定指闭环就位时目标离开安全区域："
                            f"local={measured_local}, "
                            f"translation={seating_translation:.4f} m"
                        )
                if (
                    not contact_reports_available
                    and (
                    len(measured_local) != 3
                    or measured_local[2]
                    > self.gripper_stator_support_z_max
                    )
                ):
                    raise RuntimeError(
                        "固定指闭环就位未进入接触带："
                        f"local={measured_local}, "
                        f"translation={seating_translation:.4f} m"
                    )
                self._command_scene_grasp("probe", target_object)
                if not self._wait_scene_grasp("PROBED", target_object, 2.0):
                    raise RuntimeError("Isaac 未返回闭爪前最终接触状态")
                with self._lock:
                    preclose_probe = dict(self._scene_grasp_status)
                preclose_left = preclose_probe.get(
                    "left_finger_contact_object"
                )
                preclose_right = preclose_probe.get(
                    "right_finger_contact_object"
                )
                if (
                    require_clear_fingers
                    and not bool(
                        profile.get(
                            "allow_preclose_target_contact", False
                        )
                    )
                    and bool(
                        preclose_probe.get("contact_report_available", False)
                    )
                    and target_object in (preclose_left, preclose_right)
                ):
                    raise RuntimeError(
                        "发送闭爪命令前夹指已接触苹果，已停止本次示教："
                        f"left={preclose_left}, right={preclose_right}, "
                        f"local={preclose_probe.get('local_position_m')}"
                    )
                partial_gripper_target = profile.get(
                    "preclose_partial_gripper_rad"
                )
                unilateral_target_contact = (
                    (preclose_left == target_object)
                    != (preclose_right == target_object)
                )
                if (
                    partial_gripper_target is not None
                    and unilateral_target_contact
                ):
                    stage = "鼠标单侧触点预夹紧与轴向就位"
                    partial_angle = self._execute_gripper(
                        "close", float(partial_gripper_target)
                    )
                    self.get_logger().info(
                        "Mouse pre-clamp reached: "
                        f"jointGripper={partial_angle:.4f} rad, "
                        f"left={preclose_left}, right={preclose_right}"
                    )
                    seating_step_size = float(
                        profile.get(
                            "preclose_axial_seating_step_m", 0.001
                        )
                    )
                    seating_limit = float(
                        profile.get(
                            "preclose_axial_seating_max_translation_m", 0.0
                        )
                    )
                    seating_min_x = float(
                        profile.get(
                            "preclose_axial_seating_min_x_m",
                            desired_center_x,
                        )
                    )
                    seating_y_limit = float(
                        profile.get(
                            "preclose_axial_seating_y_limit_m",
                            insertion_alignment_tolerance_y,
                        )
                    )
                    seating_z_limit = float(
                        profile.get(
                            "preclose_axial_seating_z_limit_m",
                            insertion_alignment_tolerance_z,
                        )
                    )
                    seating_total = 0.0
                    previous_seating_x = math.inf
                    stagnant_seating_steps = 0
                    while seating_total + 1.0e-6 < seating_limit:
                        self._command_scene_grasp("probe", target_object)
                        if not self._wait_scene_grasp(
                            "PROBED", target_object, 2.0
                        ):
                            raise RuntimeError(
                                "Isaac 未返回鼠标预夹紧就位状态"
                            )
                        with self._lock:
                            preclose_probe = dict(
                                self._scene_grasp_status
                            )
                        self._raise_if_finger_touching_table(
                            preclose_probe,
                            "鼠标预夹紧轴向就位前",
                        )
                        preclose_left = preclose_probe.get(
                            "left_finger_contact_object"
                        )
                        preclose_right = preclose_probe.get(
                            "right_finger_contact_object"
                        )
                        if any(
                            value not in (None, target_object)
                            for value in (
                                preclose_left,
                                preclose_right,
                            )
                        ):
                            raise RuntimeError(
                                "鼠标预夹紧就位碰到非目标物体："
                                f"left={preclose_left}, "
                                f"right={preclose_right}"
                            )
                        measured_local = tuple(
                            float(value)
                            for value in preclose_probe.get(
                                "local_position_m", []
                            )
                        )
                        if (
                            len(measured_local) != 3
                            or abs(measured_local[1] - desired_center_y)
                            > seating_y_limit
                            or abs(measured_local[2] - desired_center_z)
                            > seating_z_limit
                        ):
                            raise RuntimeError(
                                "鼠标预夹紧就位时离开 Y/Z 安全带："
                                f"local={measured_local}"
                            )
                        bilateral_now = (
                            preclose_left == target_object
                            and preclose_right == target_object
                        )
                        if bilateral_now:
                            self.get_logger().info(
                                "Mouse pre-clamp formed bilateral contact "
                                f"after {seating_total:.4f} m seating"
                            )
                            break
                        remaining_seating = (
                            measured_local[0] - seating_min_x
                        )
                        if remaining_seating <= 0.0005:
                            break
                        if (
                            math.isfinite(previous_seating_x)
                            and previous_seating_x
                            - measured_local[0]
                            < 0.00015
                        ):
                            stagnant_seating_steps += 1
                        else:
                            stagnant_seating_steps = 0
                        previous_seating_x = measured_local[0]
                        if stagnant_seating_steps >= 2:
                            self.get_logger().warning(
                                "Mouse pre-clamp axial seating stopped after "
                                "two non-progressing steps"
                            )
                            break
                        step_distance = min(
                            seating_step_size,
                            remaining_seating,
                            seating_limit - seating_total,
                        )
                        position, orientation = self._current_ee_pose()
                        stator_orientation = (
                            self._current_stator_orientation()
                        )
                        seating_world = rotate_vector(
                            stator_orientation,
                            (step_distance, 0.0, 0.0),
                        )
                        seating_position = tuple(
                            position[index] + seating_world[index]
                            for index in range(3)
                        )
                        self._execute_manual_pose(
                            seating_position,
                            orientation,
                            "鼠标预夹紧后轴向小步就位",
                            linear=True,
                            max_joint_delta=0.35,
                            position_tolerance=float(
                                profile.get(
                                    "dynamic_alignment_moveit_position_tolerance_m",
                                    0.001,
                                )
                            ),
                            orientation_tolerance=float(
                                profile.get(
                                    "insertion_orientation_tolerance_rad",
                                    0.006,
                                )
                            ),
                            velocity_scale=self.grasp_alignment_velocity_scale,
                        )
                        seating_total += step_distance
                        self.get_logger().info(
                            "Mouse pre-clamp axial seating command: "
                            f"step={step_distance:.4f} m, "
                            f"total={seating_total:.4f} m, "
                            f"local_before={measured_local}"
                        )
                    grasp_probe = preclose_probe
            object_before_close = self._current_object_position(target_object)
            stage = "闭爪形成几何夹持"
            self._set_status(
                "GUIDED_CLOSE",
                "目标已恢复动态物理，正在低速闭爪并检测运动受阻",
                False,
            )
            self._set_grasp_control_state(
                gripper_commanded_closed=True,
                gripper_obstructed=False,
                physical_test_lift_active=False,
                physical_grasp_ok=False,
                release_started=False,
                release_sequence_complete=False,
            )
            contact_angle = self._execute_gripper_close_until_obstructed(
                float(profile["gripper_rad"]),
                target_object,
                require_bilateral=bool(
                    profile.get(
                        "require_bilateral_close_obstruction", False
                    )
                ),
            )
            self._set_grasp_control_state(gripper_obstructed=True)
            if not self._wait_task_flag("gripper_obstructed", True, 2.0):
                self.get_logger().warning(
                    "Evaluator did not echo gripper_obstructed before the "
                    "deadline; continuing to the direct PhysX contact checks."
                )
            contact_closed_limit = float(
                profile.get(
                    "gripper_contact_closed_limit_rad",
                    self.gripper_contact_closed_limit,
                )
            )
            contact_open_limit = float(
                profile.get(
                    "gripper_contact_open_limit_rad",
                    self.gripper_contact_open_limit,
                )
            )
            if not (
                contact_closed_limit
                <= contact_angle
                <= contact_open_limit
            ):
                raise RuntimeError(
                    "夹爪未停在有效夹持宽度："
                    f"jointGripper={contact_angle:.3f} rad，期望范围 "
                    f"[{contact_closed_limit:.3f}, "
                    f"{contact_open_limit:.3f}] rad；"
                    f"物体局部位置={grasp_probe.get('local_position_m')}"
                )
            # First hold the measured obstruction angle without preload and
            # inspect real contacts.  Applying preload before this probe can
            # turn a one-finger push or a finger/table collision into a
            # three-body jam.
            self._hold_gripper(contact_angle, persistent=True)
            self._command_scene_grasp("probe", target_object)
            if not self._wait_scene_grasp("PROBED", target_object, 2.0):
                raise RuntimeError("Isaac 未返回预载前接触状态")
            with self._lock:
                preload_probe = dict(self._scene_grasp_status)
            self._raise_if_finger_touching_table(
                preload_probe, "闭爪受阻后、施加预载前"
            )
            contact_reports_available = bool(
                preload_probe.get("contact_report_available", False)
            )
            if contact_reports_available:
                left_object = preload_probe.get(
                    "left_finger_contact_object"
                )
                right_object = preload_probe.get(
                    "right_finger_contact_object"
                )
                contact_objects = (left_object, right_object)
                if (
                    any(
                        value not in (None, target_object)
                        for value in contact_objects
                    )
                ):
                    raise RuntimeError(
                        "预载前没有形成可安全推进的目标接触，"
                        "已保持当前开度且拒绝继续加力："
                        f"left={left_object}, right={right_object}, "
                        "table_contact=("
                        f"{preload_probe.get('left_finger_touching_table', False)}, "
                        f"{preload_probe.get('right_finger_touching_table', False)})"
                    )
                if target_object not in contact_objects:
                    self.get_logger().info(
                        "Allowing bounded no-contact-to-bilateral closure "
                        "inside the verified grasp envelope"
                    )
                elif None in contact_objects:
                    self.get_logger().info(
                        "Allowing bounded unilateral-to-bilateral closure: "
                        f"left={left_object}, right={right_object}"
                    )

            # Increase the finite closing target in small steps.  Every step
            # is allowed only while both fingers still contact the target,
            # neither finger touches the table, and the object remains in the
            # pre-close safety envelope.
            contact_reference_angle = contact_angle
            preload_applied = 0.0
            unjam_lift_applied = False
            retention_enabled = False
            contact_hold_preload = float(
                profile.get(
                    "contact_hold_preload_rad",
                    self.gripper_contact_hold_preload,
                )
            )
            contact_hold_preload_step = max(
                0.001,
                min(
                    contact_hold_preload,
                    float(
                        profile.get(
                            "contact_hold_preload_step_rad",
                            self.gripper_contact_hold_preload_step,
                        )
                    ),
                ),
            )
            contact_hold_minimum_preload = max(
                0.0,
                min(
                    contact_hold_preload,
                    float(
                        profile.get(
                            "contact_hold_minimum_preload_rad",
                            contact_hold_preload,
                        )
                    ),
                ),
            )
            while preload_applied + 1.0e-6 < contact_hold_preload:
                preload_applied = min(
                    contact_hold_preload,
                    preload_applied
                    + contact_hold_preload_step,
                )
                hold_target = min(
                    self.gripper_closed,
                    contact_reference_angle + preload_applied,
                )
                self._hold_gripper(hold_target, persistent=True)
                with self._lock:
                    held_angle = self._joint_positions.get("jointGripper")
                if held_angle is not None:
                    contact_angle = float(held_angle)
                self._command_scene_grasp("probe", target_object)
                if not self._wait_scene_grasp(
                    "PROBED", target_object, 2.0
                ):
                    raise RuntimeError("Isaac 未返回分级夹紧状态")
                with self._lock:
                    preload_probe = dict(self._scene_grasp_status)
                self._raise_if_finger_touching_table(
                    preload_probe,
                    f"分级夹紧 {preload_applied:.3f} rad 时",
                )
                if contact_reports_available:
                    left_object = preload_probe.get(
                        "left_finger_contact_object"
                    )
                    right_object = preload_probe.get(
                        "right_finger_contact_object"
                    )
                    contact_objects = (left_object, right_object)
                    if (
                        any(
                            value not in (None, target_object)
                            for value in contact_objects
                        )
                    ):
                        raise RuntimeError(
                            "分级夹紧时接触到非目标物体，已停止继续关闭："
                            f"step={preload_applied:.3f} rad, "
                            f"left={left_object}, right={right_object}"
                        )
                    if target_object not in contact_objects:
                        self.get_logger().info(
                            "Actual-mesh unilateral-to-bilateral transition "
                            "temporarily has no contact; continuing bounded "
                            f"closure at step={preload_applied:.3f} rad"
                        )
                preload_local = tuple(
                    float(value)
                    for value in preload_probe.get(
                        "local_position_m", []
                    )
                )
                margin = self.unfreeze_preclose_envelope_margin
                inside_preload_envelope = (
                    len(preload_local) == 3
                    and float(profile_envelope["x_min_m"]) - margin
                    <= preload_local[0]
                    <= float(profile_envelope["x_max_m"]) + margin
                    and abs(preload_local[1])
                    <= float(profile_envelope["y_abs_max_m"]) + margin
                    and float(profile_envelope["z_min_m"]) - margin
                    <= preload_local[2]
                    <= float(profile_envelope["z_max_m"]) + margin
                )
                capture_envelope = profile.get(
                    "preclose_capture_envelope"
                )
                inside_preload_envelope = (
                    inside_preload_envelope
                    or (
                        isinstance(capture_envelope, dict)
                        and inside_grasp_envelope(
                            preload_local, capture_envelope
                        )
                    )
                )
                if not inside_preload_envelope:
                    raise RuntimeError(
                        "分级夹紧时目标离开闭爪安全区域，已停止继续关闭："
                        f"step={preload_applied:.3f} rad, "
                        f"local={preload_local}"
                    )

                left_impulse = float(
                    preload_probe.get(
                        "left_finger_contact_impulse_ns", 0.0
                    )
                )
                right_impulse = float(
                    preload_probe.get(
                        "right_finger_contact_impulse_ns", 0.0
                    )
                )
                if (
                    target_object == "apple"
                    and not profile.get("single_position_grasp", False)
                    and not unjam_lift_applied
                    and self.gripper_contact_unjam_lift > 0.0
                    and left_object == target_object
                    and right_object == target_object
                ):
                    # Break the apple/finger/table three-body jam while the
                    # bounded close target remains active.  World +Z is used
                    # deliberately: a previous stator-local seating motion
                    # drove the fixed finger into the tabletop.
                    position, orientation = self._current_ee_pose()
                    unjam_position = (
                        position[0],
                        position[1],
                        position[2] + self.gripper_contact_unjam_lift,
                    )
                    self._set_status(
                        "GUIDED_UNJAM_LIFT",
                        "保持有限闭爪力，正在小幅上抬解除桌面卡滞",
                        False,
                    )
                    self._execute_manual_pose(
                        unjam_position,
                        orientation,
                        "闭爪接触后解除桌面卡滞",
                        linear=True,
                        max_joint_delta=0.45,
                        velocity_scale=self.grasp_alignment_velocity_scale,
                    )
                    unjam_lift_applied = True
                    time.sleep(0.2)
                    self._command_scene_grasp("probe", target_object)
                    if not self._wait_scene_grasp(
                        "PROBED", target_object, 2.0
                    ):
                        raise RuntimeError("Isaac 未返回上抬解卡状态")
                    with self._lock:
                        preload_probe = dict(self._scene_grasp_status)
                    self._raise_if_finger_touching_table(
                        preload_probe, "闭爪接触后上抬解卡"
                    )
                    left_object = preload_probe.get(
                        "left_finger_contact_object"
                    )
                    right_object = preload_probe.get(
                        "right_finger_contact_object"
                    )
                    preload_local = tuple(
                        float(value)
                        for value in preload_probe.get(
                            "local_position_m", []
                        )
                    )
                    inside_preload_envelope = (
                        len(preload_local) == 3
                        and float(profile_envelope["x_min_m"]) - margin
                        <= preload_local[0]
                        <= float(profile_envelope["x_max_m"]) + margin
                        and abs(preload_local[1])
                        <= float(profile_envelope["y_abs_max_m"]) + margin
                        and float(profile_envelope["z_min_m"]) - margin
                        <= preload_local[2]
                        <= float(profile_envelope["z_max_m"]) + margin
                    )
                    if (
                        left_object != target_object
                        or right_object != target_object
                        or not inside_preload_envelope
                    ):
                        raise RuntimeError(
                            "上抬解卡时丢失安全双指夹持："
                            f"left={left_object}, right={right_object}, "
                            f"local={preload_local}"
                        )
                    left_impulse = float(
                        preload_probe.get(
                            "left_finger_contact_impulse_ns", 0.0
                        )
                    )
                    right_impulse = float(
                        preload_probe.get(
                            "right_finger_contact_impulse_ns", 0.0
                        )
                    )
                    self.get_logger().info(
                        "Applied world-Z contact unjam lift: "
                        f"dz={self.gripper_contact_unjam_lift:.4f} m, "
                        f"impulse_ns=({left_impulse:.6f}, "
                        f"{right_impulse:.6f}), local={preload_local}"
                    )
                    if self.gripper_contact_stator_seating > 0.0:
                        position, orientation = self._current_ee_pose()
                        seating_world = rotate_vector(
                            orientation,
                            (
                                0.0,
                                0.0,
                                self.gripper_contact_stator_seating,
                            ),
                        )
                        seating_position = tuple(
                            position[index] + seating_world[index]
                            for index in range(3)
                        )
                        self._execute_manual_pose(
                            seating_position,
                            orientation,
                            "水平侧夹固定指受力就位",
                            linear=True,
                            max_joint_delta=0.45,
                            velocity_scale=(
                                self.grasp_alignment_velocity_scale
                            ),
                        )
                        time.sleep(0.2)
                        self._command_scene_grasp("probe", target_object)
                        if not self._wait_scene_grasp(
                            "PROBED", target_object, 2.0
                        ):
                            raise RuntimeError(
                                "Isaac 未返回固定指受力就位状态"
                            )
                        with self._lock:
                            preload_probe = dict(
                                self._scene_grasp_status
                            )
                        self._raise_if_finger_touching_table(
                            preload_probe, "水平侧夹固定指受力就位"
                        )
                        left_object = preload_probe.get(
                            "left_finger_contact_object"
                        )
                        right_object = preload_probe.get(
                            "right_finger_contact_object"
                        )
                        preload_local = tuple(
                            float(value)
                            for value in preload_probe.get(
                                "local_position_m", []
                            )
                        )
                        inside_preload_envelope = (
                            len(preload_local) == 3
                            and float(profile_envelope["x_min_m"]) - margin
                            <= preload_local[0]
                            <= float(profile_envelope["x_max_m"]) + margin
                            and abs(preload_local[1])
                            <= float(profile_envelope["y_abs_max_m"]) + margin
                            and float(profile_envelope["z_min_m"]) - margin
                            <= preload_local[2]
                            <= float(profile_envelope["z_max_m"]) + margin
                        )
                        if (
                            left_object != target_object
                            or right_object != target_object
                            or not inside_preload_envelope
                        ):
                            raise RuntimeError(
                                "固定指受力就位时夹持几何失效："
                                f"left={left_object}, "
                                f"right={right_object}, "
                                f"local={preload_local}"
                            )
                        left_impulse = float(
                            preload_probe.get(
                                "left_finger_contact_impulse_ns", 0.0
                            )
                        )
                        right_impulse = float(
                            preload_probe.get(
                                "right_finger_contact_impulse_ns", 0.0
                            )
                        )
                        self.get_logger().info(
                            "Applied horizontal stator seating: "
                            f"distance="
                            f"{self.gripper_contact_stator_seating:.4f} m, "
                            f"impulse_ns=({left_impulse:.6f}, "
                            f"{right_impulse:.6f}), "
                            f"local={preload_local}"
                        )
                self.get_logger().info(
                    "Applied adaptive contact preload step: "
                    f"step={preload_applied:.4f} rad, "
                    f"target={hold_target:.4f} rad, "
                    f"actual={contact_angle:.4f} rad, "
                    f"impulse_ns=({left_impulse:.6f}, "
                    f"{right_impulse:.6f}), local={preload_local}"
                )
                required_contact_frames = int(
                    preload_probe.get(
                        "bilateral_contact_required_frames", 10
                    )
                )
                sustained_mesh_contact = (
                    left_object == target_object
                    and right_object == target_object
                    and "/collisions" in str(
                        preload_probe.get(
                            "left_finger_contact_collider", ""
                        )
                    )
                    and "/collisions" in str(
                        preload_probe.get(
                            "right_finger_contact_collider", ""
                        )
                    )
                    and int(
                        preload_probe.get(
                            "bilateral_contact_consecutive_frames", 0
                        )
                    )
                    >= required_contact_frames
                )
                if (
                    sustained_mesh_contact
                    and preload_applied + 1.0e-6
                    >= contact_hold_minimum_preload
                ):
                    self.get_logger().info(
                        "Completed finite preload with sustained "
                        "actual-mesh bilateral contact: "
                        f"step={preload_applied:.4f} rad, "
                        "colliders=("
                        f"{preload_probe.get('left_finger_contact_collider')}, "
                        f"{preload_probe.get('right_finger_contact_collider')})"
                    )
                    break
            self._command_scene_grasp("probe", target_object)
            if not self._wait_scene_grasp("PROBED", target_object, 2.0):
                raise RuntimeError("Isaac 未返回闭爪后几何位置")
            with self._lock:
                closed_probe = dict(self._scene_grasp_status)
            self._raise_if_finger_touching_table(
                closed_probe, "有限预载后"
            )
            closed_local = tuple(
                float(value)
                for value in closed_probe.get("local_position_m", [])
            )
            if bool(closed_probe.get("contact_report_available", False)):
                left_object = closed_probe.get(
                    "left_finger_contact_object"
                )
                right_object = closed_probe.get(
                    "right_finger_contact_object"
                )
                if (
                    not profile.get("single_position_grasp", False)
                    and len(closed_local) == 3
                    and (left_object == target_object)
                    != (right_object == target_object)
                ):
                    position, orientation = self._current_ee_pose()
                    try:
                        stator_transform = self._tf_buffer.lookup_transform(
                            self.base_frame, "gripperStator", Time()
                        ).transform
                    except tf2_ros.TransformException as exc:
                        raise RuntimeError(
                            "无法读取固定指坐标系完成夹紧横向对中："
                            f"{exc}"
                        ) from exc
                    stator_orientation = (
                        float(stator_transform.rotation.x),
                        float(stator_transform.rotation.y),
                        float(stator_transform.rotation.z),
                        float(stator_transform.rotation.w),
                    )
                    # Preserve palm-depth margin before the lateral search.
                    # With the visual-sized fruit, sideways motion can roll
                    # the centre several millimetres toward the palm.  A
                    # bounded axial retreat toward local x=0.127 keeps the
                    # strict x_min gate meaningful while leaving both bodies
                    # fully dynamic.  The 12 mm cap comes from the measured
                    # depth consumed by the closed-mesh lateral roll.
                    if closed_local[0] < 0.127:
                        position, orientation = self._current_ee_pose()
                        axial_retreat = max(
                            -0.012, closed_local[0] - 0.127
                        )
                        retreat_world = rotate_vector(
                            stator_orientation,
                            (axial_retreat, 0.0, 0.0),
                        )
                        retreated_position = tuple(
                            position[index] + retreat_world[index]
                            for index in range(3)
                        )
                        self._execute_manual_pose(
                            retreated_position,
                            orientation,
                            "横向接触搜索前保留掌根深度余量",
                            linear=True,
                            max_joint_delta=0.45,
                            velocity_scale=(
                                self.grasp_alignment_velocity_scale
                            ),
                        )
                        time.sleep(0.1)
                        self._command_scene_grasp(
                            "probe", target_object
                        )
                        if not self._wait_scene_grasp(
                            "PROBED", target_object, 2.0
                        ):
                            raise RuntimeError(
                                "Isaac 未返回横向搜索前深度状态"
                            )
                        with self._lock:
                            closed_probe = dict(
                                self._scene_grasp_status
                            )
                        self._raise_if_finger_touching_table(
                            closed_probe, "横向搜索前轴向退让时"
                        )
                        closed_local = tuple(
                            float(value)
                            for value in closed_probe.get(
                                "local_position_m", []
                            )
                        )
                        self.get_logger().info(
                            "Applied pre-lateral axial retreat: "
                            f"dx={axial_retreat:.4f} m, "
                            f"local={closed_local}, "
                            "left="
                            f"{closed_probe.get('left_finger_contact_object')}, "
                            "right="
                            f"{closed_probe.get('right_finger_contact_object')}"
                        )
                    # A one-shot 6 mm correction was calibrated against the
                    # oversized fruit proxy and can stop short of the real
                    # visible surface.  Search with the closed physical meshes
                    # in 2 mm increments, probing after each move and stopping
                    # as soon as both actual finger colliders report contact.
                    # Moving the gripper in +stator-Y makes the measured
                    # object-local Y more negative.  Therefore the gripper
                    # must move with the measured sign to drive that value
                    # toward zero.
                    lateral_direction = _local_axis_centering_step(
                        closed_local[1], 1.0
                    )
                    lateral_total = 0.0
                    for lateral_step_index in range(6):
                        if target_bilateral_contact(
                            closed_probe, target_object
                        ):
                            break
                        position, orientation = self._current_ee_pose()
                        lateral_step = lateral_direction * 0.002
                        correction_world = rotate_vector(
                            stator_orientation,
                            (0.0, lateral_step, 0.0),
                        )
                        centered_position = tuple(
                            position[index] + correction_world[index]
                            for index in range(3)
                        )
                        self._execute_manual_pose(
                            centered_position,
                            orientation,
                            "保持夹紧力逐步搜索固定指真实接触",
                            linear=True,
                            max_joint_delta=0.45,
                            velocity_scale=(
                                self.grasp_alignment_velocity_scale
                            ),
                        )
                        lateral_total += lateral_step
                        time.sleep(0.1)
                        self._command_scene_grasp(
                            "probe", target_object
                        )
                        if not self._wait_scene_grasp(
                            "PROBED", target_object, 2.0
                        ):
                            raise RuntimeError(
                                "Isaac 未返回夹紧横向接触搜索状态"
                            )
                        with self._lock:
                            closed_probe = dict(
                                self._scene_grasp_status
                            )
                        self._raise_if_finger_touching_table(
                            closed_probe, "保持夹紧力横向接触搜索时"
                        )
                        closed_local = tuple(
                            float(value)
                            for value in closed_probe.get(
                                "local_position_m", []
                            )
                        )
                        step_left = closed_probe.get(
                            "left_finger_contact_object"
                        )
                        step_right = closed_probe.get(
                            "right_finger_contact_object"
                        )
                        if (
                            (step_left not in (None, target_object))
                            or (step_right not in (None, target_object))
                            or len(closed_local) != 3
                            or not inside_grasp_envelope(
                                closed_local, profile_envelope
                            )
                        ):
                            raise RuntimeError(
                                "固定指接触搜索离开安全夹持包络："
                                f"step={lateral_step_index + 1}, "
                                f"local={closed_local}, "
                                f"left={step_left}, right={step_right}"
                            )
                        self.get_logger().info(
                            "Closed-mesh stator contact search: "
                            f"step={lateral_step_index + 1}, "
                            f"dy_total={lateral_total:.4f} m, "
                            f"left={step_left}, right={step_right}, "
                            f"local={closed_local}"
                        )
                    left_object = closed_probe.get(
                        "left_finger_contact_object"
                    )
                    right_object = closed_probe.get(
                        "right_finger_contact_object"
                    )
                    if (
                        len(closed_local) == 3
                        and closed_local[2] < 0.016
                        and (left_object == target_object)
                        != (right_object == target_object)
                    ):
                        position, orientation = self._current_ee_pose()
                        vertical_correction = max(
                            -0.004,
                            min(0.004, closed_local[2] - 0.017),
                        )
                        correction_world = rotate_vector(
                            stator_orientation,
                            (0.0, 0.0, vertical_correction),
                        )
                        centered_position = tuple(
                            position[index] + correction_world[index]
                            for index in range(3)
                        )
                        self._execute_manual_pose(
                            centered_position,
                            orientation,
                            "保持夹紧力移动到固定指受力高度",
                            linear=True,
                            max_joint_delta=0.45,
                            velocity_scale=(
                                self.grasp_alignment_velocity_scale
                            ),
                        )
                        time.sleep(0.2)
                        self._command_scene_grasp(
                            "probe", target_object
                        )
                        if not self._wait_scene_grasp(
                            "PROBED", target_object, 2.0
                        ):
                            raise RuntimeError(
                                "Isaac 未返回固定指受力高度状态"
                            )
                        with self._lock:
                            closed_probe = dict(
                                self._scene_grasp_status
                            )
                        self._raise_if_finger_touching_table(
                            closed_probe, "固定指受力高度精调时"
                        )
                        closed_local = tuple(
                            float(value)
                            for value in closed_probe.get(
                                "local_position_m", []
                            )
                        )
                        self.get_logger().info(
                            "Applied unilateral-contact stator-height "
                            f"correction: dz={vertical_correction:.4f} m, "
                            f"left={closed_probe.get('left_finger_contact_object')}, "
                            f"right={closed_probe.get('right_finger_contact_object')}, "
                            f"local={closed_local}"
                        )
                    left_object = closed_probe.get(
                        "left_finger_contact_object"
                    )
                    right_object = closed_probe.get(
                        "right_finger_contact_object"
                    )
                    if (
                        len(closed_local) == 3
                        and closed_local[0] < 0.140
                        and (left_object == target_object)
                        != (right_object == target_object)
                    ):
                        position, orientation = self._current_ee_pose()
                        axial_correction = max(
                            -0.004,
                            min(0.004, closed_local[0] - 0.141),
                        )
                        correction_world = rotate_vector(
                            stator_orientation,
                            (axial_correction, 0.0, 0.0),
                        )
                        centered_position = tuple(
                            position[index] + correction_world[index]
                            for index in range(3)
                        )
                        self._execute_manual_pose(
                            centered_position,
                            orientation,
                            "保持夹紧力移动到双指轴向接触深度",
                            linear=True,
                            max_joint_delta=0.45,
                            velocity_scale=(
                                self.grasp_alignment_velocity_scale
                            ),
                        )
                        time.sleep(0.2)
                        self._command_scene_grasp(
                            "probe", target_object
                        )
                        if not self._wait_scene_grasp(
                            "PROBED", target_object, 2.0
                        ):
                            raise RuntimeError(
                                "Isaac 未返回双指轴向接触深度状态"
                            )
                        with self._lock:
                            closed_probe = dict(
                                self._scene_grasp_status
                            )
                        self._raise_if_finger_touching_table(
                            closed_probe, "双指轴向接触深度精调时"
                        )
                        closed_local = tuple(
                            float(value)
                            for value in closed_probe.get(
                                "local_position_m", []
                            )
                        )
                        self.get_logger().info(
                            "Applied unilateral-contact stator-depth "
                            f"correction: dx={axial_correction:.4f} m, "
                            f"left={closed_probe.get('left_finger_contact_object')}, "
                            f"right={closed_probe.get('right_finger_contact_object')}, "
                            f"local={closed_local}"
                        )
                bilateral_deadline = time.monotonic() + 1.5
                while not target_bilateral_contact(
                    closed_probe,
                    target_object,
                    require_sustained=True,
                ):
                    if time.monotonic() >= bilateral_deadline:
                        raise RuntimeError(
                            "闭爪后未形成持续双指接触："
                            f"left={closed_probe.get('left_finger_contact_object')}, "
                            f"right={closed_probe.get('right_finger_contact_object')}, "
                            "frames="
                            f"{closed_probe.get('bilateral_contact_consecutive_frames', 0)}/"
                            f"{closed_probe.get('bilateral_contact_required_frames', 0)}, "
                            "impulse_ns=("
                            f"{closed_probe.get('left_finger_contact_impulse_ns', 0.0)}, "
                            f"{closed_probe.get('right_finger_contact_impulse_ns', 0.0)})"
                        )
                    time.sleep(0.1)
                    self._command_scene_grasp("probe", target_object)
                    if not self._wait_scene_grasp(
                        "PROBED", target_object, 2.0
                    ):
                        raise RuntimeError("Isaac 未返回双指接触确认")
                    with self._lock:
                        closed_probe = dict(self._scene_grasp_status)
                    self._raise_if_finger_touching_table(
                        closed_probe, "等待持续双指接触时"
                    )
                    closed_local = tuple(
                        float(value)
                        for value in closed_probe.get("local_position_m", [])
                    )
                self.get_logger().info(
                    "Verified sustained bilateral contact before lift: "
                    f"frames={closed_probe.get('bilateral_contact_consecutive_frames')}, "
                    "impulse_ns=("
                    f"{closed_probe.get('left_finger_contact_impulse_ns')}, "
                    f"{closed_probe.get('right_finger_contact_impulse_ns')})"
                )
                if not object_loaded_grasp_is_valid(
                    closed_probe, target_object
                ):
                    raise RuntimeError(
                        "闭爪后双指接触不满足物体几何策略："
                        f"left_point={closed_probe.get('left_finger_contact_point_object_m')}, "
                        f"right_point={closed_probe.get('right_finger_contact_point_object_m')}, "
                        "impulse_ns=("
                        f"{closed_probe.get('left_finger_contact_impulse_ns')}, "
                        f"{closed_probe.get('right_finger_contact_impulse_ns')})"
                    )
                # The convex fruit can move a few tenths of a millimetre while
                # the bounded preload settles.  Keep the strict envelope
                # unchanged, but sample the real dynamic pose for up to one
                # second instead of rejecting the first transient frame.
                geometry_settle_deadline = time.monotonic() + 1.0
                while (
                    len(closed_local) != 3
                    or not inside_grasp_envelope(
                        closed_local, profile_envelope
                    )
                ) and time.monotonic() < geometry_settle_deadline:
                    time.sleep(0.1)
                    self._command_scene_grasp("probe", target_object)
                    if not self._wait_scene_grasp(
                        "PROBED", target_object, 2.0
                    ):
                        raise RuntimeError("Isaac 未返回闭爪稳定状态")
                    with self._lock:
                        closed_probe = dict(self._scene_grasp_status)
                    self._raise_if_finger_touching_table(
                        closed_probe, "闭爪几何稳定等待时"
                    )
                    closed_local = tuple(
                        float(value)
                        for value in closed_probe.get(
                            "local_position_m", []
                        )
                    )
                inside_closed_envelope = (
                    len(closed_local) == 3
                    and inside_grasp_envelope(
                        closed_local, profile_envelope
                    )
                )
                self.get_logger().info(
                    "Post-close geometry after bounded settling: "
                    f"local={closed_local}, "
                    "inside_envelope="
                    f"{inside_closed_envelope}"
                )
            post_close_valid = _post_close_geometry_is_valid(
                closed_probe,
                target_object,
                profile_envelope,
                float(profile.get("post_close_y_margin_m", 0.0)),
            )
            if post_close_valid and not inside_grasp_envelope(
                closed_local, profile_envelope
            ):
                self.get_logger().info(
                    "Apple post-close Y boundary accepted for physical lift: "
                    f"local={closed_local}, "
                    f"margin={float(profile['post_close_y_margin_m']):.4f} m"
                )
            if (
                len(closed_local) != 3
                or not post_close_valid
                or (
                    not bool(
                        closed_probe.get("contact_report_available", False)
                    )
                    and closed_local[2] > self.gripper_stator_support_z_max
                )
            ):
                contact_reports_available = bool(
                    closed_probe.get("contact_report_available", False)
                )
                geometry_requirement = (
                    f"strict_envelope={profile_envelope}"
                    if contact_reports_available
                    else
                    f"strict_envelope={profile_envelope}, "
                    f"fallback_z<={self.gripper_stator_support_z_max:.3f} m"
                )
                raise RuntimeError(
                    "闭爪后未形成有效双指几何夹持："
                    f"local={closed_local}, "
                    "reported_bilateral="
                    f"{closed_probe.get('bilateral_contact', False)}, "
                    f"{geometry_requirement}, "
                    f"contact_angle={contact_angle:.3f} rad"
                )
            object_after_close = self._current_object_position(target_object)
            close_displacement = math.sqrt(
                sum(
                    (a - b) ** 2
                    for a, b in zip(object_after_close, object_before_close)
                )
            )
            maximum_close_displacement = float(
                profile.get("maximum_close_displacement_m", 0.04)
            )
            if close_displacement > maximum_close_displacement:
                raise RuntimeError(
                    f"闭爪阶段物体移动 {close_displacement:.3f} m，拒绝抓取；"
                    f"闭爪前局部位置={grasp_probe.get('local_position_m')}，"
                    f"接触角={contact_angle:.3f} rad"
                )
            # Sustained bilateral contact inside the strict envelope is already
            # the required pre-lift state.  Do not add another closed-jaw
            # axial move: on a low object that optional motion can drive a real
            # finger mesh into the table after a valid grasp has formed.  The
            # following gravity-on vertical lift is the physical proof that the
            # object is actually carried.
            self.get_logger().info(
                "Loaded grasp is valid; proceeding directly to the gravity-on "
                "vertical lift without optional axial seating"
            )
            stage = "纯物理试抬"
            object_before_lift = self._current_object_position(target_object)
            # The grasp proof and the placement clearance motion are the same
            # useful action: lift vertically until the object clears the box
            # walls.  The old implementation continued to a fixed 220 mm,
            # stopped every 20 mm, and fell back to OMPL when the high pose
            # crossed an IK branch.  That produced a slow, visible sway with no
            # task value.  Use one continuous Cartesian lift to the actual
            # transport height instead.
            required_object_rise = float(
                profile.get("minimum_object_rise_m", 0.0)
            )
            minimum_transport_object_z = max(
                self.box_approach_object_z,
                0.14,
            )
            transport_object_z = max(
                minimum_transport_object_z,
                float(
                    profile.get(
                        "transport_object_z_m",
                        minimum_transport_object_z,
                    )
                ),
                object_before_lift[2] + required_object_rise,
            )
            lift_distance = max(
                0.0, transport_object_z - object_before_lift[2]
            )
            lift_completion_tolerance = max(
                0.001,
                float(
                    profile.get(
                        "physical_test_lift_completion_tolerance_m",
                        0.005,
                    )
                ),
            )
            safe_completion_z = min(
                transport_object_z,
                float(
                    profile.get(
                        "physical_test_lift_safe_completion_z_m",
                        transport_object_z - lift_completion_tolerance,
                    )
                ),
            )
            minimum_lift_progress = max(
                0.0,
                float(profile.get("physical_test_lift_min_progress_m", 0.0)),
            )
            maximum_stalled_iterations = max(
                1,
                int(profile.get("physical_test_lift_max_stalled_iterations", 3)),
            )
            self._set_grasp_control_state(physical_test_lift_active=True)
            if target_object == "apple":
                # Prove a local vertical lift first.  Placement plans the
                # complete carry/descent route later; do not move through
                # exploratory box-directed midpoints during grasp either.
                self._lift_apple_preplanned(transport_object_z, profile_envelope)
            else:
                self._set_status(
                    "PHYSICAL_TEST_LIFT",
                    "正在保持动态物理和完整重力，自由规划到蓝盒上方安全位",
                    False,
                )
                lift_probe = self._carry_loaded_object_toward_box(
                    target_object,
                    profile_envelope,
                    safe_completion_z,
                    "带载荷自由搬运到蓝盒方向后",
                    require_box_approach=True,
                )
            lift_segment_distance = float(
                profile.get("physical_test_lift_segment_m", lift_distance)
            )
            near_table_lift_segment_distance = float(
                profile.get(
                    "physical_test_lift_near_table_segment_m",
                    lift_segment_distance,
                )
            )
            near_table_lift_until_z = float(
                profile.get(
                    "physical_test_lift_near_table_until_z_m",
                    object_before_lift[2],
                )
            )
            near_target_lift_segment_distance = float(
                profile.get(
                    "physical_test_lift_near_target_segment_m",
                    lift_segment_distance,
                )
            )
            near_target_lift_threshold = float(
                profile.get(
                    "physical_test_lift_near_target_threshold_m",
                    0.0,
                )
            )
            remaining_lift = lift_distance
            lift_iteration = 0
            previous_object_z = self._current_object_position(target_object)[2]
            stalled_iterations = 0
            maximum_lift_iterations = max(
                1,
                int(
                    math.ceil(
                        lift_distance
                        / max(
                            min(
                                lift_segment_distance,
                                near_table_lift_segment_distance,
                            ),
                            1.0e-4,
                        )
                    )
                )
                * 4,
            )
            while target_object != "apple" and previous_object_z < safe_completion_z:
                lift_iteration += 1
                if lift_iteration > maximum_lift_iterations:
                    current_object_z = self._current_object_position(
                        target_object
                    )[2]
                    raise RuntimeError(
                        "竖直物理抬升未取得足够实际高度："
                        f"object_z={current_object_z:.4f} m, "
                        f"target_z={transport_object_z:.4f} m, "
                        f"iterations={maximum_lift_iterations}"
                    )
                current_object_z = self._current_object_position(
                    target_object
                )[2]
                if remaining_lift <= near_target_lift_threshold:
                    active_lift_segment_distance = (
                        near_target_lift_segment_distance
                    )
                elif current_object_z < near_table_lift_until_z:
                    active_lift_segment_distance = (
                        near_table_lift_segment_distance
                    )
                else:
                    active_lift_segment_distance = lift_segment_distance
                active_lift_position_tolerance = float(
                    profile.get(
                        "physical_test_lift_near_table_position_tolerance_m",
                        profile.get(
                            "physical_test_lift_position_tolerance_m",
                            0.002,
                        ),
                    )
                    if current_object_z < near_table_lift_until_z
                    else profile.get(
                        "physical_test_lift_position_tolerance_m",
                        0.002,
                    )
                )
                segment_distance = min(
                    active_lift_segment_distance, remaining_lift
                )
                minimum_retry_segment_distance = min(
                    segment_distance,
                    float(
                        profile.get(
                            "physical_test_lift_min_retry_segment_m",
                            0.005,
                        )
                    ),
                )
                lift_orientation_tolerances = tuple(
                    float(value)
                    for value in profile.get(
                        (
                            "physical_test_lift_near_target_"
                            "orientation_tolerances_rad"
                        )
                        if remaining_lift <= near_target_lift_threshold
                        else (
                            "physical_test_lift_near_table_"
                            "orientation_tolerances_rad"
                        )
                        if current_object_z < near_table_lift_until_z
                        else (
                            "physical_test_lift_"
                            "orientation_tolerances_rad"
                        ),
                        (0.04,),
                    )
                )
                while True:
                    lift_start, lift_orientation = self._current_ee_pose()
                    lift_target = (
                        lift_start[0],
                        lift_start[1],
                        lift_start[2] + segment_distance,
                    )
                    lift_error = None
                    for lift_orientation_tolerance in (
                        lift_orientation_tolerances
                    ):
                        try:
                            self._execute_manual_pose(
                                lift_target,
                                lift_orientation,
                                "连续竖直物理抬升",
                                linear=True,
                                max_joint_delta=float(
                                    profile.get(
                                        "physical_test_lift_max_joint_delta_rad",
                                        0.65,
                                    )
                                ),
                                position_tolerance=(
                                    active_lift_position_tolerance
                                ),
                                orientation_tolerance=(
                                    lift_orientation_tolerance
                                ),
                                velocity_scale=self.physical_test_lift_velocity_scale,
                            )
                            lift_error = None
                            break
                        except RuntimeError as exc:
                            lift_error = exc
                            if not _is_recoverable_lift_planning_error(exc):
                                raise
                            self.get_logger().warning(
                                "Vertical lift planner could not accept the "
                                "current Cartesian waypoint; retrying it with "
                                "bounded orientation freedom: "
                                f"segment={segment_distance:.4f} m, "
                                "tolerance="
                                f"{lift_orientation_tolerance:.3f} rad, "
                                f"reason={exc}"
                            )
                    if lift_error is None:
                        break
                    self.get_logger().warning(
                        f"Straight {segment_distance * 1000.0:.0f} mm lift "
                        "segment reached an IK branch boundary; trying one "
                        "collision-checked, joint-bounded plan to the same "
                        "nearby target"
                    )
                    try:
                        self._execute_manual_pose(
                            lift_target,
                            lift_orientation,
                            "竖直抬升 IK 分支受限过渡",
                            linear=False,
                            max_joint_delta=float(
                                profile.get(
                                    "physical_test_lift_bounded_fallback_max_joint_delta_rad",
                                    0.35,
                                )
                            ),
                            position_tolerance=(
                                active_lift_position_tolerance
                            ),
                            orientation_tolerance=max(
                                lift_orientation_tolerances
                            ),
                            velocity_scale=self.physical_test_lift_velocity_scale,
                        )
                        break
                    except RuntimeError as exc:
                        if not _is_recoverable_lift_planning_error(exc):
                            raise
                        reduced_segment_distance = (
                            _reduced_lift_segment_distance(
                                segment_distance,
                                minimum_retry_segment_distance,
                            )
                        )
                        if reduced_segment_distance is None:
                            raise RuntimeError(
                                "竖直抬升已缩短到最小安全分段仍无法规划："
                                f"segment={segment_distance:.4f} m, "
                                f"reason={exc}"
                            ) from exc
                        self.get_logger().warning(
                            "Lift fallback planning failed at the current "
                            "waypoint; preserving the verified grasp and "
                            "retrying a shorter vertical segment: "
                            f"{segment_distance:.4f} -> "
                            f"{reduced_segment_distance:.4f} m, reason={exc}"
                        )
                        segment_distance = reduced_segment_distance
                self._require_loaded_physical_grasp(
                    target_object,
                    profile_envelope,
                    "竖直抬升中",
                    use_verified_lift_reference=False,
                )
                current_object_z = self._current_object_position(
                    target_object
                )[2]
                measured_progress = current_object_z - previous_object_z
                if measured_progress < minimum_lift_progress:
                    stalled_iterations += 1
                else:
                    stalled_iterations = 0
                previous_object_z = current_object_z
                remaining_lift = max(
                    0.0, transport_object_z - current_object_z
                )
                self.get_logger().info(
                    "Physical lift height feedback: "
                    f"iteration={lift_iteration}, "
                    f"object_z={current_object_z:.4f} m, "
                    f"target_z={transport_object_z:.4f} m, "
                    f"remaining={remaining_lift:.4f} m, "
                    "completion_tolerance="
                    f"{lift_completion_tolerance:.4f} m, "
                    f"safe_completion_z={safe_completion_z:.4f} m, "
                    f"progress={measured_progress:.4f} m, "
                    f"stalled={stalled_iterations}/{maximum_stalled_iterations}"
                )
                if (
                    current_object_z < safe_completion_z
                    and stalled_iterations >= maximum_stalled_iterations
                ):
                    raise RuntimeError(
                        "竖直物理抬升连续无有效进展："
                        f"object_z={current_object_z:.4f} m, "
                        f"safe_z={safe_completion_z:.4f} m, "
                        f"minimum_progress={minimum_lift_progress:.4f} m"
                    )

            # One authoritative end-state check replaces all old per-segment
            # and top-hold threshold machinery.  It retains only facts that
            # describe a real grasp: dynamic physics, gravity, actual bilateral
            # contact, loaded geometry and table clearance.
            lift_probe = self._require_loaded_physical_grasp(
                target_object,
                profile_envelope,
                "连续直线抬升后",
                use_verified_lift_reference=False,
            )
            if bool(lift_probe.get("object_touching_table", True)):
                raise RuntimeError("连续直线抬升后目标仍接触桌面")
            object_after_lift = self._current_object_position(target_object)
            object_rise = object_after_lift[2] - object_before_lift[2]
            if object_after_lift[2] < safe_completion_z:
                raise RuntimeError(
                    "物体未达到蓝盒安全搬运净空："
                    f"object_z={object_after_lift[2]:.4f} m, "
                    f"required_z={safe_completion_z:.4f} m, "
                    f"rise={object_rise:.4f} m"
                )
            lift_local = tuple(
                float(value)
                for value in lift_probe.get("local_position_m", [])
            )
            with self._lock:
                self._loaded_grasp_reference_local = lift_local
            self._set_grasp_control_state(
                physical_test_lift_active=False,
                physical_grasp_ok=True,
            )
            if not self._wait_task_flag("physical_grasp_ok", True, 4.0):
                self.get_logger().warning(
                    "Evaluator physical_grasp_ok did not catch up before the "
                    "deadline; accepting the completed consecutive direct "
                    "PhysX verification."
                )
            self._set_status(
                "PHYSICAL_GRASP_OK",
                "苹果已在完整重力下随夹爪离桌并完成配置试抬",
                False,
            )
        except Exception as exc:
            diagnostic = ""
            if stage in (
                "安全深度直线接近",
                "瓶前安全位近水平姿态过渡",
                "对中后轴向插入",
            ):
                try:
                    self._command_scene_grasp("probe", target_object)
                    if self._wait_scene_grasp("PROBED", target_object, 2.0):
                        with self._lock:
                            failed_probe = dict(self._scene_grasp_status)
                        diagnostic = (
                            "；失败位置几何中心 local="
                            f"{failed_probe.get('local_position_m')}"
                        )
                except Exception:
                    diagnostic = "；失败位置几何中心测量失败"
            self._set_grasp_control_state(
                gripper_commanded_closed=False,
                gripper_obstructed=False,
                physical_test_lift_active=False,
                physical_grasp_ok=False,
            )
            self._command_scene_grasp("unfreeze")
            raise RuntimeError(f"{stage}失败：{exc}{diagnostic}") from exc
        self._set_status(
            "GRASP_CONFIRMED",
            "纯物理夹取已确认：双指持续受载且目标已离桌",
            False,
        )
        with self._lock:
            self._verified_grasp_sequence = True
        return "苹果已通过双指物理夹持、离桌与相对位姿验证"

    def _plan_apple_transfer(self, origin, ee_origin, orientation, height, start_state=None):
        """Look ahead through release; no physical movement during search.

        Try direct lift first, then two bounded diagonal alternatives after
        a 20 mm vertical clearance. Every leg is collision-checked LIN.
        A clearance stop is intentional; this is not a blended trajectory.
        """
        height = max(height, origin[2], self.box_approach_object_z, 0.14)
        offset = tuple(a-b for a,b in zip(ee_origin, origin))
        errors = []
        for fraction in (0.0, 0.25, 0.5):
            for tolerance in (0.025, 0.04):
                self._raise_if_stop_requested()
                destinations = []
                if fraction and origin[2] + 0.021 < height:
                    destinations.append(('clearance', (origin[0],origin[1],origin[2]+0.02)))
                lift = (origin[0] + fraction*(self.box_center[0]-origin[0]),
                        origin[1] + fraction*(self.box_center[1]-origin[1]),height)
                destinations.extend((('lift',lift),('carry',(self.box_center[0],self.box_center[1],height)),
                                     ('descent',(self.box_center[0],self.box_center[1],self.box_release_object_z))))
                previous_object, previous_ee = origin, ee_origin
                state = start_state
                route = []
                try:
                    for phase, destination in destinations:
                        if math.dist(previous_object,destination) < 0.001:
                            continue
                        self._raise_if_stop_requested()
                        target = tuple(a+b for a,b in zip(destination,offset))
                        if (target[2] < self.manual_minimum_z or target[2] > 0.75
                                or math.hypot(target[0],target[1]) > 0.78):
                            raise RuntimeError('苹果完整搬运目标超出安全工作空间')
                        plan = self._plan_candidate(
                            {'position':target,'orientation':orientation}, linear=True,
                            position_tolerance=0.002, orientation_tolerance=tolerance,
                            velocity_scale=self.physical_test_lift_velocity_scale,
                            acceleration_scale=self.physical_test_lift_velocity_scale,
                            start_state=state,
                        )
                        _,delta = self._trajectory_score(plan.planned_trajectory.joint_trajectory)
                        if delta > self.max_joint_delta:
                            raise RuntimeError('拒绝关节跳变：苹果搬运候选')
                        route.append(dict(phase=phase, object=destination, ee=target,
                                          start_object=previous_object, start_ee=previous_ee,
                                          orientation=orientation, trajectory=plan.planned_trajectory))
                        state = self._state_after_trajectory(plan)
                        previous_object, previous_ee = destination,target
                    return route
                except RuntimeError as exc:
                    if not _is_recoverable_lift_planning_error(exc):
                        raise
                    errors.append(str(exc))
        raise RuntimeError('苹果完整搬运路径预检查失败：'+'; '.join(errors))

    def _execute_apple_transfer_leg(self, leg, envelope):
        self._raise_if_stop_requested()
        self._require_loaded_physical_grasp(
            'apple',envelope,'苹果已规划路线执行前',use_verified_lift_reference=False)
        _, orientation = self._current_ee_pose()
        path = leg['trajectory'].joint_trajectory
        with self._lock:
            joints=[self._joint_positions.get(name) for name in path.joint_names]
        dot=min(1.0,abs(sum(a*b for a,b in zip(orientation,leg['orientation']))))
        if (2*math.acos(dot) > 0.10
                or any(x is None for x in joints)
                or max(abs(a-b) for a,b in zip(joints,path.points[0].positions)) > 0.03):
            raise RuntimeError('苹果预规划路线起点偏差超限')
        self._set_status('PHYSICAL_TEST_LIFT' if leg['phase'] in ('lift','clearance') else 'GUIDED_PLACE',
                         '苹果：执行预规划路线 '+leg['phase'],False)
        if self._send_arm_trajectory(leg['trajectory']):
            raise RuntimeError('苹果预规划路线已停止')
        self._require_loaded_physical_grasp(
            'apple',envelope,'苹果已规划路线执行后',use_verified_lift_reference=False)

    def _lift_apple_preplanned(self, height: float, envelope: dict) -> None:
        self._apple_place_route = None
        if not self._move_group.wait_for_server(timeout_sec=2.0):
            raise RuntimeError('MoveIt /move_action 不可用')
        origin=self._current_object_position('apple')
        ee,orientation=self._current_ee_pose()
        route=self._plan_apple_transfer(origin,ee,orientation,height)
        # Retain the validated tail, so placement never repeats planning.
        for leg in route:
            if leg['phase'] in ('clearance','lift'):
                self._execute_apple_transfer_leg(leg,envelope)
        self._apple_place_route=[leg for leg in route if leg['phase'] in ('carry','descent')]


    def _place_apple_preplanned(self, profile_envelope: dict) -> None:
        """Plan lift, one LIN carry and descent before moving the apple.

        Each candidate is rooted in the same live start state.  Chained
        planning checks descent reachability without executing trial poses.
        Stops exist only at meaningful lift/carry/descent boundaries, never
        at artificial intermediate points along the horizontal carry.
        """
        cached = getattr(self, "_apple_place_route", None)
        if cached is not None:
            self._apple_place_route = None
            for leg in cached:
                self._execute_apple_transfer_leg(leg, profile_envelope)
            return
        if not self._move_group.wait_for_server(timeout_sec=2.0):
            raise RuntimeError("MoveIt /move_action 不可用")
        origin = self._current_object_position("apple")
        ee_origin, orientation = self._current_ee_pose()
        offset = tuple(a - b for a, b in zip(ee_origin, origin))
        base_height = max(origin[2], self.box_approach_object_z, 0.14)
        plans = None
        errors = []
        self._set_status("GUIDED_PLACE", "苹果：预规划完整搬运及下放路径", False)
        for extra_height in (0.0, 0.01, 0.02):
            self._raise_if_stop_requested()
            height = base_height + extra_height
            destinations = []
            if height > origin[2] + 0.001:
                destinations.append((origin[0], origin[1], height))
            approach = (self.box_center[0], self.box_center[1], height)
            previous = destinations[-1] if destinations else origin
            if math.dist(previous, approach) > 0.001:
                destinations.append(approach)
            destinations.append((approach[0], approach[1], self.box_release_object_z))
            candidate_plans = []
            start_state = None
            try:
                for destination in destinations:
                    self._raise_if_stop_requested()
                    target = tuple(a + b for a, b in zip(destination, offset))
                    if (target[2] < self.manual_minimum_z
                            or target[2] > 0.75
                            or math.hypot(target[0], target[1]) > 0.78):
                        raise RuntimeError("苹果放置目标超出安全工作空间")
                    result = self._plan_candidate(
                        {"position": target, "orientation": orientation},
                        position_tolerance=0.002,
                        orientation_tolerance=0.025,
                        linear=True,
                        velocity_scale=self.physical_test_lift_velocity_scale,
                        acceleration_scale=self.physical_test_lift_velocity_scale,
                        start_state=start_state,
                    )
                    _, delta = self._trajectory_score(result.planned_trajectory.joint_trajectory)
                    if delta > self.max_joint_delta:
                        raise RuntimeError("拒绝关节跳变：苹果完整放置路径关节变化超限")
                    candidate_plans.append((destination, target, result.planned_trajectory))
                    start_state = self._state_after_trajectory(result)
                plans = candidate_plans
                break
            except RuntimeError as exc:
                # Only rejected plans may try another height.  Stop requests,
                # transport outages and execution failures must propagate.
                if not _is_recoverable_lift_planning_error(exc):
                    raise
                errors.append(str(exc))
        if plans is None:
            raise RuntimeError("苹果完整放置路径无法规划；未执行试探移动：" + "; ".join(errors))

        for index, (destination, target, trajectory) in enumerate(plans):
            self._raise_if_stop_requested()
            self._require_loaded_physical_grasp("apple", profile_envelope, "苹果预规划路径执行前")
            _, actual_orientation = self._current_ee_pose()
            orientation_dot = min(1.0, abs(sum(a*b for a,b in zip(orientation, actual_orientation))))
            if 2.0 * math.acos(orientation_dot) > 0.06:
                raise RuntimeError("苹果预规划路径起点偏差超限，停止本次示范")
            joint_path = trajectory.joint_trajectory
            with self._lock:
                actual_joints = [self._joint_positions.get(name) for name in joint_path.joint_names]
            if (any(value is None for value in actual_joints)
                    or max(abs(a - b) for a, b in zip(
                        actual_joints, joint_path.points[0].positions)) > 0.03):
                raise RuntimeError("苹果预规划路径关节起点偏差超限，停止本次示范")
            self._set_status("GUIDED_PLACE", f"苹果：执行已验证路径 {index + 1}/{len(plans)}", False)
            if self._send_arm_trajectory(trajectory):
                raise RuntimeError("苹果预规划放置已停止")
            self._require_loaded_physical_grasp("apple", profile_envelope, "苹果预规划路径执行后")


    def _guided_place(self) -> str:
        with self._lock:
            physical_grasp_ok = bool(
                self._task_status.get("physical_grasp_ok", False)
            )
            sequence_verified = self._verified_grasp_sequence
            target_object = self.target_object
        if not sequence_verified:
            raise RuntimeError(
                "尚未完成本次纯物理抓取、离桌和配置试抬验证"
            )
        if not physical_grasp_ok:
            # The evaluator flag is an asynchronous snapshot and can briefly
            # fall behind the real PhysX contact state.  The sequence token
            # proves this guided-grasp invocation completed; the probe below
            # is the authoritative current loaded-grasp check.
            self.get_logger().warning(
                "Evaluator physical_grasp_ok is temporarily false at place "
                "entry; verifying the current physical grasp directly."
            )
        profile = GRASP_PROFILES[target_object]
        profile_envelope = dict(GRASP_ENVELOPE)
        profile_envelope.update(
            profile.get("grasp_envelope_overrides", {})
        )
        self._require_loaded_physical_grasp(
            target_object,
            profile_envelope,
            "开始纯物理搬运前",
        )

        if target_object == "apple":
            self._place_apple_preplanned(profile_envelope)
            fallback_release = False
        else:
            object_position = self._current_object_position(target_object)
            # With the steep, inside-jaw fruit grasp, an object center at 0.20 m
            # drives the wrist outside this Z1 IK branch.  The box walls end at
            # 0.10 m; a 0.14 m centre keeps a 50 mm fruit's bottom at about
            # 0.115 m, leaving 15 mm of collision clearance above the walls.
            clearance_z = max(self.box_approach_object_z, 0.14)
            clearance_tolerance = max(
                0.0,
                float(
                    profile.get(
                        "physical_test_lift_completion_tolerance_m",
                        0.0,
                    )
                ),
            )
            minimum_clearance_z = clearance_z - clearance_tolerance
            self._carry_loaded_object_toward_box(
                target_object,
                profile_envelope,
                minimum_clearance_z,
                "开始放置前到达蓝盒上方安全位",
                require_box_approach=True,
                preferred_object_z=clearance_z,
            )
            object_position = self._current_object_position(target_object)
            if object_position[2] < minimum_clearance_z:
                self._move_object_to(
                    (object_position[0], object_position[1], clearance_z),
                    "先垂直抬升到安全搬运高度",
                    linear=True,
                )
                self._require_loaded_physical_grasp(
                    target_object,
                    profile_envelope,
                    "垂直抬升到搬运高度后",
                )
            release_xy_tolerance = BOX_RELEASE_XY_TOLERANCE_M.get(
                target_object, 0.020
            )
            object_label = {
                "apple": "苹果",
                "peach": "桃子",
                "cup": "杯子",
                "beverage_bottle": "瓶子",
            }.get(target_object, target_object)
            # Keep the original carry clearance; cup alignment/release can finish
            # above the explicitly requested115mm world-root floor.
            if target_object == "cup":
                minimum_clearance_z = CUP_DIRECT_RELEASE_MIN_ROOT_Z_M
            # The verified loaded carry has already delivered the object into the
            # box-approach corridor.  Keep its measured safe height and correct
            # only the remaining lateral error, so no waypoint is ever commanded
            # below the box walls.
            position = self._current_object_position(target_object)
            if position[2] < minimum_clearance_z:
                raise RuntimeError(
                    "目标物体放入蓝盒前低于安全搬运高度："
                    f"actual_z={position[2]:.4f} m, target_z={clearance_z:.4f} m, "
                    f"tolerance={clearance_tolerance:.4f} m, "
                    f"最低验收高度={minimum_clearance_z:.4f} m"
                )
            alignment_z = position[2]
            center_error = math.hypot(
                position[0] - self.box_center[0],
                position[1] - self.box_center[1],
            )
            single_pass_alignment = bool(
                profile.get("single_pass_box_alignment", False)
            )
            alignment_max_steps = (
                1 if single_pass_alignment else BOX_ALIGNMENT_MAX_STEPS
            )
            for step_index in range(1, alignment_max_steps + 1):
                if center_error <= release_xy_tolerance:
                    break
                step_length = (
                    center_error
                    if single_pass_alignment
                    else min(center_error, BOX_ALIGNMENT_STEP_M)
                )
                scale = step_length / center_error
                self._move_object_to(
                    (
                        position[0] + (self.box_center[0] - position[0]) * scale,
                        position[1] + (self.box_center[1] - position[1]) * scale,
                        alignment_z,
                    ),
                    (
                        f"蓝盒上方{object_label}水平对中"
                        f"（第 {step_index}/{alignment_max_steps} 步，"
                        f"步长 {step_length * 1000.0:.1f} mm）"
                    ),
                    linear=True,
                    orientation_tolerance=(
                        BOX_ALIGNMENT_ORIENTATION_TOLERANCE_RAD
                    ),
                )
                self._require_loaded_physical_grasp(
                    target_object,
                    profile_envelope,
                    f"蓝盒上方{object_label}对中第 {step_index} 步后",
                )
                previous_error = center_error
                position = self._current_object_position(target_object)
                if position[2] < minimum_clearance_z:
                    raise RuntimeError(
                        "蓝盒上方水平对中把物体带到了"
                        "安全搬运高度以下："
                        f"actual_z={position[2]:.4f} m, "
                        f"target_z={clearance_z:.4f} m, "
                        f"tolerance={clearance_tolerance:.4f} m, "
                        f"最低验收高度={minimum_clearance_z:.4f} m"
                    )
                center_error = math.hypot(
                    position[0] - self.box_center[0],
                    position[1] - self.box_center[1],
                )
                if previous_error - center_error < BOX_ALIGNMENT_MIN_PROGRESS_M:
                    step_label = f"{step_index}/{alignment_max_steps}"
                    raise RuntimeError(
                        "蓝盒对中仍超差：水平对中第 "
                        f"{step_label} 步未取得进展，"
                        f"before={previous_error:.4f} m, "
                        f"after={center_error:.4f} m, "
                        f"tolerance={release_xy_tolerance:.4f} m"
                    )
            if center_error > release_xy_tolerance:
                raise RuntimeError(
                    "蓝盒对中仍超差，已停止下降："
                    f"xy_error={center_error:.4f} m, "
                    f"tolerance={release_xy_tolerance:.4f} m, "
                    f"actual_z={position[2]:.4f} m"
                )
            fallback_release = False
            if target_object == "cup":
                _validate_cup_direct_box_release(position, self.box_center, release_xy_tolerance)
            else:
                # Descend from the converged position instead of re-targeting the box
                # centre: the final approach stays vertical and cannot drag the object
                # sideways at box-wall height.
                release_xy = (position[0], position[1])
                fallback_release = False
                try:
                    self._move_object_to(
                        (
                            release_xy[0],
                            release_xy[1],
                            self.box_release_object_z,
                        ),
                        "垂直放入蓝盒",
                        linear=True,
                    )
                except RuntimeError as exc:
                    if "MoveIt error_code=-31" not in str(exc):
                        raise
                    position = self._current_object_position(target_object)
                    xy_error = math.hypot(
                        position[0] - self.box_center[0],
                        position[1] - self.box_center[1],
                    )
                    if xy_error > release_xy_tolerance or position[2] < minimum_clearance_z:
                        raise RuntimeError(
                            "低位下降失败，回退释放位置已不安全："
                            f"xy_error={xy_error:.4f} m, z={position[2]:.4f} m"
                        ) from exc
                    if position[2] > BOX_FALLBACK_RELEASE_MAX_OBJECT_Z_M:
                        raise RuntimeError(
                            "低位下降失败，回退释放高度超限："
                            f"z={position[2]:.4f} m > "
                            f"{BOX_FALLBACK_RELEASE_MAX_OBJECT_Z_M:.4f} m"
                        ) from exc
                    fallback_release = True
                    self.get_logger().warning(
                        "Low box descent has no IK solution; releasing from the "
                        f"verified aligned pose at object z={position[2]:.4f} m"
                    )
        self._require_loaded_physical_grasp(
            target_object,
            profile_envelope,
            "杯子对中后直接释放前" if target_object == "cup" else (
                "高位回退释放前" if fallback_release else "到达蓝盒释放位后"
            ),
        )
        if target_object == "cup":
            _validate_cup_direct_box_release(
                self._current_object_position(target_object), self.box_center,
                BOX_RELEASE_XY_TOLERANCE_M.get(target_object, 0.020),
            )
        self._set_status(
            "GUIDED_RELEASE",
            "机械臂已停止，准备缓慢开爪",
            False,
        )
        self._set_grasp_control_state(release_started=True)
        self._set_status(
            "GUIDED_RELEASE",
            "正在缓慢开爪，让真实双指夹持的物体自然落入蓝盒",
            False,
        )
        self._open_gripper_for_box_release(target_object)
        # The first opening already proved that the real bilateral clamp was
        # released.  Publish that fact before moving the arm so a retreat-only
        # planning failure cannot erase a physically successful placement.
        self._set_grasp_control_state(
            gripper_commanded_closed=False,
            gripper_obstructed=False,
            release_sequence_complete=True,
        )
        # The low release pose leaves the moving finger close to a box wall.
        # A first open is enough to free the object, but the wall can stop the
        # joint just short of the evaluator's fully-released threshold.  Move
        # vertically clear of the box, then finish opening.  If this arm-only
        # cleanup fails, the final physical probe and box-stability evaluator
        # still decide whether the placement itself succeeded.
        position, orientation = self._current_ee_pose()
        retreat = (position[0], position[1], position[2] + 0.04)
        retreat_error = None
        try:
            self._execute_manual_pose(
                retreat,
                orientation,
                "释放后垂直撤离",
                linear=True,
            )
            self._execute_gripper("open")
        except RuntimeError as exc:
            retreat_error = exc
            self.get_logger().warning(
                "The object clamp was released, but post-release arm "
                f"cleanup failed; checking box stability: {exc}"
            )
        # Refresh the scene-owned contact state after release.  Without this
        # probe the evaluator keeps the last pre-release bilateral-contact
        # sample even when the open gripper is already far from the object.
        self._command_scene_grasp("probe", target_object)
        if not self._wait_scene_grasp("PROBED", target_object, 2.0):
            raise RuntimeError("Isaac 未返回自然释放后的最终物理状态")
        with self._lock:
            released_probe = dict(self._scene_grasp_status)
        # The evaluator requires 0.5 s of simulation-time stability.  When
        # headless rendering runs near 0.1x real time, a 5 s wall deadline is
        # exactly the minimum and can expire one callback too early.
        if not self._wait_task_flag("task_success", True, 45.0):
            with self._lock:
                evaluator_status = dict(self._task_status)
            if retreat_error is not None:
                raise RuntimeError(
                    "物体已解除夹持，但释放后撤离失败且未通过蓝盒稳定判定："
                    f"retreat={retreat_error}; evaluator={evaluator_status}; "
                    f"probe={released_probe}"
                ) from retreat_error
            raise RuntimeError(
                "自然释放后未通过蓝盒位置和速度判定："
                f"{evaluator_status}"
            )
        self._set_status("TASK_COMPLETE", "物体已稳定放入蓝盒", False)
        with self._lock:
            self._verified_grasp_sequence = False
            self._loaded_grasp_reference_local = None
        return "引导放置完成，任务判定成功，可以保存 Episode"

    def _open_gripper_for_box_release(self, target_object: str) -> float:
        with self._lock:
            start_angle = self._joint_positions.get("jointGripper")
        if start_angle is None:
            raise RuntimeError("盒内释放前没有 jointGripper 状态")
        action_error = None
        try:
            actual_angle = self._execute_gripper("open")
        except RuntimeError as exc:
            if (
                "夹爪打开失败：-5" not in str(exc)
                and "final point within tolerance" not in str(exc)
            ):
                raise
            action_error = exc
            with self._lock:
                actual_angle = self._joint_positions.get("jointGripper")
            if actual_angle is None:
                raise
        time.sleep(0.3)
        self._command_scene_grasp("probe", target_object)
        if not self._wait_scene_grasp("PROBED", target_object, 2.0):
            raise RuntimeError("Isaac 未返回盒内首次开爪后的释放状态")
        with self._lock:
            release_probe = dict(self._scene_grasp_status)
        if not _partial_box_release_is_valid(
            float(start_angle),
            float(actual_angle),
            release_probe,
            target_object,
            self.box_release_min_opening_delta,
        ):
            raise RuntimeError(
                "盒内首次开爪未解除真实双指夹持："
                f"start={float(start_angle):.4f}, "
                f"actual={float(actual_angle):.4f}, "
                f"left={release_probe.get('left_finger_contact_object')}, "
                f"right={release_probe.get('right_finger_contact_object')}, "
                f"bilateral={release_probe.get('bilateral_contact', False)}"
            ) from action_error
        if action_error is not None:
            self.get_logger().warning(
                "The first in-box open was wall-limited, but the real "
                "bilateral clamp is released; continuing with vertical "
                f"retreat before full opening: actual={actual_angle:.4f}"
            )
        return float(actual_angle)

    def _require_loaded_physical_grasp(
        self,
        target_object: str,
        profile_envelope: dict,
        stage: str,
        use_verified_lift_reference: bool = True,
        loaded_envelope_margin_m: Optional[float] = None,
    ) -> dict:
        with self._lock:
            reference_local = self._loaded_grasp_reference_local
        if use_verified_lift_reference and reference_local is None:
            raise RuntimeError(
                f"{stage}：缺少纯物理试抬成功时的夹持位姿基线"
            )
        valid_samples = 0
        probe = {}
        for attempt in range(1, self.loaded_grasp_max_attempts + 1):
            self._command_scene_grasp("probe", target_object)
            if not self._wait_scene_grasp("PROBED", target_object, 2.0):
                if attempt < self.loaded_grasp_max_attempts:
                    continue
                raise RuntimeError(
                    f"{stage}：Isaac 在重试窗口内未返回真实夹持状态"
                )
            with self._lock:
                probe = dict(self._scene_grasp_status)
            self._raise_if_finger_touching_table(probe, stage)
            if not bool(probe.get("physics_enabled", False)):
                raise RuntimeError(f"{stage}：目标不是动态刚体")
            if not bool(probe.get("gravity_enabled", False)):
                raise RuntimeError(f"{stage}：目标重力未启用")
            if use_verified_lift_reference:
                valid = _transport_loaded_grasp_is_valid(
                    probe,
                    target_object,
                    profile_envelope,
                    self.transport_loaded_grasp_guard_margin,
                    reference_local,
                    self.physical_test_lift_relative_position_tolerance,
                    require_envelope=_envelope_validation_enabled(
                        target_object
                    ),
                )
            else:
                valid = (
                    bool(probe.get("contact_report_available", False))
                    and _loaded_bilateral_grasp_is_valid(
                        probe,
                        target_object,
                        profile_envelope,
                        (
                            self.physical_test_lift_loaded_envelope_margin
                            if loaded_envelope_margin_m is None
                            else loaded_envelope_margin_m
                        ),
                        require_envelope=_envelope_validation_enabled(
                            target_object
                        ),
                    )
                )
            if valid:
                valid_samples += 1
                required_samples = int(
                    GRASP_PROFILES[target_object].get(
                        "loaded_samples_required",
                        self.loaded_grasp_samples_required,
                    )
                )
                if valid_samples >= required_samples:
                    object_profile = OBJECT_GRASP_PROFILES.get(target_object, {})
                    if object_profile.get("strategy") == "cylinder_side_pinch":
                        quality = cylinder_contact_quality_metrics(
                            probe, target_object
                        )
                        self.get_logger().info(
                            f"{stage}：圆柱夹持质量："
                            f"contact_heights_m={quality['contact_heights_m']}, "
                            "contact_height_midpoint_m="
                            f"{quality['contact_height_midpoint_m']}, "
                            "contact_height_delta_m="
                            f"{quality['contact_height_delta_m']}, "
                            f"contact_impulses_ns={quality['contact_impulses_ns']}, "
                            "contact_impulse_ratio="
                            f"{quality['contact_impulse_ratio']}"
                        )
                    if (
                        object_profile.get("strategy") == "cylinder_side_pinch"
                        and not cylinder_target_contact_heights_are_valid(
                            probe, target_object, object_profile,
                            require_bilateral=True,
                        )
                    ):
                        self.get_logger().warning(
                            f"{stage}：接触高度在推荐带外，双指受力检查已通过；"
                            "夹持成功仍需物理试抬、离桌及滑移验证。"
                            f"left={probe.get('left_finger_contact_point_object_m')}, "
                            f"right={probe.get('right_finger_contact_point_object_m')}, "
                            f"preferred_height_m={object_profile['grasp_height_range_m']}"
                        )
                    return probe
            else:
                valid_samples = 0
                self.get_logger().warning(
                    f"{stage}: ignoring transient loaded-grasp observation "
                    f"{attempt}/{self.loaded_grasp_max_attempts}: "
                    f"local={probe.get('local_position_m')}, "
                    f"reference={reference_local}, "
                    "contacts=("
                    f"{probe.get('left_finger_contact_object')}, "
                    f"{probe.get('right_finger_contact_object')})"
                )
            if attempt < self.loaded_grasp_max_attempts:
                time.sleep(self.physical_test_lift_verification_poll)
        raise RuntimeError(
            f"{stage}：连续夹持验证未通过（检查双指接触、受力比例、位置包络或参考位姿偏差；接触高度带仅作提示）："
            "valid="
            f"{valid_samples}/"
            f"{int(GRASP_PROFILES[target_object].get('loaded_samples_required', self.loaded_grasp_samples_required))}, "
            f"attempts={self.loaded_grasp_max_attempts}, "
            f"local={probe.get('local_position_m')}, "
            f"reference={reference_local}, "
            f"left={probe.get('left_finger_contact_object')}, "
            f"right={probe.get('right_finger_contact_object')}, "
            f"bilateral={probe.get('bilateral_contact', False)}, "
            f"contact_report={probe.get('contact_report_available', False)}, "
            f"impulses_ns=({probe.get('left_finger_contact_impulse_ns')}, "
            f"{probe.get('right_finger_contact_impulse_ns')}), "
            f"envelope={profile_envelope}"
        )

    @staticmethod
    def _raise_if_finger_touching_table(probe: dict, stage: str) -> None:
        left_touching = bool(
            probe.get("left_finger_touching_table", False)
        )
        right_touching = bool(
            probe.get("right_finger_touching_table", False)
        )
        left_impulse = float(
            probe.get("left_finger_table_impulse_ns", 0.0)
        )
        right_impulse = float(
            probe.get("right_finger_table_impulse_ns", 0.0)
        )
        loaded_table_contact = (
            left_touching and left_impulse > 1.0e-6
        ) or (
            right_touching and right_impulse > 1.0e-6
        )
        if loaded_table_contact:
            raise RuntimeError(
                f"{stage}检测到夹指与桌面碰撞，已停止闭爪："
                f"left={left_touching}, right={right_touching}, "
                "impulse_ns=("
                f"{left_impulse}, {right_impulse})"
            )

    def _move_object_to(
        self,
        destination: Vector3,
        description: str,
        linear: bool,
        orientation_tolerance: Optional[float] = None,
        max_joint_delta: Optional[float] = None,
        velocity_scale: Optional[float] = None,
        acceleration_scale: Optional[float] = None,
    ) -> None:
        object_position = self._current_object_position(self.target_object)
        ee_position, ee_orientation = self._current_ee_pose()
        offset = tuple(a - b for a, b in zip(destination, object_position))
        target = tuple(a + b for a, b in zip(ee_position, offset))
        self._set_status("GUIDED_PLACE", f"正在{description}", False)
        try:
            self._execute_manual_pose(
                target,
                ee_orientation,
                description,
                linear=linear,
                max_joint_delta=(
                    self.max_joint_delta
                    if max_joint_delta is None
                    else float(max_joint_delta)
                ),
                orientation_tolerance=orientation_tolerance,
                velocity_scale=velocity_scale,
                acceleration_scale=acceleration_scale,
            )
        except Exception as exc:
            raise RuntimeError(f"{description}失败：{exc}") from exc

    def _carry_loaded_object_toward_box(
        self,
        target_object: str,
        loaded_envelope: dict,
        minimum_object_z: float,
        stage: str,
        require_box_approach: bool = False,
        preferred_object_z: Optional[float] = None,
    ) -> dict:
        """Move a held object toward the box without prescribing a lift axis.

        The object pose is measured after every executed plan.  MoveIt may
        change the wrist orientation and take any collision-free free-space
        path.  If the direct box-approach goal has no IK solution, an elevated
        midpoint toward the box changes the arm branch before retrying it.
        """
        lift_config = GRASP_CONFIG["physical_test_lift"]
        target_profile = OBJECT_GRASP_PROFILES.get(target_object, {})
        transport_guard_margin = getattr(
            self,
            "transport_loaded_grasp_guard_margin",
            float(lift_config["transport_guard_margin_m"]),
        )
        orientation_tolerances = tuple(
            float(value)
            for value in target_profile.get(
                "loaded_transport_orientation_tolerances_rad",
                lift_config.get(
                    "goal_directed_orientation_tolerances_rad",
                    (0.20, 0.40, 0.70, 1.00),
                ),
            )
        )
        maximum_joint_delta = float(
            target_profile.get(
                "loaded_transport_max_joint_delta_rad",
                lift_config.get(
                    "goal_directed_max_joint_delta_rad",
                    self.max_joint_delta,
                ),
            )
        )
        box_xy_tolerance = BOX_TRANSPORT_XY_TOLERANCE_M.get(
            target_object,
            float(lift_config.get("goal_directed_box_xy_tolerance_m", 0.035)),
        )
        transport_velocity_scale = float(
            target_profile.get(
                "loaded_transport_velocity_scale",
                self.physical_test_lift_velocity_scale,
            )
        )
        transport_acceleration_scale = float(
            target_profile.get(
                "loaded_transport_acceleration_scale",
                transport_velocity_scale,
            )
        )
        single_pass_transport = bool(
            GRASP_PROFILES[target_object].get(
                "single_pass_box_transport", False
            )
        )
        maximum_attempts = (
            1
            if single_pass_transport
            else max(
                1,
                int(lift_config.get("goal_directed_max_attempts", 3)),
            )
        )
        destination_z = max(
            float(minimum_object_z),
            float(self.box_approach_object_z),
            float(preferred_object_z) if preferred_object_z is not None else float(minimum_object_z),
        )
        approach_xy = (self.box_center[0], self.box_center[1])
        last_error = None

        for attempt in range(1, maximum_attempts + 1):
            object_position = self._current_object_position(target_object)
            destination = (
                approach_xy[0],
                approach_xy[1],
                destination_z,
            )
            horizontal_error = math.hypot(
                object_position[0] - destination[0],
                object_position[1] - destination[1],
            )
            if (
                object_position[2] >= minimum_object_z
                and (
                    not require_box_approach
                    or horizontal_error <= box_xy_tolerance
                )
            ):
                probe = self._require_loaded_physical_grasp(
                    target_object,
                    loaded_envelope,
                    stage,
                    use_verified_lift_reference=False,
                    loaded_envelope_margin_m=transport_guard_margin,
                )
                settled = self._current_object_position(target_object)
                if settled[2] >= minimum_object_z and (
                    not require_box_approach
                    or math.hypot(settled[0] - approach_xy[0],
                                  settled[1] - approach_xy[1]) <= box_xy_tolerance
                ):
                    return probe

            for orientation_tolerance in orientation_tolerances:
                try:
                    self._move_object_to(
                        destination,
                        (
                            "带载荷自由规划到蓝盒上方安全位"
                            f"（第 {attempt}/{maximum_attempts} 次）"
                        ),
                        linear=False,
                        orientation_tolerance=orientation_tolerance,
                        max_joint_delta=maximum_joint_delta,
                        velocity_scale=transport_velocity_scale,
                        acceleration_scale=transport_acceleration_scale,
                    )
                    last_error = None
                    break
                except RuntimeError as exc:
                    last_error = exc
                    if not _is_recoverable_lift_planning_error(exc):
                        raise
                    self.get_logger().warning(
                        "Goal-directed loaded transport could not reach the "
                        "box approach with the current orientation freedom: "
                        f"attempt={attempt}, tolerance="
                        f"{orientation_tolerance:.3f} rad, reason={exc}"
                    )
            if last_error is None:
                probe = self._require_loaded_physical_grasp(
                    target_object,
                    loaded_envelope,
                    "带载荷向蓝盒搬运后",
                    use_verified_lift_reference=False,
                    loaded_envelope_margin_m=transport_guard_margin,
                )
                if bool(probe.get("object_touching_table", True)):
                    raise RuntimeError("带载荷向蓝盒搬运后目标仍接触桌面")
                continue

            if single_pass_transport:
                raise RuntimeError(
                    "瓶子单次目标导向搬运无法规划，已停止且不再执行"
                    f"斜向补齐：{last_error}"
                ) from last_error

            # A nearer diagonal waypoint changes both the Cartesian position
            # and the IK branch.  It is deliberately already above the box
            # wall clearance, so every accepted fallback makes useful task
            # progress rather than merely repeating a shorter vertical move.
            midpoint = (
                0.5 * (object_position[0] + destination[0]),
                0.5 * (object_position[1] + destination[1]),
                destination_z,
            )
            midpoint_error = last_error
            for orientation_tolerance in orientation_tolerances:
                try:
                    self._move_object_to(
                        midpoint,
                        "沿蓝盒方向带载荷斜向过渡",
                        linear=False,
                        orientation_tolerance=orientation_tolerance,
                        max_joint_delta=maximum_joint_delta,
                        velocity_scale=transport_velocity_scale,
                        acceleration_scale=transport_acceleration_scale,
                    )
                    midpoint_error = None
                    break
                except RuntimeError as exc:
                    midpoint_error = exc
                    if not _is_recoverable_lift_planning_error(exc):
                        raise
            if midpoint_error is not None:
                raise RuntimeError(
                    "无法规划到蓝盒安全位或其斜向过渡点："
                    f"{midpoint_error}"
                ) from midpoint_error
            probe = self._require_loaded_physical_grasp(
                target_object,
                loaded_envelope,
                "沿蓝盒方向斜向过渡后",
                use_verified_lift_reference=False,
                loaded_envelope_margin_m=transport_guard_margin,
            )
            if bool(probe.get("object_touching_table", True)):
                raise RuntimeError("斜向过渡后目标仍接触桌面")

        final_probe = self._require_loaded_physical_grasp(
            target_object, loaded_envelope, stage,
            use_verified_lift_reference=False,
            loaded_envelope_margin_m=transport_guard_margin,
        )
        final_position = self._current_object_position(target_object)
        final_horizontal_error = math.hypot(
            final_position[0] - approach_xy[0],
            final_position[1] - approach_xy[1],
        )
        if (
            final_position[2] < minimum_object_z
            or (
                require_box_approach
                and final_horizontal_error > box_xy_tolerance
            )
        ):
            raise TransportHeightError(
                "带载荷自由搬运未到达任务安全位："
                f"object={final_position}, required_z={minimum_object_z:.4f}, "
                f"box_xy_error={final_horizontal_error:.4f} m, "
                f"last_error={last_error}"
            )
        return final_probe

    def _wait_task_flag(
        self, key: str, expected: bool, timeout_s: float
    ) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self._raise_if_stop_requested()
            with self._lock:
                value = bool(self._task_status.get(key, False))
            if value == expected:
                return True
            time.sleep(0.02)
        return False

    def _task_flag(self, key: str) -> bool:
        with self._lock:
            return bool(self._task_status.get(key, False))

    def _command_scene_grasp(self, command: str, target: str = "") -> None:
        with self._lock:
            self._scene_grasp_status = {}
            self._scene_grasp_contact_status = {}
        message = String()
        payload = {"command": command, "target": target}
        if command == "freeze" and self.get_parameter(
            "freeze_target_keep_collisions"
        ).value:
            payload["keep_collisions"] = True
        message.data = json.dumps(
            payload, ensure_ascii=False
        )
        # Publish more than once so a just-started Isaac subscriber cannot miss
        # this safety-critical guided-grasp transition.
        for _ in range(3):
            self._scene_grasp_pub.publish(message)
            time.sleep(0.03)

    def _wait_scene_grasp(
        self, phase: str, target: str, timeout_s: float
    ) -> bool:
        # Isaac processes these commands on simulation updates.  A short
        # wall-clock deadline can expire while a slow-rendering simulation is
        # healthy, so every probe gets a small common latency budget.
        deadline = time.monotonic() + max(timeout_s, 5.0)
        while time.monotonic() < deadline:
            self._raise_if_stop_requested()
            with self._lock:
                status = dict(self._scene_grasp_status)
            if status.get("phase") == "ERROR":
                raise RuntimeError(str(status.get("message", "Isaac 抓取锁定失败")))
            # Isaac can publish PHYSICS_READY immediately after UNFROZEN;
            # its latched status may replace the earlier acknowledgement.
            phase_confirmed = status.get("phase") == phase or (
                phase == "UNFROZEN" and status.get("phase") == "PHYSICS_READY"
            )
            if phase_confirmed and (
                not target or status.get("target") == target
            ):
                if phase == "FROZEN" and self.get_parameter(
                    "freeze_target_keep_collisions"
                ).value and status.get("target_collisions_enabled") is not True:
                    raise RuntimeError("Isaac 未确认固定目标保持真实碰撞")
                return True
            time.sleep(0.02)
        return False

    def _execute_gripper_close_until_obstructed(
        self,
        target: float,
        target_object: str,
        *,
        require_bilateral: bool = False,
    ) -> float:
        """Close slowly and stop on actual bilateral mesh contact or a plateau.

        Actual PhysX stator+mover contact is the preferred stop condition.
        Waiting for a joint plateau after bilateral contact can over-drive the
        fork teeth and eject a round object laterally.  The plateau remains a
        fallback obstruction estimate; the later physical lift is still the
        physical proof.
        """
        if not self._gripper_client.wait_for_server(timeout_sec=2.0):
            raise RuntimeError("夹爪轨迹 action 不可用")
        with self._lock:
            start_position = self._joint_positions.get("jointGripper")
        if start_position is None:
            raise RuntimeError("低速闭爪前没有 jointGripper 状态")

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ["jointGripper"]
        point = JointTrajectoryPoint()
        point.positions = [float(target)]
        duration_nanoseconds = int(self.gripper_close_duration * 1.0e9)
        point.time_from_start.sec = duration_nanoseconds // 1_000_000_000
        point.time_from_start.nanosec = (
            duration_nanoseconds % 1_000_000_000
        )
        goal.trajectory.points = [point]
        send_future = self._gripper_client.send_goal_async(goal)
        goal_handle = self._wait_future(send_future, 5.0)
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError("夹爪控制器拒绝低速闭爪命令")
        with self._lock:
            self._active_goal_handle = goal_handle

        result_future = goal_handle.get_result_async()
        samples = []
        movement_started = False
        bilateral_samples = 0
        previous_contact_pair = (None, None)
        deadline = time.monotonic() + max(
            10.0, self.gripper_close_duration + 4.0
        )
        minimum_impulse_ratio = float(
            OBJECT_GRASP_PROFILES.get(target_object, {}).get(
                "minimum_contact_impulse_ratio", 0.20
            )
        )
        contact_angle_range = OBJECT_GRASP_PROFILES.get(
            target_object, {}
        ).get(
            "gripper_contact_angle_range_rad",
            (
                self.gripper_contact_closed_limit,
                self.gripper_contact_open_limit,
            ),
        )
        try:
            while time.monotonic() < deadline:
                self._raise_if_stop_requested()
                now = time.monotonic()
                with self._lock:
                    actual = self._joint_positions.get("jointGripper")
                    scene_status = dict(self._scene_grasp_status)
                    scene_status.update(self._scene_grasp_contact_status)
                if actual is not None:
                    contact_pair = (
                        scene_status.get("left_finger_contact_object"),
                        scene_status.get("right_finger_contact_object"),
                    )
                    if contact_pair != previous_contact_pair:
                        self.get_logger().info(
                            "Slow-close actual-mesh contact transition: "
                            f"joint={float(actual):.4f}, "
                            f"left={contact_pair[0]}, right={contact_pair[1]}, "
                            "colliders=("
                            f"{scene_status.get('left_finger_contact_collider')}, "
                            f"{scene_status.get('right_finger_contact_collider')})"
                        )
                        previous_contact_pair = contact_pair
                    bilateral_now = _bilateral_obstruction_evidence(
                        scene_status,
                        target_object,
                        minimum_impulse_ratio,
                    )
                    if _bilateral_preload_entry_is_valid(
                        scene_status, target_object
                    ):
                        self._wait_future(
                            goal_handle.cancel_goal_async(), 2.0
                        )
                        self._set_status(
                            "GRIPPER_PRELOAD_READY",
                            "目标物体双指首次同时受力，"
                            "立即停止并执行一次受限预紧",
                            False,
                            gripper_position_rad=round(float(actual), 6),
                            obstruction_estimated=False,
                        )
                        return float(actual)
                    bilateral_samples = (
                        bilateral_samples + 1 if bilateral_now else 0
                    )
                    if bilateral_samples >= 2:
                        self._wait_future(
                            goal_handle.cancel_goal_async(), 2.0
                        )
                        self._set_status(
                            "GRIPPER_MESH_CONTACT",
                            "左右真实夹爪网格已形成连续接触，立即停止闭合并保持",
                            False,
                            gripper_position_rad=round(float(actual), 6),
                            obstruction_estimated=False,
                        )
                        return float(actual)
                    samples.append((now, float(actual)))
                    cutoff = now - self.gripper_obstruction_window
                    samples = [
                        sample for sample in samples if sample[0] >= cutoff
                    ]
                    movement_started = movement_started or (
                        abs(float(actual) - start_position) >= 0.0005
                    )
                    plateau = (
                        movement_started
                        and len(samples) >= 3
                        and samples[-1][0] - samples[0][0]
                        >= self.gripper_obstruction_window * 0.8
                        and max(value for _, value in samples)
                        - min(value for _, value in samples)
                        <= self.gripper_obstruction_max_motion
                    )
                    valid_contact_width = (
                        self.gripper_contact_closed_limit
                        <= float(actual)
                        <= self.gripper_contact_open_limit
                    )
                    away_from_command = (
                        abs(float(actual) - target) > self.gripper_tolerance
                    )
                    if (
                        plateau
                        and valid_contact_width
                        and away_from_command
                        and not require_bilateral
                    ):
                        self._wait_future(
                            goal_handle.cancel_goal_async(), 2.0
                        )
                        self._set_status(
                            "GRIPPER_OBSTRUCTED",
                            "夹爪运动连续受阻，保持当前实际开度；"
                            "该状态不是严格双指接触传感结果",
                            False,
                            gripper_position_rad=round(float(actual), 6),
                            obstruction_estimated=True,
                        )
                        return float(actual)
                if result_future.done():
                    result = result_future.result().result
                    final_position = (
                        float(actual)
                        if actual is not None
                        else math.nan
                    )
                    window_motion = (
                        max(value for _, value in samples)
                        - min(value for _, value in samples)
                        if samples
                        else math.nan
                    )
                    stalled = (
                        movement_started
                        and math.isfinite(window_motion)
                        and window_motion
                        <= self.gripper_obstruction_max_motion
                    )
                    valid_contact_width = (
                        math.isfinite(final_position)
                        and float(contact_angle_range[0])
                        <= final_position
                        <= float(contact_angle_range[1])
                    )
                    away_from_command = (
                        math.isfinite(final_position)
                        and abs(final_position - target)
                        > self.gripper_tolerance
                    )
                    confirmed_samples = 0
                    last_probe = {}
                    if stalled and valid_contact_width and away_from_command:
                        for _ in range(4):
                            self._command_scene_grasp(
                                "probe", target_object
                            )
                            if not self._wait_scene_grasp(
                                "PROBED", target_object, 2.0
                            ):
                                confirmed_samples = 0
                                continue
                            with self._lock:
                                last_probe = dict(self._scene_grasp_status)
                            if _bilateral_preload_entry_is_valid(
                                last_probe, target_object
                            ):
                                self._set_status(
                                    "GRIPPER_PRELOAD_READY",
                                    "控制器已停止且目标物体双指同时受力，"
                                    "立即执行一次受限预紧",
                                    False,
                                    gripper_position_rad=round(
                                        final_position, 6
                                    ),
                                    obstruction_estimated=False,
                                    controller_error_code=int(
                                        result.error_code
                                    ),
                                )
                                return final_position
                            if _bilateral_obstruction_evidence(
                                last_probe,
                                target_object,
                                minimum_impulse_ratio,
                            ):
                                confirmed_samples += 1
                                if confirmed_samples >= 2:
                                    self._set_status(
                                        "GRIPPER_MESH_CONTACT",
                                        "控制器提前结束，但连续双指物理接触确认受阻成功",
                                        False,
                                        gripper_position_rad=round(
                                            final_position, 6
                                        ),
                                        obstruction_estimated=False,
                                        controller_error_code=int(
                                            result.error_code
                                        ),
                                    )
                                    return final_position
                            else:
                                confirmed_samples = 0
                            time.sleep(0.05)
                    raise RuntimeError(
                        "夹爪到达机械闭合目标或控制器提前结束，"
                        "没有检测到物体造成的持续运动受阻："
                        f"{result.error_code} {result.error_string}; "
                        f"start={start_position:.4f}, "
                        f"final={final_position:.4f}, "
                        f"window_motion={window_motion:.4f}, "
                        f"bilateral_confirmed={confirmed_samples}/2, "
                        "contacts=("
                        f"{last_probe.get('left_finger_contact_object')}, "
                        f"{last_probe.get('right_finger_contact_object')})"
                    )
                time.sleep(0.02)
            goal_handle.cancel_goal_async()
            final_position = (
                float(samples[-1][1]) if samples else math.nan
            )
            window_motion = (
                max(value for _, value in samples)
                - min(value for _, value in samples)
                if samples
                else math.nan
            )
            raise RuntimeError(
                "低速闭爪未在时限内检测到持续运动受阻："
                f"start={start_position:.4f}, final={final_position:.4f}, "
                f"window_motion={window_motion:.4f}"
            )
        finally:
            with self._lock:
                self._active_goal_handle = None

    def _execute_gripper(
        self, state: str, target_override: Optional[float] = None
    ) -> float:
        if state not in ("open", "close"):
            raise RuntimeError("夹爪命令只能是 open 或 close")
        if not self._gripper_client.wait_for_server(timeout_sec=2.0):
            raise RuntimeError("夹爪轨迹 action 不可用")
        self._cancel_persistent_gripper_hold()

        target = (
            self.gripper_open
            if state == "open"
            else self.gripper_closed
            if target_override is None
            else float(target_override)
        )
        return self._execute_gripper_target(state, target)

    def _cancel_persistent_gripper_hold(self) -> None:
        """Cancel and fully drain the previous contact-hold trajectory."""
        with self._lock:
            persistent_handle = self._persistent_gripper_goal_handle
            self._persistent_gripper_goal_handle = None
        if persistent_handle is None:
            return
        self._wait_future(persistent_handle.cancel_goal_async(), 2.0)
        # The trajectory bridge serializes gripper goals with a controller
        # lock.  A cancel response only acknowledges the request, so wait for
        # the old execute loop to exit before sending the next preload step.
        self._wait_future(persistent_handle.get_result_async(), 3.0)

    def _execute_gripper_target(
        self, state: str, target: float
    ) -> float:
        label = "打开" if state == "open" else "闭合"
        self._set_status("GRIPPER", f"正在{label}夹爪", False)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ["jointGripper"]
        point = JointTrajectoryPoint()
        point.positions = [target]
        duration = (
            self.gripper_open_duration
            if state == "open"
            else self.gripper_close_duration
        )
        if state == "open":
            with self._lock:
                actual = self._joint_positions.get("jointGripper")
                feedback = self._gripper_feedback
            now = self.get_clock().now().nanoseconds * 1e-9
            if (
                actual is not None and math.isfinite(actual)
                and abs(actual - target) <= min(self.gripper_tolerance, 0.01)
                and feedback is not None
                and 0 <= time.monotonic() - feedback[0] <= 0.2
                and -0.025 <= now - feedback[1] <= 0.1
            ):
                # Refresh the real held target for causal action recording,
                # without replaying the two-second opening motion.
                duration = 0.05
                self.get_logger().info(
                    "Already-open gripper: skipping redundant opening; "
                    f"actual={actual:.4f}, target={target:.4f}, hold=0.05 s"
                )
        duration_nanoseconds = int(duration * 1.0e9)
        point.time_from_start.sec = duration_nanoseconds // 1_000_000_000
        point.time_from_start.nanosec = (
            duration_nanoseconds % 1_000_000_000
        )
        goal.trajectory.points = [point]
        # A real contact grasp intentionally cannot reach the commanded
        # mechanical-limit angle.  The bridge therefore waits through its
        # simulated goal-tolerance window before reporting the stalled jaw;
        # allow for a headless simulator running below real time.
        # Headless RTX rendering may advance simulation below 0.5x real time.
        # Use wall-clock headroom while still validating the measured angle.
        timeout_s = 30.0
        result = None
        action_error = None
        try:
            result = self._send_action_goal(
                self._gripper_client, goal, timeout_s
            )
        except RuntimeError as exc:
            action_error = exc
            with self._lock:
                actual_after_timeout = self._joint_positions.get(
                    "jointGripper"
                )
            reached_safe_state = (
                state == "open"
                and actual_after_timeout is not None
                and actual_after_timeout <= self.gripper_release_threshold
            )
            if not reached_safe_state:
                raise
            self.get_logger().warning(
                "Gripper action timed out in slower-than-real-time simulation, "
                "but the measured joint reached the safe open threshold: "
                f"{actual_after_timeout:.4f} rad ({action_error})"
            )
        if (
            result is not None
            and result.error_code != FollowJointTrajectory.Result.SUCCESSFUL
        ):
            error = self._wait_for_joint_value(
                "jointGripper", target, self.gripper_tolerance, 2.0
            )
            with self._lock:
                actual = self._joint_positions.get("jointGripper")
            contact_closed = (
                state == "close"
                and actual is not None
                and actual >= self.gripper_grasp_threshold
            )
            safely_open = (
                state == "open"
                and actual is not None
                and actual <= self.gripper_release_threshold
            )
            if (
                error > self.gripper_tolerance
                and not contact_closed
                and not safely_open
            ):
                raise RuntimeError(
                    f"夹爪{label}失败：{result.error_code} "
                    f"{result.error_string}，关节误差 {error:.4f} rad，"
                    f"实际位置 {actual} rad"
                )
        with self._lock:
            actual = self._joint_positions.get("jointGripper")
        if actual is None:
            raise RuntimeError("夹爪动作完成后没有 jointGripper 状态")
        self._set_status("HANDED_OVER", f"夹爪已{label}", False)
        return float(actual)

    def _hold_gripper(
        self, position: float, persistent: bool = False
    ) -> None:
        """Hold the physically established jaw width before stabilization.

        The close action targets the mechanical limit.  Once object contact
        has stopped the jaw, removing object collisions for the post-verified
        transport constraint would otherwise let the jaw close through the
        object.  Retargeting the controller to the measured contact angle
        preserves the visible and kinematic grasp.
        """
        if persistent:
            self._cancel_persistent_gripper_hold()
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ["jointGripper"]
        point = JointTrajectoryPoint()
        point.positions = [float(position)]
        point.time_from_start.nanosec = 200_000_000
        goal.trajectory.points = [point]
        if persistent:
            # Keep the same target active well beyond the physical lift and
            # transport.  With only the 0.2 s point, the trajectory controller
            # exceeded its 1 s goal tolerance on intentional object contact,
            # aborted, and switched to holding the measured angle—silently
            # removing the clamp preload before the lift completed.
            hold_point = JointTrajectoryPoint()
            hold_point.positions = [float(position)]
            hold_nanoseconds = int(
                self.gripper_contact_hold_duration * 1.0e9
            )
            hold_point.time_from_start.sec = (
                hold_nanoseconds // 1_000_000_000
            )
            hold_point.time_from_start.nanosec = (
                hold_nanoseconds % 1_000_000_000
            )
            goal.trajectory.points.append(hold_point)
            send_future = self._gripper_client.send_goal_async(goal)
            goal_handle = self._wait_future(send_future, 5.0)
            if goal_handle is None or not goal_handle.accepted:
                raise RuntimeError("夹爪控制器拒绝有限接触预载")
            with self._lock:
                self._persistent_gripper_goal_handle = goal_handle
            # Leave the bounded target active.  Reaching it is not required:
            # the intentional residual position error supplies limited normal
            # force through the configured max effort.
            time.sleep(0.5)
            return
        result = self._send_action_goal(self._gripper_client, goal, 4.0)
        if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            raise RuntimeError(
                "无法保持真实接触夹持宽度："
                f"{result.error_code} {result.error_string}"
            )

    def _wait_for_joint_value(
        self,
        joint_name: str,
        target: float,
        tolerance: float,
        timeout_s: float,
    ) -> float:
        deadline = time.monotonic() + timeout_s
        error = math.inf
        while time.monotonic() < deadline:
            with self._lock:
                actual = self._joint_positions.get(joint_name)
            error = math.inf if actual is None else abs(actual - target)
            if error <= tolerance:
                break
            time.sleep(0.02)
        return error

    @staticmethod
    def _trajectory_score(trajectory) -> Tuple[float, float]:
        if not trajectory.points:
            return math.inf, math.inf
        start = trajectory.points[0].positions
        total = 0.0
        maximum = 0.0
        previous = start
        for point in trajectory.points[1:]:
            step = math.sqrt(
                sum((a - b) ** 2 for a, b in zip(point.positions, previous))
            )
            total += step
            previous = point.positions
        for point in trajectory.points:
            maximum = max(
                maximum,
                max(abs(a - b) for a, b in zip(point.positions, start)),
            )
        return total, maximum

    def _set_bottle_finger_collision(self, enabled: bool) -> None:
        client = self._bottle_collision_client
        if not client.wait_for_service(timeout_sec=3.0):
            raise RuntimeError("瓶子阶段碰撞服务不可用，停止动作")
        request = SetBool.Request()
        request.data = enabled
        result = self._wait_future(client.call_async(request), 6.0)
        if result is None or not result.success:
            raise RuntimeError("瓶子阶段碰撞许可未确认：" + (
                result.message if result is not None else "响应超时"))

    def _execute_plan(self) -> str:
        with self._lock:
            trajectory = copy.deepcopy(self._approved_trajectory)
            approved_target = self._approved_target
            candidate = copy.deepcopy(self._grasp_candidate)
        if trajectory is None or approved_target != self.target_object:
            raise RuntimeError("没有当前目标可执行的预抓取轨迹，请先生成")
        if GRASP_PROFILES[approved_target].get("single_pass_preclose", False):
            self._set_bottle_finger_collision(False)
        approved_profile = GRASP_PROFILES[approved_target]
        if approved_profile.get("freeze_before_approach", False):
            self._command_scene_grasp("freeze", approved_target)
            if not self._wait_scene_grasp("FROZEN", approved_target, 2.0):
                raise RuntimeError("Isaac 未确认接近前准备冻结")
        # Establish an explicit gripper controller target before the arm-only
        # trajectory starts.  Otherwise the trajectory bridge can retain the
        # pre-reset jaw command and silently pull the reset-open gripper closed
        # while moving to the pregrasp pose.
        self._set_status(
            "PREOPEN", "正在预先打开夹爪并建立预抓取保持目标", False
        )
        self._execute_gripper("open")
        approved_profile = GRASP_PROFILES[approved_target]
        target_command = _pregrasp_target_command(approved_profile)
        self._command_scene_grasp(target_command, approved_target)
        expected_phase = "FROZEN" if target_command == "freeze" else "UNFROZEN"
        expected_target = approved_target if target_command == "freeze" else ""
        if not self._wait_scene_grasp(expected_phase, expected_target, 2.0):
            raise RuntimeError(
                "Isaac 未确认预抓取目标场景模式："
                f"command={target_command}, target={approved_target}"
            )
        final_grasp_target = bool(
            candidate and candidate.get("final_grasp_target", False)
        )
        manual_tuning_target = bool(
            candidate and candidate.get("manual_tuning_target", False)
        )
        direct_alignment_target = bool(
            candidate and candidate.get("direct_alignment_target", False)
        )
        self._set_status(
            "EXECUTING",
            (
                "正在执行单次最终定位轨迹"
                if final_grasp_target
                else "正在执行鼠标一次定位手动调试轨迹"
                if manual_tuning_target
                else "正在一次移动到苹果安全对中位"
                if direct_alignment_target
                else "正在移动到预抓取位"
            ),
            False,
        )
        try:
            stopped = self._send_arm_trajectory(trajectory)
        except Exception:
            self._command_scene_grasp("unfreeze")
            raise
        if stopped:
            self._set_status("HANDED_OVER", "预抓取运动已停止", False)
            return "预抓取运动已停止"
        with self._lock:
            self._approved_trajectory = None
            self._approved_target = ""
        message = (
            "已到达闭爪前最终定位姿态"
            if final_grasp_target
            else "已到达鼠标手动调试预夹持位；目标保持固定，请在 RViz 微调"
            if manual_tuning_target
            else "已到达苹果安全对中位"
            if direct_alignment_target
            else "已到达自动抓取预备位"
        )
        self._set_status("HANDED_OVER", message, False)
        return message

    def _send_arm_trajectory(self, trajectory: RobotTrajectory) -> bool:
        if not self._trajectory_client.wait_for_server(timeout_sec=3.0):
            raise RuntimeError("Isaac 轨迹执行 action 不可用")
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory.joint_trajectory
        result = self._send_action_goal(
            self._trajectory_client, goal, timeout_s=90.0
        )
        with self._lock:
            stopped = self._stop_requested
        if stopped:
            return True
        if result.error_code == FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED:
            raise RuntimeError(f"运动保护中止：{result.error_string}")
        if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            error = self._joint_error(trajectory.joint_trajectory)
            if (
                result.error_code
                == FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED
                and not _trajectory_tolerance_enforcement_enabled(
                    self.target_object
                )
            ):
                self.get_logger().warning(
                    f"轨迹收敛判定已暂停（{self.target_object}）："
                    f"{result.error_code} {result.error_string}，"
                    f"最终关节误差 {error:.4f} rad，继续执行后续流程"
                )
                return False
            raise RuntimeError(
                f"预抓取执行失败：{result.error_code} "
                f"{result.error_string}，最终关节误差 {error:.4f} rad"
            )
        return False

    def _send_action_goal(self, client, goal, timeout_s: float):
        self._raise_if_stop_requested()
        send_future = client.send_goal_async(goal)
        goal_handle = self._wait_future(send_future, 5.0)
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError("轨迹控制器拒绝运动命令")
        with self._lock:
            self._active_goal_handle = goal_handle
        try:
            result_wrapper = self._wait_future(
                goal_handle.get_result_async(), timeout_s
            )
            if result_wrapper is None:
                goal_handle.cancel_goal_async()
                raise RuntimeError("轨迹执行超时")
            return result_wrapper.result
        finally:
            with self._lock:
                self._active_goal_handle = None

    def _joint_error(self, trajectory) -> float:
        if not trajectory.points:
            return math.inf
        target = trajectory.points[-1].positions
        with self._lock:
            actual = [
                self._joint_positions.get(name)
                for name in trajectory.joint_names
            ]
        if any(value is None for value in actual):
            return math.inf
        return max(
            abs(float(value) - desired)
            for value, desired in zip(actual, target)
        )

    @staticmethod
    def _wait_future(future, timeout_s: float):
        deadline = time.monotonic() + timeout_s
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        return future.result() if future.done() else None


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GraspAssistant()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
