from z1_isaac_bridge.grasp_config import (
    GRASP_CONFIG,
    cylinder_target_contact_heights_are_valid,
    cylinder_side_contact_is_valid,
    inside_grasp_envelope,
    object_loaded_grasp_is_valid,
    target_bilateral_contact,
)
from z1_isaac_bridge.grasp_evaluator import (
    GraspEvaluator,
    physical_grasp_predicate,
    physical_placement_predicate,
)
from z1_isaac_bridge.episode_recorder import (
    TASK_INSTRUCTIONS,
    episode_save_failures,
    new_episode_evidence,
    update_episode_evidence,
)
import inspect
import math
from pathlib import Path

import pytest

from z1_isaac_bridge.grasp_assistant import (
    GRASP_PROFILES,
    _adaptive_dynamic_step_limit,
    _dynamic_axis_coupling_detected,
    _dynamic_insertion_depth_allows_close,
    _ee_position_for_object_in_stator,
    _ordered_axis_correction,
    GraspAssistant,
    _bounded_axis_correction,
    _bottle_candidate_pitch_deg,
    _bottle_reference_for_orientation,
    _bilateral_obstruction_evidence,
    _safe_bilateral_preload_retry,
    _calibrated_high_cylinder_contact_is_ready,
    _candidate_sweep_blockers,
    _inside_calibrated_grasp_reference,
    _inside_preclose_corridor,
    _guard_dynamic_alignment_depth,
    _far_alignment_guard,
    _is_recoverable_lift_planning_error,
    _local_axis_centering_step,
    _loaded_bilateral_grasp_is_valid,
    _outer_alignment_is_inside_safety_corridor,
    _planar_object_axis_toward_workspace,
    _partial_box_release_is_valid,
    _preclose_envelope_correction,
    _reduced_lift_segment_distance,
    _transport_loaded_grasp_is_valid,
    orientation_for_approach,
    orientation_for_approach_and_pinch_axis,
    rotate_vector,
    _world_gripper_origin_from_object_reference,
)
from z1_isaac_bridge.trajectory_bridge import TrajectoryBridge


def test_dynamic_step_feedback_grows_after_under_tracking_and_recovers():
    assert _adaptive_dynamic_step_limit(
        0.005, 0.005, 0.0005, 0.005, 0.020
    ) == 0.020
    assert _adaptive_dynamic_step_limit(
        0.020, 0.020, 0.016, 0.005, 0.020
    ) == 0.016
    assert _adaptive_dynamic_step_limit(
        0.005, 0.0, 0.0, 0.005, 0.020
    ) == 0.005


def test_far_alignment_guard_slows_then_latches_y_only_recovery():
    assert _far_alignment_guard(
        0.020, 0.005, 0.030, 0.040, 0.030, 0.002, False
    ) == (0.005, False, False)
    assert _far_alignment_guard(
        0.035, 0.005, 0.030, 0.040, 0.030, 0.002, False
    ) == (0.002, False, True)
    assert _far_alignment_guard(
        0.041, 0.005, 0.030, 0.040, 0.030, 0.002, False
    ) == (0.0, True, True)
    assert _far_alignment_guard(
        0.035, 0.005, 0.030, 0.040, 0.030, 0.002, True
    ) == (0.0, True, True)
    assert _far_alignment_guard(
        0.029, 0.005, 0.030, 0.040, 0.030, 0.002, True
    ) == (0.005, False, False)


def test_dynamic_axis_coupling_requires_progress_and_material_cross_talk():
    assert _dynamic_axis_coupling_detected(0.0072, 0.0075)
    assert not _dynamic_axis_coupling_detected(0.0072, 0.0010)
    assert not _dynamic_axis_coupling_detected(-0.0010, 0.0075)


def test_bottle_taught_pose_is_object_relative_and_translation_invariant():
    profile = GRASP_PROFILES["beverage_bottle"]
    reference = profile["calibrated_grasp_reference"][
        "object_in_gripper_local_m"
    ]
    assert reference == [0.12583045, -0.00284520, 0.01877970]
    assert _inside_calibrated_grasp_reference(reference, profile)
    first = _world_gripper_origin_from_object_reference(
        (0.70, 0.14, 0.0), (0.0, 0.0, 0.0, 1.0), reference
    )
    second = _world_gripper_origin_from_object_reference(
        (0.76, 0.10, 0.0), (0.0, 0.0, 0.0, 1.0), reference
    )
    assert tuple(round(b - a, 6) for a, b in zip(first, second)) == (
        0.06,
        -0.04,
        0.0,
    )
    assert profile["near_grasp_elevation_steps_deg"] == ()
    assert profile["desired_center_x_m"] == reference[0]
    assert profile["desired_center_y_m"] == reference[1]
    assert profile["desired_center_z_m"] == reference[2]
    assert profile["single_pass_preclose"] is True
    assert profile["single_alignment_tolerance_m"] == (
        0.006,
        0.006,
        0.006,
    )
    assert profile["single_alignment_tolerance_m"] == tuple(
        profile["calibrated_grasp_reference"]["position_tolerance_m"]
    )


def test_bottle_safe_reference_retains_clearance_at_tolerance_edge():
    profile = GRASP_PROFILES["beverage_bottle"]
    reference = tuple(
        profile["calibrated_grasp_reference"][
            "object_in_gripper_local_m"
        ]
    )
    pitch = math.radians(profile["pitch_deg"])
    orientation = orientation_for_approach(
        (math.cos(pitch), 0.0, math.sin(pitch)), profile["roll_deg"]
    )
    bottle_center_z = GRASP_CONFIG["object_center_z_offsets_m"][
        "beverage_bottle"
    ]
    tcp_local = (0.095, 0.0, 0.021)
    tcp_world_z = bottle_center_z + rotate_vector(
        orientation,
        tuple(tcp_local[i] - reference[i] for i in range(3)),
    )[2]
    assert abs(tcp_world_z - 0.0783526776) < 1e-6

    # Exact lowest-support vertex of the fixed-finger collision STL at +5°.
    lowest_vertex = (0.095628396, 0.040441383, -0.032901041)
    tolerance = profile["single_alignment_tolerance_m"]
    worst_reference = (
        reference[0] + tolerance[0],
        reference[1] - tolerance[1],
        reference[2],
    )
    worst_clearance = bottle_center_z + rotate_vector(
        orientation,
        tuple(
            lowest_vertex[i] - worst_reference[i] for i in range(3)
        ),
    )[2]
    minimum_clearance = GRASP_CONFIG["object_grasp_profiles"][
        "beverage_bottle"
    ]["minimum_table_clearance_m"]
    assert worst_clearance > minimum_clearance


def test_bottle_pitch_survives_pregrasp_height_clamp():
    profile = GRASP_PROFILES["beverage_bottle"]

    class MidpointRng:
        @staticmethod
        def uniform(lower, upper):
            return (lower + upper) / 2.0

    assistant = object.__new__(GraspAssistant)
    assistant._rng = MidpointRng()
    assistant.azimuth_spread = 0.0
    assistant.standoff_min = 0.20
    assistant.standoff_max = 0.23
    assistant.minimum_pregrasp_z = 0.12
    candidate = assistant._sample_candidate(
        (0.55, 0.0, 0.0), "beverage_bottle"
    )
    assert candidate["position"][2] == 0.12
    local_x_world = rotate_vector(
        candidate["orientation"], (1.0, 0.0, 0.0)
    )
    assert profile["hardware_pitch_deg"] == 5.0
    assert profile["pitch_deg"] == -5.0
    assert abs(local_x_world[2] - math.sin(math.radians(-5.0))) < 1e-9


def test_bottle_guided_grasp_does_not_replan_before_close():
    dispatcher = inspect.getsource(GraspAssistant._guided_grasp)
    source = inspect.getsource(
        GraspAssistant._guided_single_pass_bottle_grasp
    )
    assert "single_pass_preclose" in dispatcher
    assert "单次最终定位后未进入闭爪允许区间" in source
    # Candidate planning owns the final pre-close pose.  The only remaining
    # manual pose request is the post-contact lift.
    assert source.count("self._execute_manual_pose(") == 1
    assert "BOTTLE_SINGLE_ALIGNMENT" not in source
    assert "for center_iteration" not in source
    assert "dynamic_alignment_correction_budget" not in source


def test_bottle_uses_one_bounded_servo_stage_without_global_replanning():
    grasp_source = inspect.getsource(
        GraspAssistant._guided_single_pass_bottle_grasp
    )
    servo_source = inspect.getsource(GraspAssistant._servo_correct_bottle_y)
    bridge_source = inspect.getsource(
        TrajectoryBridge._servo_trajectory_callback
    )
    assert grasp_source.count("self._servo_correct_bottle_y(") == 1
    assert "servo_eligible" in grasp_source
    assert "abs(aligned_error[0]) <= tolerance[0]" in grasp_source
    assert "abs(aligned_error[2]) <= tolerance[2]" in grasp_source
    assert "_move_group" not in servo_source
    assert "_plan_candidate" not in servo_source
    assert "_execute_manual_pose" not in servo_source
    assert "_publish_servo_y_twist" in servo_source
    assert "-self.bottle_servo_gain * error_y" in servo_source
    assert "Servo Y 向校正发散" in servo_source
    assert "Servo Y 向校正无有效进展" in servo_source
    assert "commanded_distance >= 0.002" in servo_source
    assert "commanded_distance >= 0.004" in servo_source
    assert "_servo_stream_enabled" in bridge_source
    assert "servo_max_joint_step" in bridge_source


def test_bottle_unprotected_trial_disables_servo_and_preclose_guards():
    init_source = inspect.getsource(GraspAssistant.__init__)
    grasp_source = inspect.getsource(
        GraspAssistant._guided_single_pass_bottle_grasp
    )
    assert 'declare_parameter("bottle_servo_enabled", False)' in init_source
    assert (
        '"bottle_preclose_alignment_guard_enabled", False'
        in init_source
    )
    assert "self.bottle_servo_enabled" in grasp_source
    assert "self.bottle_preclose_alignment_guard_enabled" in grasp_source
    assert "瓶子闭爪前保护已暂停，正在无保护闭爪尝试" in grasp_source


def test_final_bottle_pose_realizes_object_in_stator_calibration():
    object_position = (0.65, -0.25, 0.10)
    ee_orientation = orientation_for_approach(
        (math.cos(math.radians(-5.0)), 0.0, math.sin(math.radians(-5.0))),
        -90.0,
    )
    ee_to_stator = (0.051, 0.0, 0.0)
    desired_local = (0.12583045, -0.00284520, 0.01877970)
    ee_position = _ee_position_for_object_in_stator(
        object_position,
        ee_orientation,
        ee_to_stator,
        (0.0, 0.0, 0.0, 1.0),
        desired_local,
    )
    reconstructed_object = tuple(
        ee_position[index]
        + rotate_vector(ee_orientation, ee_to_stator)[index]
        + rotate_vector(ee_orientation, desired_local)[index]
        for index in range(3)
    )
    assert reconstructed_object == pytest.approx(object_position, abs=1e-9)


def test_eight_bottle_candidates_cover_reachable_pitch_edge():
    pitches = [
        _bottle_candidate_pitch_deg(index, 8, -5.0, -15.0)
        for index in range(8)
    ]
    assert pitches == [-5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -12.5, -15.0]


def test_bottle_reference_preserves_taught_world_height_across_pitch():
    profile = GRASP_PROFILES["beverage_bottle"]
    calibrated = profile["calibrated_grasp_reference"]
    captured = tuple(calibrated["object_in_gripper_local_m"])
    taught_world_z = calibrated[
        "captured_object_center_minus_stator_world_z_m"
    ]
    adjusted_y = []
    for pitch_deg in (-5.0, -10.0, -15.0):
        pitch = math.radians(pitch_deg)
        orientation = orientation_for_approach(
            (math.cos(pitch), 0.0, math.sin(pitch)),
            profile["roll_deg"],
        )
        adjusted = _bottle_reference_for_orientation(
            captured, orientation, taught_world_z
        )
        adjusted_y.append(adjusted[1])
        assert rotate_vector(orientation, adjusted)[2] == pytest.approx(
            taught_world_z, abs=1e-9
        )
    assert adjusted_y == pytest.approx(
        [0.027000995, 0.016261921, 0.005484662], abs=1e-9
    )


def test_balanced_bilateral_contact_confirms_gripper_obstruction():
    status = {
        "contact_report_available": True,
        "left_finger_contact_object": "beverage_bottle",
        "right_finger_contact_object": "beverage_bottle",
        "left_finger_contact_impulse_ns": 0.328853,
        "right_finger_contact_impulse_ns": 0.337070,
        "left_finger_touching_table": False,
        "right_finger_touching_table": False,
    }
    assert _bilateral_obstruction_evidence(
        status, "beverage_bottle", 0.20
    )
    status["right_finger_contact_object"] = None
    assert not _bilateral_obstruction_evidence(
        status, "beverage_bottle", 0.20
    )
    status["right_finger_contact_object"] = "beverage_bottle"
    status["left_finger_touching_table"] = True
    assert not _bilateral_obstruction_evidence(
        status, "beverage_bottle", 0.20
    )
    status["left_finger_touching_table"] = False
    status["left_finger_contact_impulse_ns"] = 0.01
    assert not _bilateral_obstruction_evidence(
        status, "beverage_bottle", 0.20
    )


def test_controller_error_rechecks_physical_bilateral_obstruction():
    source = inspect.getsource(
        GraspAssistant._execute_gripper_close_until_obstructed
    )
    recheck = source.index("confirmed_samples = 0")
    legacy_error = source.index("没有检测到物体造成的持续运动受阻")
    assert recheck < legacy_error
    assert "confirmed_samples >= 2" in source
    assert "stalled and valid_contact_width and away_from_command" in source


def test_only_taught_high_bottle_contact_is_a_ready_cue():
    profile = GRASP_PROFILES["beverage_bottle"]
    cylinder = GRASP_CONFIG["object_grasp_profiles"]["beverage_bottle"]
    status = {
        "local_position_m": [0.12583045, -0.00284520, 0.01877970],
        "left_finger_contact_object": "beverage_bottle",
        "left_finger_contact_point_object_m": [0.0, 0.0, 0.156694],
        "right_finger_contact_object": None,
    }
    assert _calibrated_high_cylinder_contact_is_ready(
        status, "beverage_bottle", profile, cylinder
    )
    status["left_finger_contact_point_object_m"] = [0.0, 0.0, 0.02]
    assert not _calibrated_high_cylinder_contact_is_ready(
        status, "beverage_bottle", profile, cylinder
    )
    status["left_finger_contact_point_object_m"] = [0.0, 0.0, 0.156694]
    status["local_position_m"] = [0.19, -0.00284520, 0.01877970]
    assert not _calibrated_high_cylinder_contact_is_ready(
        status, "beverage_bottle", profile, cylinder
    )


def test_bottle_coarse_approach_precedes_latched_fine_alignment():
    source = inspect.getsource(GraspAssistant._guided_grasp)
    assert "dynamic_fine_alignment_start_x_m" in source
    assert "if not fine_alignment_started:" in source
    assert "coarse_xy_correction" in source
    assert "requested_lateral_step" in source
    assert "vertical_step = 0.0" in source
    assert "axial_lateral_coupling or coupling_observed" in source
    assert "unlimited_fine_adjustment" in source
    assert "itertools.count()" in source
    assert "near_grasp_transition_alignment_y_m" in source
    assert "defer_outer_alignment" in source
    assert "axial_only_before_calibration" in source
    assert "coupled_cartesian_correction = coupled_xy_correction" in source
    assert "xy_joint_space_correction = coarse_xy_correction" in source


def test_guided_grasp_can_resume_only_from_a_valid_open_envelope():
    source = inspect.getsource(GraspAssistant._guided_grasp)
    assert "resume_inside_envelope" in source
    assert 'profile.get(\n            "open_jaw_resume_envelope", profile_envelope\n        )' in source
    assert "inside_grasp_envelope(resume_relative, resume_envelope)" in source
    assert 'not bool(task_status.get("gripper_commanded_closed", False))' in source
    assert "if not pregrasp_is_alignment_pose and not resume_inside_envelope" in source


def test_pregrasp_establishes_open_gripper_target_before_arm_motion():
    source = inspect.getsource(GraspAssistant._execute_plan)
    assert source.index('self._execute_gripper("open")') < source.index(
        "self._send_arm_trajectory(trajectory)"
    )


def test_shared_grasp_envelope_rejects_tip_side_object():
    assert inside_grasp_envelope((0.134, 0.0, 0.017))
    assert inside_grasp_envelope((0.152, 0.0, 0.017))
    assert not inside_grasp_envelope((0.155, 0.0, 0.017))
    assert inside_grasp_envelope((0.134, 0.011, 0.017))
    assert inside_grasp_envelope((0.134, 0.013, 0.017))
    assert inside_grasp_envelope((0.134, 0.016, 0.017))
    assert inside_grasp_envelope((0.134, 0.019, 0.017))
    assert inside_grasp_envelope((0.134, 0.025, 0.017))
    assert inside_grasp_envelope((0.134, 0.039, 0.017))
    assert not inside_grasp_envelope((0.134, 0.041, 0.017))
    assert inside_grasp_envelope((0.134, 0.0, 0.025))
    assert inside_grasp_envelope((0.134, 0.0, 0.030))
    assert not inside_grasp_envelope((0.134, 0.0, 0.031))


def test_bottle_cylinder_contact_requires_body_band_and_balanced_fingers():
    profile = GRASP_CONFIG["object_grasp_profiles"]["beverage_bottle"]
    status = {
        "contact_report_available": True,
        "left_finger_contact_object": "beverage_bottle",
        "right_finger_contact_object": "beverage_bottle",
        "left_finger_contact_point_object_m": [0.0, -0.024, 0.070],
        "right_finger_contact_point_object_m": [0.0, 0.024, 0.080],
        "left_finger_contact_impulse_ns": 0.4,
        "right_finger_contact_impulse_ns": 0.2,
    }

    assert cylinder_side_contact_is_valid(
        status, "beverage_bottle", profile
    )
    assert object_loaded_grasp_is_valid(status, "beverage_bottle")

    # The observed successful grasp placed the moving-finger contact at
    # 0.101235 m.  Retain a narrow 2 mm measurement allowance above the
    # nominal 0.100 m body-band limit.
    status["right_finger_contact_point_object_m"][2] = 0.101235
    assert cylinder_side_contact_is_valid(
        status, "beverage_bottle", profile
    )
    status["right_finger_contact_point_object_m"][2] = 0.080

    unilateral = dict(status)
    unilateral["left_finger_contact_object"] = None
    unilateral["left_finger_contact_point_object_m"] = None
    assert cylinder_target_contact_heights_are_valid(
        unilateral, "beverage_bottle", profile
    )
    unilateral["right_finger_contact_point_object_m"] = [
        -0.0068,
        -0.0231,
        0.1567,
    ]
    assert not cylinder_target_contact_heights_are_valid(
        unilateral, "beverage_bottle", profile
    )

    status["right_finger_contact_point_object_m"][2] = 0.130
    assert not cylinder_side_contact_is_valid(
        status, "beverage_bottle", profile
    )
    status["right_finger_contact_point_object_m"][2] = 0.080
    status["right_finger_contact_impulse_ns"] = 0.05
    assert not cylinder_side_contact_is_valid(
        status, "beverage_bottle", profile
    )


def test_bottle_uses_observed_three_degree_contact_preload():
    profile = GRASP_CONFIG["object_grasp_profiles"]["beverage_bottle"]
    assert math.isclose(
        profile["single_contact_preload_rad"],
        math.radians(3.0),
        abs_tol=1.0e-10,
    )
    assert profile["loaded_samples_required"] == 2
    assert math.isclose(
        profile["single_contact_retry_preload_rad"],
        math.radians(3.0),
        abs_tol=1.0e-10,
    )
    assert profile["single_contact_retry_attempts"] == 1


def test_bottle_retry_preload_requires_safe_real_bilateral_contact():
    envelope = GRASP_CONFIG["object_grasp_profiles"]["beverage_bottle"][
        "loaded_envelope"
    ]
    status = {
        "contact_report_available": True,
        "local_position_m": [0.12372, -0.00978, 0.02011],
        "left_finger_contact_object": "beverage_bottle",
        "right_finger_contact_object": "beverage_bottle",
        "left_finger_contact_collider": "/left/collisions/finger",
        "right_finger_contact_collider": "/right/collisions/finger",
        "left_finger_contact_impulse_ns": 0.0092,
        "right_finger_contact_impulse_ns": 0.0153,
        "left_finger_touching_table": False,
        "right_finger_touching_table": False,
    }
    assert _safe_bilateral_preload_retry(
        status, "beverage_bottle", envelope, 0.015
    )
    unsafe = dict(status, right_finger_touching_table=True)
    assert not _safe_bilateral_preload_retry(
        unsafe, "beverage_bottle", envelope, 0.015
    )
    unilateral = dict(status, left_finger_contact_object=None)
    assert not _safe_bilateral_preload_retry(
        unilateral, "beverage_bottle", envelope, 0.015
    )


def test_physical_grasp_requires_loaded_lift():
    conditions = {
        "gripper_obstructed": True,
        "inside_envelope": True,
        "object_clear_of_table": True,
        "stable_relative_pose": True,
        "object_touching_table": False,
        "finger_touching_table": False,
        "bilateral_contact": False,
        "contact_report_available": False,
    }
    assert physical_grasp_predicate(**conditions)


def test_bilateral_contact_is_required_when_reports_are_available():
    conditions = {
        "gripper_obstructed": True,
        "inside_envelope": True,
        "object_clear_of_table": True,
        "stable_relative_pose": True,
        "object_touching_table": False,
        "finger_touching_table": False,
        "bilateral_contact": False,
        "contact_report_available": True,
    }
    assert not physical_grasp_predicate(**conditions)
    conditions["bilateral_contact"] = True
    assert physical_grasp_predicate(**conditions)


def test_finger_table_contact_rejects_physical_grasp():
    conditions = {
        "gripper_obstructed": True,
        "inside_envelope": True,
        "object_clear_of_table": True,
        "stable_relative_pose": True,
        "object_touching_table": False,
        "finger_touching_table": True,
        "bilateral_contact": True,
        "contact_report_available": True,
    }
    assert not physical_grasp_predicate(**conditions)


def test_physical_placement_requires_verified_release():
    conditions = {
        "physical_grasp_was_verified": True,
        "release_sequence_complete": True,
        "released": True,
        "inside_box": True,
        "object_stationary": True,
        "failure_reason": "",
    }
    assert physical_placement_predicate(**conditions)
    conditions["release_sequence_complete"] = False
    assert not physical_placement_predicate(**conditions)


def test_guided_place_checks_real_contact():
    source = inspect.getsource(GraspAssistant._guided_place)
    assert "_require_loaded_physical_grasp" in source
    assert "自然释放后的最终物理状态" in source


def test_episode_save_uses_latched_lift_evidence_after_release():
    evidence = new_episode_evidence()
    update_episode_evidence(
        evidence,
        {
            "physical_grasp_ok": True,
            "lifted": True,
            "bilateral_contact": True,
        },
    )
    final_status = {
        "physical_grasp_ok": True,
        "lifted": False,
        "bilateral_contact": False,
        "object_touching_table": True,
        "release_sequence_complete": True,
        "task_success": True,
    }
    update_episode_evidence(evidence, final_status)
    assert episode_save_failures(evidence, final_status, True) == []


def test_episode_save_rejects_real_safety_failures_only():
    evidence = new_episode_evidence()
    update_episode_evidence(
        evidence,
        {
            "physical_grasp_ok": True,
            "lifted": False,
            "bilateral_contact": False,
            "release_sequence_complete": True,
            "task_success": True,
        },
    )
    failures = episode_save_failures(
        evidence,
        {
            "task_success": True,
        },
        True,
    )
    assert "object_never_lifted_clear_of_table" in failures
    assert "bilateral_contact_never_observed" in failures


def test_wall_limited_first_open_continues_after_real_release():
    probe = {
        "physics_enabled": True,
        "gravity_enabled": True,
        "contact_report_available": True,
        "bilateral_contact": False,
        "left_finger_contact_object": None,
        "right_finger_contact_object": None,
    }
    assert _partial_box_release_is_valid(
        -0.3938,
        -0.5521,
        probe,
        "apple",
        0.10,
    )
    probe["bilateral_contact"] = True
    probe["left_finger_contact_object"] = "apple"
    probe["right_finger_contact_object"] = "apple"
    assert not _partial_box_release_is_valid(
        -0.3938,
        -0.5521,
        probe,
        "apple",
        0.10,
    )


def test_apple_collision_proxy_is_ellipsoid():
    proxy = GRASP_CONFIG["collision_proxy"]["apple"]
    assert proxy["shape"] == "ellipsoid"
    assert len(proxy["radii_m"]) == 3
    # The full 45.60 mm render bound includes the narrow stem above z=38 mm.
    # Fit the convex proxy to the fruit body, slightly inside its 36.31 x
    # 36.12 mm visual width; the contact offset supplies the remaining skin.
    expected_radii = (0.0176, 0.0175, 0.0190)
    for actual, expected in zip(proxy["radii_m"], expected_radii):
        assert abs(actual - expected) <= 0.0001
    # A complete ellipsoid rests on one vertex and rolls as soon as it becomes
    # dynamic.  Keep a small, hidden support patch at the original bottom.
    assert 0.002 <= proxy["flat_base_height_m"] <= 0.003
    assert proxy["flat_base_height_m"] < proxy["radii_m"][2]
    assert GRASP_CONFIG["collision_proxy"]["contact_offset_m"] <= 0.00025


def test_peach_collision_proxy_matches_render_bounds():
    proxy = GRASP_CONFIG["collision_proxy"]["peach"]
    assert proxy["shape"] == "ellipsoid"
    expected_radii = (0.01545, 0.01515, 0.01450)
    for actual, expected in zip(proxy["radii_m"], expected_radii):
        assert abs(actual - expected) <= 0.0001
    assert abs(
        GRASP_CONFIG["object_center_z_offsets_m"]["peach"] - 0.0145
    ) <= 0.0001
    assert 0.0015 <= proxy["flat_base_height_m"] <= 0.0025
    assert proxy["flat_base_height_m"] < proxy["radii_m"][2]


def test_mouse_collision_proxy_has_stable_widest_sidewall():
    proxy = GRASP_CONFIG["collision_proxy"]["mouse"]
    assert proxy["shape"] == "ellipsoid"
    assert proxy["radii_m"] == [0.043, 0.0245, 0.01415]
    assert proxy["vertical_shape_power"] == 0.55
    assert proxy["flat_base_height_m"] == 0.003


def test_mouse_runtime_contact_offsets_target_the_active_proxy():
    runtime = Path(
        "/home/lp531/isaac_assets/z1/scripts/run_z1_ros2_scene.py"
    ).read_text()
    assert (
        '"mouse": f"{OBJECT_NAME_TO_PRIM[\'mouse\']}/SmolVLACollisionProxy"'
        in runtime
    )
    assert "tuple(ACTIVE_OBJECT_COLLIDER_PATHS.values())" in runtime
    assert "ACTIVE_COLLIDER_BY_OBJECT_ROOT.get(path)" in runtime
    assert "str(prim.GetPath()) == active_collider_path" in runtime


def test_apple_dynamic_insertion_is_incremental_and_stops_inside_envelope():
    profile = GRASP_PROFILES["apple"]
    assert 0.001 <= profile["dynamic_insertion_step_m"] <= 0.003
    assert 0.003 <= profile["dynamic_lateral_step_m"] <= 0.006
    assert "minimum_effective_y_step_m" not in profile
    assert profile["dynamic_alignment_gain_y"] < 1.0
    assert profile["align_before_axial_advance"] is True
    assert 0.006 <= profile["dynamic_alignment_corridor_m"] <= 0.012
    assert (
        GRASP_CONFIG["grasp_envelope"]["x_min_m"]
        < profile["dynamic_insertion_stop_x_m"]
        < GRASP_CONFIG["grasp_envelope"]["x_max_m"]
    )
    assert (
        profile["dynamic_insertion_stop_x_m"]
        > profile["desired_center_x_m"]
    )
    assert profile["desired_center_y_m"] == 0.0
    assert profile["insertion_x_tolerance_m"] == 0.003


def test_reported_apple_pose_is_accepted_as_safe_preclose_corridor():
    profile = GRASP_PROFILES["apple"]
    envelope = GRASP_CONFIG["grasp_envelope"]
    assert _inside_preclose_corridor(
        (0.13939, -0.001777, 0.016198),
        profile,
        envelope,
    )
    assert not _inside_preclose_corridor(
        (0.13939, 0.011, 0.016198),
        profile,
        envelope,
    )


def test_mouse_real_contact_accepts_the_measured_physical_stop():
    profile = GRASP_PROFILES["mouse"]
    envelope = dict(GRASP_CONFIG["grasp_envelope"])
    envelope.update(profile["grasp_envelope_overrides"])
    low_contact_pose = (0.167743, -0.002374, 0.004382)
    contacted_pose = (0.169892, 0.002058, 0.035)
    assert not _inside_preclose_corridor(
        contacted_pose, profile, envelope
    )
    assert not _inside_preclose_corridor(
        low_contact_pose, profile, envelope, contact_present=True
    )
    assert _inside_preclose_corridor(
        contacted_pose, profile, envelope, contact_present=True
    )


def test_bottle_real_contact_accepts_the_taught_grasp_pose():
    profile = GRASP_PROFILES["beverage_bottle"]
    envelope = dict(profile["grasp_envelope_overrides"])
    contacted_pose = (0.12583045, -0.00284520, 0.01877970)
    assert _inside_preclose_corridor(
        contacted_pose, profile, envelope, contact_present=False
    )
    assert _inside_preclose_corridor(
        contacted_pose, profile, envelope, contact_present=True
    )
    assert not _inside_preclose_corridor(
        (0.145, -0.00284520, 0.01877970),
        profile,
        envelope,
        contact_present=True,
    )


def test_bottle_closes_from_measured_capture_envelope_before_xy_coupling():
    profile = GRASP_PROFILES["beverage_bottle"]
    strict_envelope = dict(profile["grasp_envelope_overrides"])
    loaded_envelope = GRASP_CONFIG["object_grasp_profiles"][
        "beverage_bottle"
    ]["loaded_envelope"]

    assert profile["disable_preclose_bilateral_height_correction"] is True
    assert (
        profile["preclose_capture_envelope"]["x_max_m"]
        <= loaded_envelope["x_max_m"]
    )
    assert (
        profile["preclose_capture_envelope"]["y_abs_max_m"]
        <= loaded_envelope["y_abs_max_m"]
    )
    assert (
        profile["preclose_capture_envelope"]["z_max_m"]
        <= loaded_envelope["z_max_m"]
    )

    measured_capture_pose = (0.12583045, -0.00284520, 0.01877970)
    centered_capture_pose = (0.130, 0.003, 0.022)
    assert inside_grasp_envelope(
        measured_capture_pose, strict_envelope
    )
    assert _inside_preclose_corridor(
        measured_capture_pose, profile, strict_envelope
    )
    assert _inside_preclose_corridor(
        centered_capture_pose, profile, strict_envelope
    )
    # This controlled trial allows no-contact closure throughout the widened
    # capture volume, but never beyond the already verified loaded envelope.
    assert _dynamic_insertion_depth_allows_close(
        measured_capture_pose, profile, contact_present=False
    )
    assert _dynamic_insertion_depth_allows_close(
        (0.1317, 0.003, 0.022),
        profile,
        contact_present=False,
    )
    assert not _dynamic_insertion_depth_allows_close(
        (0.1320, 0.0, 0.022),
        profile,
        contact_present=False,
    )
    assert _dynamic_insertion_depth_allows_close(
        measured_capture_pose, profile, contact_present=True
    )

    # The transition-to-close volume is bounded and does not become another
    # broad recovery corridor.
    assert not _inside_preclose_corridor(
        (0.1320, -0.00284520, 0.01877970), profile, strict_envelope
    )
    assert not _inside_preclose_corridor(
        (0.160, 0.0151, 0.031), profile, strict_envelope
    )
    assert not _inside_preclose_corridor(
        (0.160, 0.0, 0.043), profile, strict_envelope
    )


def test_closed_mesh_lateral_search_moves_local_error_toward_zero():
    assert _local_axis_centering_step(-0.0023, 0.002) == -0.002
    assert _local_axis_centering_step(0.0023, 0.002) == 0.002
    assert _local_axis_centering_step(0.0, 0.002) == 0.0


def test_preclose_clearance_boundary_error_gets_bounded_local_correction():
    envelope = GRASP_CONFIG["grasp_envelope"]
    reported_local = (0.154252, 0.000757, 0.016249)
    assert not inside_grasp_envelope(reported_local, envelope)
    correction = _preclose_envelope_correction(
        reported_local,
        envelope,
        GRASP_CONFIG["unfreeze"][
            "clearance_correction_target_inset_m"
        ],
        GRASP_CONFIG["unfreeze"]["clearance_correction_step_m"],
    )
    assert 0.0012 <= correction[0] <= 0.0013
    assert correction[1:] == (0.0, 0.0)
    corrected_local = tuple(
        value - delta
        for value, delta in zip(reported_local, correction)
    )
    assert inside_grasp_envelope(corrected_local, envelope)
    assert abs(corrected_local[0] - 0.153) <= 1.0e-9


def test_preclose_clearance_correction_is_step_limited():
    envelope = GRASP_CONFIG["grasp_envelope"]
    step = GRASP_CONFIG["unfreeze"]["clearance_correction_step_m"]
    correction = _preclose_envelope_correction(
        (0.160, 0.045, 0.035),
        envelope,
        GRASP_CONFIG["unfreeze"][
            "clearance_correction_target_inset_m"
        ],
        step,
    )
    assert sum(value * value for value in correction) ** 0.5 <= step


def test_preclose_clearance_uses_closed_loop_safety_rechecks():
    source = inspect.getsource(GraspAssistant._guided_grasp)
    start = source.index("correction_object_origin")
    end = source.index(
        "# Enter above the fixed finger's inner support face"
    )
    correction_source = source[start:end]

    assert "while not inside_grasp_envelope" in correction_source
    assert "_preclose_envelope_correction" in correction_source
    assert "_raise_if_finger_touching_table" in correction_source
    assert "left_finger_contact_object" in correction_source
    assert "right_finger_contact_object" in correction_source
    assert "correction_object_drift" in correction_source
    assert "preclose_clearance_correction_max_attempts" in correction_source


def test_final_insertion_segment_is_probed_before_failure():
    source = inspect.getsource(GraspAssistant._guided_grasp)
    assert "复检后仍不在" in source
    assert "安全可夹取区间" in source
    assert "最大段数后的复检状态" in source


def test_reported_loaded_apple_pose_continues_to_physical_lift():
    envelope = GRASP_CONFIG["grasp_envelope"]
    local_position = (0.143956, 0.001206, 0.004312)
    assert not inside_grasp_envelope(local_position, envelope)
    probe = {
        "local_position_m": local_position,
        "physics_enabled": True,
        "gravity_enabled": True,
        "contact_report_available": True,
        "bilateral_contact": True,
        "left_finger_contact_object": "apple",
        "right_finger_contact_object": "apple",
    }
    assert _loaded_bilateral_grasp_is_valid(
        probe,
        "apple",
        envelope,
        GRASP_CONFIG["physical_test_lift"]["loaded_envelope_margin_m"],
    )
    probe["right_finger_contact_object"] = None
    assert not _loaded_bilateral_grasp_is_valid(
        probe,
        "apple",
        envelope,
        GRASP_CONFIG["physical_test_lift"]["loaded_envelope_margin_m"],
    )


def test_valid_closed_grasp_has_no_optional_pre_lift_axial_motion():
    source = inspect.getsource(GraspAssistant._guided_grasp)
    assert "prelift_axial_seating_steps" not in source
    assert "保持双指夹紧并增加抬升前轴向包络余量" not in source
    assert (
        source.index("Verified sustained bilateral contact before lift")
        < source.index("Loaded grasp is valid")
        < source.index('stage = "纯物理试抬"')
    )


def test_atomic_contact_pair_overrides_contradictory_derived_boolean():
    probe = {
        "contact_report_available": True,
        "left_finger_contact_object": "apple",
        "right_finger_contact_object": "apple",
        "bilateral_contact": False,
        "bilateral_contact_consecutive_frames": 2126,
        "bilateral_contact_required_frames": 10,
    }
    assert target_bilateral_contact(probe, "apple")
    assert target_bilateral_contact(
        probe, "apple", require_sustained=True
    )
    probe["right_finger_contact_object"] = None
    assert not target_bilateral_contact(probe, "apple")


def test_transport_accepts_reported_gravity_settling_after_verified_lift():
    envelope = GRASP_CONFIG["grasp_envelope"]
    local_position = (0.129349, -0.003623, -0.011419)
    probe = {
        "local_position_m": local_position,
        "physics_enabled": True,
        "gravity_enabled": True,
        "contact_report_available": True,
        "bilateral_contact": True,
        "left_finger_contact_object": "apple",
        "right_finger_contact_object": "apple",
        "object_touching_table": False,
    }
    # This is the real false-failure observation: it is only 2.4 mm below the
    # strict loaded envelope, while both fingers have continuously carried it.
    assert not _loaded_bilateral_grasp_is_valid(
        probe,
        "apple",
        envelope,
        GRASP_CONFIG["physical_test_lift"]["loaded_envelope_margin_m"],
    )
    assert _transport_loaded_grasp_is_valid(
        probe,
        "apple",
        envelope,
        GRASP_CONFIG["physical_test_lift"]["transport_guard_margin_m"],
        (0.1290, -0.0035, -0.0090),
        GRASP_CONFIG["physical_test_lift"][
            "relative_position_tolerance_m"
        ],
    )


def test_transport_rejects_real_grasp_loss_and_gross_pose_drift():
    envelope = GRASP_CONFIG["grasp_envelope"]
    lift = GRASP_CONFIG["physical_test_lift"]
    reference = (0.1290, 0.0, -0.0090)
    probe = {
        "local_position_m": (0.1290, 0.0, -0.0100),
        "physics_enabled": True,
        "gravity_enabled": True,
        "contact_report_available": True,
        "bilateral_contact": True,
        "left_finger_contact_object": "apple",
        "right_finger_contact_object": "apple",
        "object_touching_table": False,
    }
    arguments = (
        "apple",
        envelope,
        lift["transport_guard_margin_m"],
        reference,
        lift["relative_position_tolerance_m"],
    )
    probe["right_finger_contact_object"] = None
    assert not _transport_loaded_grasp_is_valid(probe, *arguments)
    probe["right_finger_contact_object"] = "apple"
    probe["object_touching_table"] = True
    assert not _transport_loaded_grasp_is_valid(probe, *arguments)
    probe["object_touching_table"] = False
    probe["local_position_m"] = (0.1800, 0.0, -0.0100)
    assert not _transport_loaded_grasp_is_valid(probe, *arguments)


def test_training_objects_use_collision_safe_object_specific_entry():
    for name, profile in GRASP_PROFILES.items():
        if name == "cup":
            assert profile["elevation_deg"] == (25.0, 25.0)
            assert profile["roll_deg"] == 120.0
        elif name == "beverage_bottle":
            # Legacy elevation is downward-positive; hardware pitch is
            # positive upward along the local +X approach axis.
            assert profile["hardware_pitch_deg"] == 5.0
            assert profile["pitch_deg"] == -5.0
            assert profile["elevation_deg"] == (-5.0, -5.0)
            assert profile["near_grasp_elevation_steps_deg"] == ()
            assert profile["near_grasp_transition_x_m"] == 0.145
            assert profile["dynamic_fine_alignment_start_x_m"] == 0.155
            assert profile["defer_calibrated_alignment_until_x_m"] == 0.155
            assert profile["axial_only_before_calibrated_alignment"] is True
            assert profile["unlimited_fine_adjustment"] is True
            assert profile["coarse_alignment_y_guard_m"] == 0.010
            assert profile["assume_dynamic_xy_coupling"] is False
            assert profile["dynamic_alignment_abort_corridor_m"] == 0.050
            assert profile["far_alignment_soft_corridor_m"] == 0.030
            assert profile["far_alignment_recovery_trigger_m"] == 0.040
            assert profile["far_alignment_recovery_release_m"] == 0.030
            assert profile["far_alignment_reduced_axial_step_m"] == 0.005
            assert profile["near_grasp_transition_alignment_y_m"] == 0.015
            assert profile["near_grasp_transition_alignment_z_m"] == 0.012
            assert (
                profile["dynamic_fine_alignment_start_x_m"]
                > profile["near_grasp_transition_x_m"]
            )
            transition_envelope = profile[
                "near_grasp_transition_envelope"
            ]
            assert transition_envelope["x_min_m"] == 0.135
            assert transition_envelope["x_max_m"] == 0.180
            assert profile["roll_deg"] == -90.0
            assert profile["world_z_offset_m"] == 0.0
            assert profile["insertion_center_x_m"] == 0.12583045
            assert profile["dynamic_insertion_stop_x_m"] == 0.12583045
            assert profile["maximum_insertion_distance_m"] == 0.110
            assert profile["dynamic_insertion_step_m"] == 0.005
            assert profile["maximum_dynamic_insertion_step_m"] == 0.020
            assert profile["dynamic_feedback_min_progress_ratio"] == 0.35
            assert profile["dynamic_feedback_joint_space_after_stalls"] == 2
            assert profile["desired_center_x_m"] == 0.12583045
            assert profile["desired_center_y_m"] == -0.00284520
            assert profile["lock_outer_alignment_orientation"] is True
            assert profile["lock_dynamic_insertion_orientation"] is True
            assert profile["outer_alignment_orientation_tolerance_rad"] <= 0.006
            assert profile["insertion_orientation_tolerance_rad"] <= 0.006
            assert profile["guard_dynamic_alignment_depth"] is False
            assert profile["align_before_axial_advance"] is True
            assert profile["outer_alignment_iterations"] == 20
            assert profile["outer_alignment_max_y_step_m"] == 0.015
            assert profile["alignment_axis_order"] == ("z", "y", "x")
            assert profile["alignment_tolerance_y_m"] == 0.006
            assert profile["alignment_tolerance_z_m"] == 0.006
            assert profile["outer_alignment_acceptance_y_m"] == 0.008
            assert profile["insertion_alignment_tolerance_y_m"] == 0.006
            assert profile["insertion_alignment_tolerance_z_m"] == 0.006
            assert profile["dynamic_lateral_step_m"] == 0.015
            assert profile["dynamic_vertical_step_m"] == 0.010
            assert profile["dynamic_alignment_gain_y"] == 1.5
            assert profile["contact_preclose_x_tolerance_m"] == 0.005
            assert profile["contact_alignment_tolerance_z_m"] == 0.010
            assert profile["contact_preclose_center_z_m"] == 0.01877970
            assert profile["preclose_contact_center_z_m"] == 0.01877970
            assert profile["entry_center_z_m"] == 0.01877970
            assert profile["desired_center_z_m"] == 0.01877970
            envelope = profile["grasp_envelope_overrides"]
            assert envelope["z_min_m"] == 0.014
            assert envelope["z_max_m"] == 0.02477970
            assert inside_grasp_envelope(
                tuple(
                    profile["calibrated_grasp_reference"][
                        "object_in_gripper_local_m"
                    ]
                ),
                envelope,
            )
            assert profile["dynamic_alignment_world_z_floor_margin_m"] == 0.005
            assert profile["minimum_object_rise_m"] == 0.030
            assert profile["transport_object_z_m"] == 0.150
            assert profile["gripper_contact_closed_limit_rad"] == -0.75
            assert profile["gripper_contact_open_limit_rad"] == -0.20
        elif name == "mouse":
            assert profile["elevation_deg"] == (75.0, 75.0)
            assert profile["roll_deg"] == 0.0
            assert profile["height"] == 0.020
            assert profile["use_object_axis_approach"] is True
            assert profile["object_approach_local_axis"] == (0.0, 1.0, 0.0)
            assert "object_pinch_local_axis" not in profile
            assert profile["reverse_object_axis_approach"] is False
            assert profile["alignment_axis_order"] == ("z", "y", "x")
            assert profile["decouple_outer_alignment"] is True
            assert profile["decouple_dynamic_alignment"] is True
            assert profile["alignment_tolerance_y_m"] == 0.002
            assert profile["alignment_tolerance_z_m"] == 0.002
            assert profile["grasp_envelope_overrides"]["z_min_m"] == 0.0
            assert profile["grasp_envelope_overrides"]["z_max_m"] == 0.046
            assert profile["grasp_envelope_overrides"]["x_max_m"] == 0.172
            assert (
                profile["grasp_envelope_overrides"]["y_abs_max_m"]
                == 0.010
            )
            assert profile["insertion_center_x_m"] == 0.155
            assert profile["dynamic_insertion_stop_x_m"] == 0.158
            assert profile["desired_center_x_m"] == 0.155
            assert profile["insertion_x_tolerance_m"] == 0.002
            assert profile["contact_preclose_x_tolerance_m"] == 0.019
            assert profile["contact_alignment_tolerance_y_m"] == 0.010
            assert profile["contact_alignment_tolerance_z_m"] == 0.005
            assert profile["contact_preclose_center_z_m"] == 0.035
            assert profile["preclose_contact_center_z_m"] == 0.035
            assert (
                profile[
                    "disable_preclose_bilateral_height_correction"
                ]
                is True
            )
            assert profile["preclose_partial_gripper_rad"] == -0.55
            assert profile["preclose_axial_seating_step_m"] == 0.001
            assert (
                profile["preclose_axial_seating_max_translation_m"]
                == 0.0
            )
            assert profile["preclose_axial_seating_min_x_m"] == 0.158
            assert profile["preclose_axial_seating_y_limit_m"] == 0.008
            assert profile["preclose_axial_seating_z_limit_m"] == 0.006
            assert profile["require_bilateral_close_obstruction"] is False
            assert profile["contact_hold_preload_rad"] == 0.030
            assert profile["contact_hold_preload_step_rad"] == 0.002
            assert profile["require_preclose_target_contact"] is False
            assert profile["contact_search_min_x_m"] == 0.152
            assert profile["dynamic_alignment_correction_budget"] == 24
            assert profile["approach_beyond_target_m"] == 0.060
            assert profile["minimum_object_rise_m"] == 0.100
            assert profile["transport_object_z_m"] == 0.145
            assert profile["physical_test_lift_segment_m"] == 0.010
            assert (
                profile["physical_test_lift_near_table_segment_m"]
                == 0.005
            )
            assert (
                profile["physical_test_lift_near_table_until_z_m"]
                == 0.055
            )
            assert (
                profile["physical_test_lift_near_target_segment_m"]
                == 0.005
            )
            assert (
                profile["physical_test_lift_near_target_threshold_m"]
                == 0.020
            )
            assert (
                profile["physical_test_lift_position_tolerance_m"]
                == 0.001
            )
            assert (
                profile[
                    "physical_test_lift_near_table_position_tolerance_m"
                ]
                == 0.002
            )
            assert (
                profile["physical_test_lift_max_joint_delta_rad"]
                == 0.15
            )
            assert profile[
                "physical_test_lift_orientation_tolerances_rad"
            ] == (0.04, 0.08, 0.12)
            assert profile[
                "physical_test_lift_near_table_orientation_tolerances_rad"
            ] == (0.005, 0.010, 0.020)
            assert profile[
                "physical_test_lift_near_target_orientation_tolerances_rad"
            ] == (0.005, 0.010, 0.020)
            assert (
                profile[
                    "physical_test_lift_bounded_fallback_max_joint_delta_rad"
                ]
                == 0.15
            )
            assert profile["dynamic_insertion_step_m"] == 0.001
            assert profile["entry_center_z_m"] == 0.030
            assert profile["dynamic_entry_center_z_until_x_m"] == 0.150
            assert profile["insertion_alignment_tolerance_y_m"] == 0.003
            assert profile["desired_center_z_m"] == 0.014
            assert profile["insertion_alignment_tolerance_z_m"] == 0.002
            assert profile["dynamic_axial_feedforward_z_m"] == 0.0
            assert profile["lock_outer_alignment_orientation"] is True
            assert (
                profile["outer_alignment_orientation_tolerance_rad"]
                <= 0.006
            )
            assert profile["lock_dynamic_insertion_orientation"] is True
            assert profile["insertion_orientation_tolerance_rad"] <= 0.006
        else:
            assert profile["elevation_deg"] == (75.0, 75.0)
            assert profile["roll_deg"] == 0.0
        assert profile["azimuth_spread_deg"] == 0.0
        if name == "cup":
            assert profile["dynamic_insertion_step_m"] <= 0.005
            assert profile["decouple_dynamic_alignment"] is True
            assert profile["insertion_orientation_tolerance_rad"] <= 0.01
        elif name == "mouse":
            assert profile["dynamic_insertion_step_m"] <= 0.003
        elif name != "beverage_bottle":
            assert profile["dynamic_insertion_step_m"] <= 0.002
        envelope_x_max = profile.get(
            "grasp_envelope_overrides", {}
        ).get(
            "x_max_m",
            GRASP_CONFIG["grasp_envelope"]["x_max_m"],
        )
        assert profile["dynamic_insertion_stop_x_m"] < envelope_x_max
        assert profile.get("max_joint_delta_rad", 0.0) <= 2.70
        assert (
            profile.get("outer_alignment_max_joint_delta_rad", 0.0)
            <= 0.65
        )
        maximum_y_tolerance = 0.009 if name == "beverage_bottle" else 0.008
        assert (
            profile.get("alignment_tolerance_y_m", 0.0)
            <= maximum_y_tolerance
        )
        maximum_z_tolerance = (
            0.010 if name == "beverage_bottle" else 0.004
        )
        assert (
            profile.get("alignment_tolerance_z_m", 0.0)
            <= maximum_z_tolerance
        )
        assert profile.get("minimum_insertion_distance_m", 0.02) >= 0.005


def test_peach_keeps_size_offset_but_uses_apple_style_safe_entry():
    peach = GRASP_PROFILES["peach"]
    apple = GRASP_PROFILES["apple"]

    assert peach["insertion_center_x_m"] == 0.150
    assert peach["dynamic_insertion_stop_x_m"] == 0.155
    assert peach["desired_center_x_m"] == 0.150
    assert peach["minimum_insertion_distance_m"] == 0.005
    assert peach["maximum_insertion_distance_m"] == 0.12
    assert peach["desired_center_x_m"] < 0.160
    assert peach["desired_center_x_m"] > apple["desired_center_x_m"]
    assert peach["entry_center_z_m"] == 0.015
    assert peach["desired_center_z_m"] == 0.015
    assert peach["preclose_contact_center_z_m"] == 0.021
    assert peach["outer_alignment_iterations"] >= 6
    assert peach["decouple_outer_alignment"] is True
    assert peach["decouple_dynamic_alignment"] is True
    assert "outer_alignment_min_effective_y_step_m" not in peach
    assert peach["alignment_tolerance_y_m"] == 0.003
    assert peach["alignment_tolerance_z_m"] == 0.003
    assert peach["insertion_alignment_tolerance_y_m"] == 0.004
    assert peach["insertion_alignment_tolerance_z_m"] == 0.0045
    assert peach["allow_preclose_target_contact"] is False
    assert peach["recover_unilateral_preclose_contact"] is True
    assert peach["preclose_contact_recovery_retreat_m"] == 0.006
    assert peach["preclose_contact_recovery_lift_m"] == 0.005
    assert peach["disable_preclose_live_centering"] is False
    assert peach["preclose_clearance_lift_m"] == 0.005
    assert peach["guard_dynamic_alignment_depth"] is True
    assert peach["dynamic_alignment_centering_depth_guard_m"] == 0.006
    assert (
        peach["dynamic_alignment_safe_pose_max_joint_delta_rad"]
        == 0.65
    )
    assert peach["dynamic_alignment_world_z_floor_margin_m"] == 0.003
    assert peach["transport_object_z_m"] == 0.170
    assert peach["physical_test_lift_completion_tolerance_m"] == 0.010
    assert "minimum_object_rise_m" not in peach
    assert peach["physical_test_lift_orientation_tolerances_rad"] == (
        0.04,
        0.08,
        0.12,
    )
    assert (
        peach[
            "physical_test_lift_bounded_fallback_max_joint_delta_rad"
        ]
        == 0.35
    )
    assert (
        peach["desired_center_z_m"]
        < apple["desired_center_z_m"]
    )


def test_shared_closed_loop_alignment_never_amplifies_small_residuals():
    assert _bounded_axis_correction(0.0019, 0.002, 0.85, 0.008) == 0.0
    correction = _bounded_axis_correction(
        0.0021, 0.002, 0.85, 0.008
    )
    assert 0.0 < correction < 0.0021
    assert _bounded_axis_correction(0.020, 0.002, 0.85, 0.008) == 0.008
    assert (
        _bounded_axis_correction(-0.020, 0.002, 0.85, 0.008)
        == -0.008
    )


def test_outer_alignment_hands_recoverable_residual_to_collision_loop():
    assert _outer_alignment_is_inside_safety_corridor(
        0.003809,
        0.005272,
        0.030,
    )
    assert _outer_alignment_is_inside_safety_corridor(
        0.020,
        0.025,
        0.030,
    )
    assert not _outer_alignment_is_inside_safety_corridor(
        0.031,
        0.002,
        0.030,
    )


def test_dynamic_alignment_retreats_before_centering_at_grasp_depth():
    guarded = _guard_dynamic_alignment_depth(
        0.1552,
        0.155,
        True,
        False,
        0.0,
        -0.005,
        0.0,
        0.006,
        0.008,
    )
    assert guarded == (-0.006, 0.0, 0.0, True)
    unguarded = _guard_dynamic_alignment_depth(
        0.170,
        0.155,
        True,
        False,
        0.0,
        -0.005,
        0.0,
        0.006,
        0.008,
    )
    assert unguarded == (0.0, -0.005, 0.0, False)


def test_each_object_declares_recoverable_closed_loop_geometry():
    for profile in GRASP_PROFILES.values():
        assert profile["outer_alignment_iterations"] >= 6
        assert (
            profile["outer_alignment_moveit_position_tolerance_m"]
            < profile["alignment_tolerance_y_m"]
        )
        assert (
            profile["outer_alignment_acceptance_y_m"]
            >= profile["alignment_tolerance_y_m"]
        )
        assert (
            profile["outer_alignment_acceptance_z_m"]
            >= profile["alignment_tolerance_z_m"]
        )
        assert profile["final_alignment_iterations"] >= 4
        assert "minimum_effective_y_step_m" not in profile
        assert "outer_alignment_min_effective_y_step_m" not in profile


def test_mouse_short_axis_follows_randomized_object_yaw():
    half_sqrt = 2.0 ** -0.5
    axis = _planar_object_axis_toward_workspace(
        (0.0, 0.0, half_sqrt, half_sqrt),
        (0.0, 1.0, 0.0),
        (0.50, -0.05, 0.0),
    )
    assert abs(axis[0] - 1.0) <= 1.0e-6
    assert abs(axis[1]) <= 1.0e-6


def test_mouse_steep_candidate_aligns_closing_axis_to_short_axis():
    direction = (0.25, 0.0, -0.9682458366)
    short_axis = (0.0, 1.0, 0.0)
    orientation = orientation_for_approach_and_pinch_axis(
        direction, short_axis
    )

    world_closing_axis = rotate_vector(orientation, (0.0, 0.0, 1.0))
    assert abs(world_closing_axis[0]) <= 1.0e-6
    assert abs(world_closing_axis[1] - 1.0) <= 1.0e-6
    assert abs(world_closing_axis[2]) <= 1.0e-6


def test_mouse_alignment_finishes_z_then_y_then_x():
    profile = GRASP_PROFILES["mouse"]
    axis_order = profile["alignment_axis_order"]
    correction = (0.002, -0.003, 0.004)

    assert _ordered_axis_correction(
        correction, (True, True, True), axis_order
    ) == (0.0, 0.0, 0.004)
    assert _ordered_axis_correction(
        correction, (True, True, False), axis_order
    ) == (0.0, -0.003, 0.0)
    assert _ordered_axis_correction(
        correction, (True, False, False), axis_order
    ) == (0.002, 0.0, 0.0)


def test_selected_target_may_touch_the_complete_wrist_end_effector():
    source = Path(
        __file__
    ).parents[1] / "z1_isaac_bridge" / "object_collision_sync.py"
    text = source.read_text(encoding="utf-8")

    assert (
        '["link06", "camera_link", "gripperStator", "gripperMover"]'
        in text
    )


def test_guided_grasp_unfreezes_at_outer_alignment_before_real_collision_entry():
    source = inspect.getsource(GraspAssistant._guided_grasp)
    # One transition enters real collision; the second is idempotent cleanup
    # on an exception now that the old detach command no longer exists.
    assert source.count('_command_scene_grasp("unfreeze"') == 2
    assert source.index("外侧安全对中后恢复真实碰撞") < source.index(
        "真实碰撞分段预抓取接近"
    )
    assert "outer_alignment_iterations" in source
    assert "outer_alignment_indices" in source
    assert "for alignment_iteration in outer_alignment_indices" in source
    assert "_current_stator_orientation()" in source
    assert (
        GRASP_CONFIG["gripper"]["require_clear_fingers_before_close"]
        is True
    )
    assert GRASP_CONFIG["unfreeze"]["max_position_drift_m"] <= 0.005
    assert (
        GRASP_CONFIG["unfreeze"]["clearance_correction_max_attempts"]
        == 4
    )
    assert (
        GRASP_CONFIG["unfreeze"]["clearance_correction_step_m"]
        <= 0.002
    )


def test_adaptive_preload_is_finite_and_incremental():
    gripper = GRASP_CONFIG["gripper"]
    assert 0.0 < gripper["contact_hold_preload_step_rad"]
    assert (
        gripper["contact_hold_preload_step_rad"]
        < gripper["contact_hold_preload_rad"]
    )
    assert gripper["contact_hold_preload_rad"] <= 0.15
    assert 0.0 < gripper["max_effort"] <= 800.0
    assert gripper["allow_retention_lip"] is False
    # The gripper now establishes 15 mm of table clearance before closing.
    # A second lift after bilateral contact breaks the stator contact.
    assert gripper["contact_unjam_lift_m"] == 0.0
    assert gripper["contact_hold_duration_s"] >= 30.0
    assert 0.0 <= gripper["contact_stator_seating_m"] <= 0.001
    assert gripper["preclose_seating_step_m"] <= 0.001
    assert gripper["preclose_seating_max_translation_m"] <= 0.010


def test_continuous_lift_keeps_final_loaded_evidence():
    lift = GRASP_CONFIG["physical_test_lift"]
    assert "distance_m" not in lift
    assert "minimum_object_rise_m" not in lift
    assert "verification_samples_required" not in lift
    assert "verification_max_attempts" not in lift
    assert lift["loaded_grasp_samples_required"] >= 2
    assert (
        lift["loaded_grasp_max_attempts"]
        >= 2 * lift["loaded_grasp_samples_required"]
    )
    assert 0.05 <= lift["verification_poll_period_s"] <= 0.20
    assert 0.0 < lift["loaded_envelope_margin_m"] <= 0.015
    assert (
        lift["loaded_envelope_margin_m"]
        < lift["transport_guard_margin_m"]
        <= 0.030
    )
    assert lift["relative_position_tolerance_m"] <= 0.040
    assert 0.10 <= lift["velocity_scale"] <= 0.20
    assert "initial_velocity_scale" not in lift
    assert "initial_slow_distance_m" not in lift
    assert "top_hold_duration_s" not in lift
    assert 0.25 <= GRASP_CONFIG["contact_report"]["loss_grace_s"] <= 1.0


def test_loaded_lift_uses_faster_cartesian_motion():
    source = inspect.getsource(GraspAssistant)
    assert 'self.declare_parameter("speed_scale", 0.35)' in source
    assert (
        GRASP_CONFIG["unfreeze"]["alignment_velocity_scale"] == 0.08
    )
    guided_source = inspect.getsource(GraspAssistant._guided_grasp)
    assert '"连续竖直物理抬升"' in guided_source
    assert "velocity_scale=self.physical_test_lift_velocity_scale" in (
        guided_source
    )
    assert "physical_test_lift_initial_slow_distance" not in source


def test_physical_lift_only_allows_short_joint_bounded_ik_fallback():
    source = inspect.getsource(GraspAssistant._guided_grasp)
    assert "Physical lift segment result" not in source
    lift_start = source.index('"连续竖直物理抬升"')
    final_check = source.index(
        '"连续直线抬升后"',
        lift_start,
    )
    assert "linear=True" in source[lift_start:final_check]
    assert '"竖直抬升 IK 分支受限过渡"' in source[lift_start:final_check]
    assert "linear=False" in source[lift_start:final_check]
    assert (
        "physical_test_lift_bounded_fallback_max_joint_delta_rad"
        in source[lift_start:final_check]
    )


def test_apple_lift_uses_reachable_waypoints_and_bounded_subdivision():
    apple = GRASP_PROFILES["apple"]
    assert apple["physical_test_lift_segment_m"] == 0.020
    assert apple["physical_test_lift_min_retry_segment_m"] == 0.005
    assert apple["physical_test_lift_orientation_tolerances_rad"] == (
        0.04,
        0.08,
        0.12,
    )
    assert _is_recoverable_lift_planning_error(
        RuntimeError("MoveIt error_code=99999")
    )
    assert _reduced_lift_segment_distance(0.020, 0.005) == 0.010
    assert _reduced_lift_segment_distance(0.010, 0.005) == 0.005
    assert _reduced_lift_segment_distance(0.005, 0.005) is None


def test_single_post_lift_check_replaces_top_hold_thresholds():
    source = inspect.getsource(GraspAssistant._guided_grasp)

    assert "_require_loaded_physical_grasp" in source
    assert "use_verified_lift_reference=False" in source
    assert "object_touching_table" in source
    assert "self._loaded_grasp_reference_local = lift_local" in source
    assert "verification_reference_local" not in source
    assert "pre_lift_reseating_distance" not in source
    assert "transient_invalid_reasons" not in source
    assert "hard_invalid_reasons" not in source
    assert "physical_test_lift_min_object_rise" not in source
    assert "minimum_transport_object_z" in source
    assert "lift_completion_tolerance" in source
    assert "remaining_lift > lift_completion_tolerance" in source
    assert "物体未达到蓝盒安全搬运净空" in source
    assert "当前帧目标实际抬升不足" not in source


def test_loaded_grasp_filters_transient_contact_frames():
    source = inspect.getsource(
        GraspAssistant._require_loaded_physical_grasp
    )

    assert "valid_samples += 1" in source
    assert "valid_samples = 0" in source
    assert "loaded_grasp_max_attempts" in source
    assert "contact_report_available" in source


def test_arm_action_timeout_uses_final_joint_state():
    source = inspect.getsource(GraspAssistant._send_arm_trajectory)

    assert '"轨迹执行超时" not in str(exc)' in source
    assert "_wait_for_handover_tolerance" in source
    assert "accepting physical joint error" in source


def test_evaluator_accepts_assistant_verification_without_skipping_physics():
    source = inspect.getsource(GraspEvaluator)

    assert "assistant_physical_grasp_verified" in source
    assert "assistant_verified_candidate" in source
    assert "and gripper_obstructed" in source
    assert "target_bilateral_contact" in source
    assert "bilateral_contact_recent" in source
    assert "contact_loss_grace" in source


def test_attachment_feature_is_removed_from_active_workflow():
    package_root = Path(__file__).resolve().parents[1]
    active_sources = [
        package_root / "z1_isaac_bridge" / "grasp_assistant.py",
        package_root / "z1_isaac_bridge" / "grasp_evaluator.py",
        package_root / "z1_isaac_bridge" / "episode_recorder.py",
        package_root / "z1_isaac_bridge" / "object_collision_sync.py",
        Path("/home/lp531/isaac_assets/z1/scripts/run_z1_ros2_scene.py"),
    ]
    for path in active_sources:
        normalized = path.read_text().lower()
        assert "transport_constraint" not in normalized
        assert "attached" not in normalized
        assert '"attach"' not in normalized
    assert "transport_constraint" not in GRASP_CONFIG


def test_scene_randomization_interface_is_safe_and_reproducible():
    package_root = Path(__file__).resolve().parents[1]
    runtime = Path(
        "/home/lp531/isaac_assets/z1/scripts/run_z1_ros2_scene.py"
    ).read_text()
    reset = (
        package_root / "z1_isaac_bridge" / "episode_reset.py"
    ).read_text()
    gui = (
        package_root / "z1_isaac_bridge" / "grasp_gui_bridge.py"
    ).read_text()
    randomization = GRASP_CONFIG["scene_randomization"]

    assert '"/smolvla/randomize_scene"' in runtime
    assert '"/smolvla/validate_randomized_scene"' in runtime
    assert '"/smolvla/scene_randomization_command"' in runtime
    assert "random.Random(int(seed))" in runtime
    assert "OBJECT_FOOTPRINT_RADII" in runtime
    assert "minimum_object_clearance_m" in runtime
    assert "maximum_layout_sampling_attempts" in runtime
    assert "_sample_randomized_object_poses_once" in runtime
    assert "other_names.sort(" in runtime
    assert "OBJECT_FOOTPRINT_RADII[name], reverse=True" in runtime
    assert '"/smolvla/randomize_episode"' in reset
    for level in ("easy", "medium", "full"):
        assert f'"/smolvla/randomize_scene_{level}"' in runtime
        assert f'"/smolvla/randomize_episode_{level}"' in reset
    assert '"/api/randomize_episode"' in gui
    assert "self.randomize_episode(level)" in gui
    assert set(randomization["levels"]) == {"easy", "medium", "full"}
    easy = randomization["levels"]["easy"]
    medium = randomization["levels"]["medium"]
    full = randomization["levels"]["full"]
    assert easy["mode"] == medium["mode"] == "local"
    assert full["mode"] == "workspace"
    assert 0.035 <= easy["minimum_translation_m"]
    assert easy["translation_delta_m"] < medium["translation_delta_m"]
    assert easy["yaw_delta_deg"] < medium["yaw_delta_deg"]
    assert medium["yaw_delta_deg"] < full["yaw_delta_deg"]
    assert randomization["minimum_object_clearance_m"] >= 0.02
    assert randomization["maximum_layout_sampling_attempts"] >= 2
    validation = randomization["validation"]
    assert validation["minimum_settle_duration_s"] >= 0.9
    assert validation["max_horizontal_drift_m"] <= 0.004
    assert validation["max_vertical_drift_m"] <= 0.006
    assert validation["max_linear_speed_m_s"] <= 0.01
    assert validation["max_angular_speed_rad_s"] <= 0.2
    assert validation["max_orientation_drift_deg"] <= 5.0
    assert validation["max_table_penetration_m"] <= 0.003
    assert validation["require_table_contact"] is True
    assert "minimum_separation_m" in runtime
    assert "table_penetration_m" in runtime
    assert '"validate_randomized_scene"' in reset
    validation_call = reset.index(
        'randomization_message = self._call(\n'
        '                    "validate_randomized_scene"'
    )
    assert reset.index('time.sleep(1.0)') < validation_call
    assert "auto_validate=True" in runtime
    assert set(randomization["object_footprint_radii_m"]) == {
        "cup",
        "apple",
        "peach",
        "mouse",
        "beverage_bottle",
    }
    sweep = randomization["target_sweep_clearance"]
    assert sweep["approach_before_target_m"] > 0.0
    assert sweep["robot_side_half_width_m"] > (
        sweep["target_side_half_width_m"]
    )
    assert sweep["closing_radius_m"] > 0.0
    assert sweep["lift_radius_m"] > 0.0


def test_target_sweep_rejects_failed_apple_peach_layout():
    target = (0.487408, -0.150769, 0.025)
    radial_length = (target[0] ** 2 + target[1] ** 2) ** 0.5
    pregrasp = (
        target[0] - 0.075 * target[0] / radial_length,
        target[1] - 0.075 * target[1] / radial_length,
        0.30,
    )
    blockers = _candidate_sweep_blockers(
        "apple",
        target,
        pregrasp,
        {
            "apple": target,
            "peach": (0.511098, -0.217632, 0.019),
        },
    )

    assert blockers.get("peach") in {
        "闭爪空间",
        "初始抬升空间",
        "夹爪接近扫掠通道",
    }


def test_target_sweep_keeps_original_clear_layout():
    target = (0.44, -0.18, 0.025)
    radial_length = (target[0] ** 2 + target[1] ** 2) ** 0.5
    pregrasp = (
        target[0] - 0.075 * target[0] / radial_length,
        target[1] - 0.075 * target[1] / radial_length,
        0.30,
    )
    blockers = _candidate_sweep_blockers(
        "apple",
        target,
        pregrasp,
        {
            "apple": target,
            "peach": (0.53, -0.16, 0.019),
        },
    )

    assert blockers == {}


def test_target_sweep_rejects_latest_peach_box_wall_collision_layout():
    target = (0.402675, 0.019997, 0.01574)
    radial_length = (target[0] ** 2 + target[1] ** 2) ** 0.5
    pregrasp = (
        target[0] - 0.075 * target[0] / radial_length,
        target[1] - 0.075 * target[1] / radial_length,
        0.30,
    )
    blockers = _candidate_sweep_blockers(
        "peach",
        target,
        pregrasp,
        {"peach": target},
    )

    assert blockers == {"smolvla_box": "蓝盒闭爪空间"}


def test_target_sweep_rejects_box_approach_even_outside_closing_radius():
    target = (0.47, 0.0, 0.025)
    pregrasp = (0.35, 0.0, 0.30)
    blockers = _candidate_sweep_blockers(
        "apple",
        target,
        pregrasp,
        {"apple": target},
    )

    assert blockers == {
        "smolvla_box": "蓝盒夹爪接近扫掠通道"
    }


def test_target_sweep_rejects_full_layout_blocking_camera_and_wrist():
    target = (0.425328, -0.092766, 0.025)
    radial_length = (target[0] ** 2 + target[1] ** 2) ** 0.5
    pregrasp = (
        target[0] - 0.075 * target[0] / radial_length,
        target[1] - 0.075 * target[1] / radial_length,
        0.30,
    )
    blockers = _candidate_sweep_blockers(
        "apple",
        target,
        pregrasp,
        {
            "apple": target,
            "beverage_bottle": (0.362455, -0.164001, 0.031),
        },
    )

    assert blockers == {
        "beverage_bottle": "夹爪接近扫掠通道"
    }


def test_episode_recorder_preserves_curriculum_randomization_metadata():
    package_root = Path(__file__).resolve().parents[1]
    recorder = (
        package_root / "z1_isaac_bridge" / "episode_recorder.py"
    ).read_text()

    assert '"/smolvla/scene_reset_status"' in recorder
    assert '"scene_randomization": self._episode_randomization' in recorder
    for field in (
        "curriculum_level",
        "seed",
        "object_positions_m",
        "object_displacements_mm",
        "sweep_clearance_target",
        "target_sweep_clear",
    ):
        assert f'"{field}"' in recorder


def test_all_object_episodes_are_labeled_as_physical_pick_place():
    assert set(TASK_INSTRUCTIONS) == {
        "cup",
        "apple",
        "peach",
        "mouse",
        "beverage_bottle",
    }
    for instruction in TASK_INSTRUCTIONS.values():
        normalized = instruction.lower()
        assert "lift it clear" in normalized
        assert "place it in the box" in normalized
        assert "20 cm" not in normalized


def test_force_stop_holds_measured_joints_and_unfreezes_target():
    package_root = Path(__file__).resolve().parents[1]
    trajectory_bridge = (
        package_root / "z1_isaac_bridge" / "trajectory_bridge.py"
    ).read_text()
    assistant = (
        package_root / "z1_isaac_bridge" / "grasp_assistant.py"
    ).read_text()

    assert "SAFE_HOME" not in trajectory_bridge
    assert "hold_positions" in trajectory_bridge
    assert "控制目标已锁定在实际关节位置" in trajectory_bridge
    assert 'self._command_scene_grasp("unfreeze")' in assistant
    assert "self._persistent_gripper_goal_handle = None" in assistant
    assert "用户强制停止抓取" in assistant


def test_scene_reset_replaces_stale_joint_command_with_home():
    runtime = Path(
        "/home/lp531/isaac_assets/z1/scripts/run_z1_ros2_scene.py"
    ).read_text()
    assert 'JointState, "/isaac_joint_commands"' in runtime
    assert "ArticulationAction(joint_positions=home_positions)" in runtime
    assert "self._joint_command_pub.publish(home_command)" in runtime
