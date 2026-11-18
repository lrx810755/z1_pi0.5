"""Validate one-shot acceptance and cup height transfer without robot motion."""
import copy
import json
import math
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from std_msgs.msg import String
from z1_isaac_bridge.grasp_assistant import (
    GraspAssistant, GRASP_PROFILES, _bottle_reference_for_orientation,
    _ee_position_for_object_in_stator,
    _object_in_stator_after_world_tool_offset,
    orientation_for_approach,
)
from z1_isaac_bridge.grasp_config import (
    GRASP_CONFIG, GRASP_ENVELOPE, OBJECT_GRASP_PROFILES,
    object_loaded_grasp_is_valid,
)
from z1_isaac_bridge.grasp_evaluator import rotate_vector, quaternion_multiply
from z1_isaac_bridge.grasp_gui_bridge import GraspGuiBridge


def _single_position_target(fruit):
    profile = GRASP_PROFILES[fruit]
    desired = tuple(profile[f'desired_center_{axis}_m'] for axis in 'xyz')
    if fruit != 'apple':
        return desired
    elevation = math.radians(profile['elevation_deg'][1])
    orientation = orientation_for_approach(
        (math.cos(elevation), 0.0, -math.sin(elevation)), 0.0
    )
    clearance = profile['single_position_world_clearance_lift_m']
    return _object_in_stator_after_world_tool_offset(
        desired, orientation, (0.0, 0.0, clearance)
    )


@pytest.mark.parametrize('fruit', ['apple', 'peach'])
@pytest.mark.parametrize('failure', [None, 'x', 'y', 'z', 'nan', 'table', 'physics', 'contact', 'missing'])
def test_fruit_accepts_once_or_stops_without_correction(fruit, failure):
    profile = GRASP_PROFILES[fruit]
    desired = _single_position_target(fruit)
    probe = dict(local_position_m=list(desired), physics_enabled=True,
                 gravity_enabled=True, contact_report_available=True)
    if failure in ('x', 'y', 'z'):
        probe['local_position_m']['xyz'.index(failure)] += .02
    elif failure == 'nan':
        probe['local_position_m'][0] = math.nan
    elif failure == 'physics':
        probe['physics_enabled'] = False
    elif failure == 'contact':
        probe['left_finger_contact_object'] = fruit
    elif failure == 'missing':
        probe['contact_report_available'] = False
    node = SimpleNamespace(
        _lock=threading.Lock(), _scene_grasp_status=probe,
        unfreeze_settle_duration=0., _command_scene_grasp=Mock(),
        _wait_scene_grasp=Mock(return_value=True), _set_status=Mock(),
        _raise_if_finger_touching_table=Mock(), _execute_manual_pose=Mock(),
    )
    if failure == 'table':
        node._raise_if_finger_touching_table.side_effect = RuntimeError('table')
    candidate = dict(final_grasp_target=True, desired_object_in_stator_m=desired)
    envelope = dict(GRASP_ENVELOPE, **profile.get('grasp_envelope_overrides', {}))
    if failure:
        with pytest.raises(RuntimeError):
            GraspAssistant._validate_single_position_grasp(node, fruit, candidate, envelope)
    else:
        assert GraspAssistant._validate_single_position_grasp(node, fruit, candidate, envelope) == probe
    assert [c.args[0] for c in node._command_scene_grasp.call_args_list] == ['unfreeze', 'probe']
    node._execute_manual_pose.assert_not_called()


@pytest.mark.parametrize('fruit', ['apple', 'peach'])
def test_fruit_tool_transform_reconstructs_target_with_nonidentity_mount(fruit):
    local = tuple(GRASP_PROFILES[fruit][f'desired_center_{a}_m'] for a in 'xyz')
    ee_rotation = orientation_for_approach((.3, .4, -.8660254), 32.)
    mount_rotation = (0., 0., math.sin(.2), math.cos(.2))
    mount_translation = (.05, .003, -.004)
    center = (.52, -.12, .035)
    ee = _ee_position_for_object_in_stator(center, ee_rotation, mount_translation, mount_rotation, local)
    stator = tuple(a+b for a,b in zip(ee, rotate_vector(ee_rotation, mount_translation)))
    reconstructed = tuple(a+b for a,b in zip(stator, rotate_vector(quaternion_multiply(ee_rotation, mount_rotation), local)))
    assert reconstructed == pytest.approx(center, abs=1e-9)


@pytest.mark.parametrize('pitch', [-5., -6., -7.])
def test_cup_height_transfer_targets_upper_body_across_allowed_pitch(pitch):
    cup = GRASP_CONFIG['object_grasp_profiles']['cup']
    bottle = GRASP_CONFIG['object_grasp_profiles']['beverage_bottle']
    cal = cup['calibrated_grasp_reference']
    ori = orientation_for_approach((math.cos(math.radians(pitch)), 0., math.sin(math.radians(pitch))), GRASP_PROFILES['cup']['roll_deg'])
    local = _bottle_reference_for_orientation(cal['object_in_gripper_local_m'], ori, cal['captured_object_center_minus_stator_world_z_m'])
    stator_z = GRASP_CONFIG['object_center_z_offsets_m']['cup'] - rotate_vector(ori, local)[2]
    bottle_stator_z = GRASP_CONFIG['object_center_z_offsets_m']['beverage_bottle'] - bottle['calibrated_grasp_reference']['captured_object_center_minus_stator_world_z_m']
    assert stator_z == pytest.approx(
        bottle_stator_z
        + cup['grasp_height_target_m']
        - bottle['grasp_height_target_m']
    )
    assert .0305 < cup['grasp_height_target_m'] < cup['height_m']
    assert stator_z > cup['height_m']


def test_cup_contact_height_band_is_paused_until_re_enabled():
    probe = dict(contact_report_available=True,
        left_finger_contact_object='cup', right_finger_contact_object='cup',
        left_finger_contact_impulse_ns=.1, right_finger_contact_impulse_ns=.1,
        left_finger_contact_point_object_m=[0, 0, .048],
        right_finger_contact_point_object_m=[0, 0, .048])
    assert object_loaded_grasp_is_valid(probe, 'cup')
    cup_profile = OBJECT_GRASP_PROFILES['cup']
    assert cup_profile.get('require_contact_height_band') is False
    for height in (.010, .060):
        sample = copy.deepcopy(probe)
        sample['left_finger_contact_point_object_m'][2] = height
        assert object_loaded_grasp_is_valid(sample, 'cup')
    cup_profile['require_contact_height_band'] = True
    try:
        for height in (.010, .060):
            bad = copy.deepcopy(probe)
            bad['left_finger_contact_point_object_m'][2] = height
            assert not object_loaded_grasp_is_valid(bad, 'cup')
    finally:
        cup_profile['require_contact_height_band'] = False


@pytest.mark.parametrize('phase,active,expected', [('COMPLETE', False, 'IDLE'), ('ERROR', False, 'ERROR'), ('COMPLETE', True, 'ERROR')])
def test_reset_status_clears_old_error_only_after_success_when_idle(phase, active, expected):
    node = SimpleNamespace(_lock=threading.Lock(), _auto_lock=threading.Lock(),
                           _auto_status={'phase': 'ERROR'})
    if active:
        node._auto_lock.acquire()
    GraspGuiBridge._episode_reset_status_callback(node, String(data=json.dumps({'phase': phase})))
    assert node._auto_status['phase'] == expected


@pytest.mark.parametrize('fruit', ['apple', 'peach'])
@pytest.mark.parametrize('in_tolerance', [True, False])
def test_fruit_dispatch_reaches_close_only_after_acceptance_without_arm_correction(fruit, in_tolerance):
    profile = GRASP_PROFILES[fruit]
    desired = _single_position_target(fruit)
    local = list(desired)
    if not in_tolerance:
        local[1] += .020
    candidate = dict(position=(.3, .1, .2), orientation=(0,0,0,1),
                     target_point=(.5,.1,.03), standoff_m=.2,
                     final_grasp_target=True, desired_object_in_stator_m=desired)
    node = SimpleNamespace(
        _lock=threading.Lock(), _grasp_candidate=candidate, target_object=fruit,
        _task_status={}, _scene_grasp_status=dict(local_position_m=local,
            physics_enabled=True, gravity_enabled=True, contact_report_available=True),
        _current_ee_pose=Mock(return_value=(candidate['position'], candidate['orientation'])),
        _current_object_position=Mock(return_value=(.5,.1,.01)),
        _command_scene_grasp=Mock(), _wait_scene_grasp=Mock(return_value=True),
        _set_status=Mock(), _execute_gripper=Mock(), _execute_manual_pose=Mock(),
        _raise_if_finger_touching_table=Mock(), unfreeze_settle_duration=0.,
        _set_grasp_control_state=Mock(),
        _execute_gripper_close_until_obstructed=Mock(side_effect=RuntimeError('CLOSE_REACHED')),
    )
    node._validate_single_position_grasp = GraspAssistant._validate_single_position_grasp.__get__(node)
    with pytest.raises(RuntimeError, match='CLOSE_REACHED' if in_tolerance else '一次定位超差'):
        GraspAssistant._guided_grasp(node)
    assert node._execute_gripper_close_until_obstructed.call_count == int(in_tolerance)
    node._execute_manual_pose.assert_not_called()
