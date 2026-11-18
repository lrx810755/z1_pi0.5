"""The cup must be dynamic before any approach trajectory is sent."""
import threading
from types import SimpleNamespace
import pytest
from z1_isaac_bridge.grasp_assistant import GraspAssistant


def assistant(events, acknowledge=True):
    node = SimpleNamespace(
        _lock=threading.RLock(), _approved_trajectory=object(),
        _approved_target='cup', target_object='cup', _grasp_candidate={},
        _set_bottle_finger_collision=lambda _: None,
        _set_status=lambda *args: None,
        _execute_gripper=lambda name: events.append(('gripper', name)),
        _command_scene_grasp=lambda command, *args: events.append(('scene', command)),
        _wait_scene_grasp=lambda phase, *args: acknowledge or phase=='FROZEN',
        _send_arm_trajectory=lambda _: events.append(('arm', 'approach')) or False,
    )
    return node


def test_cup_unfreezes_before_approach_and_never_refreezes_during_motion():
    events=[]
    GraspAssistant._execute_plan(assistant(events))
    assert events == [('scene','freeze'),('gripper','open'),
                      ('scene','unfreeze'),('arm','approach')]


def test_cup_does_not_move_without_unfreeze_acknowledgement():
    events=[]
    with pytest.raises(RuntimeError, match='场景模式'):
        GraspAssistant._execute_plan(assistant(events, False))
    assert ('arm','approach') not in events
