"""Already-open confirmation must not replay a two-second trajectory."""
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from z1_isaac_bridge.grasp_assistant import GraspAssistant


def assistant(actual, wall_age=0., sensor_age=0.):
    node = object.__new__(GraspAssistant)
    node._lock = threading.RLock()
    node._joint_positions = {'jointGripper': actual}
    node._gripper_feedback = (time.monotonic()-wall_age, 5.-sensor_age)
    node.gripper_open = -.9
    node.gripper_closed = 0.
    node.gripper_tolerance = .03
    node.gripper_open_duration = 2.
    node.gripper_close_duration = 1.5
    node._gripper_client = Mock()
    node._cancel_persistent_gripper_hold = Mock()
    node._set_status = Mock()
    node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=5_000_000_000))
    node.get_logger = lambda: SimpleNamespace(info=Mock())
    node._send_action_goal = Mock(return_value=SimpleNamespace(error_code=0))
    return node


def trajectory_duration(node, state='open'):
    node._execute_gripper(state)
    goal = node._send_action_goal.call_args.args[1]
    t = goal.trajectory.points[0].time_from_start
    return t.sec+t.nanosec*1e-9


def test_already_open_refreshes_real_command_without_two_second_wait():
    node = assistant(-.8963)
    assert trajectory_duration(node) == pytest.approx(.05)
    node._cancel_persistent_gripper_hold.assert_called_once()
    goal = node._send_action_goal.call_args.args[1]
    assert goal.trajectory.points[0].positions == pytest.approx([-.9])


@pytest.mark.parametrize('actual,wall_age,sensor_age', [(-.88,0.,0.),(-.6,0.,0.),(-.8963,1.,0.),(-.8963,0.,1.),(float('nan'),0.,0.)])
def test_partial_stale_or_invalid_feedback_keeps_normal_open(actual,wall_age,sensor_age):
    assert trajectory_duration(assistant(actual,wall_age,sensor_age)) == pytest.approx(2.)


def test_missing_feedback_keeps_normal_open():
    node = assistant(-.8963)
    node._gripper_feedback = None
    assert trajectory_duration(node) == pytest.approx(2.)


def test_close_keeps_original_duration():
    assert trajectory_duration(assistant(-.8963),'close') == pytest.approx(1.5)
