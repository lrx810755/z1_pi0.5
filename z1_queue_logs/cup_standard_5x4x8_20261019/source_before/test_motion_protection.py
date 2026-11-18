"""Exercise protection against simulated feedback, without moving Isaac."""
import inspect
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from std_msgs.msg import Bool, String
from z1_isaac_bridge.trajectory_bridge import (
    ALL_JOINTS, ARM_JOINTS, GRIPPER_JOINTS, TrajectoryBridge,
)
from z1_isaac_bridge.grasp_assistant import GraspAssistant


def bridge():
    node = SimpleNamespace(
        _state_lock=threading.RLock(), _motion_fault="", _fault_hold={},
        _scene_reset_active=False, _joint_state_times={},
        _stop_generation=0, _servo_stream_enabled=True,
        _policy_control_active=False,
        _recording=False, _recording_status_time=0.0,
        _actual=dict.fromkeys(ALL_JOINTS, 0.0),
        _command=dict.fromkeys(ALL_JOINTS, 0.0),
        _last_state_time=time.monotonic(), _contact_time=time.monotonic(),
        _touching_table=False, _path_error_since=None,
        path_error_limit=0.08, path_error_hard_limit=0.20,
        path_error_duration=0.20, state_timeout=3.0,
        arm_goal_stable_duration=0.20,
        _controller_locks={"arm": threading.Lock(), "gripper": threading.Lock()},
        _publisher=Mock(), get_logger=lambda: Mock(),
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: None)),
        _simulation_seconds=Mock(return_value=1.0),
    )
    # JointState header needs an actual builtin_interfaces Time.
    from builtin_interfaces.msg import Time
    node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=Time))
    for name in ("_publish_command", "_trip_fault", "_check_motion",
                 "_contact_callback", "_clear_fault_callback",
                 "_begin_scene_reset_callback"):
        setattr(node, name, getattr(TrajectoryBridge, name).__get__(node))
    return node


def test_recording_idle_hold_preserves_target_and_keeps_causal_actions_fresh():
    from builtin_interfaces.msg import Time
    from z1_isaac_bridge.recording_sync import RecordingSynchronizer
    node = bridge()
    node._servo_stream_enabled = False
    node._recording_status_callback = TrajectoryBridge._recording_status_callback.__get__(node)
    node._recording_status_callback(String(data='{"recording":true}'))
    node._command = dict.fromkeys(ALL_JOINTS, 0.01)
    sync = RecordingSynchronizer((), action_age=0.4)
    for tick in range(11):
        stamp = Time(sec=10 + tick // 10, nanosec=(tick % 10) * 100_000_000)
        node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: stamp))
        TrajectoryBridge._motion_watchdog(node)
        message = node._publisher.publish.call_args.args[0]
        assert list(message.position) == [0.01] * len(ALL_JOINTS)
        assert message.header.stamp == stamp
        t = stamp.sec + stamp.nanosec * 1e-9
        sync.push('action', t, message.position)
        sync.push('state', t, [0.0] * len(ALL_JOINTS))
        sync.push('status', t, {})
        assert sync.select(t)['action'][0] == t
    node._recording_status_callback(String(data='{"recording":false}'))
    node._publisher.reset_mock()
    TrajectoryBridge._motion_watchdog(node)
    node._publisher.publish.assert_not_called()


@pytest.mark.parametrize('owner', ['reset', 'servo', 'policy', 'arm', 'gripper',
                                    'not_recording', 'stale_recording', 'incomplete_target'])
def test_recording_hold_respects_command_owner_and_recording_state(owner):
    node = bridge()
    node._servo_stream_enabled = False
    node._recording = True
    node._recording_status_time = time.monotonic()
    if owner == 'reset': node._scene_reset_active = True
    elif owner == 'servo': node._servo_stream_enabled = True
    elif owner == 'policy': node._policy_control_active = True
    elif owner in ('arm', 'gripper'): node._controller_locks[owner].acquire()
    elif owner == 'not_recording': node._recording = False
    elif owner == 'stale_recording': node._recording_status_time = time.monotonic() - 4.0
    elif owner == 'incomplete_target': node._command.pop('jointGripper')
    TrajectoryBridge._motion_watchdog(node)
    node._publisher.publish.assert_not_called()


def test_recording_hold_keeps_fault_hold_instead_of_old_trajectory_target():
    node = bridge()
    node._servo_stream_enabled = False
    node._recording = True
    node._recording_status_time = time.monotonic()
    node._motion_fault = 'tracking error'
    node._fault_hold = dict.fromkeys(ALL_JOINTS, -0.02)
    node._command = dict.fromkeys(ALL_JOINTS, 0.5)
    TrajectoryBridge._motion_watchdog(node)
    assert list(node._publisher.publish.call_args.args[0].position) == [-0.02] * len(ALL_JOINTS)


def test_policy_control_ownership_stops_servo_and_blocks_reenable():
    node = bridge()
    node._policy_control_callback = (
        TrajectoryBridge._policy_control_callback.__get__(node)
    )
    node._enable_servo_stream_callback = (
        TrajectoryBridge._enable_servo_stream_callback.__get__(node)
    )
    node._policy_control_callback(Bool(data=True))
    assert node._policy_control_active
    assert not node._servo_stream_enabled
    assert node._stop_generation == 1
    response = node._enable_servo_stream_callback(
        SimpleNamespace(data=True), SimpleNamespace()
    )
    assert not response.success


def test_scene_reset_suppresses_old_hold_until_fresh_contact_free_feedback():
    node = bridge()
    node._trip_fault("contact")
    assert node._begin_scene_reset_callback(None, SimpleNamespace()).success
    node._publisher.publish.reset_mock()
    node._trip_fault("stale contact during reset")
    node._publish_command(ALL_JOINTS, [0.8] * 7)
    node._publisher.publish.assert_not_called()
    assert not node._clear_fault_callback(None, SimpleNamespace()).success
    node._actual = dict.fromkeys(ALL_JOINTS, 0.1)
    node._last_state_time = node._contact_time = time.monotonic()
    node._touching_table = True
    assert not node._clear_fault_callback(None, SimpleNamespace()).success
    node._touching_table = False
    assert node._clear_fault_callback(None, SimpleNamespace()).success
    assert not node._scene_reset_active
    assert list(node._publisher.publish.call_args.args[0].position) == [0.1] * 7


def test_scene_reset_rejects_active_controller():
    node = bridge()
    node._controller_locks["arm"].acquire()
    assert not node._begin_scene_reset_callback(None, SimpleNamespace()).success
    assert not node._scene_reset_active


def test_contact_latches_hold_and_blocks_old_arm_and_gripper_commands():
    node = bridge()
    node._command["joint1"] = 0.5
    node._contact_callback(String(data=json.dumps({
        "contact_report_available": True,
        "left_finger_touching_table": True,
        "right_finger_touching_table": False,
    })))
    assert node._motion_fault
    assert not node._servo_stream_enabled
    assert node._stop_generation == 1
    node._publish_command(ALL_JOINTS, [0.8] * len(ALL_JOINTS))
    assert list(node._publisher.publish.call_args.args[0].position) == [0.0] * 7
    node._actual["joint1"] = 0.01
    node._trip_fault("another fault")
    assert node._command["joint1"] == 0.0  # hold must not chase contact drift


def test_tracking_error_requires_persistence_but_gross_error_stops_now():
    node = bridge()
    node._command["joint1"] = 0.10
    assert not node._check_motion(ARM_JOINTS)
    node._simulation_seconds.return_value = 1.1
    assert not node._check_motion(ARM_JOINTS)
    node._command["joint1"] = 0.0
    assert not node._check_motion(ARM_JOINTS)
    node._command["joint1"] = 0.10
    assert not node._check_motion(ARM_JOINTS)
    node._simulation_seconds.return_value = 1.31
    assert "tracking error" in node._check_motion(ARM_JOINTS)
    node = bridge()
    node._command["joint1"] = 0.21
    assert "tracking error" in node._check_motion(ARM_JOINTS)


@pytest.mark.parametrize("field", ["_last_state_time", "_contact_time"])
def test_stale_feedback_stops_motion(field):
    node = bridge()
    setattr(node, field, time.monotonic() - 4)
    assert "timeout" in node._check_motion(ARM_JOINTS)


def test_gripper_preload_is_not_arm_tracking_error():
    node = bridge()
    node._command["jointGripper"] = 0.3
    assert not node._check_motion(GRIPPER_JOINTS)


def test_fault_clear_requires_idle_fresh_no_contact_and_does_not_resume():
    node = bridge()
    node._trip_fault("contact")
    node._touching_table = True
    assert not node._clear_fault_callback(None, SimpleNamespace()).success
    node._touching_table = False
    node._controller_locks["arm"].acquire()
    assert not node._clear_fault_callback(None, SimpleNamespace()).success
    node._controller_locks["arm"].release()
    node._actual["joint1"] = 0.02
    assert node._clear_fault_callback(None, SimpleNamespace()).success
    assert not node._motion_fault
    assert not node._servo_stream_enabled
    assert node._command["joint1"] == 0.02


def test_latched_contact_aborts_action_without_executing_waypoint():
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from control_msgs.action import FollowJointTrajectory
    node = bridge()
    node._wait_for_state = lambda joints: True
    node._ordered_positions = TrajectoryBridge._ordered_positions.__get__(node)
    node._result = TrajectoryBridge._result
    node._trip_fault("Finger/table contact detected")
    endpoint = [0.1, 0.1, -0.1, 0.1, 0.1, 0.1]
    trajectory = JointTrajectory(joint_names=list(ARM_JOINTS))
    trajectory.points = [JointTrajectoryPoint(positions=endpoint)]
    goal = Mock(request=SimpleNamespace(trajectory=trajectory), is_cancel_requested=False)
    result = TrajectoryBridge._execute(node, goal, ARM_JOINTS, "arm")
    assert result.error_code == FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
    goal.abort.assert_called_once()
    goal.succeed.assert_not_called()
    assert not node._controller_locks["arm"].locked()
    assert all(value == 0 for value in node._command.values())


def test_endpoint_miss_keeps_commanding_planned_endpoint():
    from control_msgs.action import FollowJointTrajectory
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    node = bridge()
    node._servo_stream_enabled = False
    node._wait_for_state = lambda joints: True
    node._ordered_positions = TrajectoryBridge._ordered_positions.__get__(node)
    node._result = TrajectoryBridge._result
    node._check_motion = lambda joints: ""
    node._feedback = lambda goal, joints, desired, elapsed: None
    node.command_rate = 10000.0
    node.goal_tolerance = 0.005
    node.goal_time_tolerance = 0.02
    simulation_time = iter((0.0, 0.01, 0.02, 0.03, 0.04, 0.05))
    node._simulation_seconds = lambda: next(simulation_time)

    endpoint = [0.1, 0.1, -0.1, 0.1, 0.1, 0.1]
    trajectory = JointTrajectory(joint_names=list(ARM_JOINTS))
    trajectory.points = [JointTrajectoryPoint(positions=endpoint)]
    goal = Mock(request=SimpleNamespace(trajectory=trajectory),
                is_cancel_requested=False)

    result = TrajectoryBridge._execute(node, goal, ARM_JOINTS, "arm")

    assert result.error_code == FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED
    assert [node._command[name] for name in ARM_JOINTS] == endpoint
    assert not node._motion_fault


def test_arm_goal_tolerance_violation_is_not_accepted_as_handover():
    from control_msgs.action import FollowJointTrajectory
    from trajectory_msgs.msg import JointTrajectory

    result = SimpleNamespace(
        error_code=FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED,
        error_string="endpoint remained outside tolerance",
    )
    node = SimpleNamespace(
        _trajectory_client=SimpleNamespace(
            wait_for_server=lambda timeout_sec: True
        ),
        _send_action_goal=lambda client, goal, timeout_s: result,
        _lock=threading.Lock(),
        _stop_requested=False,
        target_object="beverage_bottle",
        _joint_error=lambda trajectory: 0.01,
    )
    trajectory = SimpleNamespace(joint_trajectory=JointTrajectory())

    with pytest.raises(RuntimeError, match="预抓取执行失败"):
        GraspAssistant._send_arm_trajectory(node, trajectory)


def test_cup_goal_tolerance_violation_is_only_logged_when_paused():
    from control_msgs.action import FollowJointTrajectory
    from trajectory_msgs.msg import JointTrajectory

    result = SimpleNamespace(
        error_code=FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED,
        error_string="endpoint remained outside tolerance",
    )
    logger = Mock()
    node = SimpleNamespace(
        _trajectory_client=SimpleNamespace(
            wait_for_server=lambda timeout_sec: True
        ),
        _send_action_goal=lambda client, goal, timeout_s: result,
        _lock=threading.Lock(),
        _stop_requested=False,
        target_object="cup",
        _joint_error=lambda trajectory: 0.0052,
        get_logger=lambda: logger,
    )
    trajectory = SimpleNamespace(joint_trajectory=JointTrajectory())

    assert GraspAssistant._send_arm_trajectory(node, trajectory) is False
    logger.warning.assert_called_once()
    assert "轨迹收敛判定已暂停" in logger.warning.call_args[0][0]


def test_cup_path_tolerance_violation_still_aborts_when_paused():
    from control_msgs.action import FollowJointTrajectory
    from trajectory_msgs.msg import JointTrajectory

    result = SimpleNamespace(
        error_code=FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED,
        error_string="path deviated",
    )
    node = SimpleNamespace(
        _trajectory_client=SimpleNamespace(
            wait_for_server=lambda timeout_sec: True
        ),
        _send_action_goal=lambda client, goal, timeout_s: result,
        _lock=threading.Lock(),
        _stop_requested=False,
        target_object="cup",
        _joint_error=lambda trajectory: 0.01,
        get_logger=lambda: Mock(),
    )
    trajectory = SimpleNamespace(joint_trajectory=JointTrajectory())

    with pytest.raises(RuntimeError, match="运动保护中止"):
        GraspAssistant._send_arm_trajectory(node, trajectory)


def test_arm_action_timeout_is_not_accepted_as_handover():
    from trajectory_msgs.msg import JointTrajectory

    def timeout(client, goal, timeout_s):
        raise RuntimeError("轨迹执行超时")

    node = SimpleNamespace(
        _trajectory_client=SimpleNamespace(
            wait_for_server=lambda timeout_sec: True
        ),
        _send_action_goal=timeout,
        _lock=threading.Lock(),
        _stop_requested=False,
        get_logger=lambda: Mock(),
    )
    trajectory = SimpleNamespace(joint_trajectory=JointTrajectory())

    with pytest.raises(RuntimeError, match="轨迹执行超时"):
        GraspAssistant._send_arm_trajectory(node, trajectory)


def test_transient_endpoint_tolerance_does_not_finish_arm_goal():
    from control_msgs.action import FollowJointTrajectory
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    node = bridge()
    node._servo_stream_enabled = False
    node._actual.update(dict.fromkeys(ARM_JOINTS, 0.096))
    node._wait_for_state = lambda joints: True
    node._ordered_positions = TrajectoryBridge._ordered_positions.__get__(node)
    node._result = TrajectoryBridge._result
    node._check_motion = lambda joints: ""
    node._feedback = lambda goal, joints, desired, elapsed: None
    node.command_rate = 10000.0
    node.goal_tolerance = 0.005
    node.goal_time_tolerance = 0.03
    node.arm_goal_stable_duration = 0.015
    calls = 0

    def simulation_time():
        nonlocal calls
        calls += 1
        if calls == 5:
            node._actual.update(dict.fromkeys(ARM_JOINTS, 0.08))
        return calls * 0.01

    node._simulation_seconds = simulation_time
    endpoint = [0.1, 0.1, -0.1, 0.1, 0.1, 0.1]
    node._actual["joint3"] = -0.096
    trajectory = JointTrajectory(joint_names=list(ARM_JOINTS))
    trajectory.points = [JointTrajectoryPoint(positions=endpoint)]
    goal = Mock(request=SimpleNamespace(trajectory=trajectory),
                is_cancel_requested=False)

    result = TrajectoryBridge._execute(node, goal, ARM_JOINTS, "arm")

    assert result.error_code == FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED
    goal.succeed.assert_not_called()


def test_fault_clear_reports_each_blocking_condition_and_keeps_reset_exclusive():
    node = bridge()
    node._scene_reset_active = True
    node._motion_fault = 'reset'
    node._last_state_time = node._contact_time = time.monotonic() - 4.
    node._touching_table = True
    node._controller_locks['arm'].acquire()
    response = node._clear_fault_callback(None, SimpleNamespace())
    assert not response.success
    for detail in ('关节反馈', '接触反馈', '夹指仍接触桌面', '控制器尚未停止'):
        assert detail in response.message
    assert node._scene_reset_active
    assert node._motion_fault == 'reset'
    node._publisher.publish.assert_not_called()


def test_joint6_drive_holds_wide_azimuth_pregrasp_endpoint():
    scene = (
        Path(__file__).resolve().parents[2]
        / "isaac_assets/z1/scripts/run_z1_ros2_scene.py"
    ).read_text()

    assert '"joint6": (80000.0, 800.0, 800.0)' in scene


def test_joint3_drive_and_sim_goal_tolerance_cover_measured_endpoint_residual():
    repository = Path(__file__).resolve().parents[2]
    scene = (
        repository / "isaac_assets" / "z1" / "scripts" / "run_z1_ros2_scene.py"
    ).read_text(encoding="utf-8")
    bridge_source = inspect.getsource(TrajectoryBridge.__init__)

    assert '"joint3": (200000.0, 1200.0, 1600.0)' in scene
    assert 'declare_parameter("goal_tolerance", 0.012)' in bridge_source
