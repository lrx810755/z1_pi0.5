import math
from types import SimpleNamespace
import threading

import pytest
from z1_isaac_bridge import smolvla_rollout_core as core
from z1_isaac_bridge.smolvla_sim_rollout import SmolVLASimRollout


def quaternion(degrees):
    angle = math.radians(degrees) / 2
    return (math.sin(angle), 0., 0., math.cos(angle))


def feed(guard, start, end, angle=80, placing=False):
    result = False
    for i in range(round((end-start)*100)+1):
        result = guard.observe(quaternion(angle), start+i/100, placing)
    return result


def test_strict_threshold_and_one_continuous_simulation_second():
    guard = core.BottleTipGuard()
    assert not feed(guard, 0, 2, 70)
    assert not feed(guard, 2.01, 2.99)
    assert guard.observe(quaternion(80), 3.01, False)
    assert guard.tripped


def test_upright_sample_and_missing_feedback_restart_countdown():
    guard = core.BottleTipGuard()
    assert not feed(guard, 0, .8)
    assert not guard.observe(quaternion(0), .81, False)
    assert not feed(guard, .82, 1.5)
    assert not feed(guard, 5, 5.99)
    assert guard.observe(quaternion(80), 6, False)


def test_placement_latches_off_but_cannot_erase_failure():
    guard = core.BottleTipGuard()
    assert not feed(guard, 0, .8)
    assert not guard.observe(quaternion(90), .81, True)
    assert not feed(guard, .82, 3)
    failed = core.BottleTipGuard()
    assert feed(failed, 0, 1)
    assert failed.observe(quaternion(0), 1.01, True)


def test_pause_reordered_messages_and_invalid_quaternion():
    guard = core.BottleTipGuard()
    assert not guard.observe(quaternion(90), 1, False)
    for _ in range(100):
        assert not guard.observe(quaternion(90), 1, False)
    assert not guard.observe(quaternion(0), .5, False)
    assert not guard.observe((0, 0, 0, 0), 1.1, False)
    assert not feed(guard, 1.11, 2.1)
    assert guard.observe(quaternion(90), 2.11, False)


def test_truth_callback_ignores_other_objects_and_opening_away_from_box():
    guard = core.BottleTipGuard()
    node = SimpleNamespace(_lock=threading.RLock(), _bottle_tip_guard=guard,
        _box_geometry=(.32, .14, .055), _status={'was_grasped': True, 'release_started': True},
        _evidence={})
    def message(t, x=.5, frame='BeverageBottle'):
        q = quaternion(90)
        return SimpleNamespace(transforms=[SimpleNamespace(child_frame_id=frame,
            header=SimpleNamespace(stamp=SimpleNamespace(sec=int(t), nanosec=round((t-int(t))*1e9))),
            transform=SimpleNamespace(translation=SimpleNamespace(x=x,y=.14,z=.2),
            rotation=SimpleNamespace(x=q[0],y=q[1],z=q[2],w=q[3])))])
    SmolVLASimRollout._bottle_pose_callback(node, message(0, frame='Cup'))
    assert guard.last_stamp is None
    for i in range(101):
        SmolVLASimRollout._bottle_pose_callback(node, message(i/100))
    assert guard.tripped
    node._bottle_tip_guard = core.BottleTipGuard()
    for i in range(201):
        SmolVLASimRollout._bottle_pose_callback(node, message(i/100, x=.32))
    assert node._bottle_tip_guard.placement_started
    assert not node._bottle_tip_guard.tripped


def test_fall_during_inference_stops_before_next_motion_and_returns_failure(tmp_path, monkeypatch):
    import numpy as np
    from z1_isaac_bridge import smolvla_sim_rollout as module
    monkeypatch.setattr(module, 'VideoSet', lambda *a: SimpleNamespace(write=lambda _: None, close=lambda: None))
    args = module.parse_args(['--checkpoint', str(tmp_path)])
    joints = np.array([0.] * 6 + [-.9])
    commands, active, errors = [], [], []
    node = SimpleNamespace(args=args, _lock=threading.RLock(), _randomization_status={},
        _joint_sim_stamp=0., _joints=dict(zip(core.JOINT_NAMES, joints)),
        _episode_seed=lambda *a: 123, _set_active=active.append,
        _snapshot=lambda: ({}, joints.copy(), {}, {}), _policy_batch=lambda *a: {},
        _publish_grasp_phase=lambda *a: None, _publish_command=commands.append,
        get_logger=lambda: SimpleNamespace(error=errors.append))
    def predict(_):
        assert feed(node._bottle_tip_guard, 0, 1)
        return np.array([[.06]*6 + [-.9]])
    node._predict = predict
    result = SmolVLASimRollout.run_episode(node, 'beverage_bottle', 1, tmp_path)
    assert result['success'] is False
    assert result['failure_category'] == result['stop_reason'] == 'bottle_tipped'
    assert len(commands) == 1  # Cleanup holds current pose; no policy action is sent.
    assert commands[0] == pytest.approx(joints)
    assert active[-1] is False
    assert len(errors) == 1
    assert node._bottle_tip_guard is None
    assert not (next((tmp_path/'episodes').iterdir())/'first_contact.json').exists()
