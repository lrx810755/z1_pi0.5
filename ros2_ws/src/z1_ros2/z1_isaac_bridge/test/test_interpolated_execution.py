from types import SimpleNamespace
import threading
import numpy as np
import pytest
from z1_isaac_bridge import smolvla_rollout_core as core
from z1_isaac_bridge.smolvla_sim_rollout import SmolVLASimRollout, parse_args


def sample_gate(gate, command, current, status, now):
    # Match production: the first command starts waiting; callbacks then update contact.
    if gate.started is None:
        gate.apply(command, current, now)
    gate.observe(status, now)
    return gate.apply(command, current, now)


def test_arm_waits_for_continuous_bilateral_contact_and_keeps_closing():
    assert hasattr(core, 'PreliftContactGate')
    gate = core.PreliftContactGate(0.5, 2.0)
    current = np.array([0.] * 6 + [-.9])
    requested = np.array([.03] * 6 + [-.65])
    assert sample_gate(gate, requested, current, {}, 0.)[:6] == pytest.approx(current[:6])
    assert sample_gate(gate, requested, current, {}, .1)[-1] == -.65
    sample_gate(gate, requested, current, {'bilateral_contact': True}, .2)
    sample_gate(gate, requested, current, {'bilateral_contact': False}, .6)
    sample_gate(gate, requested, current, {'bilateral_contact': True}, .7)
    assert sample_gate(gate, requested, current, {'bilateral_contact': True}, 1.1)[:6] == pytest.approx(current[:6])
    assert sample_gate(gate, requested, current, {'bilateral_contact': True}, 1.2) == pytest.approx(requested)


def test_contact_timeout_does_not_allow_lift():
    assert hasattr(core, 'PreliftContactGate')
    gate = core.PreliftContactGate(.5, 2.)
    request = np.array([.03]*6 + [-.65])
    sample_gate(gate, request, np.zeros(7), {}, 0.)
    with pytest.raises(RuntimeError, match='prelift_contact_not_stable'):
        sample_gate(gate, request, np.zeros(7), {}, 2.)


def test_60hz_execution_preserves_chunk_duration_and_endpoint(tmp_path, monkeypatch):
    from z1_isaac_bridge import smolvla_sim_rollout as module
    # A policy stub isolates execution timing from GPU inference and ROS transport.
    monkeypatch.setattr(module, 'VideoSet', lambda *a: SimpleNamespace(write=lambda _: None, close=lambda: None))
    args = parse_args(['--checkpoint', str(tmp_path)])
    args.execution_rate = 60.
    args.prelift_contact_stable_s = 0.
    args.prelift_contact_timeout_s = 2.
    joints = np.array([0.] * 6 + [-.9])
    commands = []
    stop = threading.Event()
    node = SimpleNamespace(args=args, _lock=threading.RLock(), _randomization_status={},
        _joint_sim_stamp=0., _joints=dict(zip(core.JOINT_NAMES,joints)),
        _episode_seed=lambda *a: 123, _set_active=lambda _: None,
        _snapshot=lambda: ({}, joints.copy(), {}, {}), _policy_batch=lambda *a: {},
        _publish_grasp_phase=lambda *a: None, _predict=lambda _: np.array([[.06]*6+[-.9], [.12]*6+[-.9]]))
    def publish(command):
        commands.append((node._joint_sim_stamp, np.array(command).copy()))
        joints[:] = command
        if len(commands) == 12: stop.set()
    def wait(stamp):
        node._joint_sim_stamp = stamp + 1./60.

    node._publish_command = publish
    node._wait_sim_period = wait
    SmolVLASimRollout.run_episode(node, 'beverage_bottle', 1, tmp_path, stop_event=stop)
    motion = commands[:-1]  # finally holds the measured pose.
    assert len(motion) == 12
    assert [x[0] for x in motion] == pytest.approx(np.arange(1,13)/60.)
    assert [x[1][0] for x in motion] == pytest.approx(np.arange(1,13)*.01)
    assert node._joint_sim_stamp == pytest.approx(.2)


def test_release_detects_slow_opening_at_60hz():
    tracker = core.GraspPhaseTracker()
    tracker.update(-.65,-.72,{'physical_grasp_ok':True})
    phase = None
    for jaw in [-.652,-.654,-.656]:
        phase = tracker.update(jaw,-.72,{'physical_grasp_ok':True})
    assert phase['release_started']


@pytest.mark.parametrize('rate', ['0','-1','nan','inf','55'])
def test_execution_rate_rejects_invalid_or_fractional_substeps(tmp_path, rate):
    with pytest.raises(SystemExit):
        parse_args(['--checkpoint', str(tmp_path), '--execution-rate', rate])


def test_contact_settling_after_deadline_is_rejected():
    gate=core.PreliftContactGate(.5,2.)
    command=np.array([.03]*6+[-.65])
    sample_gate(gate, command,np.zeros(7),{},0.)
    sample_gate(gate, command,np.zeros(7),{'bilateral_contact':True},1.8)
    with pytest.raises(RuntimeError,match='prelift_contact_not_stable'):
        sample_gate(gate, command,np.zeros(7),{'bilateral_contact':True},2.3)


def test_callback_observation_resets_contact_during_inference():
    assert hasattr(core.PreliftContactGate,'observe')
    gate=core.PreliftContactGate(.5,2.)
    command=np.array([.03]*6+[-.65]);current=np.zeros(7)
    sample_gate(gate, command,current,{'bilateral_contact':True},0.)
    gate.observe({'bilateral_contact':False},.3)
    gate.observe({'bilateral_contact':True},.4)
    assert sample_gate(gate, command,current,{'bilateral_contact':True},.6)[:6] == pytest.approx(current[:6])
    assert sample_gate(gate, command,current,{'bilateral_contact':True},.9) == pytest.approx(command)


def test_contact_gate_requires_fresh_status_telemetry(tmp_path):
    import time
    args=parse_args(['--checkpoint',str(tmp_path),'--prelift-contact-stable-s','.5'])
    node=SimpleNamespace(_lock=threading.RLock(),args=args,_stamps={'joint':time.monotonic()},
        camera_keys=[],_images={},_joints=dict(zip(core.JOINT_NAMES,np.zeros(7))),
        _status={'bilateral_contact':True},_evidence={})
    with pytest.raises(RuntimeError,match='stale_grasp_status'):
        SmolVLASimRollout._snapshot(node)


def test_native_gate_does_not_overwrite_contact_callback_with_old_snapshot():
    gate=core.PreliftContactGate(.5,2.)
    command=np.array([.03]*6+[-.65]);current=np.zeros(7)
    sample_gate(gate, command,current,{'bilateral_contact':True},0.)
    gate.observe({'bilateral_contact':False},.3)
    gate.apply(command,current,.31)
    assert gate.contact_since is None


def test_contact_telemetry_gap_restarts_continuous_interval():
    gate=core.PreliftContactGate(.5,2.)
    command=np.array([.03]*6+[-.65]);current=np.zeros(7)
    sample_gate(gate, command,current,{'bilateral_contact':True},0.)
    gate.observe({'bilateral_contact':True},1.)
    assert sample_gate(gate, command,current,{'bilateral_contact':True},1.1)[:6] == pytest.approx(current[:6])


def test_waiting_for_contact_preserves_arm_action_points(tmp_path, monkeypatch):
    from z1_isaac_bridge import smolvla_sim_rollout as module
    monkeypatch.setattr(module,'VideoSet',lambda *a:SimpleNamespace(write=lambda _:None,close=lambda:None))
    args=parse_args(['--checkpoint',str(tmp_path),'--execution-rate','60','--prelift-contact-stable-s','.5'])
    joints=np.array([0.]*6+[-.74]);commands=[];stop=threading.Event()
    node=SimpleNamespace(args=args,_lock=threading.RLock(),_randomization_status={},_joint_sim_stamp=0.,
        _joints=dict(zip(core.JOINT_NAMES,joints)),_episode_seed=lambda *a:123,_set_active=lambda _:None,
        _snapshot=lambda:({},joints.copy(),{'bilateral_contact':True},{}),_policy_batch=lambda *a:{},
        _publish_grasp_phase=lambda *a:None,_predict=lambda _:np.array([[.06]*6+[-.65],[.12]*6+[-.65]]))
    def wait(stamp):
        node._joint_sim_stamp=stamp+1/60.
        node._contact_gate.observe({'bilateral_contact':True},node._joint_sim_stamp)
    def publish(command):
        commands.append((node._joint_sim_stamp,np.array(command).copy()));joints[:]=command
        if command[0]>=.12-1e-8:stop.set()
    node._wait_sim_period=wait;node._publish_command=publish
    SmolVLASimRollout.run_episode(node,'beverage_bottle',1,tmp_path,stop_event=stop)
    motion=[(t,c) for t,c in commands[:-1] if c[0]>1e-8]
    assert motion[0][0]>=.5
    assert [c[0] for _,c in motion] == pytest.approx(np.arange(1,13)*.01)
    assert [c[-1] for _,c in motion] == pytest.approx([-.65]*12)
