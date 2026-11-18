"""Model observations must share sensor time; safety feedback stays current."""
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from z1_isaac_bridge.recording_sync import RecordingSynchronizer
from z1_isaac_bridge.smolvla_sim_rollout import SmolVLASimRollout, parse_args, JOINT_NAMES


def populated():
    sync = RecordingSynchronizer(('wrist', 'global'))
    for name, value in [('wrist', 10), ('global', 20)]:
        sync.push(name, 1.0, np.full((2, 2, 3), value, np.uint8))
    sync.push('global', 1.2, np.full((2, 2, 3), 99, np.uint8))
    sync.push('state', .99, [0.] * 7)
    sync.push('state', 1.01, [2.] * 7)
    sync.push('state', 1.2, [99.] * 7)
    return sync


def test_pick_common_camera_time_and_interpolate_state():
    sample = populated().select_latest_observation(1.2, .21)
    assert sample['time'] == 1.0
    assert sample['state'] == pytest.approx([1.] * 7)
    assert sample['state_bracket'] == (.99, 1.01)
    assert {name: value[0] for name, value in sample['images'].items()} == {'wrist': 1., 'global': 1.}


def test_reject_old_or_misaligned_observations():
    with pytest.raises(ValueError):
        populated().select_latest_observation(1.3, .1)
    sync = populated()
    sync.buffers['wrist'].clear()
    sync.push('wrist', .95, 'wrong time')
    with pytest.raises(ValueError):
        sync.select_latest_observation(1.2, .3)


def test_reject_large_state_gap_and_reused_images():
    sync = populated()
    sync.buffers['state'].clear()
    sync.push('state', .7, [0.] * 7)
    sync.push('state', 1.3, [1.] * 7)
    with pytest.raises(ValueError):
        sync.select_latest_observation(1.2, .21)
    sync = populated()
    sync.commit(sync.select_latest_observation(1.2, .21))
    with pytest.raises(ValueError):
        sync.select_latest_observation(1.2, .21)


def test_camera_spread_is_bounded_across_three_views():
    sync = RecordingSynchronizer(('wrist', 'global', 'vla_wrist'))
    for camera, stamp in [('wrist', .98), ('global', 1.), ('vla_wrist', 1.02)]:
        sync.push(camera, stamp, camera)
    sync.push('state', .95, [0.] * 7)
    sync.push('state', 1.05, [1.] * 7)
    with pytest.raises(ValueError):
        sync.select_latest_observation(1.05, .2)


def offline_node(tmp_path):
    node = object.__new__(SmolVLASimRollout)
    node._lock = threading.RLock()
    node._observation_sync = populated()
    node._joint_sim_stamp = 1.2
    node._stamps = {name: time.monotonic() for name in ['wrist', 'global', 'joint']}
    node._joints = {'joint1': 99.}
    node.camera_keys = ['wrist', 'global']
    node.state_dim = 7
    node.device = torch.device('cpu')
    node.args = SimpleNamespace(policy_type='pi05', input_sync_max_age=.21, input_sync_timeout=.015,
                                stale_timeout=.5, reject_blank_cameras=False)
    node._input_sync_path = tmp_path/'input_sync.jsonl'
    node._input_sync_index = 0
    node._last_input_sync = None
    return node


def test_policy_batch_uses_paired_data_and_preserves_latest_feedback(tmp_path):
    node = offline_node(tmp_path)
    wrong = {name: np.full((2, 2, 3), 255, np.uint8) for name in node.camera_keys}
    batch = node._policy_batch(wrong, np.array([99.] * 7), {}, 'pick bottle')
    assert batch['observation.state'].tolist()[0] == pytest.approx([1.] * 7)
    assert float(batch['observation.images.wrist'][0, 0, 0, 0]) == pytest.approx(10/255)
    assert float(batch['observation.images.global'][0, 0, 0, 0]) == pytest.approx(20/255)
    assert node._joints['joint1'] == 99.
    assert node._last_input_sync['sensor_skew_s'] == 0.
    assert node._input_sync_path.exists()


def test_sync_timeout_does_not_build_wrong_time_batch(tmp_path):
    node = offline_node(tmp_path)
    node._observation_sync.buffers['wrist'].clear()
    with pytest.raises(RuntimeError, match='input_sync_timeout'):
        node._policy_batch({}, np.array([99.] * 7), {}, 'pick bottle')


def test_sync_arguments_reject_invalid_values(tmp_path):
    args = parse_args(['--checkpoint', str(tmp_path), '--input-sync-tolerance', '.02'])
    assert args.input_sync_tolerance == .02
    for flag, value in [('--input-sync-tolerance', '0'), ('--input-sync-max-age', 'nan'),
                        ('--input-sync-timeout', '-1')]:
        with pytest.raises(SystemExit):
            parse_args(['--checkpoint', str(tmp_path), flag, value])


def joint_message(stamp, value):
    seconds = int(stamp)
    return SimpleNamespace(name=list(JOINT_NAMES), position=[value]*7,
                           header=SimpleNamespace(stamp=SimpleNamespace(
                               sec=seconds, nanosec=round((stamp-seconds)*1e9))))


@pytest.mark.parametrize('stamp', [1.19, .1])
def test_out_of_order_joint_does_not_roll_back_safety_feedback(tmp_path, stamp):
    node = offline_node(tmp_path)
    node._joint_callback(joint_message(stamp, -5.))
    assert node._joint_sim_stamp == 1.2
    assert node._joints['joint1'] == 99.
    assert node._observation_sync.buffers['state'][-1][0] == 1.2


def test_clock_rollback_does_not_resume_using_old_epoch_images(tmp_path):
    node = offline_node(tmp_path)
    node._observation_sync.commit(node._observation_sync.select_latest_observation(1.2, .21))
    node._joint_callback(joint_message(.1, -2.))
    assert node._joint_sim_stamp == 1.2
    assert node._joints['joint1'] == 99.
    with pytest.raises(RuntimeError, match='input_sync_timeout'):
        node._policy_batch({}, np.array([99.] * 7), {}, 'pick bottle')


def test_input_timeout_holds_and_disables_control_before_unfreeze_wait(tmp_path, monkeypatch):
    import z1_isaac_bridge.smolvla_sim_rollout as rollout
    node = offline_node(tmp_path)
    node.args = parse_args(['--checkpoint', str(tmp_path), '--freeze-target-until-close',
                            '--input-sync-timeout', '.015'])
    node._observation_sync.buffers['wrist'].clear()
    node._randomization_status = {}
    node._joints = {name: .25 for name in JOINT_NAMES}
    node._episode_seed = lambda *args: 1
    node._snapshot = lambda: ({}, np.array([.25]*7), {}, {})
    events = []
    node._set_active = lambda active: events.append(('active', active))
    node._publish_command = lambda positions: events.append(('hold', list(positions)))
    node._publish_grasp_phase = lambda *args: None
    node.get_logger = lambda: SimpleNamespace(error=lambda text: None)
    def freeze(target, frozen):
        events.append(('freeze', frozen))
        if not frozen:
            raise RuntimeError('scene_grasp_timeout:PHYSICS_READY')
    node._set_target_frozen = freeze
    monkeypatch.setattr(rollout, 'VideoSet', lambda *args: SimpleNamespace(write=lambda *args: None,
                                                                        close=lambda: None))
    result = node.run_episode('beverage_bottle', 1, tmp_path)
    assert events.index(('hold', [.25]*7)) < events.index(('freeze', False))
    assert events.index(('active', False)) < events.index(('freeze', False))
    assert result['success'] is False
    assert result['stop_reason'] == 'input_sync_timeout'
    assert result['cleanup_error'] == 'scene_grasp_timeout:PHYSICS_READY'


def test_three_camera_pairing_requires_all_views_and_preserves_right_view():
    from z1_isaac_bridge.smolvla_sim_rollout import CAMERA_TOPICS
    assert CAMERA_TOPICS['global_right'] == '/global_right_camera/color/image_raw'
    sync = RecordingSynchronizer(('wrist', 'global', 'global_right'))
    for name in ('wrist', 'global'):
        sync.push(name, 1., np.full((2, 2, 3), 10, np.uint8))
    sync.push('state', .99, [0.] * 7)
    sync.push('state', 1.01, [2.] * 7)
    with pytest.raises(ValueError):
        sync.select_latest_observation(1.01, .2)
    sync.push('global_right', 1., np.full((2, 2, 3), 30, np.uint8))
    sample = sync.select_latest_observation(1.01, .2)
    assert sample['images']['global_right'][1][0, 0, 0] == 30
    assert sample['state'] == pytest.approx([1.] * 7)
