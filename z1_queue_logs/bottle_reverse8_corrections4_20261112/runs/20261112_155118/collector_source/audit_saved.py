"""Independently audit saved correction episodes without changing raw recordings."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import xml.etree.ElementTree as ET

from protocol import (DATA, QUOTAS, atomic_json, command_labels, training_mask,
                      validate_category, validate_final, validate_preparation, validate_withdrawal)

RUN = Path('/home/lp531/z1_queue_logs/bottle_reverse8_corrections4_20261112')
URDF = Path('/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/generated_dual_camera/z1_with_dual_d435_resolved.urdf')


def audit():
    plan = json.loads((RUN / 'collection_plan.json').read_text())
    tasks = {t['id']: t for t in plan['tasks']}
    robot = ET.parse(URDF).getroot()
    chain = []
    for i in range(1, 7):
        joint = robot.find(f"joint[@name='joint{i}']")
        origin = Rotation.from_euler('xyz', np.fromstring(joint.find('origin').get('rpy'), sep=' ')).as_matrix()
        chain.append((origin, np.fromstring(joint.find('axis').get('xyz'), sep=' ')))
    verified, seen = [], set()
    for path in sorted(DATA.glob('episode_[0-9]*')):
        meta = json.loads((path / 'correction.json').read_text())
        summary = json.loads((path / 'episode_summary.json').read_text())
        task = tasks[meta['task_id']]
        assert meta['task_id'] not in seen
        seen.add(meta['task_id'])
        assert meta['accepted'] and meta['plan_hash'] == plan['hash']
        assert all(summary[k] for k in ('task_success', 'physical_grasp_ok', 'release_sequence_complete', 'placement_stable'))
        assert summary['scene_randomization']['target_grid'] == task['grid']
        reference = json.loads((Path(meta['task']['reference_basic_episode']) / 'episode_summary.json').read_text())
        assert summary['scene_randomization']['seed'] == reference['scene_randomization']['seed']
        raw = (path / 'frames.jsonl').read_bytes()
        import sys
        sys.path.insert(0,'/home/lp531/robot/reverse8_queue_20261112')
        from quality_gate import screen
        screen(path)
        timing = json.loads((path / 'time_screen.json').read_text())
        assert timing['passed'] and timing['longest_candidate_wait_s'] < 8
        assert timing['frames_sha256'] == hashlib.sha256(raw).hexdigest()
        rows = [json.loads(line) for line in raw.splitlines()]
        assert len(rows) == summary['frame_count'] and summary['dropped_frames'] == 0
        assert rows and all(row['target_object'] == 'beverage_bottle' for row in rows)
        for key in ('observation.state', 'action'):
            values = np.asarray([row[key] for row in rows], dtype=float)
            assert values.shape == (len(rows), 7) and np.isfinite(values).all()
        assert np.allclose(np.diff([row['simulation_time'] for row in rows]), .1, atol=1e-6)
        assert all(row['action_stamp'] <= row['simulation_time'] + 1e-7 for row in rows)
        labels, close, release = command_labels(rows, meta['events'])
        saved_labels = json.loads((path / 'gripper_labels.json').read_text())
        assert labels == saved_labels['labels']
        mask, anchors = training_mask(rows, meta['events'], horizon=30)
        saved_mask = json.loads((path / 'training_mask.json').read_text())
        assert mask == saved_mask['expert_frame_mask']
        assert anchors == saved_mask['valid_action_horizon30_anchors']
        rotation = np.eye(3)
        for angle, (origin, axis) in zip(rows[close - 1]['observation.state'][:6], chain):
            rotation = rotation @ origin @ Rotation.from_rotvec(angle * axis).as_matrix()
        x, z = rotation[:, 0], np.array([0., 0., 1.])
        z = z - (z @ x) * x
        z /= np.linalg.norm(z)
        y = np.cross(z, x)
        roll = float(np.degrees(np.arctan2(rotation[:, 1] @ z, rotation[:, 1] @ y)))
        validate_final(meta['final_status'], roll, meta['table_contact'],
                       meta['dynamic_samples'], meta['dynamic_violations'], len(rows) - release)
        validate_category(task, meta)
        validate_preparation(meta['measured_finger_table_clearance_m'], meta['preparation_table_contact_seen'])
        validate_withdrawal(meta)
        physics = [json.loads(line) for line in (path / 'physics_status.jsonl').read_text().splitlines()]
        assert physics and len(physics) == meta['dynamic_samples']
        assert all(p['status']['frozen'] is False and p['status']['target_collisions_enabled'] is True for p in physics)
        image_hashes = hashlib.sha256()
        for row in rows:
            for camera, expected in [('wrist', (640, 360)), ('global', (640, 480)), ('global_right', (640, 480))]:
                with Image.open(path / row['observation.images.' + camera]) as image:
                    assert image.size == expected
                    image.load()
                image_hashes.update(hashlib.sha256((path / row['observation.images.' + camera]).read_bytes()).digest())
        verified.append(dict(path=str(path), task_id=task['id'], kind=task['kind'], grid=task['grid'],
            rerecord_source=task['rerecord_source'], source_pool=task['source_pool'],
            split=task['split'], frames=len(rows), close_frame=close, release_frame=release,
            roll_deg=roll, withdrawal_rise_m=meta['withdrawal_rise_m'],
            expert_frames=sum(mask), valid_horizon30_anchors=sum(anchors),
            frames_sha256=hashlib.sha256(raw).hexdigest(), image_content_digest=image_hashes.hexdigest(),
            preclose_images={c: str(path / rows[close - 1]['observation.images.' + c])
                             for c in ('wrist', 'global', 'global_right')}))
    counts = Counter(row['kind'] for row in verified)
    result = dict(verified=True, accepted=len(verified), complete=counts == QUOTAS,
                  by_kind=dict(counts), quotas=QUOTAS, episodes=verified)
    atomic_json(RUN / 'saved_integrity.json', result)
    atomic_json(RUN / 'replacement_map.json', [dict(old=e['rerecord_source'],new=e['path'],pool=e['source_pool']) for e in verified])
    print(json.dumps({k: v for k, v in result.items() if k != 'episodes'}, ensure_ascii=False))
    return result


if __name__ == '__main__':
    audit()
