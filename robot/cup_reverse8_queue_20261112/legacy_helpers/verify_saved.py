"""Audit saved episodes independently against archived scenes and actual joint FK."""
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
from protocol import atomic_json, full_labels, tool_roll, validate_saved

RUN = Path(__file__).resolve().parent.parent
PLAN = json.loads((RUN/'collection_plan.json').read_text())
DATA = Path(PLAN['data_root'])
TASKS = {t['id']:t for t in PLAN['tasks']}
URDF = Path('/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/generated_dual_camera/z1_with_dual_d435_resolved.urdf')


def verify():
    robot = ET.parse(URDF).getroot(); chain = []
    for i in range(1,7):
        j = robot.find(f"joint[@name='joint{i}']")
        origin = Rotation.from_euler('xyz', np.fromstring(j.find('origin').get('rpy'),sep=' ')).as_matrix()
        chain.append((origin,np.fromstring(j.find('axis').get('xyz'),sep=' ')))
    episodes = []; seen = set()
    for folder in ['basic_replacements','corrections_full']:
        for path in sorted((DATA/folder).glob('episode_[0-9]*')):
            meta = json.loads((path/'rerecord.json').read_text())
            summary = json.loads((path/'episode_summary.json').read_text())
            start = json.loads((path/'episode_start.json').read_text())
            labels = json.loads((path/'gripper_labels.json').read_text())
            task = TASKS[meta['task_id']]
            assert meta['task_id'] not in seen
            seen.add(meta['task_id'])
            assert meta['accepted'] and meta['plan_hash']==PLAN['hash']
            assert meta['split']==task['split'] and summary['task_success'] and summary['physical_grasp_ok']
            assert start['scene_randomization']['seed']==task['seed']
            assert start['scene_randomization']['target_grid']==task['grid']
            raw = (path/'frames.jsonl').read_bytes()
            rows = [json.loads(l) for l in raw.splitlines()]
            assert summary['frame_count']==len(rows) and summary['dropped_frames']==0
            expected_labels,close,release = full_labels(rows)
            assert labels['labels']==expected_labels and labels['close_command_frame']==close and labels['open_command_frame']==release
            rotation = np.eye(3)
            for angle,(origin,axis) in zip(rows[max(0,close-1)]['observation.state'][:6],chain):
                rotation = rotation @ origin @ Rotation.from_rotvec(angle*axis).as_matrix()
            angle = tool_roll(rotation)
            assert abs(angle-meta['measured_preclose_roll_deg'])<1e-6
            validate_saved(meta['final_status'],angle,meta['table_contact'],len(rows)-release)
            for row in rows:
                assert all((path/row['observation.images.'+cam]).is_file() for cam in ['wrist','global','global_right'])
                assert 0<=row['action_age_s']<=.04
            for row in [rows[0],rows[-1]]:
                for cam in ['wrist','global','global_right']:
                    with Image.open(path/row['observation.images.'+cam]) as image:
                        image.verify()
            episodes.append(dict(path=str(path),task_id=meta['task_id'],source=task['source'],split=task['split'],
                frames=len(rows),measured_roll_deg=angle,close_frame=close,release_frame=release,
                frames_sha256=hashlib.sha256(raw).hexdigest()))
    counts=Counter(TASKS[e['task_id']]['phase'] for e in episodes)
    report=dict(verified=True,basic_replacements=counts['basic'],full_corrections=counts['correction'],
        complete=counts['basic']==36 and counts['correction']==89,episodes=episodes,
        missing=[t['id'] for t in PLAN['tasks'] if t['id'] not in seen])
    atomic_json(RUN/'saved_integrity.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['episodes','missing']},ensure_ascii=False))
    return report


if __name__=='__main__':
    verify()
