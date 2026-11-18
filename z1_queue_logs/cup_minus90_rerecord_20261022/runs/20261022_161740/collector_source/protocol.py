"""Acceptance rules for original-scene, minus-90, complete demonstrations."""
import json
import math
import os
from pathlib import Path
import numpy as np


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def check_scene(expected, actual):
    if actual.get('seed') != expected['seed'] or actual.get('target_grid') != expected['grid']:
        raise ValueError('Scene seed/grid differs from archived demonstration')
    scene = expected['scene_randomization']
    positions = scene.get('original_object_positions_m', scene['object_positions_m'])
    errors = {}; angles = {}
    for name, xyz in positions.items():
        errors[name] = float(np.linalg.norm(np.array(xyz) - actual['object_positions_m'][name]))
        q = np.array(scene['object_orientations_xyzw'][name], float)
        other = np.array(actual['object_orientations_xyzw'][name], float)
        cosine = min(1., abs(float(q @ other)) / np.linalg.norm(q) / np.linalg.norm(other))
        angles[name] = math.degrees(2 * math.acos(cosine))
    if max(errors.values()) > .002 or max(angles.values()) > 2.:
        raise ValueError(f'Archived scene reproduction mismatch: positions={errors}, angles={angles}')
    return dict(maximum_position_error_m=max(errors.values()), maximum_orientation_error_deg=max(angles.values()),
                position_errors_m=errors, orientation_errors_deg=angles)


def full_labels(rows):
    angles = np.array([r['action'][-1] for r in rows])
    strong = np.flatnonzero(angles > -.75)
    if not len(strong):
        raise ValueError('No confirmed closing command')
    held = int(strong[0]); before = np.flatnonzero(angles[:held] <= -.895)
    if not len(before):
        raise ValueError('No open command before closing')
    close = int(before[-1] + 1)
    opened = np.flatnonzero((np.arange(len(angles)) > held) & (angles <= -.899))
    if not len(opened):
        raise ValueError('Complete demonstration is missing release')
    end = int(opened[0]); start = end
    while start > held and angles[start] < angles[start - 1] - 1e-4:
        start -= 1
    release = start + 1
    if not close < held < release <= end or max(angles[end:]) >= -.88:
        raise ValueError('Ambiguous close/release command sequence')
    labels = [int(close <= i < release) for i in range(len(rows))]
    return labels, close, release


def tool_roll(rotation):
    x = rotation[:, 0]; up = np.array([0., 0., 1.])
    z = up - float(up @ x) * x; z /= np.linalg.norm(z)
    y = np.cross(z, x)
    return math.degrees(math.atan2(rotation[:, 1] @ z, rotation[:, 1] @ y))


def validate_saved(status, roll_deg, table_contact, release_tail_frames):
    required = ['task_success', 'physical_grasp_ok', 'release_sequence_complete',
                'placement_stable', 'released', 'target_collisions_enabled']
    if status.get('target_object') != 'cup' or any(status.get(k) is not True for k in required):
        raise ValueError('Physical grasp, stable box placement and release must all pass')
    if status.get('holding_observed') or status.get('frozen') or table_contact:
        raise ValueError('Holding/frozen/finger-table contact disqualifies demonstration')
    if abs(roll_deg + 90.) > 3.:
        raise ValueError(f'Preclose grasp roll is not minus90: {roll_deg:.3f}')
    if release_tail_frames < 16:
        raise ValueError('Require at least 16 frames from release through episode end')
