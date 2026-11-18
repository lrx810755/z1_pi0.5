"""Acceptance rules for original-scene, minus-90, complete demonstrations."""
import json
import math
import os
import itertools
from pathlib import Path
import numpy as np


def round_indices(until_complete):
    return itertools.count(1) if until_complete else range(1,4)


class ProgressGuard:
    def __init__(self):
        self.stalled_rounds = 0

    def observe(self, before, after):
        self.stalled_rounds = self.stalled_rounds+1 if after <= before else 0
        return self.stalled_rounds < 3


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


def preclose_correction(actual, desired, tolerance, envelope):
    actual = np.asarray(actual, float); desired = np.asarray(desired, float)
    if actual.shape != (3,) or desired.shape != (3,) or not np.isfinite(actual).all():
        raise ValueError('Invalid preclose measurement')
    inside = (envelope['x_min_m'] <= actual[0] <= envelope['x_max_m']
        and abs(actual[1]) <= envelope['y_abs_max_m']
        and envelope['z_min_m'] <= actual[2] <= envelope['z_max_m'])
    if inside and np.all(abs(actual-desired) <= tolerance):
        return None
    correction = actual-desired
    if np.linalg.norm(correction) > .012:
        raise ValueError('Preclose deviation exceeds bounded 12mm correction')
    return correction.tolist()


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


def archived_shift_position(original):
    """Compare setup relocation against archived setup, before approach contact."""
    return original['prepared_cup_shift']['after']
