"""Quota and acceptance rules for complete, explicitly synthetic recoveries."""
import json
import math
import os
import time
from pathlib import Path

QUOTAS = {'approach':20,'low_clearance':10}
DATA = Path('/home/lp531/smolvla_data/bottle_approach_height30_20261112')


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def build_tasks():
    return json.loads((Path(__file__).resolve().parent/'retake_tasks.json').read_text())


def command_labels(rows, events):
    commands = sorted((e for e in events if e.get('event') == 'gripper_intent'),
                      key=lambda e: e['time'])
    labels, cursor, state = [], 0, 0
    for row in rows:
        stamp = float(row['simulation_time'])
        while cursor < len(commands) and commands[cursor]['time'] <= stamp + 1e-7:
            state = int(commands[cursor]['state'] == 'close')
            cursor += 1
        labels.append(state)
    rises = [i for i in range(1, len(labels)) if labels[i] and not labels[i - 1]]
    if not rises or not labels or labels[0] or labels[-1]:
        raise ValueError('Missing full open/close/open command sequence')
    close = rises[-1]
    release = next((i for i in range(close + 1, len(labels)) if not labels[i]), None)
    if release is None or not any(float(r['action'][-1]) > -.75 for r in rows[close:release]):
        raise ValueError('Final closing command has no recorded closing action')
    if not any(float(r['action'][-1]) <= -.88 for r in rows[release:]):
        raise ValueError('Final release has no recorded opening action')
    return labels, close, release


def training_mask(rows, events, horizon=16):
    intervals, start = [], None
    for event in sorted(events, key=lambda e: e['time']):
        if event['event'] == 'intervention_start':
            if start is not None:
                raise ValueError('Overlapping disturbance interventions')
            start = event['time']
        elif event['event'] == 'intervention_end':
            if start is None:
                raise ValueError('Disturbance end without start')
            intervals.append((start, event['time']))
            start = None
    if start is not None:
        raise ValueError('Unfinished disturbance intervention')
    mask = [not any(a - 1e-7 <= float(r['simulation_time']) <= b + 1e-7
                    for a, b in intervals) for r in rows]
    anchors = [i + horizon <= len(mask) and all(mask[i:i + horizon])
               for i in range(len(mask))]
    return mask, anchors


def validate_category(task, meta):
    if not meta.get('recovery_complete'):
        raise ValueError('Correction recovery incomplete')
    kind = task['kind']
    if kind == 'approach' and meta.get('measured_start_deviation_m', 0.) < .006:
        raise ValueError('Missing measured approach deviation')
    if kind == 'approach':
        delta=[float(a)-float(b) for a,b in zip(meta['actual_start_position_m'],meta['reference_position_m'])]
        if math.hypot(delta[0],delta[1]) < .006:
            raise ValueError('Approach requires measured >=6mm horizontal error')
    if kind == 'low_clearance':
        dz=meta.get('measured_world_z_deviation_m',0.)
        if not -.015 <= dz <= -.0015:
            raise ValueError('Bottle height deviation must be measured between -15 and -1.5mm')
    if kind == 'close_transition' and not meta.get('close_transition_verified'):
        raise ValueError('Missing close transition')
    if kind == 'shifted_target':
        shift = meta.get('prepared_bottle_shift', {})
        if not .006 <= shift.get('distance_m', 0.) <= .018 or not meta.get('relocalized'):
            raise ValueError('Missing measured target relocation and recovery')
    if kind == 'low_clearance' and meta.get('measured_world_z_deviation_m', 0.) > -.0005:
        raise ValueError('Missing measured low approach state')
    if kind == 'regrasp' and not (meta.get('retry_performed') and meta.get('retry_contact_seen')):
        raise ValueError('Retry requires measured contact and reopening')
    if kind == 'regrasp' and not meta.get('retry_unilateral_contact'):
        raise ValueError('Retry requires measured unilateral contact before recovery')
    if kind in ('transport', 'placement'):
        if meta.get('measured_disturbance_m', 0.) < .008:
            raise ValueError('Missing measured loaded disturbance')
        if not meta.get('recovery_after_disturbance'):
            raise ValueError('Missing loaded recovery')
    if kind == 'placement' and not (
            meta.get('placement_hover_duration_s', 0.) >= .3
            and meta.get('placement_xy_error_m', 1.) <= .02
            and meta.get('placement_vertical_excess_m', 0.) >= .015):
        raise ValueError('Placement requires measured centered high hover before descent')


def validate_preparation(clearance, table_contact):
    if not math.isfinite(clearance) or clearance < .003 or table_contact:
        raise ValueError('Actual preparation must have3mm clearance and no finger/table contact')


def validate_withdrawal(meta):
    if meta.get('withdrawal_rise_m', 0.) < .03:
        raise ValueError('Missing measured successful post-release withdrawal of3cm')


def wait_simulation_duration(now, duration, *, sleep=time.sleep,
                             wall_now=time.monotonic, wall_timeout=120., check_stop=None):
    start = now()
    deadline = wall_now() + wall_timeout
    while now() - start < duration:
        if check_stop is not None:
            check_stop()
        if wall_now() >= deadline:
            raise TimeoutError('Simulation clock did not advance in the allowed wall time')
        sleep(.05)
    return now() - start


def validate_final(status, roll, table_contact, dynamic_samples, dynamic_violations, tail):
    for key in ('task_success', 'physical_grasp_ok', 'placement_stable',
                'release_sequence_complete', 'released', 'target_collisions_enabled'):
        if status.get(key) is not True:
            raise ValueError('Final physical acceptance failed: ' + key)
    if status.get('holding_observed') or status.get('frozen') or table_contact:
        raise ValueError('Holding/frozen/finger-table contact invalidates episode')
    if not math.isfinite(roll) or abs(roll + 90.) > 3.:
        raise ValueError('Final preclose FK roll must be -90±3 degrees')
    if dynamic_samples < 1 or dynamic_violations:
        raise ValueError('Missing/invalid dynamic physics evidence')
    if tail < 16:
        raise ValueError('At least16 release-tail frames required')
