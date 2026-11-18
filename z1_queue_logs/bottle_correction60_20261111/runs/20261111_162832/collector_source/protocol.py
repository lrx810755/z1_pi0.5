"""Quota and acceptance rules for complete, explicitly synthetic recoveries."""
import json
import math
import os
import time
from pathlib import Path

QUOTAS = {kind: 12 for kind in ('close_transition', 'regrasp', 'shifted_target', 'transport', 'placement')}
DATA = Path('/home/lp531/smolvla_data/bottle_correction60_20261111')


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
    reference_root = Path('/home/lp531/smolvla_data/bottle_horizontal50_after_basic_20261109')
    references = {}
    for folder in sorted(reference_root.glob('episode_[0-9]*')):
        summary = json.loads((folder / 'episode_summary.json').read_text())
        if summary.get('task_success') and summary.get('target_object') == 'beverage_bottle':
            cell = tuple(summary['scene_randomization']['target_grid'])
            references.setdefault(cell, []).append((folder, summary))
    kinds = list(QUOTAS)
    counts = dict.fromkeys(kinds, 0)
    tasks = []
    for cell in range(20):
        grid = [cell % 5, cell // 5]
        if tuple(grid) not in references:
            raise ValueError('Missing physically successful reference cell: ' + str(grid))
        for slot in range(3):
            kind = kinds[(cell * 3 + slot) % len(kinds)]
            i = counts[kind]
            counts[kind] += 1
            direction = 1 if i % 2 == 0 else -1
            folder, summary = references[tuple(grid)][slot % len(references[tuple(grid)])]
            task = dict(id=f'bottle_correction_{len(tasks):03d}_{kind}', kind=kind,
                grid=grid, split='train', source_type='synthetic_coverage_gap_curriculum',
                source_report='/home/lp531/pi05_lora_workspace/four_object_correction_audit_20261110/report.json',
                synthesis_note='Synthetic bounded recoverable state; not an exact policy failure replay. Bottle production calibration retained.',
                route_kind='close_ready' if kind == 'close_transition' else kind,
                offset_approach_m=[0., direction * .004 if kind == 'regrasp' else 0., 0.],
                rotation_tool_deg=[0., 0., 0.],
                preparation_retreat_m=0. if kind == 'close_transition' else .004,
                requested_shift_world_m=([direction * (.008 + .002 * (i % 3)), 0., 0.]
                    if i % 4 < 2 else [0., direction * (.008 + .002 * (i % 3)), 0.]),
                loaded_offset_world_m=([0.,0.,.020 + .005 * (i % 3)] if kind == 'placement'
                    else [direction * (.016 + .004 * (i % 3)), direction * .010, .008]),
                variant=i, reference_basic_episode=str(folder),
                seed=summary['scene_randomization']['seed'], reference_scene=summary['scene_randomization'])
            tasks.append(task)
    if counts != QUOTAS:
        raise AssertionError('Category schedule is unbalanced')
    return tasks


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
