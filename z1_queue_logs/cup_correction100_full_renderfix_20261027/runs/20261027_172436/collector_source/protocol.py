"""Quota and acceptance rules for complete, explicitly synthetic recoveries."""
import json
import math
import os
import time
from pathlib import Path

QUOTAS = {'approach': 20, 'close_transition': 15, 'shifted_target': 20,
          'low_clearance': 10, 'regrasp': 10, 'transport': 10, 'placement': 15}
DATA = Path('/home/lp531/smolvla_data/z1_cup_correction100_full_renderfix_20261027')


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
    reference_root = Path('/home/lp531/smolvla_data/z1_tabletop_raw_cup_basic_4x5x20_renderfix_20261026')
    references = {}
    for folder in sorted(reference_root.glob('episode_[0-9]*')):
        summary = json.loads((folder / 'episode_summary.json').read_text())
        if summary.get('task_success'):
            cell = tuple(summary['scene_randomization']['target_grid'])
            references.setdefault(cell, []).append((folder, summary))
    tasks = []
    for kind, quota in QUOTAS.items():
        for i in range(quota):
            n = len(tasks)
            cell = n % 20
            grid = [cell % 5, cell // 5]
            direction = -1 if i % 2 else 1
            offset = [0., 0., 0.]
            rotation = [0., 0., 0.]
            if kind == 'approach':
                variant = i % 5
                if variant < 2:
                    offset[1] = direction * (.012 + .006 * ((i // 5) % 3))
                elif variant == 2:
                    offset[0] = -.025 - .005 * ((i // 5) % 2)
                elif variant == 3:
                    offset[0] = .004
                    offset[1] = direction * .012
                else:
                    offset[1] = direction * .01
                    rotation[1] = direction * 2.
            elif kind == 'low_clearance':
                offset = [-.004, direction * .004, -.001 - .0005 * (i % 3)]
            elif kind == 'regrasp':
                offset = [0., direction * .004, 0.]
            elif kind in ('transport', 'placement'):
                offset = [-.002, direction * .003, 0.]
            task = {
                'id': f'cup_correction_{n:03d}_{kind}', 'kind': kind, 'grid': grid,
                'split': 'validation' if tuple(grid) in ((1, 1), (2, 3)) else 'train',
                'source_type': 'synthetic_failure_informed',
                'source_report': '/home/lp531/smolvla_eval/act_cup_only_frustum_steps15_paired30_20261025/失败原因分类.md',
                'synthesis_note': 'Bounded perturbation around a newly planned minus90 reference, not exact policy-state replay.',
                'route_kind': 'close_ready' if kind == 'close_transition' else kind,
                'offset_approach_m': offset, 'rotation_tool_deg': rotation,
                'preparation_retreat_m': 0. if kind == 'close_transition' else .004,
                'requested_shift_world_m': [direction * (.008 + .002 * (i % 4)), 0., 0.],
                'loaded_offset_world_m': [direction * (.018 + .004 * (i % 3)), direction * .012,
                                          .025 if i % 3 == 0 else .012],
                'variant': i,
            }
            if kind == 'shifted_target':
                task['offset_approach_m'] = [0., 0., 0.]
            if kind == 'placement':
                task['loaded_offset_world_m'] = [0., 0., .025 + .005 * (i % 3)]
            choices = references[tuple(grid)]
            folder, summary = choices[(n // 20) % len(choices)]
            task['reference_basic_episode'] = str(folder)
            task['seed'] = summary['scene_randomization']['seed']
            task['reference_scene'] = summary['scene_randomization']
            tasks.append(task)
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
        shift = meta.get('prepared_cup_shift', {})
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
