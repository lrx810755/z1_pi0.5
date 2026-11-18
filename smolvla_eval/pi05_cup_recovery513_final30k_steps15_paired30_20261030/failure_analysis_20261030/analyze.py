"""Read-only event-level audit of PI05 cup30 evaluations and paired12-step trials."""
import collections
import csv
import json
import math
from pathlib import Path

import numpy as np

A = Path(__file__).resolve().parent
R = A.parent
PREVIOUS = Path('/home/lp531/smolvla_eval/pi05_cup_recovery513_final30k_steps12_random30_20261030')


def load(path):
    return json.loads(Path(path).read_text())


def is_pair(row):
    s = row['contact_status']
    return s.get('left_finger_contact_object') == s.get('right_finger_contact_object') == 'cup'


def longest_span(rows, predicate):
    start = previous = None
    longest = 0.
    for row in rows:
        stamp = row['simulation_elapsed_s']
        if predicate(row):
            if start is None or previous is None or stamp-previous > .15:
                start = stamp
            longest = max(longest, stamp-start)
        else:
            start = None
        previous = stamp
    return round(longest, 4)


def envelope_error(row):
    s = row['contact_status']
    p = s.get('relative_position_gripper_m')
    e = s.get('grasp_envelope')
    if not p or not e:
        return math.inf
    x, y, z = p
    return float(np.linalg.norm([max(e['x_min_m']-x, 0, x-e['x_max_m']),
                                max(abs(y)-e['y_abs_max_m'], 0),
                                max(e['z_min_m']-z, 0, z-e['z_max_m'])]))


def snapshot(rows, idx):
    row = rows[idx]
    s = row['contact_status']
    return dict(trace_index=idx, simulation_elapsed_s=row['simulation_elapsed_s'],
        wall_elapsed_s=row['elapsed_s'], actual_gripper_rad=row['joint_state'][-1],
        binary_target_rad=row['execution_target'][-1], prediction=row['policy_action'][-1],
        relative_position_gripper_m=s.get('relative_position_gripper_m'),
        envelope_error_m=envelope_error(row), object_position_m=s.get('object_position_m'),
        gripper_world_position_m=s.get('gripper_world_position_m'), tilt_deg=s.get('cylinder_tilt_deg'),
        inside_envelope=s.get('inside_grasp_envelope'), bilateral=is_pair(row),
        contact_angle_valid=s.get('contact_angle_valid'), geometry_contact_valid=s.get('geometry_contact_valid'),
        object_touching_table=s.get('object_touching_table'), relative_pose_stable=s.get('relative_pose_stable'),
        physical_test_lift_active=s.get('physical_test_lift_active'))


def audit(item):
    folder = Path(item['source_result']).parent
    result = load(folder/'result.json')
    rows = [json.loads(s) for s in (folder/'trace.jsonl').read_text().splitlines()]
    assert rows and all(np.isfinite(r['simulation_elapsed_s']) for r in rows)
    close = [i for i, r in enumerate(rows) if r['execution_target'][-1] > -.2]
    transitions = [i for i in close if i == 0 or rows[i-1]['execution_target'][-1] <= -.2]
    opened = [i for i in range(1, len(rows)) if rows[i]['execution_target'][-1] < -.8 and rows[i-1]['execution_target'][-1] > -.2]
    pair = [i for i, r in enumerate(rows) if is_pair(r)]
    best = min(close or range(len(rows)), key=lambda i: envelope_error(rows[i]))
    maximum_height = max(range(len(rows)), key=lambda i: rows[i]['contact_status']['object_position_m'][2])
    firstclose = close[0] if close else 0
    initial = np.array(result['initial_scene_randomization']['object_positions_m']['cup'])
    firstpos = np.array(rows[firstclose]['contact_status']['object_position_m'])
    closed_rows = [r for r in rows if r['execution_target'][-1] > -.2]
    frame_fields = ['inside_grasp_envelope','contact_angle_valid','geometry_contact_valid',
                    'relative_pose_stable','physical_test_lift_active','physical_grasp_candidate',
                    'physical_grasp_ok','cylinder_tilt_valid']
    ever = {key: any(r['contact_status'].get(key) is True for r in closed_rows) for key in frame_fields}
    lifted = lambda r: r['contact_status']['object_position_m'][2]-initial[2] >= .03 and r['contact_status'].get('object_touching_table') is False
    loaded = [r for r in closed_rows if is_pair(r) and lifted(r)]
    blockers = collections.Counter()
    gate_names = ['inside_grasp_envelope','contact_angle_valid','geometry_contact_valid',
                  'relative_pose_stable','physical_test_lift_active','cylinder_tilt_valid']
    for row in loaded:
        for gate in gate_names:
            if row['contact_status'].get(gate) is not True:
                blockers[gate] += 1
    best_loaded = None
    if loaded:
        row = min(loaded, key=lambda r: sum(r['contact_status'].get(g) is not True for g in gate_names))
        best_loaded = dict(simulation_elapsed_s=row['simulation_elapsed_s'],
            failed_gates=[g for g in gate_names if row['contact_status'].get(g) is not True],
            relative_position_gripper_m=row['contact_status']['relative_position_gripper_m'],
            actual_gripper_rad=row['joint_state'][-1], tilt_deg=row['contact_status']['cylinder_tilt_deg'])
    events = dict(first_close=snapshot(rows, firstclose), best_closed_alignment=snapshot(rows,best),
                  maximum_height=snapshot(rows,maximum_height), final=snapshot(rows,len(rows)-1))
    if pair:
        events['first_bilateral_contact']=snapshot(rows,pair[0])
        events['last_bilateral_contact']=snapshot(rows,pair[-1])
    if opened:
        events['first_reopen']=snapshot(rows,opened[0])
    trajectory_targets = {side: dict(collections.Counter(r['contact_status'].get(side+'_finger_contact_object')
                          for r in rows)) for side in ['left','right']}
    prediction = [json.loads(s) for s in (folder/'prediction_chunks.jsonl').read_text().splitlines()]
    tracking = np.array([np.array(r['command'][:6])-np.array(r['joint_state'][:6]) for r in rows])
    record = dict(episode=item['episode'],seed=item['seed'],final_box_success=item['success'],
        strict_success=item['legacy_success'],native_category=item['legacy_failure_category'],
        source_result=item['source_result'],trace_rows=len(rows),predictions=len(prediction),
        first_close_sim_s=rows[firstclose]['simulation_elapsed_s'],
        first_close_wall_s=rows[firstclose]['elapsed_s'],
        first_close_envelope_error_m=envelope_error(rows[firstclose]),
        first_close_inside_envelope=rows[firstclose]['contact_status'].get('inside_grasp_envelope'),
        first_close_cup_displacement_xy_m=float(np.linalg.norm((firstpos-initial)[:2])),
        min_closed_envelope_error_m=envelope_error(rows[best]),
        close_transitions=len(transitions),reopen_transitions=len(opened),ever_bilateral=bool(pair),
        ever_unilateral_only=any((r['contact_status'].get('left_finger_contact_object')=='cup') !=
                                (r['contact_status'].get('right_finger_contact_object')=='cup') for r in rows),
        longest_bilateral_sim_s=longest_span(rows,is_pair),
        longest_loaded_bilateral_sim_s=longest_span(rows,lambda r:is_pair(r) and lifted(r)),
        loaded_bilateral_rows=len(loaded),loaded_bilateral_blocker_rows=dict(blockers),best_loaded=best_loaded,
        ever_gates_during_close=ever,
        max_object_z_m=max(r['contact_status']['object_position_m'][2] for r in rows),
        max_tilt_deg=max(r['contact_status'].get('cylinder_tilt_deg') or 0 for r in rows),
        preclose_max_tilt_deg=max(r['contact_status'].get('cylinder_tilt_deg') or 0 for r in rows[:firstclose+1]),
        actual_gripper_range_rad=[min(r['joint_state'][-1] for r in rows),max(r['joint_state'][-1] for r in rows)],
        hold_blocked_open_rows=sum(r['gripper_close_hold_blocked_open'] for r in rows),
        peak_command_tracking_step_rad=float(np.max(np.abs(tracking))),
        finger_contact_objects=trajectory_targets,events=events)
    return record


def main():
    current=load(R/'结果与失败分类.json')['episodes']
    old={x['seed']:x for x in load(PREVIOUS/'结果与失败分类.json')['episodes']}
    records=[];paired=[]
    for item in current:
        row=audit(item);records.append(row)
        previous=audit(old[item['seed']]);paired.append(dict(episode=item['episode'],seed=item['seed'],
            steps15_strict=row['strict_success'],steps12_strict=previous['strict_success'],
            steps15_final=row['final_box_success'],steps12_final=previous['final_box_success'],
            steps15_firstclose_s=row['first_close_sim_s'],steps12_firstclose_s=previous['first_close_sim_s'],
            steps15_firstclose_error_m=row['first_close_envelope_error_m'],steps12_firstclose_error_m=previous['first_close_envelope_error_m'],
            steps15_loaded_pair_span=row['longest_loaded_bilateral_sim_s'],steps12_loaded_pair_span=previous['longest_loaded_bilateral_sim_s']))
        print(item['episode'], row['strict_success'],row['final_box_success'],row['ever_bilateral'],
              round(row['first_close_envelope_error_m']*1000,1),round(row['min_closed_envelope_error_m']*1000,1),
              row['longest_loaded_bilateral_sim_s'],row['best_loaded'],flush=True)
    groups={}
    for name, subset in [('all',records),('final_failed',[r for r in records if not r['final_box_success']]),
                          ('strict_failed',[r for r in records if not r['strict_success']]),
                          ('final_only',[r for r in records if r['final_box_success'] and not r['strict_success']]),
                          ('strict_success',[r for r in records if r['strict_success']])]:
        groups[name]=dict(count=len(subset),no_bilateral=[r['episode'] for r in subset if not r['ever_bilateral']],
            never_entered_envelope_closed=[r['episode'] for r in subset if not r['ever_gates_during_close']['inside_grasp_envelope']],
            no_loaded_bilateral=[r['episode'] for r in subset if not r['loaded_bilateral_rows']],
            no_stable_relative_pose=[r['episode'] for r in subset if not r['ever_gates_during_close']['relative_pose_stable']],
            initial_close_outside_envelope=[r['episode'] for r in subset if not r['first_close_inside_envelope']],
            first_close_displaced_gt9mm=[r['episode'] for r in subset if r['first_close_cup_displacement_xy_m']>.009],
            first_close_error_m_median=float(np.median([r['first_close_envelope_error_m'] for r in subset])))
    comparison={}
    for name in ['strict','final']:
        comparison[name]=dict(improved=[r['episode'] for r in paired if r[f'steps15_{name}'] and not r[f'steps12_{name}']],
            regressed=[r['episode'] for r in paired if not r[f'steps15_{name}'] and r[f'steps12_{name}']],
            both_success=[r['episode'] for r in paired if r[f'steps15_{name}'] and r[f'steps12_{name}']])
    report=dict(groups=groups,paired_comparison=comparison,episodes=records,paired_episodes=paired,
        caveat='Recorded trace samples only; unlogged inference intervals are not assumed to preserve contact. Envelope coordinates refer to the cup center in the gripper frame, not fingertip distance.')
    (A/'failure_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    fields=[k for k in records[0] if k not in ['events','finger_contact_objects','ever_gates_during_close','best_loaded','loaded_bilateral_blocker_rows']]
    with (A/'failure_metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fields,extrasaction='ignore');writer.writeheader();writer.writerows(records)
    print(json.dumps(dict(groups=groups,paired_comparison=comparison),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
