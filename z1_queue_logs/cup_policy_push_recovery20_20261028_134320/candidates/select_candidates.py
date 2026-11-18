#!/usr/bin/env python3
"""Read-only source trace audit; write reproducible, upright push takeover candidates."""
import json, math
from pathlib import Path

OUT = Path(__file__).resolve().parent
SOURCES = {
    'current8hz': Path('/home/lp531/smolvla_eval/act_cup_chunk30_batch64_best70k_steps15_8hz_paired30_20261028_102158'),
    'old10hz': Path('/home/lp531/smolvla_eval/act_cup_renderfix500_best80k_steps5_random30_20261027'),
}
BOUNDS = (.32, .72, -.30, .12)

def motion_distance(row, initial):
    return math.dist(row['contact_status']['object_position_m'][:2], initial[:2])

def prefix_violation(row, initial):
    c = row['contact_status']
    if not c.get('cylinder_tilt_valid', False) or c.get('cylinder_tilt_deg', math.inf) > 10:
        return 'previous_tilt_over10deg_or_invalid'
    if abs(c['object_position_m'][2] - initial[2]) > .025:
        return 'previous_vertical_excursion_over25mm'
    if c.get('was_grasped') or c.get('physical_grasp_ok') or row.get('lifted'):
        return 'previous_grasp_or_lift'
    return None

def good_row(row, initial, threshold=.015, bounds=BOUNDS):
    c = row['contact_status']; p = c['object_position_m']; a,b,d,e = bounds
    return (row['simulation_elapsed_s'] > 2
        and motion_distance(row, initial) >= threshold
        and abs(p[2] - initial[2]) <= .008
        and c.get('object_speed_m_s', math.inf) <= .02
        and c.get('object_touching_table') is True
        and not c.get('gripper_commanded_closed', True)
        and row['joint_state'][-1] <= -.75
        and not c.get('physical_grasp_ok') and not c.get('was_grasped')
        and c.get('target_collisions_enabled') is True
        and row.get('command_target_collisions_enabled') is True
        and c.get('frozen') is False and row.get('command_target_frozen') is False
        and a <= p[0] <= b and d <= p[1] <= e)

def select_window(rows, initial, threshold=.015, duration=.4, bounds=BOUNDS):
    start = None; violation = None
    for i,row in enumerate(rows):
        violation = violation or prefix_violation(row, initial)
        if violation:
            return None, {'first_prefix_violation_index': i, 'first_prefix_violation_time_s': row['simulation_elapsed_s'], 'reason': violation}
        if good_row(row, initial, threshold, bounds):
            if start is None: start = i
        else:
            start = None
        if start is not None and row['simulation_elapsed_s'] - rows[start]['simulation_elapsed_s'] >= duration - 1e-9:
            # Poses are asynchronous; require independent pose timestamps to span the window too.
            stamps = [x['contact_status']['object_pose_stamp_s'] for x in rows[start:i+1]]
            if max(stamps) - min(stamps) >= duration - .035:
                return (start, i), None
    return None, {'reason': 'no_displaced_upright_open_dynamic_slow_window_before_prefix_failure'}

def wxyz_to_xyzw(q):
    return [q[1],q[2],q[3],q[0]]

def summarize_candidate(group, path, rows, result, window, threshold):
    start,i=window; row=rows[i]; c=row['contact_status']; initial=result['target_position_m']; prefix=rows[:i+1]
    first1 = next(j for j,x in enumerate(prefix) if motion_distance(x,initial)>=.001)
    motion_slice=prefix[max(0,first1-60):]
    future = [(j,x) for j,x in enumerate(rows[i+1:],i+1) if prefix_violation(x,initial)]
    q=c['object_orientation_xyzw']; scene=result['initial_scene_randomization']
    ident=f"{group}_ep{result['episode_number']:03d}_seed{result['seed']}"
    commands_path=OUT/(ident+'_prefix_commands.jsonl')
    keys=['simulation_elapsed_s','elapsed_s','joint_state','policy_action','execution_target','command','policy_action_index','execution_substep','execution_rate_hz','phase','command_target_collisions_enabled','command_target_frozen']
    with commands_path.open('w') as f:
        for x in prefix:
            f.write(json.dumps({k:x.get(k) for k in keys},separators=(',',':'))+'\n')
    layout={}
    for obj,p in scene['object_positions_m'].items():
        raw=scene['object_orientations_xyzw'][obj]
        layout[obj]={'position_m':p,'orientation_root_wxyz':raw,'orientation_xyzw':wxyz_to_xyzw(raw)}
    assert len(layout)==5
    p=c['object_position_m']
    return {
        'candidate_id':ident,'group':group,'episode_number':result['episode_number'],'seed':result['seed'],
        'source_eval_path':str(SOURCES[group]),'source_episode_path':str(path.parent),'source_result_path':str(path.with_name('result.json')),
        'source_trace_path':str(path),'source_input_sync_path':str(path.with_name('input_sync.jsonl')),'source_prediction_chunks_path':str(path.with_name('prediction_chunks.jsonl')),
        'source_prefix_commands_path':str(commands_path),'source_prefix_trace_inclusive_indices':[0,i],
        'takeover_simulation_elapsed_s':row['simulation_elapsed_s'],'takeover_trace_index_zero_based':i,'takeover_source_elapsed_s':row['elapsed_s'],
        'stable_window':{'start_trace_index_zero_based':start,'end_trace_index_zero_based':i,'simulation_start_s':rows[start]['simulation_elapsed_s'],'simulation_end_s':row['simulation_elapsed_s'],'duration_s':row['simulation_elapsed_s']-rows[start]['simulation_elapsed_s'],'max_speed_m_s':max(x['contact_status']['object_speed_m_s'] for x in rows[start:i+1]),'max_tilt_deg':max(x['contact_status']['cylinder_tilt_deg'] for x in rows[start:i+1]),'max_z_drift_m':max(abs(x['contact_status']['object_position_m'][2]-initial[2]) for x in rows[start:i+1]),'unique_object_pose_timestamps':len(set(x['contact_status']['object_pose_stamp_s'] for x in rows[start:i+1]))},
        'cup_initial_xyz_m':initial,'cup_xyz_m':p,'cup_orientation_xyzw':q,'cup_root_orientation_wxyz':[q[3],q[0],q[1],q[2]],'joints_rad':row['joint_state'],'command_rad':row['command'],
        'cup_displacement_xy_m':motion_distance(row,initial),'selection_displacement_threshold_m':threshold,
        'supplemental_under15mm':threshold<.015,'outside_initial_evaluation_sampling_bounds':not (.36<=p[0]<=.69 and -.27<=p[1]<=.074),
        'workspace_screen_bounds_m':{'x':[BOUNDS[0],BOUNDS[1]],'y':[BOUNDS[2],BOUNDS[3]]},'requires_reachability_execution_verification':True,
        'initial_five_object_layout':layout,'raw_initial_scene_randomization':scene,
        'prefix_max_tilt_deg':max(x['contact_status']['cylinder_tilt_deg'] for x in prefix),'prefix_max_speed_m_s':max(x['contact_status']['object_speed_m_s'] for x in prefix),'prefix_max_vertical_excursion_m':max(abs(x['contact_status']['object_position_m'][2]-initial[2]) for x in prefix),
        'push_not_initial_settling_evidence':{'first_1mm_displacement_time_s':rows[first1]['simulation_elapsed_s'],'first_1mm_displacement_trace_index':first1,'max_displacement_first2s_m':max(motion_distance(x,initial) for x in prefix if x['simulation_elapsed_s']<=2),'initial_randomization_settle_metrics':scene['settle_metrics']['cup'],'minimum_gripper_cup_distance_during_movement_m':min(x['contact_status']['gripper_object_distance_m'] for x in motion_slice),'source_motion_frame_count':sum(x['contact_status']['object_speed_m_s']>.005 for x in motion_slice),'any_reported_finger_object_contact_before_takeover':any(x['contact_status'].get('left_finger_contact_object') or x['contact_status'].get('right_finger_contact_object') for x in prefix)},
        'takeover_contact_status':c,
        'first_later_prefix_safety_violation':({'trace_index':future[0][0],'simulation_elapsed_s':future[0][1]['simulation_elapsed_s'],'reason':prefix_violation(future[0][1],initial)} if future else None),
        'original_rollout_final_cup_position_m':result['final_status'].get('object_position_m'),'original_rollout_final_tilt_deg':result['final_status'].get('cylinder_tilt_deg'),'original_rollout_failure_category':result.get('failure_category'),
        'quaternion_convention_note':'Use evaluator contact_status object_orientation_xyzw for world tilt. Isaac get_world_pose returns WXYZ; initial_scene_randomization.object_orientations_xyzw is historically mislabeled and contains WXYZ. Both explicit conventions are saved here; never classify raw root quaternion by assuming it is XYZW.',
    }

def run():
    accepted=[];audit=[]
    for group,root in SOURCES.items():
        final_box_records = {x['seed']: x for x in json.load((root/'final_box_results.json').open())['episodes']}
        for path in sorted(root.rglob('trace.jsonl')):
            rows=[json.loads(l) for l in path.open()]; result=json.load(path.with_name('result.json').open());initial=result['target_position_m']
            final_box = final_box_records[result['seed']]
            if final_box['success'] or result['success']:
                audit.append({'group':group,'episode_number':result['episode_number'],'seed':result['seed'],'source_trace_path':str(path),'accepted':False,'rejection':{'reason':'successful_source_rollout_excluded','final_box_record':final_box}})
                continue
            window,reason=select_window(rows,initial)
            threshold=.015
            if window is None:
                window,reason=select_window(rows,initial,.010)
                threshold=.010
            entry={'group':group,'episode_number':result['episode_number'],'seed':result['seed'],'source_trace_path':str(path)}
            if window:
                candidate=summarize_candidate(group,path,rows,result,window,threshold)
                candidate['source_final_box_success'] = final_box['success']
                candidate['source_final_box_record'] = final_box
                outfile=OUT/(candidate['candidate_id']+'.json');outfile.write_text(json.dumps(candidate,indent=2)+'\n');candidate['candidate_json_path']=str(outfile);accepted.append(candidate)
                entry.update(accepted=True,candidate_json_path=str(outfile),takeover_time_s=candidate['takeover_simulation_elapsed_s'],displacement_m=candidate['cup_displacement_xy_m'])
            else:
                entry.update(accepted=False,rejection=reason,max_displacement_xy_m=max(motion_distance(x,initial) for x in rows))
            audit.append(entry)
    summary={'total_distinct_rollout_episodes':len(accepted),'distinct_seed_count':len(set(c['seed'] for c in accepted)),'counts_by_group':{g:sum(c['group']==g for c in accepted) for g in SOURCES},'primary15mm_count':sum(not c['supplemental_under15mm'] for c in accepted),'supplemental10mm_count':sum(c['supplemental_under15mm'] for c in accepted),'candidates':accepted,'episode_audit':audit}
    (OUT/'candidates_manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['group,episode,seed,takeover_s,trace_index,displacement_mm,max_prefix_tilt_deg,window_max_speed_m_s,supplemental,outside_initial_sampling,source_trace_path,candidate_json_path']
    for c in accepted:
        lines.append(','.join(map(str,[c['group'],c['episode_number'],c['seed'],round(c['takeover_simulation_elapsed_s'],6),c['takeover_trace_index_zero_based'],round(c['cup_displacement_xy_m']*1000,3),round(c['prefix_max_tilt_deg'],3),c['stable_window']['max_speed_m_s'],c['supplemental_under15mm'],c['outside_initial_evaluation_sampling_bounds'],c['source_trace_path'],c['candidate_json_path']])))
    (OUT/'candidates_summary.csv').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('candidates','episode_audit')},indent=2))
if __name__=='__main__':run()
