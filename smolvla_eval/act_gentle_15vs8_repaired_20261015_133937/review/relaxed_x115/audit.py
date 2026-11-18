"""Audit saved evidence; never overwrite historical evaluator outcomes.

Missing contact impulses, object yaw and release acknowledgement prevent an exact
counterfactual replay. Report task evidence separately from official scores.
"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

def longest(mask, times):
    start = None
    best = (0., None, None)
    for i, ok in enumerate(mask):
        if ok and (start is None or (i and times[i]-times[i-1] > .12)):
            start = i
        if ok:
            value = (float(times[i]-times[start]), start, i)
            if value[0] > best[0]:
                best = value
        else:
            start = None
    return best

def audit(path, steps):
    result = json.loads(path.read_text())
    rows = [json.loads(line) for line in (path.parent/'trace.jsonl').read_text().splitlines()]
    times = np.array([r['simulation_elapsed_s'] for r in rows])
    contacts = [r['contact_status'] for r in rows]
    rel = np.array([c['relative_position_gripper_m'] for c in contacts])
    obj = np.array([c['object_position_m'] for c in contacts])
    grip = np.array([r['joint_state'][-1] for r in rows])
    bilateral = np.array([c.get('left_finger_contact_object') == 'beverage_bottle'
                          and c.get('right_finger_contact_object') == 'beverage_bottle' for c in contacts])
    table = np.array([bool(c.get('left_finger_touching_table') or c.get('right_finger_touching_table')) for c in contacts])
    lifted = np.array([r['lifted'] for r in rows])
    closing = np.array([r['execution_target'][-1] == 0 for r in rows])
    transport_mask = bilateral & lifted & ~table & closing
    base_mask = transport_mask & (grip >= -.8) & (grip <= -.2)
    geometric = (rel[:,0] <= .152) & (abs(rel[:,1]) <= .032) & (rel[:,2] >= .008) & (rel[:,2] <= .042)
    # Recover the bottle axis from its logged center and base. Yaw is unobservable.
    center = np.array([c['gripper_world_position_m'] for c in contacts]) + Rotation.from_quat(
        np.array([c['gripper_world_orientation_xyzw'] for c in contacts])).apply(rel)
    axis = (center-obj)/.0785
    axis_norm = np.linalg.norm(axis, axis=1)
    tilt = np.degrees(np.arccos(np.clip(abs(axis[:,2])/np.maximum(axis_norm,1e-12),0,1)))
    axis_consistent = abs(axis_norm-1) < .002
    envelope = {}
    for name, xmin in [('old',.12),('new',.115)]:
        valid = base_mask & geometric & (rel[:,0] >= xmin)
        interval = longest(valid,times)
        upright = longest(valid & axis_consistent & (tilt <= 5),times)
        envelope[name] = dict(rows=int(valid.sum()), longest_continuous_s=interval[0],
            upright_longest_s=upright[0],
            first_time_s=float(times[np.flatnonzero(valid)[0]]) if valid.any() else None)
    final = result['final_status']
    in_box = (abs(obj[:,0]-.32)<=.055) & (abs(obj[:,1]-.14)<=.055) & (obj[:,2]>=0) & (obj[:,2]<=.12)
    released = grip<=-.8
    placement = longest(in_box & released & ~closing,times)
    carried = longest(transport_mask,times)
    # Fixed 0.4 s stability evidence using actual elapsed time and 6 mm tolerance.
    stable = np.zeros(len(rows),dtype=bool)
    for i in np.flatnonzero(transport_mask):
        j = int(np.searchsorted(times,times[i]-.4))
        if times[i]-times[j] >= .36 and transport_mask[j:i+1].all():
            stable[i] = np.max(np.linalg.norm(rel[j:i+1]-rel[j],axis=1))<=.006
    evidence = bool((stable & lifted).any() and placement[0] >= .5)
    # This is a task-completion evidence category, not an exact evaluator replay.
    return dict(steps=steps,episode=result['episode_number'],seed=result['seed'],
        official_success=result['success'],official_failure=result['failure_category'],
        final_inside_box=final.get('inside_success_volume'),final_released=final.get('released'),
        final_position=final.get('object_position_m'),bilateral_lift_duration_s=carried[0],
        stable_bilateral_lift_rows=int(stable.sum()),placement_inside_released_duration_s=placement[0],
        sustained_transport_and_placement_evidence=evidence,envelope=envelope,
        stable_lift_local_x_range_mm=[float(rel[stable,0].min()*1000),float(rel[stable,0].max()*1000)] if stable.any() else None,
        stable_lift_tilt_range_deg=[float(tilt[stable].min()),float(tilt[stable].max())] if stable.any() else None,
        result_path=str(path), video=str(path.parent/'videos/global_third_person.mp4'))

if __name__ == '__main__':
    records=[]
    for steps in [15,8]:
        files=sorted((ROOT/f'steps{steps}').glob('act_sim_eval_*/episodes/*/result.json'))
        assert len(files)==30
        records.extend(audit(p,steps) for p in files)
    summaries={}
    for steps in [15,8]:
        rr=[r for r in records if r['steps']==steps]
        summaries[str(steps)]=dict(official_successes=sum(r['official_success'] for r in rr),
            task_evidence_episodes=[r['episode'] for r in rr if r['sustained_transport_and_placement_evidence']],
            additional_task_evidence_episodes=[r['episode'] for r in rr if r['sustained_transport_and_placement_evidence'] and not r['official_success']],
            expanded_envelope_sustained_lift_episodes=[r['episode'] for r in rr if r['envelope']['new']['longest_continuous_s']>=.4 and r['envelope']['old']['longest_continuous_s']<.4])
    report=dict(summary=summaries,episodes=records,old_xmin_m=.12,new_xmin_m=.115,
        exact_replay_possible=False,
        missing=['per-frame contact impulse ratio','per-frame bottle quaternion/yaw','per-frame grasp-control release acknowledgement'],
        notes=['Task evidence is independent of evaluator success and requires video review.',
               'Reconstructed tilt uses rounded asynchronous telemetry; it is diagnostic only.',
               'Only X minimum changed. Other original evaluator requirements remain active.'])
    (HERE/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(summaries,ensure_ascii=False,indent=2))
