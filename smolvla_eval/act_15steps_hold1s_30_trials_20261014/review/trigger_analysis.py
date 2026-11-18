from pathlib import Path
import json,collections
import numpy as np
from scipy.spatial.transform import Rotation
old=Path('/home/lp531/pi05_lora_workspace/act_correction_20261013/check_failed_grasp_pose.py')
ns={'__file__':str(old)};exec(compile(old.read_text().split("root=Path('/home/lp531/smolvla_eval')")[0],str(old),'exec'),ns)
profile=ns['profile'];joints=ns['joints'];upref=ns['upref'];limit=ns['angle_limit'];env=profile['preclose_envelope']
W=Path(__file__).resolve().parent;root=W.parent/'act_sim_eval_20261014_161455'
def angles(q):
 R=np.broadcast_to(np.eye(3),(len(q),3,3)).copy()
 for i in range(6):
  j=joints['joint'+str(i+1)];o=j.find('origin');base=Rotation.from_euler('xyz',np.fromstring(o.attrib.get('rpy','0 0 0'),sep=' ')).as_matrix();axis=np.fromstring(j.find('axis').attrib['xyz'],sep=' ')
  R=R@base@Rotation.from_rotvec(q[:,i,None]*axis).as_matrix()
 j=joints['gripperStatorLink'];R=R@Rotation.from_euler('xyz',np.fromstring(j.find('origin').attrib.get('rpy','0 0 0'),sep=' ')).as_matrix()
 return np.degrees(np.arccos(np.clip(R[:,2,:]@upref,-1,1)))
def span(mask,t):
 best=run=0
 for i,v in enumerate(mask):
  dt=min(max(t[i+1]-t[i],0),1/60) if i+1<len(t) else 0
  run=run+dt if v else 0;best=max(best,run)
 return float(best)
records=[]
for p in sorted(root.rglob('result.json')):
 r=json.loads(p.read_text());rows=[json.loads(x) for x in (p.parent/'trace.jsonl').read_text().splitlines()];queries=[json.loads(x) for x in (p.parent/'prediction_chunks.jsonl').read_text().splitlines()]
 valid=[x for x in rows if x.get('contact_status',{}).get('relative_position_gripper_m') is not None]
 v=np.array([x['contact_status']['relative_position_gripper_m'] for x in valid]);q=np.array([x['joint_state'][:6] for x in valid]);a=angles(q);t=np.array([x['simulation_elapsed_s'] for x in valid]);cmd=np.array([x['execution_target'][-1] for x in valid]);pred=np.array([x['policy_action'][-1] for x in valid]);pre=np.cumsum(cmd==0)==0
 err=np.column_stack([np.maximum(np.maximum(env['x_min_m']-v[:,0],v[:,0]-env['x_max_m']),0),np.maximum(abs(v[:,1])-env['y_abs_max_m'],0),np.maximum(np.maximum(env['z_min_m']-v[:,2],v[:,2]-env['z_max_m']),0)])
 pos=err.max(axis=1)<1e-9;candidate=pos&(a<=limit)&pre
 idx=np.where(pre)[0];best=idx[np.argmin(np.linalg.norm(err[idx],axis=1))] if len(idx) else 0
 first=np.flatnonzero(cmd==0)
 query_first=[x['first_close_step_06'] for x in queries];outside=[i for i,x in enumerate(queries) if x['first_close_step_06'] is not None and x['first_close_step_06']>=15]
 item=dict(episode=r['episode_number'],seed=r['seed'],success=r['success'],failure=r['failure_category'],target=r['target_position_m'],ever_close=bool(len(first)),ever_position_before_close=bool((pos&pre).any()),ever_candidate_before_close=bool(candidate.any()),candidate_max_contiguous_s=span(candidate,t),candidate_prediction_range=[float(pred[candidate].min()),float(pred[candidate].max())] if candidate.any() else None,closest_before_close=dict(time_s=float(t[best]),relative_position_mm=(v[best]*1000).tolist(),excess_mm=(err[best]*1000).tolist(),tilt_deg=float(a[best]),prediction=float(pred[best])),full_chunk_max=float(max(max(x['gripper_prediction']) for x in queries)),queries=len(queries),queries_with_close=sum(x is not None for x in query_first),queries_close_only_step16=outside,ever_finger_contact=any((x['contact_status'].get('target_finger_contact_pair_count') or 0)>0 for x in rows),first_close=dict(time_s=float(t[first[0]]),relative_position_mm=(v[first[0]]*1000).tolist(),tilt_deg=float(a[first[0]])) if len(first) else None,result=str(p))
 records.append(item)
records.sort(key=lambda r:r['episode']);fail=[r for r in records if not r['success']];no=[r for r in fail if not r['ever_close']]
summary=dict(total=len(records),success=sum(r['success'] for r in records),failures=len(fail),never_close=len(no),never_close_never_position=sum(not r['ever_position_before_close'] for r in no),never_close_position=sum(r['ever_position_before_close'] for r in no),never_close_candidate=sum(r['ever_candidate_before_close'] for r in no),never_close_candidate_02s=sum(r['candidate_max_contiguous_s']>=.2 for r in no),never_close_all_chunk_below04=sum(r['full_chunk_max']<.4 for r in no),failures_never_candidate_before_close=sum(not r['ever_candidate_before_close'] for r in fail),failures_close_only_step16_ever=sum(bool(r['queries_close_only_step16']) for r in fail),no_close_failure_categories=dict(collections.Counter(r['failure'] for r in no)),success_ids=[r['episode'] for r in records if r['success']])
(W/'trigger_analysis.json').write_text(json.dumps(dict(summary=summary,preclose_envelope=env,orientation_reference_max_deg=limit,episodes=records,limitations=['Envelope is configuration-based, not an empirical demonstration position distribution or proof of graspability.','Orientation reference uses 21 held-out demonstration close frames. Bottle orientation not logged per frame.','Trace predictions refer to executed actions; exact inference-time camera input not saved.','Approximate 60Hz durations can include repeated asynchronous status.']),indent=2))
print(json.dumps(summary,indent=2))
for x in records: print(json.dumps(x,ensure_ascii=False))
