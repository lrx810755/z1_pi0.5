"""Offline pose/close cross-check using saved ACT rollouts and URDF FK."""
import json,xml.etree.ElementTree as ET,csv
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
RUN=Path(__file__).resolve().parent;OUT=RUN/'failed_pose_check_20261014';OUT.mkdir(exist_ok=True)
CONFIG=Path('/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge/config/grasp_physics.json')
config=json.loads(CONFIG.read_text());profile=config['object_grasp_profiles']['beverage_bottle']
URDF=Path('/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/generated_dual_camera/z1_with_dual_d435_resolved.urdf')
tree=ET.parse(URDF).getroot();joints={j.attrib['name']:j for j in tree.findall('joint')}
def orient(q):
 R=np.eye(3)
 for i in range(6):
  j=joints['joint'+str(i+1)];o=j.find('origin');rpy=np.fromstring(o.attrib.get('rpy','0 0 0'),sep=' ');axis=np.fromstring(j.find('axis').attrib['xyz'],sep=' ')
  R=R@Rotation.from_euler('xyz',rpy).as_matrix()@Rotation.from_rotvec(axis*q[i]).as_matrix()
 j=joints['gripperStatorLink'];R=R@Rotation.from_euler('xyz',np.fromstring(j.find('origin').attrib.get('rpy','0 0 0'),sep=' ')).as_matrix()
 return R
reference=profile['calibrated_grasp_reference'];Rref=orient(reference['captured_arm_joint_positions_rad']);upref=Rref.T@np.array([0.,0.,1.])
qref=Rotation.from_quat(reference['captured_gripper_stator_world_quaternion_xyzw']).as_matrix();fk_reference_error=float(np.degrees(Rotation.from_matrix(qref.T@Rref).magnitude()))
# Compare yaw-invariant pitch/roll to held-out preclose examples, not a fabricated hard physical angle limit.
split=json.loads((RUN/'split_manifest.json').read_text());demo=[]
for ep in split['validation_episodes']:
 rec=next(x for x in split['source_records'] if x['episode_index']==ep);raw=[json.loads(x) for x in (Path(rec['source_path'])/'frames.jsonl').read_text().splitlines()];q=raw[rec['close_frame']]['observation.state'];u=orient(q).T@np.array([0.,0.,1.]);angle=float(np.degrees(np.arccos(np.clip(u@upref,-1,1))));demo.append(angle)
angle_limit=float(max(demo))
def excess(v,e):
 return np.array([max(e['x_min_m']-v[0],0,v[0]-e['x_max_m']),max(abs(v[1])-e['y_abs_max_m'],0),max(e['z_min_m']-v[2],0,v[2]-e['z_max_m'])])
root=Path('/home/lp531/smolvla_eval');batch=root/'act_30_trials_20261014_113709/act_sim_eval_20261014_113714'
latest=sorted((root/'act_correction_chunk16_20261013_50000').rglob('result.json'),key=lambda p:p.stat().st_mtime,reverse=True)[:3]
results=sorted(batch.rglob('result.json'))+list(reversed(latest));reports=[]
for p in results:
 result=json.loads(p.read_text());trace=p.parent/'trace.jsonl'
 if not trace.exists():continue
 rows=[json.loads(x) for x in trace.read_text().splitlines()];valid=[r for r in rows if r.get('contact_status',{}).get('relative_position_gripper_m') is not None]
 vals=np.array([r['contact_status']['relative_position_gripper_m'] for r in valid]);q=np.array([r['joint_state'][:6] for r in valid]);angles=np.array([np.degrees(np.arccos(np.clip((orient(x).T@np.array([0.,0.,1.]))@upref,-1,1))) for x in q])
 masks={};errors={}
 for name,e in [('preclose',profile['preclose_envelope']),('loaded',profile['loaded_envelope']),('generic',config['grasp_envelope'])]:
  errors[name]=np.array([excess(v,e) for v in vals]);masks[name]=np.max(errors[name],axis=1)<1e-9
 pred=np.array([r['policy_action'][-1] for r in valid]);cmd=np.array([r['execution_target'][-1] for r in valid]);times=np.array([r['simulation_elapsed_s'] for r in valid]);dt=np.r_[np.diff(times),0];dt=np.clip(dt,0,1/60)
 closest=int(np.argmin(np.linalg.norm(errors['preclose'],axis=1)));contact=np.array([(r['contact_status'].get('target_finger_contact_pair_count') or 0)>0 for r in valid]);jointclose=np.array([r['joint_state'][-1]>=-.75 for r in valid]);both=masks['preclose']&(angles<=angle_limit)
 before_first_close=np.cumsum(cmd==0)==0
 open_candidate=both&before_first_close
 def span(mask):
  runs=[];begin=None
  for i,ok in enumerate(mask):
   if ok and begin is None:begin=i
   if begin is not None and (not ok or i==len(mask)-1):
    end=i if ok else i-1;runs.append(float(times[end]-times[begin]+dt[end]));begin=None
  return max(runs,default=0.)
 def stats(mask):
  return dict(rows=int(mask.sum()),duration_s=float(dt[mask].sum()),max_contiguous_s=span(mask),pred_min=float(pred[mask].min()) if mask.any() else None,pred_max=float(pred[mask].max()) if mask.any() else None,close_target_rows=int((mask&(cmd==0)).sum()),orientation_error_min_deg=float(angles[mask].min()) if mask.any() else None,orientation_error_max_deg=float(angles[mask].max()) if mask.any() else None)
 reports.append(dict(group='batch30' if batch in p.parents else 'latest3',result=str(p),success=result['success'],failure=result['failure_category'],seed=result['seed'],execute_steps=result['execute_steps'],simulation_duration_s=result['simulation_duration_s'],preclose=stats(masks['preclose']),loaded=stats(masks['loaded']),generic=stats(masks['generic']),preclose_and_demo_orientation=stats(both),candidate_before_first_close=stats(open_candidate),closest_to_preclose=dict(time_s=float(times[closest]),relative_position_m=vals[closest].tolist(),outside_distance_mm=float(np.linalg.norm(errors['preclose'][closest])*1000),axis_excess_mm=(errors['preclose'][closest]*1000).tolist(),orientation_error_deg=float(angles[closest]),prediction=float(pred[closest]),target=float(cmd[closest])),prediction_range=[float(pred.min()),float(pred.max())],ever_close_target=bool((cmd==0).any()),ever_actual_closed=bool(jointclose.any()),ever_contact=bool(contact.any())))
 print('checked',p.parent.name,flush=True)
summary={}
for group in ['batch30','latest3']:
 rr=[r for r in reports if r['group']==group and not r['success']];summary[group]=dict(failures=len(rr),never_preclose=sum(r['preclose']['rows']==0 for r in rr),ever_preclose=sum(r['preclose']['rows']>0 for r in rr),ever_loaded=sum(r['loaded']['rows']>0 for r in rr),ever_preclose_demo_orientation=sum(r['preclose_and_demo_orientation']['rows']>0 for r in rr),ever_close_target=sum(r['ever_close_target'] for r in rr),ever_contact=sum(r['ever_contact'] for r in rr),candidate_before_first_close=sum(r['candidate_before_first_close']['rows']>0 for r in rr),candidate_at_least_02s_before_close=sum(r['candidate_before_first_close']['max_contiguous_s']>=.2 for r in rr),preclose_without_any_close=sum(r['preclose']['rows']>0 and not r['ever_close_target'] for r in rr))
report=dict(summary=summary,preclose_envelope=profile['preclose_envelope'],loaded_envelope=profile['loaded_envelope'],demo_orientation_errors_deg=demo,demo_orientation_max_deg=angle_limit,fk_reference_rotation_error_deg=fk_reference_error,urdf=str(URDF),episodes=reports,limitations=['Position envelope is a candidate criterion, not proof of physical graspability.','Yaw-invariant pitch/roll compared to held-out close-onset examples; ignores bottle-axis rotation and no saved per-frame bottle tilt.','Durations approximate based on 60Hz trace; repeated status values may overcount samples.','Observational comparison cannot prove why policy chooses open.'])
(OUT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(summary,indent=2));print('FK ref error',fk_reference_error,'demo tilt max',angle_limit)
