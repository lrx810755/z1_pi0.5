from pathlib import Path
import json,importlib.util,struct,csv,sys
from collections import defaultdict
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('audit','/home/lp531/smolvla_eval/three_camera_dataset_strict_audit_20261006/audit.py');audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
mesh={}
for name in ('Stator','Mover'):
 d=Path('/home/lp531/ros2_ws/src/z1_ros2/z1_description/meshes/collision/z1_Gripper'+name+'.STL').read_bytes();dt=np.dtype([('n','<f4',(3,)),('v','<f4',(3,3)),('a','<u2')]);v=np.unique(np.frombuffer(d,dtype=dt,count=struct.unpack_from('<I',d,80)[0],offset=84)['v'].reshape(-1,3),axis=0);mesh[name]=v[ConvexHull(v).vertices]
def height(q):
 p,r=audit.fk(q[:6]);fixed=mesh['Stator']@r.T+p;rot=Rotation.from_rotvec([0,q[-1],0]).as_matrix();moving=(mesh['Mover']@rot.T+[.049,0,0])@r.T+p
 return float(min(fixed[:,2].min(),moving[:,2].min()))
results=[]
for p in sorted(OUT.glob('steps*/**/result.json')):
 r=json.loads(p.read_text());root=p.parent
 if r.get('seed') is None:raise RuntimeError(f'Setup failure: {r}')
 tr=[json.loads(x) for x in (root/'trace.jsonl').read_text().splitlines()];pr=[json.loads(x) for x in (root/'predictions.jsonl').read_text().splitlines()];sy=[json.loads(x) for x in (root/'input_sync.jsonl').read_text().splitlines()]
 assert len(pr)==len(sy) and all(len(x['full_actions'])==15 for x in pr)
 full=[]
 for pred in pr:
  raw=np.array(pred['full_actions']);mapped=raw.copy();mapped[:,-1]=np.where(raw[:,-1]>=.5,0,-.9);clipped=np.clip(mapped,audit.limits[:,0],audit.limits[:,1]);rh=np.array([height(q) for q in mapped]);ch=np.array([height(q) for q in clipped]);steps=r['execute_steps']
  first=lambda h:next((i for i,z in enumerate(h) if z<.004),None)
  full.append(dict(prediction=pred['prediction'],input_state=pred['input_state'],raw_min=float(rh.min()),clipped_min=float(ch.min()),selected_raw_min=float(rh[:steps].min()),selected_clipped_min=float(ch[:steps].min()),first_raw_under4mm=first(rh),first_clipped_under4mm=first(ch),q5_exceeds=bool(np.any((raw[:,4]<audit.limits[4,0]-1e-5)|(raw[:,4]>audit.limits[4,1]+1e-5)))))
 firstclose=next((x for x in tr if x['execution_target'][-1]==0),None)
 firstdanger=next((x for x in full if x['selected_raw_min']<.004 or x['selected_clipped_min']<.004),None)
 j=np.array([x['joint_state'] for x in tr]);c=np.array([x['command'] for x in tr])
 item=dict(group=p.parts[len(OUT.parts)],seed=r['seed'],steps=r['execute_steps'],success=r['success'],failure=r['failure_category'],sim_s=r['simulation_duration_s'],wall_s=r['duration_s'],predictions=len(pr),first_close_sim_s=None if firstclose is None else firstclose['simulation_elapsed_s'],first_close_relative=None if firstclose is None else firstclose['contact_status']['relative_position_gripper_m'],target=r['target_position_m'],initial_joints=r['initial_joint_state'],initial_scene=r['initial_scene_randomization'],final_relative=r['final_status']['relative_position_gripper_m'],physical_grasp=r['final_status']['physical_grasp_ok'],raw_risk_predictions=sum(x['selected_raw_min']<.004 for x in full),clipped_risk_predictions=sum(x['selected_clipped_min']<.004 for x in full),first_risk_prediction=firstdanger,q5_exceed_predictions=sum(x['q5_exceeds'] for x in full),actual_min_height_mm=min(height(q) for q in j)*1000,command_min_height_mm=min(height(q) for q in c)*1000,max_arm_tracking_rad=float(np.max(abs(j[:,:6]-c[:,:6]))),sync_max_skew=max(x['sensor_skew_s'] for x in sy),sync_max_age=max(x['oldest_sensor_age_s'] for x in sy),infer_median_ms=float(np.median([x['inference_wall_s'] for x in pr])*1000),record_median_ms=float(np.median([x['recording_wall_s'] for x in pr])*1000),result_path=str(p))
 (root/'prediction_geometry.json').write_text(json.dumps(full,indent=2)+'\n');results.append(item)
 assert len(list((root/'policy_inputs').glob('*.npz')))==len(pr)
complete=len(results)==8
if not complete and "--partial" not in sys.argv:raise AssertionError(len(results))
checks=[]
for seed in (489361075,774196979):
 group=[r for r in results if r['seed']==seed]
 if len(group)<4:continue
 assert len(group)==4
 assert sorted(r['steps'] for r in group)==[5,5,10,10]
 poses=np.array([r['target'] for r in group]);joints=np.array([r['initial_joints'] for r in group]);spread=np.max(np.linalg.norm(poses[:,None,:]-poses[None,:,:],axis=2))
 checks.append(dict(seed=seed,target_position_spread_mm=float(spread*1000),initial_arm_joint_spread_rad=float(np.ptp(joints[:,:6],axis=0).max())))
summary={'episodes':results,'pairing_checks':checks,'scope':'8 actual closed-loop simulator trials; geometry is an approximate URDF reconstruction','completed':complete}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
fields=['group','seed','steps','success','failure','sim_s','predictions','first_close_sim_s','raw_risk_predictions','clipped_risk_predictions','q5_exceed_predictions','actual_min_height_mm','command_min_height_mm','infer_median_ms','record_median_ms']
with (OUT/'comparison.csv').open('w') as f:
 w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(results)
print(json.dumps(summary,indent=2))
