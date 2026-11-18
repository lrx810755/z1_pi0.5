import sys,json,csv
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,'/home/lp531/ros2_ws/src/z1_ros2/scripts/cup_failure_recovery_20261019')
from geometry import Geometry
P=Path(__file__).resolve().parent;g=Geometry();probes=json.loads((P/'probes.json').read_text())['samples'];a=np.load(P/'predictions.npz');pr=a['predicted'];gt=a['expert'];out=[]
for i,s in enumerate(probes):
 p=np.array(s['actual_position']);r=np.array(s['actual_rotation']);axis=r[:,0];pf=[g.fk(q[:6]) for q in pr[i]];gf=[g.fk(q[:6]) for q in gt[i]];pp=np.array([v[0] for v in pf]);gp=np.array([v[0] for v in gf]);record=dict(s)
 for step,name in [(14,'15'),(15,'16')]:
  pd=pp[step]-p;gd=gp[step]-p;norm=np.linalg.norm(gd);progress=float(pd@gd/norm) if norm>1e-8 else 0
  record.update({f'pred_forward_mm_{name}':float(pd@axis*1000),f'expert_forward_mm_{name}':float(gd@axis*1000),f'pred_distance_mm_{name}':float(np.linalg.norm(pd)*1000),f'expert_distance_mm_{name}':float(norm*1000),f'progress_along_expert_mm_{name}':progress*1000,f'endpoint_error_mm_{name}':float(np.linalg.norm(pp[step]-gp[step])*1000),f'rotation_error_deg_{name}':float(np.rad2deg(Rotation.from_matrix(pf[step][1]@gf[step][1].T).magnitude())),f'pred_gripper_{name}':float(pr[i,step,6]),f'expert_gripper_{name}':float(gt[i,step,6])})
 record['pred_delta_local_mm_15']=(r.T@(pp[14]-p)*1000).tolist()
 record['expert_delta_local_mm_15']=(r.T@(gp[14]-p)*1000).tolist()
 record['pred_max_movement_mm']=float(np.linalg.norm(pp-p,axis=1).max()*1000)
 record['pred_forward_max_mm']=float(((pp-p)@axis).max()*1000)
 record['pred_close_max']=float(pr[i,:,6].max());record['expert_close_any']=bool(gt[i,:,6].max()>.5)
 record['joint_l1_rad']=float(abs(pr[i,:,:6]-gt[i,:,:6]).mean())
 record['depth_gap_mm']=(s['local_xyz'][0]-.12083045)*1000
 record['advance_required']=record['expert_forward_mm_15']>=2
 record['not_advancing']=record['pred_forward_mm_15']<1
 record['retreating']=record['pred_forward_mm_15']< -1
 record['stalled_all16']=record['pred_max_movement_mm']<2
 record['half_forward']=record['pred_forward_mm_15']>=.5*record['expert_forward_mm_15']
 record['gt_motion_required']=record['expert_distance_mm_15']>=3
 record['right_general_direction']=record['progress_along_expert_mm_15']>1
 out.append(record)
def stats(rows):
 adv=[s for s in rows if s['advance_required']];motion=[s for s in rows if s['gt_motion_required']]
 return dict(samples=len(rows),episodes=len(set(s['episode_index'] for s in rows)),expert_advance_samples=len(adv),expert_advance_episodes=len(set(s['episode_index'] for s in adv)),model_forward_ge1mm=sum(not s['not_advancing'] for s in adv),model_forward_ge_half_expert=sum(s['half_forward'] for s in adv),model_not_advancing=sum(s['not_advancing'] for s in adv),model_retreating=sum(s['retreating'] for s in adv),model_stalled_all16=sum(s['stalled_all16'] for s in adv),median_expert_forward_mm=float(np.median([s['expert_forward_mm_15'] for s in adv])) if adv else None,median_pred_forward_mm=float(np.median([s['pred_forward_mm_15'] for s in adv])) if adv else None,median_endpoint_error_mm=float(np.median([s['endpoint_error_mm_15'] for s in adv])) if adv else None,expert_motion_samples=len(motion),model_moves_in_expert_direction=sum(s['right_general_direction'] for s in motion),expert_will_close=sum(s['expert_close_any'] for s in rows),model_predicts_close=sum(s['pred_close_max']>=.6 for s in rows))
groups={}
for split in ['train','validation']:
 for source in ['all','old_correction','new_correction']:
  for phase in ['all','initial_wait','after_recovery_start']:
   rr=[s for s in out if s['split']==split and (source=='all' or s['source']==source) and (phase=='all' or s['initial_wait']==(phase=='initial_wait'))]
   groups[f'{split}/{source}/{phase}']=stats(rr)
for depth in [5,10,18,30]:
 for split in ['train','validation']:groups[f'{split}/gap_ge_{depth}mm']=stats([s for s in out if s['split']==split and s['depth_gap_mm']>=depth])
(P/'results.json').write_text(json.dumps(dict(groups=groups,samples=out),ensure_ascii=False,indent=2))
with (P/'predictions_vs_expert.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
assert pr.shape==gt.shape==(len(probes),16,7) and np.isfinite(pr).all()
print(json.dumps({k:v for k,v in groups.items() if k in ['train/all/all','validation/all/all','train/new_correction/all','validation/new_correction/all','train/all/after_recovery_start','validation/all/after_recovery_start']},ensure_ascii=False,indent=2))
print('BAD',json.dumps(sorted([s for s in out if s['advance_required']],key=lambda s:s['pred_forward_mm_15']-s['expert_forward_mm_15'])[:4],ensure_ascii=False,indent=2))
