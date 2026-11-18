import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,'/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge')
from z1_isaac_bridge.smolvla_rollout_core import JOINT_LIMITS
OUT=Path(__file__).resolve().parent
labels=json.loads(Path('/home/lp531/pi05_lora_workspace/binary_gripper_20261007/episode_labels.json').read_text())
rows=[];allerr=[];allstep=[];age=[]
for lab in labels:
 p=Path(lab['raw_root']); f=[json.loads(x) for x in (p/'frames.jsonl').read_text().splitlines()]
 a=np.array([x['action'] for x in f]);q=np.array([x['observation.state'] for x in f]);c=np.clip(a,JOINT_LIMITS[:,0],JOINT_LIMITS[:,1]);e=np.abs(c[:,:6]-q[:,:6]);d=np.max(abs(np.diff(a[:,:6],axis=0)),axis=1)
 start=json.loads((p/'episode_start.json').read_text());scene=start['scene_randomization']
 motion=np.flatnonzero(np.max(abs(a[:,:6]-a[0,:6]),axis=1)>.02)
 r=dict(**lab,seed=scene['seed'],grid=scene['target_grid'],position=scene['object_positions_m']['beverage_bottle'],limit_changed_frames=int(np.sum(np.max(abs(c[:,:6]-a[:,:6]),axis=1)>1e-6)),step_limited_frames=int(np.sum(e.max(axis=1)>.1+1e-6)),max_action_state_rad=float(e.max()),p95_action_state_rad=float(np.percentile(e.max(axis=1),95)),max_consecutive_action_step_rad=float(d.max()),first_motion_frame=int(motion[0]) if len(motion) else None,q5_margin_min_rad=float(np.min(JOINT_LIMITS[4,1]-a[:,4])),close_state=q[lab['close_frame']-1].tolist())
 rows.append(r);allerr.extend(e.max(axis=1));allstep.extend(d);age.extend(x['action_age_s'] for x in f)
selected=[]
for r in sorted(rows,key=lambda x:x['max_action_state_rad'],reverse=True):
 if len(selected)<2:selected.append(r)
for seed in (489361075,774196979):
 found=[r for r in rows if r['seed']==seed]
 if found and found[0] not in selected:selected.append(found[0])
while len(selected)<6:
 choice=max((r for r in rows if r not in selected),key=lambda r:min(np.linalg.norm(np.array(r['position'])[:2]-np.array(s['position'])[:2]) for s in selected))
 selected.append(choice)
summary=dict(episodes=len(rows),frames=sum(r['frames'] for r in rows),limit_changed_frames=sum(r['limit_changed_frames'] for r in rows),step_limited_frames=sum(r['step_limited_frames'] for r in rows),action_state_rad=dict(median=float(np.median(allerr)),p95=float(np.percentile(allerr,95)),max=float(max(allerr))),consecutive_action_step_rad=dict(p95=float(np.percentile(allstep,95)),max=float(max(allstep))),action_age_s=dict(median=float(np.median(age)),p95=float(np.percentile(age,95)),max=float(max(age))),selected=[dict(episode=r['raw_episode'],seed=r['seed'],grid=r['grid'],max_action_state_rad=r['max_action_state_rad']) for r in selected])
(OUT/'dataset_summary.json').write_text(json.dumps(summary,indent=2));(OUT/'dataset_episodes.json').write_text(json.dumps(rows,indent=2));(OUT/'selected.json').write_text(json.dumps(selected,indent=2));print(json.dumps(summary,indent=2))
