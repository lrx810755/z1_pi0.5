from pathlib import Path
import json,importlib.util
import numpy as np
from scipy.spatial import cKDTree
OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('audit','/home/lp531/smolvla_eval/three_camera_dataset_strict_audit_20261006/audit.py');audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
from sys import path
path.insert(0,'/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge')
from z1_isaac_bridge.smolvla_rollout_core import JOINT_LIMITS
labels=json.loads(Path('/home/lp531/pi05_lora_workspace/binary_gripper_20261007/episode_labels.json').read_text())
states=[];actions=[];initial=[];close=[]
for e in labels:
 rows=[json.loads(x) for x in (Path(e['raw_root'])/'frames.jsonl').read_text().splitlines()]
 states.extend(r['observation.state'][:6] for r in rows);actions.extend(r['action'][:6] for r in rows)
 initial.append(rows[0]['observation.state']);close.append(rows[e['close_frame']]['observation.state'])
states=np.array(states);actions=np.array(actions);tree=cKDTree(states)
np.savez(OUT/'reference.npz',states=states,actions=actions)
result=[]
for episode in sorted(Path('/home/lp531/smolvla_eval/pi05_bottle_binary_chunk15_attn_ffn').glob('*/episodes/*')):
 rows=[json.loads(x) for x in (episode/'trace.jsonl').read_text().splitlines()]
 if not rows: continue
 r=json.loads((episode/'result.json').read_text()) if (episode/'result.json').exists() else {}
 sync=[json.loads(x) for x in (episode/'input_sync.jsonl').read_text().splitlines()]
 unique=[x for x in rows if x['execution_substep']==0]
 q=np.array([x['joint_state'][:6] for x in rows]);cmd=np.array([x['command'][:6] for x in rows]);raw=np.array([x['policy_action'][:6] for x in unique]);actual=np.array([x['joint_state'][:6] for x in unique])
 nearest,idx=tree.query(actual)
 clip=raw-np.clip(raw,JOINT_LIMITS[:6,0],JOINT_LIMITS[:6,1])
 closed=[x for x in rows if x['execution_target'][-1]==0]
 contacts=[x for x in rows if x['contact_status']['left_finger_touching_table'] or x['contact_status']['right_finger_touching_table']]
 last=rows[-1]
 fk_stats={}
 for name,arr in [('actual',q),('command',cmd),('policy',raw)]:
  poses=[audit.fk(x) for x in arr];zs=np.array([p[0][2] for p in poses]);rolls=np.array([audit.roll(p[1]) for p in poses]);fk_stats[name]={'z_min':float(zs.min()),'z_final':float(zs[-1]),'roll_final':float(rolls[-1])}
 def stamp(x):return dict(sim=x['simulation_elapsed_s'],wall=x['elapsed_s'],relative=x['contact_status']['relative_position_gripper_m'],q=x['joint_state'],pred=x['policy_action'],cmd=x['command'],index=x['policy_action_index'])
 item={'run':episode.parent.parent.name,'completed':bool(r),'seed':r.get('seed'),'stop':r.get('stop_reason'),'success':r.get('success'),'execute_steps':r.get('execute_steps'),'sim_duration':r.get('simulation_duration_s'),'target':r.get('target_position_m'),'trace_rows':len(rows),'predictions':len(sync),'first_close':stamp(closed[0]) if closed else None,'first_table_trace':stamp(contacts[0]) if contacts else None,'last':stamp(last),'final_status':r.get('final_status'),
 'tracking_error_arm_rad':{'median':float(np.median(np.max(np.abs(cmd-q),axis=1))),'max':float(np.max(np.abs(cmd-q)))},'joint_clip_counts':np.sum(abs(clip)>1e-5,axis=0).tolist(),'joint_clip_max_rad':np.max(abs(clip),axis=0).tolist(),'state_distance_to_training_l2_rad':{'initial':float(nearest[0]),'final':float(nearest[-1]),'max':float(nearest.max())},'nearest_training_last':states[idx[-1]].tolist(),'fk':fk_stats,'sync_max_skew':max(x['sensor_skew_s'] for x in sync),'sync_max_age':max(x['oldest_sensor_age_s'] for x in sync)}
 result.append(item)
print(json.dumps(result,indent=2))
(OUT/'analysis.json').write_text(json.dumps({'episodes':result,'training_initial_q_median':np.median(initial,axis=0).tolist(),'training_close_q_median':np.median(close,axis=0).tolist(),'training_action_min':actions.min(axis=0).tolist(),'training_action_max':actions.max(axis=0).tolist()},indent=2)+'\n')
