import json,sys
from pathlib import Path
import numpy as np
from geometry import audit,height
sys.path.insert(0,'/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge')
from z1_isaac_bridge.smolvla_rollout_core import JOINT_LIMITS,safe_joint_target
ROOT=Path(__file__).resolve().parent;SRC=Path('/home/lp531/smolvla_eval/pi05_execute5_vs10_20261007')
results=[]
for p in sorted(SRC.glob('steps*/**/result.json')):
 r=json.loads(p.read_text());tr=[json.loads(x) for x in (p.parent/'trace.jsonl').read_text().splitlines()];pr=[json.loads(x) for x in (p.parent/'predictions.jsonl').read_text().splitlines()];sy=[json.loads(x) for x in (p.parent/'input_sync.jsonl').read_text().splitlines()]
 lookup={tuple(a):(k,j) for k,pred in enumerate(pr) for j,a in enumerate(pred['full_actions'][:r['execute_steps']])}
 actions=[]
 for i,t in enumerate(tr):
  if t['execution_substep']!=0:continue
  raw=np.array(t['execution_target']);q=np.array(t['joint_state']);clip=np.clip(raw,JOINT_LIMITS[:,0],JOINT_LIMITS[:,1]);endpoint=safe_joint_target(raw,q,.1)
  k,j=lookup[tuple(t['policy_action'])]
  # Reconstruct endpoint from the sixth interpolated command, when present.
  block=tr[i:i+6];complete=len(block)==6 and [x['execution_substep'] for x in block]==list(range(6))
  endpoint_actual=np.array(block[-1]['command']) if complete else None
  nextq=np.array(tr[i+6]['joint_state']) if i+6<len(tr) and complete else None
  armheight=lambda a:height(np.r_[a[:6],q[-1]])
  rh=armheight(raw);ch=armheight(clip);eh=armheight(endpoint)
  item=dict(prediction=k,action_index=j,time=t['simulation_elapsed_s'],raw=raw.tolist(),state=q.tolist(),limit_change_rad=float(np.max(abs(raw[:6]-clip[:6]))),step_change_rad=float(np.max(abs(clip[:6]-endpoint[:6]))),limit_position_change_mm=float(np.linalg.norm(audit.fk(raw[:6])[0]-audit.fk(clip[:6])[0])*1000),step_position_change_mm=float(np.linalg.norm(audit.fk(clip[:6])[0]-audit.fk(endpoint[:6])[0])*1000),raw_height_m=rh,clipped_height_m=ch,endpoint_height_m=eh,actual_height_m=armheight(q),end_command_error_rad=None if endpoint_actual is None else float(np.max(abs(endpoint_actual[:6]-endpoint[:6]))),next_actual_error_rad=None if nextq is None else float(np.max(abs(nextq[:6]-endpoint[:6]))))
  actions.append(item)
 first=lambda fn:next((x for x in actions if fn(x)),None)
 limited=[a for a in actions if a['limit_change_rad']>.001];stepped=[a for a in actions if a['step_change_rad']>.001]
 item=dict(group=p.parts[len(SRC.parts)],seed=r['seed'],success=r['success'],failure=r['failure_category'],actions=len(actions),limit_changed_gt1mrad=len(limited),step_changed_gt1mrad=len(stepped),max_limit_change_rad=max(a['limit_change_rad'] for a in actions),max_step_change_rad=max(a['step_change_rad'] for a in actions),max_limit_position_change_mm=max(a['limit_position_change_mm'] for a in actions),max_step_position_change_mm=max(a['step_position_change_mm'] for a in actions),first_limit_change_gt02rad=first(lambda x:x['limit_change_rad']>.02),first_step_change_gt02rad=first(lambda x:x['step_change_rad']>.02),first_raw_under4mm=first(lambda x:x['raw_height_m']<.004),first_clip_only_under4mm=first(lambda x:x['raw_height_m']>=.004 and x['clipped_height_m']<.004),max_endpoint_reconstruction_error_rad=max((a['end_command_error_rad'] or 0 for a in actions)),p95_endpoint_tracking_error_rad=float(np.percentile([a['next_actual_error_rad'] for a in actions if a['next_actual_error_rad'] is not None],95)),sync_max_skew_s=max(x['sensor_skew_s'] for x in sy))
 results.append(item);(ROOT/(item['group']+'_'+str(r['seed'])+'_actions.json')).write_text(json.dumps(actions,indent=2))
( ROOT/'execution_summary.json').write_text(json.dumps(results,indent=2));print(json.dumps([{k:v for k,v in r.items() if not k.startswith('first')} for r in results],indent=2))
