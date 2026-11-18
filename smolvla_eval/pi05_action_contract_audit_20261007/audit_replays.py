import json,sys
from pathlib import Path
import numpy as np
from geometry import audit
sys.path.insert(0,'/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge')
from z1_isaac_bridge.smolvla_rollout_core import JOINT_LIMITS
ROOT=Path(__file__).resolve().parent;labels={x['raw_episode']:x for x in json.loads((ROOT/'selected.json').read_text())}
results=[]
for p in sorted((ROOT/'replays').glob('*/episodes/*/result.json')):
 r=json.loads(p.read_text());name=p.parts[-4][:14];lab=labels[name];source=Path(lab['raw_root']);frames=[json.loads(x) for x in (source/'frames.jsonl').read_text().splitlines()];tr=[json.loads(x) for x in (p.parent/'trace.jsonl').read_text().splitlines()]
 start=[t for t in tr if t['execution_substep']==0];events=[]
 for i,t in enumerate(start):
  f=frames[min(i,len(frames)-1)];assert np.max(abs(np.array(f['action'])[:6]-np.array(t['policy_action'])[:6]))<1e-5
  q=np.array(t['joint_state']);goal=np.array(t['execution_target']);limited=float(max(0,np.max(abs(goal[:6]-q[:6]))-.1));ref=np.array(f['observation.state']);pos=float(np.linalg.norm(audit.fk(q[:6])[0]-audit.fk(ref[:6])[0])*1000)
  events.append(dict(frame=i,time=t['simulation_elapsed_s'],step_excess_rad=limited,reference_state_position_error_mm=pos,gripper_rad=q[-1],command_gripper=goal[-1],object=t['contact_status']['object_position_m'],relative=t['contact_status']['relative_position_gripper_m']))
 near=events[max(0,lab['close_frame']-1)]
 z=[t['contact_status']['object_position_m'][2] for t in tr if t['contact_status']['object_position_m']]
 item=dict(episode=name,variant=p.parts[-4][15:],success=r['success'],physical_predicates=r['physical_predicates'],step_limited_actions=sum(e['step_excess_rad']>.001 for e in events),max_step_excess_rad=max(e['step_excess_rad'] for e in events),p95_reference_state_position_error_mm=float(np.percentile([e['reference_state_position_error_mm'] for e in events],95)),preclose=near,max_object_z_m=max(z),final_object=r['final_status']['object_position_m'],source_hold_angle=frames[min(lab['close_frame']+15,len(frames)-1)]['action'][-1],first_physical_grasp_time=next((t['simulation_elapsed_s'] for t in tr if t['physical_grasp_ok']),None))
 results.append(item)
(ROOT/'replay_analysis.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
