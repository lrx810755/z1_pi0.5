from pathlib import Path
import json,collections
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
records=[]
for p in sorted((ROOT/'act_sim_eval_20261014_161455').rglob('result.json')):
 r=json.loads(p.read_text());rows=[json.loads(x) for x in (p.parent/'trace.jsonl').read_text().splitlines()]
 closed=[x for x in rows if x['execution_target'][-1]==0];lift=[x for x in rows if x['lifted']];physical=[x for x in rows if x['physical_grasp_ok']];switch=[];last=-.9
 for x in rows:
  z=x['execution_target'][-1]
  if z!=last:switch.append({'time_s':x['simulation_elapsed_s'],'target':z});last=z
 hold=sum(x.get('gripper_close_hold_blocked_open',False) for x in rows)
 maxz=max(x['contact_status']['object_position_m'][2] for x in rows if x['contact_status'].get('object_position_m'))
 contact=[x for x in rows if (x['contact_status'].get('target_finger_contact_pair_count') or 0)>0]
 first=closed[0] if closed else None
 records.append(dict(episode=r['episode_number'],seed=r['seed'],success=r['success'],failure=r['failure_category'],ever_close=bool(closed),close_count=sum(x['target']==0 for x in switch),open_count=sum(x['target']==-.9 for x in switch),switches=switch,hold_blocked_trace_rows=hold,ever_lift=bool(lift),ever_verified_grasp=bool(physical),final_was_grasped=r['final_status'].get('was_grasped'),max_object_root_z_m=maxz,first_close_time_s=first['simulation_elapsed_s'] if first else None,first_close_relative_position_m=first['contact_status']['relative_position_gripper_m'] if first else None,contact_trace_rows=len(contact),result=str(p)))
f=[r for r in records if not r['success']]
summary=dict(episodes=len(records),successes=sum(r['success'] for r in records),failure_counts=dict(collections.Counter(r['failure'] for r in f)),failed_without_close=sum(not r['ever_close'] for r in f),failed_with_close=sum(r['ever_close'] for r in f),failed_with_lift=sum(r['ever_lift'] for r in f),failed_with_verified_grasp=sum(r['ever_verified_grasp'] for r in f),episodes_hold_blocked=[r['episode'] for r in records if r['hold_blocked_trace_rows']],success_episodes=[r['episode'] for r in records if r['success']],multiple_close_episodes=[r['episode'] for r in records if r['close_count']>1])
(OUT/'metrics.json').write_text(json.dumps(dict(summary=summary,episodes=records),indent=2));print(json.dumps(summary,indent=2))
for r in f:
 if r['ever_close']:print(r['episode'],'maxz',round(r['max_object_root_z_m'],3),'firstpos',r['first_close_relative_position_m'])
