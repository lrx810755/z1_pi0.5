from pathlib import Path
import json,collections,math
import numpy as np
ROOT=Path('/home/lp531/smolvla_eval/pi05_gentle_correction_base_20261010')
OUT=Path(__file__).resolve().parent
results=[]
for run in sorted(ROOT.glob('pi05_sim_eval_*')):
 for ep in sorted((run/'episodes').glob('*')):
  rows=[json.loads(x) for x in (ep/'trace.jsonl').read_text().splitlines() if x.strip()]
  sync=[json.loads(x) for x in (ep/'input_sync.jsonl').read_text().splitlines() if x.strip()]
  result=json.loads((ep/'result.json').read_text()) if (ep/'result.json').exists() else {}
  if not rows:continue
  t=np.array([r['simulation_elapsed_s'] for r in rows]);obj=np.array([r['contact_status']['object_position_m'] for r in rows]);q=np.array([r['joint_state'] for r in rows])
  closed=np.array([r['execution_target'][-1]>-.45 for r in rows]);raw=np.array([r['policy_action'][-1] for r in rows]);trans=np.flatnonzero(closed[1:]!=closed[:-1])+1
  contacts=np.array([r['contact_status'].get('left_finger_contact_object')=='beverage_bottle' and r['contact_status'].get('right_finger_contact_object')=='beverage_bottle' for r in rows])
  opens=[i for i in trans if not closed[i]];closes=[i for i in trans if closed[i]]
  intervals=[]
  for i in closes:
   j=next((j for j in opens if j>i),len(rows)-1)
   intervals.append({'start':round(float(t[i]),3),'end':round(float(t[j]),3),'duration':round(float(t[j]-t[i]),3),'max_z':float(obj[i:j+1,2].max()),'bilateral':bool(contacts[i:j+1].any()),'open_position':obj[j].tolist(),'raw_close':float(raw[i]),'raw_open':float(raw[j]),'boundary_open':rows[j]['policy_action_index']==0,'boundary_close':rows[i]['policy_action_index']==0})
  pred=np.array([r['observation_time_s'] for r in sync]);capture=(obj[:,2]>.04)&contacts
  rel=np.array([r['contact_status']['relative_position_gripper_m'] for r in rows]);envelope=((rel[:,0]>=.110)&(rel[:,0]<=.154)&(np.abs(rel[:,1])<=.04)&(rel[:,2]>=.006)&(rel[:,2]<=.030))
  physical={k:bool(any(r.get(k,False) for r in rows)) for k in ['physical_grasp_ok','lifted','release_sequence_complete']}
  d=dict(run=run.name[-6:],path=str(ep),complete=bool(result),success=result.get('success'),failure=result.get('failure_category','incomplete/interrupted'),initial_position=result.get('target_position_m',obj[0].tolist()),duration_sim=float(t[-1]),trace_rows=len(rows),predictions=len(sync),prediction_interval_median=float(np.median(np.diff(pred))) if len(pred)>1 else None,
   max_z=float(obj[:,2].max()),final_position=obj[-1].tolist(),final_xy_shift=float(np.linalg.norm(obj[-1,:2]-obj[0,:2])),any_bilateral=bool(contacts.any()),any_high_bilateral=bool(capture.any()),high_bilateral_rows=int(capture.sum()),in_envelope_rows=int(envelope.sum()),physical=physical,closed_intervals=intervals,open_close_transitions=len(trans),
   changes_at_chunk_boundary=sum(rows[i]['policy_action_index']==0 for i in trans),raw_gripper_range=[float(raw.min()),float(raw.max())],raw_near_threshold_fraction=float(np.mean((raw>.4)&(raw<.6))),max_sensor_skew=max((r['sensor_skew_s'] for r in sync),default=0),max_sensor_age=max((r['oldest_sensor_age_s'] for r in sync),default=0),table_contact=bool(result.get('failure_category')=='finger_table_contact' or any(r['contact_status'].get('left_finger_touching_table') or r['contact_status'].get('right_finger_touching_table') for r in rows)))
  results.append(d)
(OUT/'metrics.json').write_text(json.dumps(results,indent=2))
for d in results:print(json.dumps({k:v for k,v in d.items() if k not in ['path','closed_intervals']},ensure_ascii=False));print('INTERVALS',json.dumps(d['closed_intervals']))
