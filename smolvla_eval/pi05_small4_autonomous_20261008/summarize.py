import json,collections,csv
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
records=[json.loads(x) for x in (OUT/'offline_predictions.jsonl').read_text().splitlines()] if (OUT/'offline_predictions.jsonl').exists() else []
offline=[]
for model in sorted({x['model'] for x in records}):
 for stage in dict.fromkeys(x['stage'] for x in records):
  rows=[x for x in records if x['model']==model and x['stage']==stage]
  if not rows:continue
  errors=np.array([x['position_error_mm'] for x in rows]);correct=np.array([x['gripper_correct'] for x in rows])
  offline.append(dict(model=model,stage=stage,n=len(rows),first_action_mm=float(errors[:,0].mean()),first5_mm=float(errors[:,:5].mean()),all15_mm=float(errors.mean()),first_action_max_mm=float(errors[:,0].max()),joint_mae_deg=float(np.rad2deg(np.mean([x['joint_mae_rad'] for x in rows]))),gripper_accuracy=float(correct.mean()),limit_exceeded_chunks=sum(max(x['raw_limit_change_rad'])>.001 for x in rows),raw_under4mm_chunks=sum(min(x['raw_height_m'])<.004 for x in rows),close_in_first5=sum(x['first_close_frame'] is not None and x['first_close_frame']<=5 for x in rows),close_in15=sum(x['first_close_frame'] is not None for x in rows)))
(OUT/'offline_summary.json').write_text(json.dumps(offline,indent=2))
if offline:
 with (OUT/'offline_summary.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(offline[0]));w.writeheader();w.writerows(offline)
results=json.loads((OUT/'results.json').read_text()) if (OUT/'results.json').exists() else []
autonomous=[]
for r in results:
 paths=list((OUT/f"repeat{r['repetition']}"/r['source_episode']).glob('episodes/*/trace.jsonl'));assert len(paths)==1
 p=paths[0];tr=[json.loads(x) for x in p.read_text().splitlines()];pred=[json.loads(x) for x in (p.parent/'predictions.jsonl').read_text().splitlines()];sync=[json.loads(x) for x in (p.parent/'input_sync.jsonl').read_text().splitlines()]
 firstclose=next((t for t in tr if t['execution_target'][-1]==0),None)
 objectposes=np.array([t['contact_status']['object_position_m'] for t in tr if t['contact_status'].get('object_position_m') is not None]);final=r['final_status']['object_position_m'];pos=np.array(final)
 env=r['final_status'].get('grasp_envelope',dict(x_min_m=.11,x_max_m=.154,y_abs_max_m=.04,z_min_m=.006,z_max_m=.03))
 def inside(v):return v is not None and env['x_min_m']<=v[0]<=env['x_max_m'] and abs(v[1])<=env['y_abs_max_m'] and env['z_min_m']<=v[2]<=env['z_max_m']
 ranges=[]
 for t in tr:
  v=t['contact_status'].get('relative_position_gripper_m')
  if v is not None:
   lo=np.array([env['x_min_m'],-env['y_abs_max_m'],env['z_min_m']]);hi=np.array([env['x_max_m'],env['y_abs_max_m'],env['z_max_m']]);ranges.append(float(np.linalg.norm(np.array(v)-np.clip(v,lo,hi))*1000))
 item=dict(episode=r['source_episode'],repeat=r['repetition'],success=r['success'],failure=r['failure_category'],stop_reason=r['stop_reason'],sim_s=r['simulation_duration_s'],physical_grasp=r['physical_predicates'].get('physical_grasp_ok',False),lifted=r['physical_predicates'].get('lifted',False),released=r['physical_predicates'].get('release_sequence_complete',False),predictions=len(pred),first_close_sim_s=None if firstclose is None else firstclose['simulation_elapsed_s'],first_close_relative=None if firstclose is None else firstclose['contact_status']['relative_position_gripper_m'],first_close_in_envelope=False if firstclose is None else inside(firstclose['contact_status']['relative_position_gripper_m']),ever_in_position_envelope=any(inside(t['contact_status'].get('relative_position_gripper_m')) for t in tr),min_distance_to_position_envelope_mm=min(ranges) if ranges else None,max_object_z_m=float(objectposes[:,2].max()) if len(objectposes) else None,final_object=final,interior_floor_check=bool(r['success'] and .280<=pos[0]<=.360 and .100<=pos[1]<=.180 and .007<=pos[2]<=.014),frozen_ticks=sum(t['contact_status'].get('frozen') is True for t in tr),table_contact_ticks=sum(bool(t['contact_status'].get('left_finger_touching_table')) or bool(t['contact_status'].get('right_finger_touching_table')) for t in tr),prediction_closes_in15=sum(any(a[-1]>=.5 for a in x['full_actions']) for x in pred),prediction_closes_in5=sum(any(a[-1]>=.5 for a in x['full_actions'][:5]) for x in pred),sensor_max_skew_s=max(x['sensor_skew_s'] for x in sync),sensor_max_age_s=max(x['oldest_sensor_age_s'] for x in sync),infer_median_ms=float(np.median([x['inference_wall_s'] for x in pred])*1000),logging_median_ms=float(np.median([x['recording_wall_s'] for x in pred])*1000),max_layout_error_mm=max(r['layout_errors_m'].values())*1000,result_path=str(p.parent/'result.json'))
 autonomous.append(item)
summary=dict(completed=(OUT/'completed.json').exists(),trials=len(autonomous),successes=sum(x['success'] for x in autonomous),physical_grasps=sum(x['physical_grasp'] for x in autonomous),interior_floor_successes=sum(x['interior_floor_check'] for x in autonomous),failures=dict(collections.Counter(x['failure'] for x in autonomous)),episodes=autonomous,offline=offline)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k not in ['episodes','offline']},indent=2))
