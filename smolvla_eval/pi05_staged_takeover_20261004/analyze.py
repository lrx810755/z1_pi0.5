import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
out=[]
for result in json.loads((ROOT/'results.json').read_text()):
 stage=result['stage']; rep=result['repetition']; folder=ROOT/stage/f'trial_{rep:02d}'
 rows=[json.loads(s) for s in next(folder.glob('episodes/*/trace.jsonl')).read_text().splitlines()]
 item={'stage':stage,'repeat':rep,'success':result['success'],'sim_duration_s':result['simulation_duration_s'],'lifted':result['physical_predicates']['lifted']}
 if stage!='reference':
  predictions=[json.loads(s) for s in (folder/'full_predictions.jsonl').read_text().splitlines()]
  offsets=[next((i for i,a in enumerate(r['actions'][0]) if a[-1]>=-.75),None) for r in predictions]
  prefix=[json.loads(s) for s in (folder/'prefix_trace.jsonl').read_text().splitlines()]
  end=prefix[-1]['command']; same=next(i for i,r in enumerate(prefix) if max(abs(a-b) for a,b in zip(r['command'],end))<1e-6)
  ref=result['handoff']['reference_joints']; env=result['handoff']['status']['grasp_envelope']
  def inside(row):
   p=row['contact_status']['relative_position_gripper_m']
   return env['x_min_m']<=p[0]<=env['x_max_m'] and abs(p[1])<=env['y_abs_max_m'] and env['z_min_m']<=p[2]<=env['z_max_m']
  item.update(first_close_index=offsets[0],close_offsets=offsets,first_chunk_jaw_max=max(a[-1] for a in predictions[0]['actions'][0]),raw_executed_jaw_range=[min(r['policy_action'][-1] for r in rows),max(r['policy_action'][-1] for r in rows)],actual_jaw_range=[min(r['joint_state'][-1] for r in rows),max(r['joint_state'][-1] for r in rows)],handoff_arm_error_rad=result['handoff']['arm_error_rad'],endpoint_hold_s=prefix[-1]['elapsed_s']-prefix[same]['elapsed_s'],first_outside_envelope_s=next((r['simulation_elapsed_s'] for r in rows if not inside(r)),None),first_arm_deviation_over_002_rad_s=next((r['simulation_elapsed_s'] for r in rows if max(abs(a-b) for a,b in zip(r['joint_state'][:6],ref[:6]))>.02),None))
 out.append(item)
(ROOT/'analysis_metrics.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
for x in out:
 print(x['stage'],x['repeat'],'success',x['success'],'time',x['sim_duration_s'],'first close idx',x.get('first_close_index'),'exit envelope',x.get('first_outside_envelope_s'),'jaw',x.get('actual_jaw_range'))
