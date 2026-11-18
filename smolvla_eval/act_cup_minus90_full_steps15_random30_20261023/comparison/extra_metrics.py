import json,collections
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;d=json.loads((R/'comparison.json').read_text())
TRAIN=Path('/home/lp531/pi05_lora_workspace/act_cup_minus90_full_cached_20261023')
records=json.loads((TRAIN/'split_manifest.json').read_text())['source_records']
with np.load(TRAIN/'sampling_arrays.npz') as a:
 weights=a['weights']
 mass={}
 for label,when in [('closing','close_frame'),('release','open_frame')]:
  mask=np.zeros(len(weights),bool)
  for r in records:
   i=r['dataset_from_index']+r[when];mask[max(r['dataset_from_index'],i-15):i]=True
  mass[label+'_transition_window_mass']=float(weights[mask].sum())
ages=[];spans=[];jointclips=[];grippergap=[];evercaptured_withdraw=[]
for e in d['episodes']:
 p=Path(e['path']);tr=[json.loads(l) for l in (p/'trace.jsonl').read_text().splitlines()]
 if e['ever_physical']:
  final=e['final_status'];print('PHYSICAL',e['episode'],'firstphysical',e['first_physical_time'],'events',[(x['kind'],round(x['time'],2)) for x in e['events']],'final', {k:final.get(k) for k in ['object_position_m','holding_observed','released','physical_grasp_ok','placement_stable']})
  inbox=[t for t in tr if t['contact_status'].get('object_position_m') and abs(t['contact_status']['object_position_m'][0]-.32)<=.055 and abs(t['contact_status']['object_position_m'][1]-.14)<=.055 and t['simulation_elapsed_s']>=e['first_physical_time']]
  if inbox:
   chunks=[json.loads(l) for l in (p/'prediction_chunks.jsonl').read_text().splitlines()]
   after=[t for t in tr if t['simulation_elapsed_s']>=inbox[0]['simulation_elapsed_s']]
   item=dict(episode=e['episode'],first_above_box_s=inbox[0]['simulation_elapsed_s'],last_above_box_s=inbox[-1]['simulation_elapsed_s'],final_position=final['object_position_m'],open_command_after_box=any(t['execution_target'][-1]<-.895 for t in after),minimum_policy_gripper_after_box=min(t['policy_action'][-1] for t in after),final_policy_gripper=tr[-1]['policy_action'][-1])
   spans.append(item)
 # Distinguish preclose movement from movements later in a never-closed episode.
 movements=[]
 initial=np.array(e['target_position'])
 for t in tr:
  pos=t['contact_status'].get('object_position_m')
  if pos and np.linalg.norm(np.array(pos)[:2]-initial[:2])>.005:
   movements.append(dict(time=t['simulation_elapsed_s'],actual_jaw=t['joint_state'][-1],target_jaw=t['execution_target'][-1],left=t['contact_status'].get('left_finger_contact_object'),right=t['contact_status'].get('right_finger_contact_object'),position=pos));break
 e['first_shift_over5mm']=movements[0] if movements else None
 first=e['first_close'];end=first['trace_index'] if first else len(tr)
 opens=tr[:end]
 if first:
  i=first['trace_index'];target=first['command_jaw'];look=[t for t in tr[i:] if t['simulation_elapsed_s']-first['time']<=.35]
  e['jaw_after_first_close_035s']=[dict(t=round(t['simulation_elapsed_s']-first['time'],3),actual=t['joint_state'][-1],command=t['command'][-1]) for t in look[::6]]
 clipping=np.array([np.abs(np.array(t['execution_target'][:6])-t['joint_state'][:6]).max() for t in tr if t['execution_substep']==0])
 jointclips.append(dict(episode=e['episode'],fraction_arm_requests_over_point1rad=float(np.mean(clipping>.10001)),p95_policy_arm_error=float(np.quantile(clipping,.95))))
summary=dict(weighted_sampling_mass=mass,above_box=spans,clipping=jointclips,preclose_shift=collections.Counter('never_closed' if not e['first_close'] else 'closed' for e in d['episodes'] if e['first_shift_over5mm'] and (not e['first_close'] or e['first_shift_over5mm']['time']<e['first_close']['time'])))
(R/'comparison.json').write_text(json.dumps(d,ensure_ascii=False,indent=2));(R/'extra_metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='clipping'},ensure_ascii=False,indent=2))
