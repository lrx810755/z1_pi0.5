from pathlib import Path
import json, math, collections
ROOT=Path('/home/lp531/smolvla_eval')
OUT=ROOT/'latest_model_review_20261013'
rows=[]
for model,folder in [('new120k','pi05_fine_grip_recovery_base_20261012_120000'),('old80k','pi05_gentle_correction_base_20261010')]:
 for p in sorted((ROOT/folder).rglob('result.json')):
  result=json.loads(p.read_text()); trace=p.with_name('trace.jsonl')
  data=[json.loads(l) for l in trace.read_text().splitlines() if l.strip()] if trace.exists() else []
  acts=[r for r in data if r.get('execution_substep',0)==0]
  def t(r):return r.get('simulation_elapsed_s',r['elapsed_s'])
  bouts=[]; start=None; last=None
  for r in data:
   close=r['execution_target'][-1]>=-.75
   if close and start is None:start=t(r)
   if not close and start is not None:bouts.append([start,t(r)]);start=None
   last=t(r)
  if start is not None:bouts.append([start,last])
  contacts=[r for r in data if any(r.get('contact_status',{}).get(k)=='beverage_bottle' for k in ['left_finger_contact_object','right_finger_contact_object'])]
  dual=[r for r in data if all(r.get('contact_status',{}).get(k)=='beverage_bottle' for k in ['left_finger_contact_object','right_finger_contact_object'])]
  pos=result['target_position_m']
  displacements=[math.dist(r['contact_status']['object_position_m'][:2],pos[:2]) for r in data if r.get('contact_status',{}).get('object_position_m')]
  near=[r for r in acts if (r.get('contact_status',{}).get('gripper_object_distance_m') or 99)<.17]
  vals=[r['policy_action'][-1] for r in acts]
  # Each trace row holds the same action during interpolated substeps.
  row=dict(model=model,run=p.parents[2].name,path=str(p.parent),seed=result['seed'],position=pos,
    result=result['failure_category'],duration=result.get('simulation_duration_s'),actions=len(acts),
    closed_actions=sum(v>=.5 for v in vals),close_bouts=bouts,raw_min=min(vals,default=None),raw_max=max(vals,default=None),
    near_actions=len(near),near_closed=sum(r['policy_action'][-1]>=.5 for r in near),
    near_max=max([r['policy_action'][-1] for r in near],default=None),
    first_contact=t(contacts[0]) if contacts else None,first_dual=t(dual[0]) if dual else None,
    max_displacement=max(displacements,default=0),physical=result['physical_predicates'],
    sync_max_skew=None,sync_max_age=None)
  sync=p.with_name('input_sync.jsonl')
  if sync.exists():
   inp=[json.loads(l) for l in sync.read_text().splitlines() if l.strip()]
   row['sync_sample']=inp[0] if inp else None
  rows.append(row)
(OUT/'metrics.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
for r in rows:
 print(r['model'],r['run'].split('_')[-1],str(r['position'][:2]),r['result'],
       'acts',r['actions'],'close',r['closed_actions'],'bouts',[[round(v,2) for v in x] for x in r['close_bouts']],
       'rawmax',round(r['raw_max'] or 0,3),'near',r['near_actions'],r['near_closed'],
       'contact',round(r['first_contact'] or 0,2),'dual',round(r['first_dual'] or 0,2),
       'push_cm',round(r['max_displacement']*100,1))
for model in ['new120k','old80k']:
 group=[r for r in rows if r['model']==model and r['result']!='blank_camera']
 print('SUMMARY',model,'n',len(group),'neverclose',sum(not r['close_bouts'] for r in group),'contact',sum(r['first_contact'] is not None for r in group),'dual',sum(r['first_dual'] is not None for r in group),'verified',sum(r['physical']['physical_grasp_ok'] for r in group),'push>5cm',sum(r['max_displacement']>.05 for r in group))
print('SYNC SAMPLE',rows[0]['sync_sample'])
