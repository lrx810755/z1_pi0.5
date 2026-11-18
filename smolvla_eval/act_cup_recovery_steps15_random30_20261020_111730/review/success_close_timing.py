import json,csv,statistics
from pathlib import Path
R=Path(__file__).resolve().parent;ms=json.loads((R/'metrics.json').read_text());out=[]
def stable_end(rows,start,end,predicate,duration=.2):
 begin=None
 for q in rows:
  t=q['simulation_elapsed_s']
  if t<start or t>=end:continue
  if predicate(q):
   if begin is None:begin=t
   if t-begin>=duration-1e-6:return begin,t
  else:begin=None
 return None,None
for m in ms:
 if not m['final_box_success']:continue
 rows=[json.loads(x) for x in (Path(m['path'])/'trace.jsonl').read_text().splitlines()]
 # Select the close that actually supports the first >5cm lift and final transport.
 opens=[e['time'] for e in m['events'] if e['event']=='open']
 first_lift=m['first_high_s'];close=max(e['time'] for e in m['events'] if e['event']=='close' and e['time']<=first_lift)
 release=next(t for t in opens if t>close)
 during=[q for q in rows if close<=q['simulation_elapsed_s']<release]
 both=lambda q:all(q['contact_status'].get(k)=='cup' for k in ['left_finger_contact_object','right_finger_contact_object'])
 first_both=next(q['simulation_elapsed_s'] for q in during if both(q))
 continuous_start,continuous_confirm=stable_end(rows,close,release,both)
 first_verified=next((q['simulation_elapsed_s'] for q in during if q['contact_status'].get('physical_grasp_ok')),None)
 lift_begin,lift_confirm=stable_end(rows,close,release,lambda q:q['contact_status']['object_position_m'][2]>=.05 and both(q))
 assert continuous_confirm is not None and lift_begin is not None
 item=dict(episode=m['episode'],successful_close_s=close,first_bilateral_contact_delay_s=first_both-close,
  bilateral_contact_0p2s_confirm_delay_s=continuous_confirm-close,first_5cm_lift_delay_s=first_lift-close,
  bilateral_5cm_lift_0p2s_confirm_delay_s=lift_confirm-close,physical_grasp_verified_delay_s=None if first_verified is None else first_verified-close,
  close_to_release_s=release-close,previous_unsuccessful_close_count=sum(e['event']=='close' and e['time']<close for e in m['events']))
 out.append(item)
summary={}
for key in ['first_bilateral_contact_delay_s','bilateral_contact_0p2s_confirm_delay_s','first_5cm_lift_delay_s','bilateral_5cm_lift_0p2s_confirm_delay_s','close_to_release_s']:
 v=[d[key] for d in out];summary[key]={'mean':statistics.mean(v),'median':statistics.median(v),'min':min(v),'max':max(v)}
result={'definition':'6 final-box successes; use the last close before first 5cm cup-root lift. Continuous-contact .2s is an offline timing definition, not a changed evaluation gate. Episode21 succeeds under final-box rule but not legacy physical_grasp_ok.','episodes':out,'summary':summary}
(R/'success_close_timing.json').write_text(json.dumps(result,indent=2))
with (R/'成功闭爪耗时.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
print(json.dumps(result,ensure_ascii=False,indent=2))
