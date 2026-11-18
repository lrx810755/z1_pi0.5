"""Distinguish reaching closure from the production service's full loaded carry."""
import json

def failure_kind(r):
 if r.get('passed'):return 'passed'
 if r.get('failed_stage') in ('reset','planning','approach'):return r['failed_stage']
 error=r.get('error','')
 if '闭爪前' in error or '校正' in error or '最终位置验收' in error:return 'preclose'
 if '纯物理抬升' in error or '搬运' in error or '抬升' in error:return 'lift_or_transport'
 if '闭爪' in error or '预紧' in error:return 'closure'
 return 'other'

def read_lines(path):
 if not path.exists():return
 with path.open() as f:
  for line in f:
   try:yield json.loads(line)
   except json.JSONDecodeError:pass

def evidence(folder):
 events=list(read_lines(folder/'assistant_events.jsonl'))
 closed=any(r.get('phase')=='GUIDED_CLOSE' for r in events)
 heights=[]
 for r in read_lines(folder/'scene_events.jsonl'):
  if r.get('bilateral_contact') and r.get('left_finger_contact_object')=='cup' and r.get('right_finger_contact_object')=='cup':
   pts=[r.get(side+'_finger_contact_point_object_m') for side in ['left','right']]
   if all(isinstance(p,list) and len(p)==3 for p in pts):heights=[p[2]*1000 for p in pts];break
 return dict(closure_attempted=closed,first_bilateral_contact_height_mm=heights)
