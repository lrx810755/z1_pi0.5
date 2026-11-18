"""Durable read-only audit of this collection, including failed attempts."""
from pathlib import Path
import json,time,urllib.request,os,collections,datetime
ROOT=Path(__file__).parent
DATA=Path('/home/lp531/smolvla_data/z1_tabletop_raw_cup_standard_three_camera_20261019')
launch=json.loads((ROOT/'launch.json').read_text());pid=launch['pid']
last_auto=None
while Path(f'/proc/{pid}').exists():
 now=datetime.datetime.now().astimezone().isoformat()
 try:
  d=json.load(urllib.request.urlopen('http://127.0.0.1:8767/api/status',timeout=3))
  counts=collections.Counter()
  for p in DATA.glob('episode_*/episode_summary.json'):
   s=json.loads(p.read_text());g=s.get('scene_randomization',{}).get('target_grid')
   if s.get('target_object')=='cup' and s.get('task_success') and g:counts[tuple(g)]+=1
  status={'updated_at':now,'saved':sum(counts.values()),'goal':160,'grid':{f'{c},{r}':counts[c,r] for r in range(4) for c in range(5)},'live':d}
  tmp=ROOT/'live_status.tmp';tmp.write_text(json.dumps(status,ensure_ascii=False));tmp.replace(ROOT/'live_status.json')
  with (ROOT/'live_trace.jsonl').open('a') as f:f.write(json.dumps({'at':now,'status':d},ensure_ascii=False)+'\n')
  auto=d.get('auto_demonstration',{});key=(auto.get('phase'),auto.get('message'))
  if key!=last_auto:
   with (ROOT/'stage_events.jsonl').open('a') as f:f.write(json.dumps({'at':now,'auto':auto,'recording':d.get('recording'),'position':d.get('object_position_m')},ensure_ascii=False)+'\n')
   last_auto=key
 except Exception as e:
  with (ROOT/'watch.log').open('a') as f:f.write(now+' '+str(e)+'\n')
 time.sleep(2)
