"""Read-only collection gate and unchanged final-box evaluation rules."""
import collections,json
from pathlib import Path

def collection_ready(data,log):
 counts=collections.Counter();saved=0
 for p in Path(data).glob('episode_*/episode_summary.json'):
  s=json.loads(p.read_text())
  if s.get('target_object')=='cup' and s.get('task_success') and s.get('physical_grasp_ok'):
   counts[tuple(s['scene_randomization']['target_grid'])]+=1;saved+=1
 info=dict(saved=saved,goal=160,grid={f'{c},{r}':counts[c,r] for r in range(4) for c in range(5)})
 if saved!=160 or len(counts)!=20 or any(counts[c,r]!=8 for r in range(4) for c in range(5)):return False,info
 complete=Path(log)/'completed.json';verified=Path(log)/'final_verified.json'
 if not complete.exists() or not verified.exists():return False,info
 d=json.loads(complete.read_text());v=json.loads(verified.read_text())
 ready=d.get('complete') is True and d.get('exit_code')==0 and v.get('quota_verified') is True and v.get('recordings_verified')==160
 return ready,info

def classify_error(record):
 detail=record.get('error_detail','');cleanup=record.get('cleanup_error','')
 if not detail and not cleanup:return 'valid'
 if detail.startswith('blank_camera:') and not cleanup:return 'replace'
 return 'fatal'

def in_box_released(s):
 x,y,z=s['object_position_m']
 return bool(abs(x-.32)<=.055 and abs(y-.14)<=.055 and 0<=z<=.12 and s.get('released') and not s.get('holding_observed'))
