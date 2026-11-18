"""Independent checks of saved20Hz episodes. Never repairs timestamps."""
import json,math,hashlib
from pathlib import Path
CAMERAS=['wrist','global','global_right']
def check(ok,msg):
 if not ok:raise ValueError(msg)
def audit_rows(rows):
 check(len(rows)>1,'empty/short episode');base=rows[0]['simulation_time'];previous={};spreads=[];ages=[];gaps=[]
 for i,x in enumerate(rows):
  t=x['simulation_time'];check(math.isfinite(t),'nonfinite time')
  check(x['frame_index']==i and abs(x['timestamp']-i/20)<1e-7,'frame index/time grid')
  check(abs(t-base-i/20)<2e-6,'simulation grid gap')
  check(abs(x['joint_state_stamp']-t)<2e-6,'state stamp mismatch')
  stamps=[x[f'{c}_image_stamp'] for c in CAMERAS]
  check(all(math.isfinite(s) for s in stamps),'nonfinite camera stamp')
  spread=max([t,*stamps])-min([t,*stamps]);check(spread<=.020001,'camera/state spread >20ms');spreads.append(spread)
  for c,s in zip(CAMERAS,stamps):
   check(s>previous.get(c,-1),'reused/out-of-order camera');previous[c]=s
  a,b=x['joint_state_source_stamps'];check(a<=t+1e-8 and b>=t-1e-8 and 0<=b-a<=.050001,'invalid state bracket');gaps.append(b-a)
  age=t-x['action_stamp'];check(math.isfinite(age) and -1e-8<=age<=.050001,'future/stale action');check(abs(age-x['action_age_s'])<2e-6,'action age inconsistency');ages.append(age)
 return dict(frames=len(rows),fps=20,max_camera_state_spread_s=max(spreads),max_action_age_s=max(ages),max_state_bracket_gap_s=max(gaps),no_reused_images=True,continuous_simulation_grid=True,causal_actions=True)
def audit_episode(folder):
 folder=Path(folder);summary=json.loads((folder/'episode_summary.json').read_text());start=json.loads((folder/'episode_start.json').read_text());raw=(folder/'frames.jsonl').read_bytes();rows=[json.loads(x) for x in raw.splitlines()];report=audit_rows(rows)
 check(summary['fps']==20 and start['fps']==20,'wrong fps');check(summary['dropped_frames']==0,'dropped frames');check(summary['frame_count']==len(rows),'frame count')
 for key in ['task_success','physical_grasp_ok','bilateral_contact_ever','object_lifted_clear_of_table_ever','release_sequence_complete','placement_stable']:check(summary.get(key) is True,'missing physical success '+key)
 check(start['target_object']=='cup','wrong target')
 count=0
 for x in rows:
  for c in CAMERAS:
   p=folder/x[f'observation.images.{c}'];check(p.is_file() and p.stat().st_size>100,'missing/empty image');count+=1
 report.update(passed=True,episode=str(folder),seed=summary['scene_randomization']['seed'],grid=summary['scene_randomization']['target_grid'],images=count,frames_sha256=hashlib.sha256(raw).hexdigest(),physical_success_verified=True)
 return report
