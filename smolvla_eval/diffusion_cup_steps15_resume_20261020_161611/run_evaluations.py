"""Resume preserved trials; blank-camera runs stay archived and are replaced."""
import json,subprocess,time,os,signal,datetime
from pathlib import Path
R=Path(__file__).resolve().parent;BASE=Path((R/'base_root.txt').read_text());valid=[Path(p) for p in json.loads((R/'existing_valid_results.json').read_text())];invalid=[Path(p) for p in json.loads((R/'previous_invalid_results.json').read_text())]
def save(root,name,data):
 p=root/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(data,ensure_ascii=False,indent=2));t.replace(p)
def status(**data):
 for root in [R,BASE]:save(root,'status.json',dict(updated=datetime.datetime.now().isoformat(),total=30,**data))
def stop(p):
 if p.poll() is not None:return
 p.send_signal(signal.SIGINT)
 try:p.wait(timeout=10)
 except subprocess.TimeoutExpired:
  p.kill();p.wait(timeout=5)
def all_results(folder):return sorted(folder.glob('diffusion_sim_eval_*/episodes/*/result.json'))
try:
 for batch in range(1,6):
  remaining=30-len(valid)
  if remaining<=0:break
  excluded=set(json.loads((R/'excluded_seeds.json').read_text()))
  for path in valid+invalid:
   seed=json.loads(path.read_text()).get('seed')
   if seed is not None:excluded.add(seed)
  save(R,'excluded_seeds.json',sorted(excluded))
  folder=R/f'batch_{batch:02d}';cmd=json.loads((R/'command_template.json').read_text());cmd[1]=str(R/'eval_entry.py');cmd[cmd.index('--episodes-per-object')+1]=str(remaining);cmd[cmd.index('--output')+1]=str(folder);save(R,f'command_batch_{batch:02d}.json',cmd)
  with (R/f'batch_{batch:02d}.log').open('w') as log:
   p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
   while p.poll() is None:
    files=all_results(folder);good=[];bad=[]
    for path in files:
     d=json.loads(path.read_text())
     if d.get('error_detail') or d.get('cleanup_error'):
      bad.append(path)
      if d.get('error_detail') not in ['blank_camera:wrist','blank_history_camera:wrist'] or d.get('cleanup_error'):
       stop(p);raise RuntimeError(f'Evaluation chain error: {path}: {d.get("error_detail")}')
     else:good.append(path)
    status(status='running',pid=p.pid,completed=len(valid)+len(good),invalid=len(invalid)+len(bad),batch=batch,execute_steps=15)
    time.sleep(3)
   if p.returncode:raise RuntimeError(f'Evaluator exit {p.returncode}')
  for path in all_results(folder):
   d=json.loads(path.read_text());(invalid if d.get('error_detail') or d.get('cleanup_error') else valid).append(path)
  save(R,'valid_results.json',[str(p) for p in valid]);save(R,'invalid_results.json',[str(p) for p in invalid])
 if len(valid)!=30:raise RuntimeError(f'Only {len(valid)} valid trials after five batches')
 rows=[];health=[]
 for number,path in enumerate(valid,1):
  d=json.loads(path.read_text());s=d['final_status'];x,y,z=s['object_position_m'];inside=abs(x-.32)<=.055 and abs(y-.14)<=.055 and 0<=z<=.12;success=bool(inside and s.get('released') and not s.get('holding_observed'))
  videos={name:(path.parent/'videos'/name).stat().st_size for name in ['wrist.mp4','global_third_person.mp4','global_right.mp4']};assert min(videos.values())>0
  sync=[json.loads(t) for t in (path.parent/'input_sync.jsonl').read_text().splitlines()];assert sync
  for item in sync:
   assert abs(item['observation_time_s']-item['history_observation_time_s']-.1)<1e-7
   for cam,stamp in item['image_stamps_s'].items():assert .05<=stamp-item['history_image_stamps_s'][cam]<=.15
  assert (path.parent/'trace.jsonl').stat().st_size and (path.parent/'prediction_chunks.jsonl').stat().st_size
  health.append(dict(episode=number,predictions=len(sync),videos=videos,history_verified=True))
  rows.append(dict(episode=number,source_episode=d['episode_number'],seed=d['seed'],success=success,inside_box=inside,final_position_m=[x,y,z],released=s.get('released'),holding=s.get('holding_observed'),legacy_success=d['success'],legacy_failure_category=d['failure_category'],source_result=str(path)))
 assert len(set(r['seed'] for r in rows))==30
 summary=dict(completed=True,total=30,execute_steps=15,successes=sum(r['success'] for r in rows),success_rate=sum(r['success'] for r in rows)/30,episodes=rows,excluded_invalid_trials=[str(p) for p in invalid],finished=datetime.datetime.now().isoformat())
 for root in [R,BASE]:
  save(root,'completed.json',summary);save(root,'final_box_results.json',summary);save(root,'trace_video_history_verified.json',health)
 save(BASE,'supervisor_status.json',dict(stage='completed',resume_root=str(R),total=30))
 status(status='complete',completed=30,successes=summary['successes'],success_rate=summary['success_rate'],invalid=len(invalid))
except BaseException as e:
 if 'p' in locals():stop(p)
 status(status='failed',completed=len(valid),error=str(e));raise
