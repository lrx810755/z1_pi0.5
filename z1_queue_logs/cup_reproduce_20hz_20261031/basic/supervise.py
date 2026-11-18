import json,os,sys,time,subprocess,signal,fcntl,datetime,hashlib,threading
from pathlib import Path
R=Path(__file__).resolve().parent
DATA=Path('/home/lp531/smolvla_data/z1_cup_basic400_20hz_20261031')
SOURCE=Path('/home/lp531/ros2_ws/src/z1_ros2')
sys.path.insert(0,str(SOURCE/'z1_isaac_bridge/z1_isaac_bridge'))
from continuous_training import JsonApiClient,ContinuousTrainingRunner
from audit import audit_episode
from fault_latch import check_fault
sys.path.insert(0,str(R.parent))
from schedule import pending_cells
API=JsonApiClient('http://127.0.0.1:8820')
def save(name,d):
 p=R/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def status(stage,**kw):save('status.json',dict(updated=datetime.datetime.now().astimezone().isoformat(),stage=stage,total=400,**kw))
def output(message):
 check_fault(R/'sync_fault.json')
 print(message,flush=True)
 if any(s in message for s in ['时间对齐失败','no fresh image','timestamp spread','interpolation gap','no recent causal','simulation clock moved']):raise RuntimeError('Synchronization fault; collection stopped before next attempt: '+message)
def stop(p):
 if p is None:return
 for sig,delay in [(signal.SIGINT,4),(signal.SIGTERM,3),(signal.SIGKILL,0)]:
  try:os.killpg(p.pid,sig)
  except ProcessLookupError:break
  if delay:time.sleep(delay)
 try:p.wait(timeout=3)
 except subprocess.TimeoutExpired:pass
def main():
 scene=None;reports=[]
 check_fault(R/'sync_fault.json')
 def watch():
  while True:
   if (R/'sync_fault.json').exists():
    os.kill(os.getpid(),signal.SIGTERM);return
   time.sleep(.1)
 threading.Thread(target=watch,daemon=True).start()
 signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
 with (R/'.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   DATA.mkdir(exist_ok=True,parents=True)
   # Resume only after re-auditing every already-saved episode.
   for p in sorted(DATA.glob('episode_*')):reports.append(audit_episode(p))
   counts={tuple([c,r]):0 for r in range(4) for c in range(5)}
   for x in reports:counts[tuple(x['grid'])]+=1
   assert all(n<=20 for n in counts.values())
   cmd=['ros2','launch',str(R/'collection.launch.py'),'external_camera_layout:=dual','headless:=true','rviz:=false','task_gui:=true','run_smolvla_policy:=false','freeze_target_keep_collisions:=true',f'dataset_root:={DATA}','target_object:=cup','wrist_rgb_topic:=/left_wrist_camera/color/image_raw','vla_wrist_rgb_topic:=none','recording_fps:=20.0','recording_sync_tolerance_s:=0.020','recording_sync_delay_s:=0.5','recording_action_age_s:=0.050','task_gui_port:=8820']
   save('scene_command.json',cmd)
   with (R/'scene.log').open('a') as f:scene=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   status('scene_starting',scene_pid=scene.pid,completed=len(reports))
   deadline=time.monotonic()+300
   while time.monotonic()<deadline:
    if scene.poll() is not None:raise RuntimeError('Scene exited')
    try:
     s=API.get('/api/status')
     if s.get('connected') and all(s.get(k,{}).get('ready') for k in ['recording','episode_reset','collision_scene','object_collisions']):break
    except Exception:pass
    time.sleep(2)
   else:raise RuntimeError('Scene readiness timeout')
   info=json.loads((DATA/'dataset_info.json').read_text());assert info['fps']==20 and set(info['image_shapes'])=={'wrist','global','global_right'}
   assert "Close-range glass occlusion: {'/rtx/translucency/worldEps': 0.0001}" in (R/'scene.log').read_text()
   stalled=0;round_number=0
   while len(reports)<400:
    round_number+=1;round_before=len(reports)
    for col,row in pending_cells(counts):
     while counts[col,row]<20:
      before=set(DATA.glob('episode_*'));status('collecting',completed=len(reports),current_grid=[col,row],round=round_number,cell_counts={f'{c},{r}':v for (c,r),v in counts.items()})
      runner=ContinuousTrainingRunner(API,'cup','full',1,target_grid=(col,row),max_failures=3,output=output)
      code=runner.run()
      check_fault(R/'sync_fault.json')
      if code in (3,4):
       print('DEFER CELL',col,row,'remaining',20-counts[col,row],flush=True);break
      if code:raise RuntimeError(f'Expert collection exited {code} at cell {col},{row}')
      new=set(DATA.glob('episode_*'))-before
      assert len(new)==1,new
      p=new.pop();report=audit_episode(p);assert report['grid']==[col,row]
      assert report['seed'] not in {x['seed'] for x in reports},'duplicate new dataset seed'
      reports.append(report);counts[col,row]+=1
      save('episode_audits.json',reports)
      print('AUDITED',len(reports),'/400',p.name,flush=True)
    stalled=stalled+1 if len(reports)==round_before else 0
    if stalled>=3:
     print('No new basic episodes this round; continuing missing cells',flush=True);time.sleep(5)
   assert len(reports)==400 and all(x==20 for x in counts.values())
   final=dict(completed=True,total=400,fps=20,dataset=str(DATA),all_episode_audits_passed=True,images=sum(x['images'] for x in reports),frames=sum(x['frames'] for x in reports),max_camera_state_spread_s=max(x['max_camera_state_spread_s'] for x in reports),max_action_age_s=max(x['max_action_age_s'] for x in reports),cell_counts={f'{c},{r}':v for (c,r),v in counts.items()},finished=datetime.datetime.now().astimezone().isoformat())
   save('completed.json',final);status('completed',completed=400,**{k:v for k,v in final.items() if k not in ['completed','total']})
  except BaseException as e:
   status('stopped' if isinstance(e,KeyboardInterrupt) else 'failed',completed=len(reports),error=str(e));raise
  finally:stop(scene)
if __name__=='__main__':main()
