import json,os,time,signal,subprocess,urllib.request,traceback
from pathlib import Path
from experiment_core import cases,valid_trial
from report import report
R=Path(__file__).resolve().parent;L=Path('/home/lp531/z1_queue_logs/cup_shape_height_test_20261108')
def atomic(p,data):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(data,ensure_ascii=False,indent=2));t.replace(p)
def stop(process):
 if process.poll() is not None:return
 os.killpg(process.pid,signal.SIGINT)
 try:process.wait(timeout=25)
 except subprocess.TimeoutExpired:
  os.killpg(process.pid,signal.SIGTERM)
  try:process.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
def completed(case):
 p=L/'cases'/case['id']/'result.json'
 return p.exists() and valid_trial(json.loads(p.read_text()))

def main():
 import fcntl
 lock=(L/'.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
 assert free>=18000,'Other GPU workload detected; no experiment scene started'
 allcases=cases();atomic(L/'cases.json',allcases)
 try:
  for shape in ['frustum','cylinder']:
   for height in [34,32]:
    group=[c for c in allcases if c['shape']==shape and c['height_mm']==height]
    missing=[c for c in group if not completed(c)]
    if not missing:continue
    d=L/f'{shape}_h{height}';d.mkdir(exist_ok=True)
    casefile=d/'cases.json';atomic(casefile,group);atomic(L/'current_case.json',missing[0])
    env=dict(os.environ,CUP_EXPERIMENT_SHAPE=shape,CUP_EXPERIMENT_CASE=str(L/'current_case.json'),CUP_EXPERIMENT_GROUP_DIR=str(d),Z1_ISAAC_GRASP_CONFIG=str(R/f'grasp_h{height}.json'))
    cmd=['ros2','launch',str(R/'collection.launch.py'),'isaac_script:='+str(R/'scene.py'),'headless:=false','rviz:=false','task_gui:=true','task_gui_port:=8838','external_camera_layout:=dual','run_smolvla_policy:=false','freeze_target_keep_collisions:=true','target_object:=cup','dataset_root:='+str(d/'inactive_recorder')]
    atomic(L/'status.json',dict(state='STARTING',shape=shape,height_mm=height,completed=report()['completed']))
    for setup_attempt in range(1,4):
     try:
      with (d/'scene.log').open('a') as scene_log:
       p=subprocess.Popen(cmd,env=env,stdout=scene_log,stderr=subprocess.STDOUT,start_new_session=True)
       try:
        deadline=time.monotonic()+420;ready=False
        while time.monotonic()<deadline:
         if p.poll() is not None:raise RuntimeError(f'Scene exited {p.returncode}; see {d}/scene.log')
         try:
          with urllib.request.urlopen('http://127.0.0.1:8838/api/status',timeout=2) as response:s=json.load(response)
          ready=bool(s.get('connected') and all(s.get(k,{}).get('ready') for k in ['episode_reset','collision_scene','object_collisions']))
          if ready:break
         except Exception:pass
         time.sleep(2)
        if not ready:raise RuntimeError('Scene readiness deadline exceeded')
        atomic(d/'startup_status.json',s)
        atomic(L/'status.json',dict(state='RUNNING',shape=shape,height_mm=height,completed=report()['completed']))
        with (d/'trials.log').open('a') as output:
         client=subprocess.Popen(['/usr/bin/python3',str(R/'run_cases.py'),str(casefile)],env=env,stdout=output,stderr=subprocess.STDOUT)
         try:
          rc=client.wait()
         finally:
          if client.poll() is None:client.terminate();client.wait(timeout=15)
         if rc:raise RuntimeError(f'Trial runner exited {rc}; see {d}/trials.log')
       finally:stop(p);report()
      break
     except Exception as exc:
      atomic(d/f'infrastructure_retry_{setup_attempt}.json',dict(error=str(exc),completed=report()['completed']))
      if setup_attempt==3:raise
      time.sleep(4)
    time.sleep(4)
  payload=report();assert payload['complete'] and not payload['production_sources_changed'],payload
  atomic(L/'status.json',dict(state='COMPLETED',**payload))
 except BaseException as exc:
  atomic(L/'status.json',dict(state='STOPPED_WITH_ERROR',error=str(exc),traceback=traceback.format_exc(),**report()));raise
if __name__=='__main__':main()
