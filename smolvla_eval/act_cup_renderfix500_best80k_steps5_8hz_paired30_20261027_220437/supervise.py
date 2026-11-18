import os,signal,subprocess,json,time,urllib.request,fcntl
from pathlib import Path
R=Path(__file__).resolve().parent
PORT=8810

def save(**v):
 p=R/'supervisor_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def status():
 with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/api/status',timeout=3) as f:return json.load(f)
def stop(p):
 if p is None:return
 for sig,delay in [(signal.SIGINT,5),(signal.SIGTERM,3),(signal.SIGKILL,0)]:
  try:os.killpg(p.pid,sig)
  except ProcessLookupError:break
  if delay:time.sleep(delay)
 try:p.wait(timeout=3)
 except subprocess.TimeoutExpired:pass

def main():
 signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
 scene=None;runner=None
 with (R/'.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   cmd=['ros2','launch','z1_isaac_bridge','z1_isaac.launch.py','scene:=/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/generated_dual_camera/smolvla_dual_wrist_camera.usd','isaac_script:=/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/scripts/run_z1_ros2_scene.py','external_camera_layout:=dual','target_object:=cup','headless:=true','rviz:=false','task_gui:=true','run_smolvla_policy:=false','freeze_target_keep_collisions:=true','wrist_rgb_topic:=/left_wrist_camera/color/image_raw','vla_wrist_rgb_topic:=none',f'task_gui_port:={PORT}',f'dataset_root:={R}/inactive_recorder','recording_fps:=10.0','recording_sync_tolerance_s:=0.045','recording_sync_delay_s:=0.5','recording_action_age_s:=0.40']
   (R/'scene_command.json').write_text(json.dumps(cmd,indent=2))
   with (R/'scene.log').open('w') as f:scene=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   save(stage='scene_starting',scene_pid=scene.pid)
   deadline=time.monotonic()+240
   while time.monotonic()<deadline:
    if scene.poll() is not None:raise RuntimeError('scene exited during startup')
    try:
     s=status()
     if s.get('connected') and all(s.get(k,{}).get('ready') for k in ['recording','episode_reset','collision_scene','object_collisions']):break
    except Exception:pass
    time.sleep(2)
   else:raise RuntimeError('scene readiness timeout')
   assert "Close-range glass occlusion: {'/rtx/translucency/worldEps': 0.0001}" in (R/'scene.log').read_text(), 'Renderer fix missing at startup'
   eval_env=os.environ.copy();eval_env.pop('PYTHONNOUSERSITE',None)
   with (R/'runner.log').open('w') as f:runner=subprocess.Popen(['/usr/bin/python3','-u',str(R/'run_evaluations.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=eval_env)
   save(stage='evaluating',scene_pid=scene.pid,runner_pid=runner.pid,total=30,steps=[5])
   code=runner.wait()
   if code:raise RuntimeError(f'evaluation runner exit {code}; inspect status.json')
   save(stage='generating_report',total=30)
   with (R/'report_generation.log').open('w') as report_log:
    subprocess.run(['/usr/bin/python3',str(R/'summarize.py')],stdout=report_log,stderr=subprocess.STDOUT,check=True)
   save(stage='completed',total=30)
  except BaseException as e:
   save(stage='stopped' if isinstance(e,KeyboardInterrupt) else 'failed',error=str(e));raise
  finally:
   stop(runner);stop(scene)
if __name__=='__main__':main()
