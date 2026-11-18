"""Launch an isolated simulator and run ten training-scene closed-loop diagnostics."""
import datetime,fcntl,hashlib,json,os,signal,subprocess,sys,time,urllib.request
from pathlib import Path
R=Path(__file__).resolve().parent
PORT=8796

def save_status(stage,**extra):
 data=dict(stage=stage,updated=datetime.datetime.now().astimezone().isoformat(),diagnostic_only=True,**extra)
 p=R/'status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(p)
 print(json.dumps(data,ensure_ascii=False),flush=True)

def stop(p):
 if p is None or p.poll() is not None:return
 for sig,seconds in [(signal.SIGINT,8),(signal.SIGTERM,4),(signal.SIGKILL,1)]:
  try:os.killpg(p.pid,sig)
  except ProcessLookupError:break
  try:p.wait(timeout=seconds);break
  except subprocess.TimeoutExpired:pass

def main():
 scene=runner=None
 with (R/'.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (R/'diagnostic_results.json').exists():
   d=json.loads((R/'diagnostic_results.json').read_text());assert d['completed'];print('Diagnostic already completed; inspect diagnostic_results.json');return 0
  if list((R/'rollouts').glob('act_sim_eval_*')):raise RuntimeError('Partial rollout files already exist; inspect them before restarting')
  # Isaac requires a real NVIDIA device, even when policy inference can fall back to CPU.
  try:
   gpu=subprocess.run(['nvidia-smi','-L'],capture_output=True,text=True,timeout=15)
   gpu_check=dict(returncode=gpu.returncode,stdout=gpu.stdout.strip(),stderr=gpu.stderr.strip())
  except (OSError,subprocess.TimeoutExpired) as e:gpu_check=dict(returncode=-1,error=str(e))
  (R/'environment_preflight.json').write_text(json.dumps(dict(gpu=gpu_check,nvidia_device_paths=[str(p) for p in Path('/dev').glob('nvidia*')],ros_domain=os.environ.get('ROS_DOMAIN_ID')),ensure_ascii=False,indent=2))
  if gpu_check['returncode']!=0:
   save_status('environment_unavailable',completed_episodes=0,reason='NVIDIA driver/device is inaccessible to this execution environment; Isaac Sim cannot run',preflight=str(R/'environment_preflight.json'));return 2
  plan=json.loads((R/'selected_scenes.json').read_text());model=Path(plan['checkpoint'])/'model.safetensors'
  assert hashlib.sha256(model.read_bytes()).hexdigest()==plan['policy_sha256'],'Checkpoint changed after diagnostic selection'
  cmd=['ros2','launch','z1_isaac_bridge','z1_isaac.launch.py','external_camera_layout:=dual','target_object:=cup','headless:=true','rviz:=false','task_gui:=true','run_smolvla_policy:=false','freeze_target_keep_collisions:=true','wrist_rgb_topic:=/left_wrist_camera/color/image_raw','vla_wrist_rgb_topic:=none',f'task_gui_port:={PORT}',f'dataset_root:={R}/inactive_recorder','recording_fps:=10.0','recording_sync_tolerance_s:=0.045','recording_sync_delay_s:=0.5','recording_action_age_s:=0.40']
  (R/'scene_command.json').write_text(json.dumps(cmd,indent=2))
  signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
  try:
   with (R/'scene.log').open('w') as log:scene=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   save_status('scene_starting',scene_pid=scene.pid,completed_episodes=0)
   deadline=time.monotonic()+240
   while time.monotonic()<deadline:
    if scene.poll() is not None:raise RuntimeError('Simulator launch exited; inspect scene.log')
    try:
     with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/api/status',timeout=2) as response:s=json.load(response)
     if s.get('connected') and all(s.get(k,{}).get('ready') for k in ['recording','episode_reset','collision_scene','object_collisions']):break
    except (OSError,ValueError):pass
    time.sleep(2)
   else:raise RuntimeError('Simulator readiness timed out')
   command=json.loads((R/'command.json').read_text());env=os.environ.copy();env.pop('PYTHONNOUSERSITE',None)
   with (R/'runner.log').open('w') as log:runner=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
   save_status('evaluating',scene_pid=scene.pid,runner_pid=runner.pid,completed_episodes=0)
   last=0
   while runner.poll() is None:
    paths=list((R/'rollouts').glob('act_sim_eval_*/episodes/*/result.json'))
    for p in paths:
     d=json.loads(p.read_text())
     if d.get('failure_category')=='episode_setup_failed' or d.get('error_detail') or d.get('cleanup_error'):raise RuntimeError(f'Diagnostic setup/control chain failed: {p}')
    if len(paths)!=last:last=len(paths);save_status('evaluating',scene_pid=scene.pid,runner_pid=runner.pid,completed_episodes=last)
    time.sleep(2)
   if runner.returncode:raise RuntimeError(f'Rollout process exited {runner.returncode}; inspect runner.log')
   from summarize import summarize
   report=summarize();save_status('completed',completed_episodes=10,successes=report['successes'],success_rate=report['success_rate'],results=str(R/'diagnostic_results.json'));return 0
  except BaseException as e:
   save_status('interrupted' if isinstance(e,KeyboardInterrupt) else 'failed',reason=str(e));raise
  finally:stop(runner);stop(scene)
if __name__=='__main__':raise SystemExit(main())
