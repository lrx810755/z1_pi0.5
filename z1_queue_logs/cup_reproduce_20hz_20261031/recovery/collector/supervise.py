import pathlib,subprocess,json,time,os,signal,urllib.request,datetime
ROOT=pathlib.Path(__file__).resolve().parent.parent;PORT=8822
DATA=pathlib.Path('/home/lp531/smolvla_data/z1_cup_push_recovery13_20hz_20261031')
def write(name,d):
 p=ROOT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2));t.replace(p)
def main():
 signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
 DATA.mkdir(exist_ok=True);scene=None
 env=os.environ.copy();env['Z1_PUSH_RECOVERY_CURRENT']=str(ROOT/'current.json');env['Z1_ISAAC_GRASP_CONFIG']=str(ROOT/'grasp_physics_minus90.json')
 cmd=['ros2','launch',str(ROOT/'collector/collection.launch.py'),'external_camera_layout:=dual','headless:=true','rviz:=false','target_object:=cup','task_gui:=true',f'task_gui_port:={PORT}','run_smolvla_policy:=false','freeze_target_keep_collisions:=true',f'dataset_root:={DATA}','wrist_rgb_topic:=/left_wrist_camera/color/image_raw','vla_wrist_rgb_topic:=none','recording_fps:=20.0','recording_sync_tolerance_s:=0.020','recording_sync_delay_s:=0.7','recording_action_age_s:=0.050']
 write('scene_command.json',cmd)
 try:
  if (ROOT/'scene.log').exists():(ROOT/'scene.log').rename(ROOT/('scene_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.log'))
  with (ROOT/'scene.log').open('w') as f:scene=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=env)
  write('workflow_status.json',dict(stage='scene_starting',scene_pid=scene.pid));end=time.monotonic()+300
  while time.monotonic()<end:
   if scene.poll() is not None:raise RuntimeError('Scene exited')
   try:
    with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/api/status',timeout=2) as f:s=json.load(f)
    if s.get('connected') and all(s.get(k,{}).get('ready') for k in ['recording','episode_reset','collision_scene','object_collisions']):break
   except Exception:pass
   time.sleep(2)
  else:raise RuntimeError('Scene startup timeout')
  assert "Close-range glass occlusion: {'/rtx/translucency/worldEps': 0.0001}" in (ROOT/'scene.log').read_text()
  write('workflow_status.json',dict(stage='collecting',scene_pid=scene.pid))
  with (ROOT/'collection.log').open('a') as f:subprocess.run(['/usr/bin/python3','-u',str(ROOT/'collector/replay.py')],stdout=f,stderr=subprocess.STDOUT,check=True,env=env)
  write('workflow_status.json',dict(stage='independent_collection_audit'))
  with (ROOT/'audit.log').open('w') as f:subprocess.run(['/usr/bin/python3',str(ROOT/'collector/audit_saved.py')],stdout=f,stderr=subprocess.STDOUT,check=True,env=env)
 finally:
  if scene:
   for sig,wait in [(signal.SIGINT,10),(signal.SIGTERM,4),(signal.SIGKILL,1)]:
    try:os.killpg(scene.pid,sig);scene.wait(timeout=wait)
    except ProcessLookupError:break
    except subprocess.TimeoutExpired:continue
 write('workflow_status.json',dict(stage='completed',total=13))
if __name__=='__main__':
 try:main()
 except BaseException as exc:write('workflow_status.json',dict(stage='failed',error=str(exc)));raise
