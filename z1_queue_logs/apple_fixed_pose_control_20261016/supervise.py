import os,subprocess,json,time,pathlib
r=pathlib.Path(__file__).parent
base=os.environ.copy()
for k in ('PYTHONPATH','AMENT_PREFIX_PATH','COLCON_PREFIX_PATH','CMAKE_PREFIX_PATH','ROS_DISTRO','ROS_VERSION'):base.pop(k,None)
base['PYTHONUNBUFFERED']='1';base['RMW_IMPLEMENTATION']='rmw_fastrtps_cpp';base['LD_LIBRARY_PATH']='/home/lp531/isaac-sim/exts/isaacsim.ros2.bridge/humble/lib'
for group in ('baseline','P_half','P_double','D_half','D_double','effort_double','position_iterations_double','velocity_iterations_increase','baseline_repeat'):
 env=base.copy();env['APPLE_CONTROL_GROUP']=group
 out=r/group;out.mkdir(exist_ok=True)
 with (out/'scene.log').open('w') as log:
  p=subprocess.Popen(['/home/lp531/isaac-sim/python.sh',str(r/'run_independent.py'),'--scene','/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/generated_dual_camera/smolvla_dual_wrist_camera.usd','--external-camera-layout','dual'],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
 (r/'status.json').write_text(json.dumps({'group':group,'pid':p.pid,'started':time.time()}))
 try:code=p.wait(timeout=120)
 except subprocess.TimeoutExpired:
  import signal
  os.killpg(p.pid,signal.SIGKILL);code=-9
 (out/'exit.json').write_text(json.dumps({'code':code}))
 if code:raise SystemExit(f'{group} failed with {code}')
(r/'all_completed.json').write_text(json.dumps({'complete':True,'time':time.time()}))
