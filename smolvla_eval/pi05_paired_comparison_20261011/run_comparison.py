"""Sequential fixed-scene PI05 comparison using the existing live simulator."""
import datetime, hashlib, json, os, subprocess, time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
MODELS={
 'old_080000':'pi05_gentle_correction_base_20261010/checkpoints/080000/pretrained_model',
 'new_080000':'pi05_grip_stage_base_20261011/checkpoints/080000/pretrained_model',
 'new_096000':'pi05_grip_stage_base_20261011/checkpoints/096000/pretrained_model',
 'new_120000':'pi05_grip_stage_base_20261011/checkpoints/120000/pretrained_model',
}
SCENES=[
 (431927079,'previous_old_success'),
 (1420929974,'previous_new_success'),
 (1778049255,'side_short_close'),
 (1437796936,'side_repeated_close'),
 (381017666,'far_open_only'),
 (1174105379,'far_table_contact'),
 (276137783,'near_side'),
 (517570180,'middle_open_only'),
]
INFERENCE_SEED=20261011

def write(name,value):
 p=ROOT/name;t=p.with_suffix(p.suffix+'.tmp')
 t.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');t.replace(p)

def now():return datetime.datetime.now().astimezone().isoformat()

def main():
 env=os.environ.copy()
 env.update(RMW_IMPLEMENTATION='rmw_fastrtps_cpp',
  FASTRTPS_DEFAULT_PROFILES_FILE='/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge/config/fastdds_camera.xml',
  FASTDDS_DEFAULT_PROFILES_FILE='/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge/config/fastdds_camera.xml',
  Z1_PI05_INFERENCE_SEED=str(INFERENCE_SEED),PYTHONUNBUFFERED='1',PYTORCH_ALLOC_CONF='expandable_segments:True')
 manifest=dict(created_at=now(),models={},scenes=[dict(seed=s,label=n) for s,n in SCENES],
  inference_seed=INFERENCE_SEED,expected_episodes=32,execute_steps=10,action_rate=10,execution_rate=60,
  episode_timeout_sim_s=60,selection='diagnostic matched scenarios including historical successes and failures; not a random population sample',
  current_simulator_reused=True)
 for name,rel in MODELS.items():
  p=Path('/home/lp531/smolvla_training')/rel
  assert all((p/n).stat().st_size>0 for n in ['adapter_model.safetensors','config.json','adapter_config.json'])
  manifest['models'][name]=dict(checkpoint=str(p),adapter_sha256=hashlib.sha256((p/'adapter_model.safetensors').read_bytes()).hexdigest())
 for name,path in [('physics','/home/lp531/ros2_ws/install/z1_isaac_bridge/share/z1_isaac_bridge/config/grasp_physics.json'),('scene_script','/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/scripts/run_z1_ros2_scene.py')]:
  manifest[name]=dict(path=path,sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
 write('manifest.json',manifest)
 done=[]
 try:
  for name,model in manifest['models'].items():
   output=ROOT/name
   output.mkdir(exist_ok=True)
   if list(output.glob('*/report.json')):raise RuntimeError('Existing run needs inspection: '+str(output))
   cmd=['/bin/bash','/home/lp531/ros2_ws/src/z1_ros2/scripts/run_smolvla_sim_eval.sh',
    '--checkpoint',model['checkpoint'],'--policy-type','pi05','--objects','beverage_bottle',
    '--episodes-per-object',str(len(SCENES)),'--eval-seeds',*[str(s) for s,n in SCENES],
    '--wrist-rgb-topic','/left_wrist_camera/color/image_raw',
    '--training-dataset-root','/home/lp531/smolvla_data/z1_tabletop_raw_left_wrist_three_camera_v1',
    '--max-training-target-distance-m','0.03','--reject-blank-cameras',
    '--rate','10','--execution-rate','60','--execute-steps','10','--timeout','60',
    '--prelift-contact-stable-s','0','--prelift-contact-timeout-s','2.0',
    '--binary-gripper','--gripper-action-encoding','binary','--no-freeze-target-until-close',
    '--preview-episodes','8','--output',str(output)]
   write('progress.json',dict(state='running',started_at=manifest['created_at'],updated_at=now(),current_model=name,completed_models=done,command=cmd))
   print(now(),'START',name,flush=True)
   with (ROOT/(name+'.log')).open('w') as log:
    result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
   if result.returncode:raise RuntimeError(f'{name} exited {result.returncode}')
   records=[json.loads(p.read_text()) for p in sorted(output.glob('*/episodes/*/result.json'))]
   assert len(records)==8,(name,len(records))
   if any(r.get('failure_category')=='episode_setup_failed' for r in records):
    raise RuntimeError(name+' has setup failure; do not count infrastructure failure as policy failure')
   assert {r['seed'] for r in records}=={s for s,n in SCENES}
   done.append(name)
   print(now(),'DONE',name,'successes',sum(bool(r['success']) for r in records),flush=True)
   subprocess.run(['/usr/bin/python3',str(ROOT/'summarize.py')],check=True)
   time.sleep(3)
  write('progress.json',dict(state='complete',started_at=manifest['created_at'],completed_at=now(),completed_models=done,expected_episodes=32))
  print(now(),'COMPARISON COMPLETE',flush=True)
 except BaseException as exc:
  write('progress.json',dict(state='error',updated_at=now(),completed_models=done,error=f'{type(exc).__name__}: {exc}'))
  raise

if __name__=='__main__':main()
