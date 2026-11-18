import os,sys,json,time,signal,subprocess,fcntl,datetime,hashlib
from pathlib import Path
from training_gate import training_ready
R=Path(__file__).resolve().parent
TRAIN=Path('/home/lp531/pi05_lora_workspace/pi05_cup513_20hz_chunk50_lora_20261031')
CHECKPOINT=Path('/home/lp531/smolvla_training/pi05_cup513_20hz_chunk50_lora_20261031/checkpoints/240000/pretrained_model')
TRAIN_UNIT='pi05-cup513-20hz-chunk50-lora-20261031.service'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(name,d):
 p=R/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def state(stage,**kw):save('waiting_status.json',dict(stage=stage,total=50,action_rate_hz=20,execute_steps=15,updated=datetime.datetime.now().astimezone().isoformat(),**kw))
def exclude_seeds():
 seeds=set();sources=[]
 # All available training manifests, including this full513 train/validation set.
 for p in Path('/home/lp531/pi05_lora_workspace').glob('*/split_manifest.json'):
  d=json.loads(p.read_text());rows=d.get('source_records',[])
  for x in rows:
   if type(x.get('seed')) is int:seeds.add(x['seed'])
  sources.append(str(p))
 # Prior exclusion baselines preserve earlier experiments outside present training roots.
 for p in Path('/home/lp531/smolvla_eval').glob('*/excluded_seed_baseline.json'):
  if p.parent==R:continue
  seeds.update(json.loads(p.read_text()))
 for p in Path('/home/lp531/smolvla_eval').glob('*/**/episodes/*/result.json'):
  d=json.loads(p.read_text())
  if type(d.get('seed')) is int:seeds.add(d['seed'])
 save('excluded_seed_baseline.json',sorted(seeds));save('seed_exclusion_sources.json',dict(training_manifests=sources,count=len(seeds),rule='All training and heldout source seeds plus all prior evaluation result seeds'))
def prepare():
 excluded_seed_sources=exclude_seeds()
 val=json.loads((TRAIN/'validation_latest.json').read_text())
 selected=dict(checkpoint=str(CHECKPOINT),optimizer_updates=30000,microsteps=240000,training_fps=20,chunk_size=50,execute_steps=15,selection='final model after complete training and final reload verification',heldout_loss=val['heldout_loss'],adapter_sha256=sha(CHECKPOINT/'adapter_model.safetensors'))
 save('selected_checkpoint.json',selected)
 source=Path('/home/lp531/ros2_ws/src/z1_ros2')
 files=list(R.glob('*.py'))+[R/'start.sh',R/'selected_checkpoint.json',R/'excluded_seed_baseline.json',TRAIN/'split_manifest.json',TRAIN/'final_checkpoint_reload_verified.json',CHECKPOINT/'config.json',CHECKPOINT/'adapter_model.safetensors',CHECKPOINT/'policy_preprocessor.json',source/'isaac_assets/z1/generated_dual_camera/smolvla_dual_wrist_camera.usd',source/'isaac_assets/z1/scripts/run_z1_ros2_scene.py',source/'z1_isaac_bridge/z1_isaac_bridge/pi05_policy_bridge.py',Path('/home/lp531/z1_queue_logs/cup_policy_push_recovery20_20261028_134320/grasp_physics_minus90.json')]
 save('source_snapshot.json',{str(p):sha(p) for p in files})
def main():
 signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
 with (R/'.waiting.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   while True:
    active=subprocess.run(['systemctl','--user','is-active','--quiet',TRAIN_UNIT]).returncode==0
    if training_ready(TRAIN,CHECKPOINT,active):
     free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
     if free>=18000:break
     state('waiting_for_gpu',free_memory_mib=free)
    else:
     live=json.loads((TRAIN/'live_training.json').read_text())
     state('waiting_for_training',optimizer_updates=live.get('optimizer_updates'),target_updates=30000)
    time.sleep(10)
   prepare();state('starting_evaluation',checkpoint=str(CHECKPOINT))
   subprocess.run(['/usr/bin/python3','-u',str(R/'supervise.py')],check=True)
   result=json.loads((R/'completed.json').read_text());assert result['completed'] and result['total']==50
   state('completed',successes=result['successes'],strict_successes=result['strict_successes'])
  except BaseException as e:state('stopped' if isinstance(e,KeyboardInterrupt) else 'failed',error=str(e));raise
if __name__=='__main__':main()
