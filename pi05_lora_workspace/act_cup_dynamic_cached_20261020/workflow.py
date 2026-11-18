"""Wait for the complete dynamic cup set, cache, train ACT, then evaluate 50 trials."""
from pathlib import Path
import argparse,collections,datetime,fcntl,hashlib,json,os,signal,subprocess,time
from workflow_core import collection_ready
RUN=Path(__file__).resolve().parent
COLLECTION=Path('/home/lp531/z1_queue_logs/cup_standard_dynamic_5x4x8_20261020')
RAW=Path('/home/lp531/smolvla_data/z1_tabletop_raw_cup_standard_dynamic_three_camera_20261020')
EVAL=Path('/home/lp531/smolvla_eval/act_cup_dynamic_steps15_random50_20261020')
CHECKPOINT=Path('/home/lp531/smolvla_training/act_cup_dynamic_gentle_cached_chunk16_20261020/checkpoints/050000/pretrained_model')
PY='/home/lp531/pi05_lora_workspace/.venv/bin/python'

def save(name,value):
 p=RUN/name;tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(p)
def state(stage,**extra):
 save('workflow_status.json',dict(stage=stage,updated=datetime.datetime.now().astimezone().isoformat(),**extra))
def train_env():
 env=os.environ.copy()
 for key in ['PYTHONPATH','PYTHONHOME']:env.pop(key,None)
 env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8',PYTHONUNBUFFERED='1',PYTORCH_ALLOC_CONF='expandable_segments:True')
 return env

def step(name,command,env=None):
 with (RUN/(name+'.log')).open('a',buffering=1) as log:
  p=subprocess.Popen(command,cwd=RUN,env=env or train_env(),stdout=log,stderr=subprocess.STDOUT)
  state(name,pid=p.pid)
  try:code=p.wait()
  except BaseException:
   p.terminate()
   try:p.wait(timeout=10)
   except subprocess.TimeoutExpired:p.kill();p.wait()
   raise
 if code:raise RuntimeError(f'{name} exited {code}; inspect {name}.log')

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--dry-run',action='store_true');a=parser.parse_args()
 if a.dry_run:
  ready,info=collection_ready(RAW,COLLECTION);print(json.dumps(dict(ready=ready,collection=info,stages=['wait_for_160_and_collection_shutdown','prepare249','cache_and_verify','smoke_and_reload','train50000','final_reload','random50_steps15'],checkpoint=str(CHECKPOINT),evaluation=str(EVAL)),ensure_ascii=False,indent=2));return
 with (RUN/'.workflow.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (RUN/'workflow_completed.json').exists():return
  try:
   verification_wait_started=None
   while True:
    ready,info=collection_ready(RAW,COLLECTION)
    state('waiting_for_collection',**info)
    if ready:break
    done=COLLECTION/'completed.json'
    if done.exists():
     result=json.loads(done.read_text())
     if result.get('complete') is not True or result.get('exit_code')!=0:raise RuntimeError('Collection exited before exact 160/160 quota')
     if verification_wait_started is None:verification_wait_started=time.monotonic()
     if time.monotonic()-verification_wait_started>120:raise RuntimeError('Collection finished but final quota verification did not complete')
    else:
     launch=json.loads((COLLECTION/'launch.json').read_text());pid=launch['pid']
     if not Path(f'/proc/{pid}').exists():raise RuntimeError('Collection process exited without completed marker')
    time.sleep(15)
   save('collection_ready_verified.json',info)
   step('prepare',[PY,'-u','prepare.py'])
   manifest=json.loads((RUN/'split_manifest.json').read_text());records=manifest['source_records']
   counts=collections.Counter(r['source'] for r in records)
   assert counts==dict(original=160,old_correction=55,new_correction=34),counts
   assert all(str(RAW) in r['source_path'] for r in records if r['source']=='original')
   frozen=[dict(source=r['source'],path=r['source_path'],frames_sha256=r['source_frames_sha256']) for r in records]
   save('frozen_sources.json',dict(collection_stopped=True,episodes=249,manifest=frozen))
   step('cache_build',[PY,'-u','build_cache.py'])
   step('recipe_verification',[PY,'-u','run_tests.py'])
   step('cache_verification',[PY,'-u','verify_cache.py'])
   step('smoke',[PY,'-u','train.py','--smoke'])
   step('checkpoint_verification',[PY,'-u','verify_checkpoint.py'])
   save('launch.json',dict(target_optimizer_updates=50000,batch_size=16,gradient_accumulation=1,fresh_training=True,cache_verified=True,smoke_verified=True,output=str(CHECKPOINT.parents[2]),seed=20261013))
   step('training',['bash',str(RUN/'run_training.sh')])
   completed=json.loads((RUN/'completed.json').read_text());assert completed['optimizer_updates']==50000
   assert (CHECKPOINT/'model.safetensors').stat().st_size>0
   env=train_env();env['Z1_ACT_VERIFY_CHECKPOINT']=str(CHECKPOINT)
   step('final_checkpoint_verification',[PY,'-u','verify_checkpoint.py'],env)
   (RUN/'latest_eval.txt').write_text(str(EVAL)+'\n')
   step('evaluation50',['bash',str(EVAL/'start.sh')],os.environ.copy())
   results=json.loads((EVAL/'final_box_results.json').read_text());assert results['completed'] and results['total']==50 and len(results['episodes'])==50
   summary=dict(completed=True,training_updates=50000,evaluations=50,successes=results['successes'],success_rate=results['success_rate'],checkpoint=str(CHECKPOINT),results=str(EVAL/'final_box_results.json'),finished=datetime.datetime.now().astimezone().isoformat())
   save('workflow_completed.json',summary);state('completed',**summary)
  except BaseException as exc:
   state('failed',error=str(exc));raise
if __name__=='__main__':main()
