"""Supervise cache verification, training smoke, and the authorized full run."""
from pathlib import Path
import subprocess,json,time,datetime,os
W=Path(__file__).resolve().parent
PY='/home/lp531/pi05_lora_workspace/.venv/bin/python'
CACHE=Path('/home/lp531/smolvla_data/act_apple_rgb224_cache_20261018')
def status(stage,**extra):
 (W/'pipeline_status.json').write_text(json.dumps(dict(stage=stage,updated=datetime.datetime.now().isoformat(),**extra),indent=2))
def run(args,name):
 status(name)
 with (W/(name+'.log')).open('a',buffering=1) as f:
  p=subprocess.Popen(args,cwd=W,stdout=f,stderr=subprocess.STDOUT)
  status(name,pid=p.pid)
  code=p.wait()
 if code:raise RuntimeError(f'{name} exited {code}; see {name}.log')
try:
 status('cache_build_wait')
 pid=json.loads((W/'cache_launch.json').read_text())['pid'];deadline=time.monotonic()+7200
 while not (CACHE/'completed.json').exists():
  if time.monotonic()>deadline:raise RuntimeError('Cache build timed out')
  try:os.kill(pid,0)
  except ProcessLookupError:raise RuntimeError('Cache builder exited without complete manifest')
  time.sleep(5)
 run([PY,'-u','verify_cache.py'],'cache_verification')
 run([PY,'-u','train.py','--smoke'],'smoke')
 run([PY,'-u','verify_checkpoint.py'],'checkpoint_verification')
 status('training_launch')
 with (W/'training.log').open('a',buffering=1) as f:
  p=subprocess.Popen(['bash',str(W/'run_training.sh')],stdout=f,stderr=subprocess.STDOUT)
  launch=dict(pid=p.pid,started=datetime.datetime.now().isoformat(),target_optimizer_updates=50000,batch_size=16,gradient_accumulation=1,fresh_training=True,cache_verified=True,smoke_verified=True,checkpoint=str('/home/lp531/smolvla_training/act_apple_gentle_cached_chunk16_20261018'),seed=20261018)
  (W/'launch.json').write_text(json.dumps(launch,indent=2));status('training',pid=p.pid)
  code=p.wait()
 if code:raise RuntimeError(f'Training exited {code}')
 status('completed',optimizer_updates=50000)
except Exception as e:
 status('failed',error=str(e));raise
