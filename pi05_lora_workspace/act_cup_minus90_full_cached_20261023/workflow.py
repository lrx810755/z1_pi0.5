"""Predecode every source, verify release labels/pixels, smoke-check, then train ACT."""
import datetime,fcntl,json,os,subprocess,sys,traceback
from pathlib import Path
RUN=Path(__file__).resolve().parent
PYTHON='/home/lp531/pi05_lora_workspace/.venv/bin/python'

def status(stage,**values):
 data=dict(stage=stage,updated_at=datetime.datetime.now().astimezone().isoformat(),**values)
 p=RUN/'workflow_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2));t.replace(p)
 print(json.dumps(data),flush=True)

def command(stage,args,log,extra=None):
 status(stage)
 env=os.environ.copy()
 if extra:env.update(extra)
 with (RUN/log).open('a') as output:
  subprocess.run([PYTHON,'-u',*args],cwd=RUN,env=env,stdout=output,stderr=subprocess.STDOUT,check=True)

with (RUN/'.workflow.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  command('predecoding',['build_cache.py'],'cache_build.log')
  command('verifying_cache',['verify_cache.py'],'cache_verification.log')
  command('smoke_training',['train.py','--smoke'],'smoke.log')
  command('verifying_smoke_checkpoint',['verify_checkpoint.py'],'checkpoint_verification.log')
  command('training',['train.py','--train'],'training.log')
  checkpoint=json.loads((RUN/'latest_checkpoint.json').read_text())['checkpoint']
  command('verifying_final_checkpoint',['verify_checkpoint.py'],'final_checkpoint_verification.log',{'Z1_ACT_VERIFY_CHECKPOINT':checkpoint})
  status('completed',optimizer_updates=50000,checkpoint=checkpoint)
 except BaseException as exc:
  status('failed',error=str(exc),traceback=traceback.format_exc())
  raise
