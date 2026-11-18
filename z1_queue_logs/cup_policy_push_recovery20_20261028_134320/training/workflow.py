"""Explicit gated handoff: CPU preparation default; GPU smoke/train require flags."""
import argparse,datetime,fcntl,json,os,subprocess,sys,shutil,traceback
from pathlib import Path
from prepare import RUN,NEW,write,sha,collection_gate,load
PYTHON='/home/lp531/pi05_lora_workspace/.venv/bin/python'
LEROBOT=Path('/home/lp531/pi05_lora_workspace/lerobot')

def command(stage,args,log):
 write('workflow_status.json',dict(stage=stage,updated_at=datetime.datetime.now().astimezone().isoformat()));print(stage,flush=True)
 env=os.environ.copy();env.update(PYTHONNOUSERSITE='1',PYTHONPATH=str(LEROBOT/'src'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
 with (RUN/log).open('a') as output:subprocess.run([PYTHON,'-u',*args],cwd=RUN,env=env,stdout=output,stderr=subprocess.STDOUT,check=True)

def provenance():
 pin=RUN/'code_pins';pin.mkdir(exist_ok=True);hashes={}
 for path in sorted(RUN.glob('*.py')):
  digest=sha(path);dest=pin/path.name
  if dest.exists() and sha(dest)!=digest:raise ValueError('Pipeline changed after pin; use a new run directory')
  if not dest.exists():shutil.copyfile(path,dest)
  hashes[path.name]=digest
 import torch,torchvision
 backbone=Path('/home/lp531/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth')
 tracked={str(p.relative_to(LEROBOT)):sha(p) for p in [LEROBOT/'src/lerobot/scripts/lerobot_train.py',LEROBOT/'src/lerobot/datasets/sampler.py',LEROBOT/'src/lerobot/policies/act/configuration_act.py',LEROBOT/'src/lerobot/policies/act/modeling_act.py']}
 evidence=dict(pipeline_files=hashes,lerobot_files=tracked,python=sys.version,interpreter=sys.executable,torch=torch.__version__,torchvision=torchvision.__version__,backbone_sha256=sha(backbone),completion_sha256=sha(NEW/'completion_verified.json'))
 if (RUN/'runtime_provenance.json').exists() and load(RUN/'runtime_provenance.json')!=evidence:raise ValueError('Runtime or acceptance evidence changed')
 write('runtime_provenance.json',evidence)

def main():
 p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');p.add_argument('--cache-only',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--train',action='store_true');p.add_argument('--workers',type=int,default=2);args=p.parse_args()
 with (RUN/'.workflow.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   collection_gate();provenance()
   command('cpu_tests',['-m','unittest','discover','-s','tests','-v'],'cpu_tests.log')
   command('prepare',['prepare.py'],'prepare.log')
   if args.prepare_only:write('workflow_status.json',dict(stage='prepared'));return
   command('cache_new_only',['build_cache.py','--workers',str(args.workers)],'cache_build.log')
   command('verify_cache',['verify_cache.py'],'cache_verification.log')
   if args.cache_only or not(args.smoke or args.train):write('workflow_status.json',dict(stage='cache_verified_waiting_gpu_handoff'));return
   if not (RUN/'checkpoint_reload_verified.json').exists():
    command('smoke_batch64',['train.py','--smoke','--batch-size','64'],'smoke_batch64.log')
    smoke=load(RUN/'smoke_live.json')
    if smoke['allocated_gib']>20:raise ValueError('Smoke exceeds20GiB memory budget')
    command('reload_smoke',['verify_checkpoint.py'],'checkpoint_verification.log')
   if not args.train:write('workflow_status.json',dict(stage='smoke_reload_verified'));return
   audit=load(RUN/'sampling_audit.json')
   write('selected_hyperparameters.json',dict(batch_size=64,steps=90000,chunk_size=30,learning_rate=1e-5,backbone_learning_rate=1e-5,fp32=True,fresh_policy=True,pretrained_backbone='ResNet18_Weights.IMAGENET1K_V1',source_mass=audit['source_mass'],phase_sampling_mass=audit['phase_sampling_mass']))
   command('training',['train.py','--train'],'training.log')
   checkpoint=load(RUN/'latest_checkpoint.json')['checkpoint'];os.environ['Z1_ACT_VERIFY_CHECKPOINT']=checkpoint
   command('reload_final',['verify_checkpoint.py'],'final_checkpoint_verification.log')
   write('workflow_status.json',dict(stage='completed',optimizer_updates=90000,checkpoint=checkpoint))
  except BaseException as exc:
   write('workflow_status.json',dict(stage='failed',error=str(exc),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
