"""Gated prepare -> full JPEG decode -> exact cache audit -> smoke -> fresh ACT round."""
import argparse,datetime,fcntl,json,os,shutil,subprocess,sys,traceback
from pathlib import Path
from prepare import RUN,write,sha,collection_gate
PYTHON='/home/lp531/pi05_lora_workspace/.venv/bin/python'
LEROBOT=Path('/home/lp531/pi05_lora_workspace/lerobot')

def status(stage,**values):
 data=dict(stage=stage,updated_at=datetime.datetime.now().astimezone().isoformat(),**values);write('workflow_status.json',data);print(json.dumps(data),flush=True)

def command(stage,args,log,extra=None):
 status(stage);env=os.environ.copy();env.update(PYTHONNOUSERSITE='1',PYTHONPATH=str(LEROBOT/'src'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
 if extra:env.update(extra)
 with (RUN/log).open('a') as output:subprocess.run([PYTHON,'-u',*args],cwd=RUN,env=env,stdout=output,stderr=subprocess.STDOUT,check=True)

def provenance():
 pin=RUN/'code_pins';pin.mkdir(exist_ok=True);hashes={}
 for path in sorted(RUN.glob('*.py')):
  digest=sha(path);dest=pin/path.name
  if dest.exists() and sha(dest)!=digest:raise ValueError(f'Pipeline code changed since run pin:{path}; review before continuing')
  if not dest.exists():shutil.copyfile(path,dest)
  hashes[path.name]=digest
 import torch,torchvision,lerobot
 tracked={str(p.relative_to(LEROBOT)):sha(p) for p in [LEROBOT/'src/lerobot/scripts/lerobot_train.py',LEROBOT/'src/lerobot/datasets/sampler.py',LEROBOT/'src/lerobot/policies/act/configuration_act.py',LEROBOT/'src/lerobot/policies/act/modeling_act.py']}
 backbone=Path('/home/lp531/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth')
 if not backbone.is_file():raise ValueError('Expected locally cached ImageNet ResNet18 weights missing')
 versions=dict(backbone_weight_sha256=sha(backbone),interpreter=sys.executable,python=sys.version,torch=torch.__version__,torchvision=torchvision.__version__,lerobot=lerobot.__file__,pipeline_files=hashes,lerobot_files=tracked)
 previous=RUN/'runtime_provenance.json'
 if previous.exists() and json.loads(previous.read_text())!=versions:raise ValueError('Runtime/provenance changed since pin')
 write('runtime_provenance.json',versions)

def main():
 p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');p.add_argument('--cache-only',action='store_true');p.add_argument('--workers',type=int,default=2);args=p.parse_args()
 with (RUN/'.workflow.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   collection_gate();provenance()
   command('preparing',['prepare.py'],'prepare.log')
   if args.prepare_only:status('prepared');return
   command('decoding_all_three_cameras',['build_cache.py','--workers',str(args.workers)],'cache_build.log')
   command('verifying_cache',['verify_cache.py'],'cache_verification.log')
   if args.cache_only:status('cache_verified');return
   batch=32
   try:command('smoke_batch32',['train.py','--smoke','--batch-size','32'],'smoke_batch32.log')
   except subprocess.CalledProcessError:
    log=(RUN/'smoke_batch32.log').read_text()
    if 'out of memory' not in log.lower():raise
    batch=16;command('smoke_batch16',['train.py','--smoke','--batch-size','16'],'smoke_batch16.log')
   smoke=json.loads((RUN/'smoke_live.json').read_text())
   if smoke['allocated_gib']>20:raise RuntimeError('Insufficient GPU memory headroom after smoke')
   audit=json.loads((RUN/'sampling_audit.json').read_text());steps=90000
   write('selected_hyperparameters.json',dict(batch_size=batch,steps=steps,learning_rate=1e-5,backbone_learning_rate=1e-5,weight_decay=1e-4,chunk_size=16,n_action_steps=5,
     total_sampled_anchors=steps*batch,equivalent_unweighted_window_passes=steps*batch/audit['training_windows'],train_windows=audit['training_windows'],source_mass={'basic':.75,'correction':.25},phase_multiplier=1.5,
     fresh_policy=True,pretrained_backbone='ResNet18_Weights.IMAGENET1K_V1',smoke_peak_allocated_gib=smoke['allocated_gib'],selection_rule='90k updates; batch32 if four real-data updates fit, otherwise16 on CUDA OOM; all state/actions statistics train-only'))
   command('verifying_smoke_checkpoint',['verify_checkpoint.py'],'checkpoint_verification.log')
   command('training',['train.py','--train'],'training.log')
   checkpoint=json.loads((RUN/'latest_checkpoint.json').read_text())['checkpoint']
   command('verifying_final_checkpoint',['verify_checkpoint.py'],'final_checkpoint_verification.log',{'Z1_ACT_VERIFY_CHECKPOINT':checkpoint})
   status('completed',optimizer_updates=steps,batch_size=batch,checkpoint=checkpoint)
  except BaseException as exc:
   status('failed',error=str(exc),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
