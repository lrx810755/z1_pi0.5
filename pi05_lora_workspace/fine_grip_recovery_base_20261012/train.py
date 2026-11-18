"""Fresh base LoRA: recovery plus fine grip stage weighting, 120k microsteps, accumulation 8."""
import argparse,copy,importlib.util,json,math,sys
from pathlib import Path
import numpy as np
RUN=Path(__file__).resolve().parent;WORK=RUN.parent
SOURCE=WORK/'weights/pi05_base'
OUTPUT=Path('/home/lp531/smolvla_training/pi05_fine_grip_recovery_base_20261012')
from prepare import MERGED,REPO,write
from sampling import GentleSampler
spec=importlib.util.spec_from_file_location('binary_training',WORK/'binary_gripper_20261007/train.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)

def make_config():
 cfg=previous.make_config();cfg.dataset.root=str(MERGED);cfg.dataset.repo_id=REPO
 records=json.loads((RUN/'phase_manifest.json').read_text())
 cfg.dataset.episodes=[r['episode_index'] for r in records if r['split']=='train']
 cfg.dataset.eval_split=8/len(records) # Actual episode IDs are fixed by datasets(), not a random split.
 cfg.policy.pretrained_path=SOURCE;cfg.policy.use_peft=False
 cfg.policy.optimizer_lr=5e-5;cfg.policy.scheduler_decay_lr=5e-6
 cfg.steps=120000;cfg.policy.scheduler_warmup_steps=4000;cfg.policy.scheduler_decay_steps=120000
 cfg.save_freq=4000;cfg.eval_steps=4000;cfg.max_eval_samples=0
 cfg.log_freq=8;cfg.num_workers=2;cfg.output_dir=OUTPUT
 cfg.job_name='pi05_fine_grip_recovery_base';cfg.seed=20261010
 return cfg

def datasets(cfg):
 from lerobot.datasets.factory import make_dataset
 records=json.loads((RUN/'phase_manifest.json').read_text())
 val_cfg=copy.deepcopy(cfg);val_cfg.dataset.episodes=[r['episode_index'] for r in records if r['split']=='validation']
 return make_dataset(cfg),make_dataset(val_cfg)

def verify(cfg):
 import torch
 from lerobot.datasets.factory import make_dataset
 data=make_dataset(cfg)
 sampler=GentleSampler(data.meta.episodes['dataset_from_index'],data.meta.episodes['dataset_to_index'],episode_indices_to_use=data.episodes,shuffle=True,seed=cfg.seed,absolute_to_relative_idx=data.absolute_to_relative_idx)
 order=list(iter(sampler));sampler.load_state_dict({'epoch':0,'start_index':123})
 assert list(iter(sampler))==order[123:],'weighted resume sequence mismatch'
 import pyarrow.parquet as pq
 table=pq.read_table(MERGED/'data').sort_by([('index','ascending')])
 actions=np.asarray(table['action'].to_pylist(),dtype=np.float32)
 # Decode each episode at its close boundary, including all held-out corrections.
 all_cfg=copy.deepcopy(cfg);all_cfg.dataset.episodes=None;full=make_dataset(all_cfg)
 records=json.loads((RUN/'phase_manifest.json').read_text());checked=0
 for r in records:
  ep=r['episode_index'];start=int(full.meta.episodes[ep]['dataset_from_index']);end=int(full.meta.episodes[ep]['dataset_to_index'])
  anchor=start+max(0,r['close_frame']-1);item=full[anchor]
  expected=actions[np.minimum(np.arange(anchor,anchor+15),end-1)]
  assert np.array_equal(item['action'].numpy(),expected),(ep,'action chunk')
  for key in full.meta.camera_keys:
   im=item[key];assert im.shape[0]==3 and torch.isfinite(im).all() and im.float().std()>1, (ep,key)
  checked+=1
 stats=json.loads((MERGED/'meta/stats.json').read_text())
 old=json.loads((Path('/home/lp531/smolvla_data/z1_bottle_three_camera_pi05_binary_20261007')/'meta/stats.json').read_text())
 assert stats==old
 audit=json.loads((RUN/'sampling_audit.json').read_text())
 write('training_setup_verification.json',dict(passed=True,decoded_episode_boundaries=checked,all_checked_chunks_exact=True,
     deterministic_sampler_resume=True,sampling=audit,train_episodes=len(cfg.dataset.episodes),heldout_new_episodes=8,
     fresh_optimizer=True,base_weights=str(SOURCE),fresh_adapter=True,peak_lr=5e-5,optimizer_updates=15000,gradient_accumulation=8,effective_batch=8,normalization_preserved=True))
 print('DATA AND SAMPLER VERIFIED',json.dumps(audit),flush=True)

def install_hooks(cfg):
 import torch
 from safetensors.torch import load_file
 from peft import get_peft_model_state_dict
 import lerobot.scripts.lerobot_train as trainer
 trainer.make_train_eval_datasets=datasets
 trainer.EpisodeAwareSampler=GentleSampler
 original_loaders=trainer.make_dataloaders
 def loaders(cfg,dataset,eval_dataset,step,parallel_dims):
  train_dl,eval_dl=original_loaders(cfg,dataset,eval_dataset,step,parallel_dims)
  selected=[]
  for ep in eval_dataset.episodes:
   with np.load(RUN/'sampling_arrays.npz') as arrays:
    candidates=np.flatnonzero((arrays['episode']==ep)&arrays['eligible_validation_anchor'])
   assert len(candidates)>0
   absolute=candidates[np.linspace(0,len(candidates)-1,16,dtype=int)]
   selected.extend(eval_dataset.absolute_to_relative_idx[int(i)] for i in absolute)
  val=torch.utils.data.Subset(eval_dataset,selected)
  eval_dl=torch.utils.data.DataLoader(val,batch_size=cfg.batch_size,shuffle=False,num_workers=0)
  write('validation_windows.json',dict(episodes=eval_dataset.episodes,relative_indices=selected,frames_per_episode=16))
  return train_dl,eval_dl
 trainer.make_dataloaders=loaders
 original_policy=trainer.make_policy
 def checked_policy(*args,**kwargs):
  policy=original_policy(*args,**kwargs)
  from dataclasses import asdict
  from safetensors.torch import save_file
  assert (SOURCE/'model.safetensors').is_file() and not (SOURCE/'adapter_model.safetensors').exists()
  full_names=cfg.peft.full_training_modules
  before={n:p.detach().cpu().clone() for n,p in policy.named_parameters() if any('.'+suffix+'.' in n for suffix in full_names)}
  assert len(before)==8
  policy=policy.wrap_with_peft(peft_cli_overrides=asdict(cfg.peft))
  names=dict(policy.named_parameters())
  b=[p for n,p in names.items() if '.lora_B.' in n]
  assert len(b)==252 and all(torch.count_nonzero(p)==0 for p in b),'Fresh LoRA B matrices must be zero'
  for n,expected in before.items():
   prefix,param=n.rsplit('.',1)
   key='base_model.model.'+prefix+'.modules_to_save.default.'+param
   assert torch.equal(names[key].detach().cpu(),expected),key
  actual=get_peft_model_state_dict(policy)
  save_file({k:v.detach().cpu().contiguous() for k,v in actual.items()},str(RUN/'initial_adapter.safetensors'))
  trainable=[(n,p.numel()) for n,p in policy.named_parameters() if p.requires_grad]
  assert trainable and all('lora_' in n or 'modules_to_save' in n for n,_ in trainable)
  assert not any('vision_tower' in n for n,_ in trainable)
  write('initialization_verified.json',dict(base_weights=str(SOURCE),fresh_adapter=True,zero_lora_b_matrices=len(b),full_projection_tensors_match_base=True,tensors=len(actual),
          trainable_parameters=sum(n for _,n in trainable),trainable_names=[n for n,_ in trainable],vision_frozen=True))
  print('BASE INITIALIZATION AND FRESH LORA VERIFIED',len(actual),flush=True)
  return policy
 trainer.make_policy=checked_policy
 original_update=trainer.update_policy;calls=0
 def checked_update(*args,**kwargs):
  nonlocal calls
  result=original_update(*args,**kwargs);calls+=1
  loss=result[0].loss.val if hasattr(result[0].loss,'val') else float(result[0].loss)
  if not math.isfinite(loss):raise RuntimeError('Nonfinite training loss')
  if calls%8==0:
   accelerator=kwargs['accelerator']
   assert accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped
   optimizer=args[3]
   actual_steps={int(state['step'].item()) for state in optimizer.state.values() if 'step' in state}
   assert actual_steps=={calls//8},actual_steps
   scheduler=kwargs['lr_scheduler']
   assert scheduler.state_dict()['last_epoch']==calls
   write('live_training.json',dict(microsteps=calls,optimizer_updates=calls//8,actual_optimizer_steps=sorted(actual_steps),
         scheduler_microsteps=scheduler.state_dict()['last_epoch'],learning_rate=optimizer.param_groups[0]['lr'],loss=loss))
  return result
 trainer.update_policy=checked_update
 original_save=trainer.save_checkpoint
 def checked_save(**kwargs):
  kwargs['cfg'].save_freq=4000
  original_save(**kwargs)
  checkpoint=Path(kwargs['checkpoint_dir'])/'pretrained_model'
  tensors=load_file(str(checkpoint/'adapter_model.safetensors'))
  assert all(torch.isfinite(v).all() for v in tensors.values())
  if kwargs['step']==8:
   source=load_file(str(RUN/'initial_adapter.safetensors'))
   changed=[k for k,v in tensors.items() if not torch.equal(v,source[k].to(dtype=v.dtype))]
   assert changed,'No adapter parameters changed after first optimizer update'
   state=kwargs['optimizer'].state_dict()
   actual_steps={int(v['step'].item()) for v in state['state'].values() if 'step' in v}
   assert actual_steps=={1},actual_steps
   assert kwargs['scheduler'].state_dict()['last_epoch']==8
   write('first_update_verified.json',dict(checkpoint=str(checkpoint),microstep=8,optimizer_updates=1,actual_optimizer_steps=sorted(actual_steps),
         scheduler_microsteps=8,changed_tensors=len(changed),finite=True))
  write('latest_checkpoint.json',dict(checkpoint=str(checkpoint),microstep=kwargs['step'],optimizer_updates=kwargs['step']//8,finite=True))
 trainer.save_checkpoint=checked_save
 return trainer

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--train',action='store_true');args=parser.parse_args();sys.argv=[sys.argv[0]]
 cfg=make_config();cfg.validate()
 if not args.train:verify(cfg);return
 assert json.loads((RUN/'training_setup_verification.json').read_text())['passed']
 write('resolved_train_config.json',cfg.to_dict())
 trainer=install_hooks(cfg);cfg.save_freq=8;trainer.train(cfg)
 write('completed.json',dict(microsteps=120000,optimizer_updates=15000,output=str(OUTPUT)))
if __name__=='__main__':main()
