"""Isolated Diffusion Policy run using the frozen cup ACT cache and sampling distribution."""
import argparse,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
from prepare import RUN,DATA,write
from sampling import BalancedSampler
OUTPUT=Path('/home/lp531/smolvla_training/diffusion_cup_gentle_cached_h16_20261020')
def make_config(smoke=False):
 from lerobot.configs.accelerator import AcceleratorConfig,GradientAccumulationConfig
 from lerobot.configs.default import DatasetConfig,WandBConfig,EMAConfig
 from lerobot.configs.train import TrainPipelineConfig
 from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
 split=json.loads((RUN/'split_manifest.json').read_text())
 cfg=TrainPipelineConfig(dataset=DatasetConfig(repo_id='lp531/z1_cup_diffusion_cached_20261020',root=str(DATA),episodes=split['train_episodes'],video_backend='pyav',eval_split=25/249),policy=DiffusionConfig(device='cuda',push_to_hub=False,n_obs_steps=2,horizon=16,n_action_steps=15,drop_n_last_frames=0,down_dims=(512,1024,2048),resize_shape=None,crop_shape=None,crop_ratio=1.0,crop_is_random=False,num_train_timesteps=100,num_inference_steps=100,noise_scheduler_type='DDPM',do_mask_loss_for_padding=True,optimizer_lr=1e-4,optimizer_weight_decay=1e-6),accelerator=AcceleratorConfig(mixed_precision='no',gradient_accumulation=GradientAccumulationConfig(steps=1)),ema=EMAConfig(enable=True),batch_size=16,num_workers=4,prefetch_factor=2,steps=20 if smoke else 50000,save_freq=20 if smoke else 5000,log_freq=1 if smoke else 50,eval_steps=0 if smoke else 5000,max_eval_samples=0,env_eval_freq=0,output_dir=RUN/'smoke' if smoke else OUTPUT,job_name='diffusion_cup_gentle_cached_h16',seed=20261013,wandb=WandBConfig(enable=False))
 cfg.dataset.image_transforms.enable=False;cfg.persistent_workers=True;cfg.dataloader_multiprocessing_context='spawn';cfg.validate();cfg.optimizer.grad_clip_norm=1.0
 # Smoke warmup must match the formal scheduler for a useful speed/stability test.
 cfg.scheduler.num_decay_steps=50000
 return cfg

def datasets(cfg):
 from cache_data import CupDataset
 split=json.loads((RUN/'split_manifest.json').read_text())
 return CupDataset(split['train_episodes']),CupDataset(split['validation_episodes'])

def install_hooks(cfg,smoke):
 import lerobot.scripts.lerobot_train as trainer
 from lerobot.processor import ImageCropResizeProcessorStep
 trainer.make_train_eval_datasets=datasets
 original_loaders=trainer.make_dataloaders
 def loaders(cfg,dataset,eval_dataset,step,parallel_dims):
  trainer.EpisodeAwareSampler=BalancedSampler
  train_dl,_=original_loaders(cfg,dataset,None,step,parallel_dims)
  with np.load(RUN/'sampling_arrays.npz') as a:
   selected=[]
   for ep in eval_dataset.episodes:
    ids=np.flatnonzero((a['episode']==ep)&a['eligible_validation'])
    selected.extend(eval_dataset.absolute_to_relative_idx[int(i)] for i in ids[np.linspace(0,len(ids)-1,16,dtype=int)])
  eval_dl=torch.utils.data.DataLoader(torch.utils.data.Subset(eval_dataset,selected),batch_size=16,num_workers=0)
  write('validation_windows.json',dict(episodes=eval_dataset.episodes,relative_indices=selected))
  assert len(train_dl.sampler)==44723
  return train_dl,eval_dl
 trainer.make_dataloaders=loaders
 orig_processors=trainer.make_pre_post_processors
 def processors(*args,**kwargs):
  pre,post=orig_processors(*args,**kwargs)
  if not cfg.resume:
   crop={'observation.images.wrist':(-140,0,640,640),'observation.images.global':(-80,0,640,640),'observation.images.global_right':(-80,0,640,640)}
   pre.steps.insert(3,ImageCropResizeProcessorStep(crop_params_dict=crop,resize_size=(224,224)))
  return pre,post
 trainer.make_pre_post_processors=processors
 original_preprocess=trainer._preprocess_dataset_batch
 def preprocess(batch,camera_keys,rename_map,preprocessor):
  for cam in camera_keys:assert batch[cam].shape[1:]==(2,3,224,224) and batch[cam].dtype==torch.float32
  steps=preprocessor.steps;preprocessor.steps=[s for s in steps if not isinstance(s,ImageCropResizeProcessorStep)]
  try:return original_preprocess(batch,camera_keys,rename_map,preprocessor)
  finally:preprocessor.steps=steps
 trainer._preprocess_dataset_batch=preprocess
 original_update=trainer.update_policy;calls=0;started=time.monotonic();times=[]
 def update(*args,**kwargs):
  nonlocal calls
  tensor=next(p for n,p in args[1].named_parameters() if 'unet.final_conv' in n and p.requires_grad)
  before=tensor.detach().clone() if calls==1 else None
  torch.cuda.synchronize();tick=time.monotonic();result=original_update(*args,**kwargs);torch.cuda.synchronize();times.append(time.monotonic()-tick);calls+=1
  loss=float(result[0].loss.val);assert math.isfinite(loss)
  accelerator=kwargs['accelerator'];assert accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped
  if calls==1 or calls%20==0:
   actual={int(s['step'].item()) for s in args[3].state.values() if 'step' in s};assert actual=={calls}
   status=dict(optimizer_updates=calls,target_updates=cfg.steps,loss=loss,learning_rate=args[3].param_groups[0]['lr'],elapsed_s=time.monotonic()-started,median_update_s=float(np.median(times[-100:])),allocated_gib=torch.cuda.max_memory_allocated()/2**30,reserved_gib=torch.cuda.max_memory_reserved()/2**30,actual_optimizer_steps=sorted(actual))
   write('smoke_live.json' if smoke else 'live_training.json',status)
  if calls==2:
   assert not torch.equal(before,tensor.detach());write('smoke_first_update.json' if smoke else 'first_update_verified.json',dict(action_unet_changed=True,actual_optimizer_updates=2,loss=loss,first_step_lr_zero_due_to_warmup=True))
  return result
 trainer.update_policy=update
 original_save=trainer.save_checkpoint
 def save(**kwargs):
  if not smoke:kwargs['cfg'].save_freq=5000
  original_save(**kwargs)
  ck=Path(kwargs['checkpoint_dir'])/'pretrained_model';assert (ck/'model.safetensors').is_file()
  write('smoke_checkpoint.json' if smoke else 'latest_checkpoint.json',dict(checkpoint=str(ck),optimizer_updates=kwargs['step']))
 trainer.save_checkpoint=save
 return trainer

def main():
 p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');args=p.parse_args();sys.argv=[sys.argv[0]];torch.set_num_threads(4)
 cfg=make_config(args.smoke);write('smoke_config.json' if args.smoke else 'resolved_train_config.json',cfg.to_dict());trainer=install_hooks(cfg,args.smoke)
 if not args.smoke:cfg.save_freq=20
 trainer.train(cfg)
 write('smoke_completed.json' if args.smoke else 'completed.json',dict(optimizer_updates=cfg.steps,output=str(cfg.output_dir),ema_model=str(Path(cfg.output_dir)/'checkpoints'/f'{cfg.steps:06d}'/'pretrained_model_ema')))
if __name__=='__main__':main()
