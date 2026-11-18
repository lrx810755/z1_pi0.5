"""ACT baseline: 50k optimizer updates, nominal batch 16, approved source-balanced windows."""
import argparse,copy,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
from prepare import RUN,DATA,write
from sampling import BalancedSampler
OUTPUT=Path('/home/lp531/smolvla_training/act_cup_dynamic_gentle_cached_chunk16_20261020')

def make_config(smoke=False):
    from lerobot.configs.accelerator import AcceleratorConfig,GradientAccumulationConfig
    from lerobot.configs.default import DatasetConfig,WandBConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.act.configuration_act import ACTConfig
    split=json.loads((RUN/'split_manifest.json').read_text())
    cfg=TrainPipelineConfig(
        dataset=DatasetConfig(repo_id='lp531/z1_cup_dynamic_raw_cached_20261020',root=str(DATA),
            episodes=split['train_episodes'],video_backend='pyav',eval_split=len(split["validation_episodes"])/len(split["source_records"])),
        policy=ACTConfig(device='cuda',push_to_hub=False,chunk_size=16,n_action_steps=5,
            optimizer_lr=1e-5,optimizer_lr_backbone=1e-5,optimizer_weight_decay=1e-4),
        accelerator=AcceleratorConfig(mixed_precision='no',gradient_accumulation=GradientAccumulationConfig(steps=1)),
        batch_size=16,num_workers=4,prefetch_factor=2,steps=4 if smoke else 50000,
        save_freq=4 if smoke else 5000,log_freq=1 if smoke else 50,eval_steps=0 if smoke else 5000,
        max_eval_samples=0,env_eval_freq=0,output_dir=RUN/'smoke' if smoke else OUTPUT,
        job_name='act_cup_gentle_cached_chunk16',seed=20261013,wandb=WandBConfig(enable=False))
    cfg.dataset.image_transforms.enable=False
    cfg.persistent_workers=True
    cfg.dataloader_multiprocessing_context="spawn"
    cfg.validate()
    cfg.optimizer.grad_clip_norm=1.0
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
        write('validation_windows.json',dict(episodes=eval_dataset.episodes,relative_indices=selected,windows_per_episode=16))
        sampler=train_dl.sampler
        order=list(iter(sampler));sampler.load_state_dict({'epoch':0,'start_index':123})
        assert list(iter(sampler))==order[123:]
        sampler.load_state_dict({'epoch':0,'start_index':0})
        write('sampler_verified.json',dict(deterministic_resume=True,training_windows=len(sampler),validation_windows=len(selected)))
        return train_dl,eval_dl
    trainer.make_dataloaders=loaders
    original_processors=trainer.make_pre_post_processors
    def processors(*args,**kwargs):
        pre,post=original_processors(*args,**kwargs)
        if not cfg.resume:
            # torchvision crop with a negative top pads to a 640x640 square without losing edge objects.
            crop={'observation.images.wrist':(-140,0,640,640),
                  'observation.images.global':(-80,0,640,640),
                  'observation.images.global_right':(-80,0,640,640)}
            pre.steps.insert(3,ImageCropResizeProcessorStep(crop_params_dict=crop,resize_size=(224,224)))
        return pre,post
    trainer.make_pre_post_processors=processors
    # Cache already contains exact crop/resize output. Keep deployment processor
    # serialized unchanged, but bypass this single step for cached training/eval batches.
    original_preprocess=trainer._preprocess_dataset_batch
    def preprocess(batch,camera_keys,rename_map,preprocessor):
        assert all(batch[k].dtype==torch.float32 and tuple(batch[k].shape[-2:])==(224,224) for k in camera_keys)
        steps=preprocessor.steps
        preprocessor.steps=[s for s in steps if not isinstance(s,ImageCropResizeProcessorStep)]
        try:return original_preprocess(batch,camera_keys,rename_map,preprocessor)
        finally:preprocessor.steps=steps
    trainer._preprocess_dataset_batch=preprocess
    original_update=trainer.update_policy
    calls=0;started=time.monotonic()
    def update(*args,**kwargs):
        nonlocal calls
        model=args[1]
        tensor=next(p for n,p in model.named_parameters() if 'action_head' in n and p.requires_grad)
        before=tensor.detach().clone() if calls==0 else None
        result=original_update(*args,**kwargs);calls+=1
        loss=float(result[0].loss.val)
        if not math.isfinite(loss):raise RuntimeError('Nonfinite loss')
        accelerator=kwargs['accelerator']
        assert accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped
        if calls==1 or calls%20==0:
            actual={int(s['step'].item()) for s in args[3].state.values() if 'step' in s}
            assert actual=={calls},actual
            status=dict(optimizer_updates=calls,target_updates=cfg.steps,loss=loss,
                learning_rate=args[3].param_groups[0]['lr'],elapsed_s=time.monotonic()-started,
                allocated_gib=torch.cuda.max_memory_allocated()/2**30,reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                actual_optimizer_steps=sorted(actual))
            write('smoke_live.json' if smoke else 'live_training.json',status)
        if calls==1:
            assert not torch.equal(before,tensor.detach()),'Action head did not update'
            write('smoke_first_update.json' if smoke else 'first_update_verified.json',dict(actual_optimizer_updates=1,action_head_changed=True,loss=loss))
        return result
    trainer.update_policy=update
    original_save=trainer.save_checkpoint
    def save(**kwargs):
        if not smoke:kwargs['cfg'].save_freq=5000
        original_save(**kwargs)
        checkpoint=Path(kwargs['checkpoint_dir'])/'pretrained_model'
        assert (checkpoint/'model.safetensors').is_file()
        pre=json.loads((checkpoint/'policy_preprocessor.json').read_text())
        assert any(x.get('registry_name')=='image_crop_resize_processor' for x in pre['steps'])
        write('smoke_checkpoint.json' if smoke else 'latest_checkpoint.json',dict(checkpoint=str(checkpoint),optimizer_updates=kwargs['step']))
    trainer.save_checkpoint=save
    return trainer

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');parser.add_argument('--train',action='store_true')
    args=parser.parse_args();sys.argv=[sys.argv[0]]
    cfg=make_config(args.smoke)
    write('smoke_config.json' if args.smoke else 'resolved_train_config.json',cfg.to_dict())
    if not (args.smoke or args.train):return
    trainer=install_hooks(cfg,args.smoke)
    if not args.smoke:cfg.save_freq=20
    trainer.train(cfg)
    write('smoke_completed.json' if args.smoke else 'completed.json',dict(optimizer_updates=cfg.steps,output=str(cfg.output_dir)))
if __name__=='__main__':main()
