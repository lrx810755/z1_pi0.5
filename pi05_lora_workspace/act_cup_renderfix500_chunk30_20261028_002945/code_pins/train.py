"""ACT baseline: 90k optimizer updates, nominal batch32, approved source-balanced windows."""
import argparse,copy,json,logging,math,re,sys,time
from pathlib import Path
import numpy as np
import torch
from prepare import RUN,DATA,write,collection_gate,assert_pins,sha
from sampling import BalancedSampler
OUTPUT=RUN/'output'

def make_config(smoke=False):
    from lerobot.configs.accelerator import AcceleratorConfig,GradientAccumulationConfig
    from lerobot.configs.default import DatasetConfig,WandBConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.act.configuration_act import ACTConfig
    split=json.loads((RUN/'split_manifest.json').read_text())
    cfg=TrainPipelineConfig(
        dataset=DatasetConfig(repo_id='lp531/z1_cup_renderfix500_raw_cached_20261027',root=str(DATA),
            episodes=split['train_episodes'],video_backend='pyav',eval_split=len(split["validation_episodes"])/len(split["source_records"])),
        policy=ACTConfig(device='cuda',push_to_hub=False,chunk_size=30,n_action_steps=5,pretrained_backbone_weights='ResNet18_Weights.IMAGENET1K_V1',
            optimizer_lr=1e-5,optimizer_lr_backbone=1e-5,optimizer_weight_decay=1e-4),
        accelerator=AcceleratorConfig(mixed_precision='no',gradient_accumulation=GradientAccumulationConfig(steps=1)),
        batch_size=16 if smoke else 32,num_workers=2,prefetch_factor=2,steps=4 if smoke else 90000,
        save_freq=4 if smoke else 5000,log_freq=1 if smoke else 50,eval_steps=0 if smoke else 5000,
        max_eval_samples=0,env_eval_freq=0,output_dir=RUN/'smoke' if smoke else OUTPUT,
        job_name='act_cup_renderfix500_cached_chunk30',seed=20261027,wandb=WandBConfig(enable=False))
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
    class ValidationEvidence(logging.Handler):
        def emit(self,record):
            match=re.search(r'step (\d+): eval_loss=([0-9.eE+\-]+)',record.getMessage())
            if match:
                item=dict(optimizer_updates=int(match.group(1)),heldout_loss=float(match.group(2)))
                if not math.isfinite(item['heldout_loss']):raise RuntimeError('Nonfinite validation loss')
                write('validation_latest.json',item)
                with (RUN/'validation_history.jsonl').open('a') as f:f.write(json.dumps(item)+'\n')
    import lerobot.scripts.lerobot_train as trainer
    original_logging=trainer.init_logging
    def init_logging(*args,**kwargs):
        original_logging(*args,**kwargs)
        logging.getLogger().addHandler(ValidationEvidence())
    trainer.init_logging=init_logging
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
                if not len(ids):raise ValueError(f'No approved validation windows in episode{ep}')
                selected.extend(eval_dataset.absolute_to_relative_idx[int(i)] for i in ids[np.unique(np.linspace(0,len(ids)-1,min(32,len(ids)),dtype=int))])
        eval_dl=torch.utils.data.DataLoader(torch.utils.data.Subset(eval_dataset,selected),batch_size=cfg.batch_size,num_workers=0)
        write('validation_windows.json',dict(episodes=eval_dataset.episodes,relative_indices=selected,maximum_windows_per_episode=32))
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
    calls=0;peak_allocated=0.;peak_reserved=0.;started=time.monotonic()
    def update(*args,**kwargs):
        nonlocal calls,peak_allocated,peak_reserved
        model=args[1]
        tensor=next(p for n,p in model.named_parameters() if 'action_head' in n and p.requires_grad)
        before=tensor.detach().clone() if calls==0 else None
        result=original_update(*args,**kwargs);calls+=1
        loss=float(result[0].loss.val)
        if not math.isfinite(loss):raise RuntimeError('Nonfinite loss')
        accelerator=kwargs['accelerator']
        assert accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped
        peak_allocated=max(peak_allocated,torch.cuda.max_memory_allocated()/2**30)
        peak_reserved=max(peak_reserved,torch.cuda.max_memory_reserved()/2**30)
        if calls==1 or calls%20==0 or (smoke and calls==cfg.steps):
            actual={int(s['step'].item()) for s in args[3].state.values() if 'step' in s}
            assert actual=={calls},actual
            status=dict(optimizer_updates=calls,target_updates=cfg.steps,loss=loss,
                learning_rate=args[3].param_groups[0]['lr'],elapsed_s=time.monotonic()-started,
                allocated_gib=peak_allocated,reserved_gib=peak_reserved,
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
        evidence=dict(checkpoint=str(checkpoint),optimizer_updates=kwargs['step'])
        write('smoke_checkpoint.json' if smoke else 'latest_checkpoint.json',evidence)
        if not smoke and (RUN/'validation_latest.json').exists():
            val=json.loads((RUN/'validation_latest.json').read_text())
            if val['optimizer_updates']==kwargs['step']:
                evidence.update(val);best=RUN/'best_validation_checkpoint.json'
                if not best.exists() or evidence['heldout_loss']<json.loads(best.read_text())['heldout_loss']:write('best_validation_checkpoint.json',evidence)
    trainer.save_checkpoint=save
    return trainer

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');parser.add_argument('--train',action='store_true');parser.add_argument('--batch-size',type=int,choices=[16,32])
    args=parser.parse_args();sys.argv=[sys.argv[0]]
    collection_gate()
    split=json.loads((RUN/'split_manifest.json').read_text());assert_pins(split['source_records'])
    for name in ['cache_verified.json']:
        evidence=json.loads((RUN/name).read_text())
        if not evidence.get('passed') or not evidence.get('digests_verified') or evidence['manifest_sha256']!=sha(RUN/'split_manifest.json'):raise RuntimeError(f'Missing current successful{name}')
    cfg=make_config(args.smoke)
    if not args.smoke and (RUN/'selected_hyperparameters.json').exists():
        selected=json.loads((RUN/'selected_hyperparameters.json').read_text());cfg.batch_size=selected['batch_size'];cfg.steps=selected['steps']
    if args.batch_size:cfg.batch_size=args.batch_size
    if args.smoke:cfg.output_dir=RUN/f'smoke_b{cfg.batch_size}'
    if cfg.resume or cfg.policy.pretrained_path is not None:raise RuntimeError('This round must initialize fresh ACT with ImageNet backbone')
    if (args.smoke or args.train) and Path(cfg.output_dir).exists():raise RuntimeError(f'Output already exists; preserve artifacts and inspect before restart:{cfg.output_dir}')
    write('smoke_config.json' if args.smoke else 'resolved_train_config.json',cfg.to_dict())
    if not (args.smoke or args.train):return
    trainer=install_hooks(cfg,args.smoke)
    if not args.smoke:
        if not json.loads((RUN/'checkpoint_reload_verified.json').read_text()).get('normalization_roundtrip'):raise RuntimeError('Smoke checkpoint not verified')
        cfg.save_freq=20
    trainer.train(cfg)
    write('smoke_completed.json' if args.smoke else 'completed.json',dict(optimizer_updates=cfg.steps,output=str(cfg.output_dir)))
if __name__=='__main__':main()
