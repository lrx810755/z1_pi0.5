"""Native LeRobot ACT training with peach data, balanced sampler and evidence."""
import argparse
import json
import math
import sys
import time
from pathlib import Path
import torch
from common import RUN, CROPS, CAMERAS, read, write, sha, verify_prepared, require_training_evidence
from config import make_config
from dataset import PeachDataset
from sampling import BalancedSampler


def datasets(cfg):return PeachDataset(),None


def install_hooks(cfg,smoke):
    import lerobot.scripts.lerobot_train as trainer
    from lerobot.processor import ImageCropResizeProcessorStep
    prefix='smoke_' if smoke else ''
    trainer.make_train_eval_datasets=datasets
    trainer.EpisodeAwareSampler=BalancedSampler
    original_loaders=trainer.make_dataloaders
    def loaders(cfg,dataset,eval_dataset,step,parallel_dims):
        if cfg.batch_size!=64 or parallel_dims.dp_world_size!=1:
            raise ValueError('This schedule requires single GPU and batch64')
        return original_loaders(cfg,dataset,None,step,parallel_dims)
    trainer.make_dataloaders=loaders
    def optimizer_and_scheduler(cfg,policy):
        # Native factory ignores policy.get_optim_params when presets are disabled.
        optimizer=cfg.optimizer.build(policy.get_optim_params())
        return optimizer,cfg.scheduler.build(optimizer,cfg.steps)
    trainer.make_optimizer_and_scheduler=optimizer_and_scheduler
    original_processors=trainer.make_pre_post_processors
    def processors(*args,**kwargs):
        pre,post=original_processors(*args,**kwargs)
        if not cfg.resume:
            pre.steps.insert(3,ImageCropResizeProcessorStep(crop_params_dict=CROPS,resize_size=(224,224)))
        return pre,post
    trainer.make_pre_post_processors=processors
    original_preprocess=trainer._preprocess_dataset_batch
    def preprocess(batch,camera_keys,rename_map,preprocessor):
        if not all(batch[k].dtype==torch.float32 and tuple(batch[k].shape[-2:])==(224,224) for k in camera_keys):
            raise ValueError('Training must consume lossless predecoded float32RGB224')
        original_steps=preprocessor.steps
        preprocessor.steps=[s for s in original_steps if not isinstance(s,ImageCropResizeProcessorStep)]
        try:return original_preprocess(batch,camera_keys,rename_map,preprocessor)
        finally:preprocessor.steps=original_steps
    trainer._preprocess_dataset_batch=preprocess
    original_policy=trainer.make_policy
    def policy(*args,**kwargs):
        result=original_policy(*args,**kwargs)
        checkpoint=Path(torch.hub.get_dir())/'checkpoints/resnet18-f37072fd.pth'
        expected=torch.load(checkpoint,map_location='cpu',weights_only=True)['conv1.weight']
        if not torch.equal(result.model.backbone.conv1.weight.detach().cpu(),expected):
            raise ValueError('ImageNet convolution initialization differs')
        if cfg.resume or cfg.policy.pretrained_path is not None:
            raise ValueError('Fresh ACT initialization required')
        write(prefix+'initialization_verified.json',dict(passed=True,fresh_act=True,imagenet_conv1_exact=True,
              pretrained_resnet_sha256=sha(checkpoint),trainable_parameters=sum(p.numel() for p in result.parameters() if p.requires_grad),
              prediction_steps=30,temporal_ensemble_coeff=.01,normalization='per-peach MEAN_STD',prepared_sha256=sha(RUN/'prepared.json')))
        return result
    trainer.make_policy=policy
    original_update=trainer.update_policy
    calls=0;started=time.monotonic()
    def update(*args,**kwargs):
        nonlocal calls
        model=kwargs['accelerator'].unwrap_model(args[1])
        before=model.model.action_head.bias.detach().clone() if calls==0 else None
        result=original_update(*args,**kwargs);calls+=1
        accelerator=kwargs['accelerator']
        if not accelerator.sync_gradients or accelerator.optimizer_step_was_skipped:
            raise ValueError('Actual optimizer update skipped')
        actual={int(s['step'].item()) for s in args[3].state.values() if 'step' in s}
        if actual!={calls}:raise ValueError('Optimizer step count differs: '+str(actual))
        rates=[g['lr'] for g in args[3].param_groups]
        if len(rates)!=2 or not math.isclose(rates[1]/rates[0],.1,rel_tol=1e-6):
            raise ValueError('Backbone learning-rate group is not0.1x')
        if kwargs['lr_scheduler'].state_dict()['last_epoch']!=calls:
            raise ValueError('Scheduler update count mismatch')
        loss=float(result[0].loss.val)
        if not math.isfinite(loss):raise ValueError('Nonfinite loss')
        evidence=dict(optimizer_updates=calls,target_updates=cfg.steps,loss=loss,learning_rate=rates[0],
                      backbone_learning_rate=rates[1],elapsed_s=time.monotonic()-started,
                      allocated_gib=torch.cuda.max_memory_allocated()/2**30,reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                      actual_optimizer_steps=sorted(actual),batch_size=64,gradient_accumulation=1)
        if calls==1:
            changed=not torch.equal(before,model.model.action_head.bias.detach())
            if not changed:raise ValueError('Action head did not change after actual first update')
            write(prefix+'first_update_verified.json',dict(evidence,passed=True,action_head_changed=True))
        if calls==1 or calls%100==0 or (smoke and calls==cfg.steps):
            write(prefix+'live_training.json',evidence)
        return result
    trainer.update_policy=update
    original_save=trainer.save_checkpoint
    def save(**kwargs):
        if not smoke:kwargs['cfg'].save_freq=5000
        original_save(**kwargs)
        path=Path(kwargs['checkpoint_dir'])/'pretrained_model'
        if not (path/'model.safetensors').is_file():raise ValueError('ACT weights not saved')
        pre=read(path/'policy_preprocessor.json')
        if not any(s['registry_name']=='image_crop_resize_processor' for s in pre['steps']):
            raise ValueError('Deployment crop not serialized')
        write(prefix+'latest_checkpoint.json',dict(checkpoint=str(path),optimizer_updates=kwargs['step']))
    trainer.save_checkpoint=save
    return trainer


def main():
    parser=argparse.ArgumentParser();flags=parser.add_mutually_exclusive_group(required=True)
    flags.add_argument('--smoke',action='store_true');flags.add_argument('--train',action='store_true');args=parser.parse_args()
    sys.argv=[sys.argv[0]]
    torch.set_num_threads(8)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    verify_prepared()
    require_training_evidence(formal=args.train)
    cfg=make_config(args.smoke)
    if Path(cfg.output_dir).exists():raise ValueError('Output already exists; refusing overwrite')
    write('smoke_config.json' if args.smoke else 'resolved_train_config.json',cfg.to_dict())
    trainer=install_hooks(cfg,args.smoke)
    if args.train:cfg.save_freq=100
    trainer.train(cfg)
    write('smoke_completed.json' if args.smoke else 'completed.json',dict(optimizer_updates=cfg.steps,output=str(cfg.output_dir)))


if __name__=='__main__':main()
