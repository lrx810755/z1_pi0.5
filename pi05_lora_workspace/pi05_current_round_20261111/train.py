"""Native LeRobot training with immutable data, fresh LoRA and launch evidence."""
import argparse
import json
import logging
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

from config import RUN, BASE, ACCUMULATION, make_config, peft_overrides
from dataset_prepare import write, sha, CAMERAS, CROPS
from dataset import CupDataset
from sampling import BalancedSampler
from guards import strict_base_load, validation_evidence


def datasets(cfg):
    split = json.loads((RUN/'split_manifest.json').read_text())
    return CupDataset(split['train_episodes']), None


def install_hooks(cfg, smoke):
    import lerobot.scripts.lerobot_train as trainer
    from lerobot.processor import ImageCropResizeProcessorStep
    from safetensors.torch import load_file
    prefix = 'smoke_' if smoke else ''
    trainer.make_train_eval_datasets = datasets
    trainer.EpisodeAwareSampler = BalancedSampler
    original_loaders = trainer.make_dataloaders

    def loaders(cfg, dataset, eval_dataset, step, parallel_dims):
        train_dl, _ = original_loaders(cfg, dataset, None, step, parallel_dims)
        if parallel_dims.dp_world_size != 1:
            raise RuntimeError('Strict four-object schedule requires single GPU')
        return train_dl, None
    trainer.make_dataloaders = loaders

    original_processors = trainer.make_pre_post_processors
    def processors(*args, **kwargs):
        pre, post = original_processors(*args, **kwargs)
        if not cfg.resume:
            pre.steps.insert(3, ImageCropResizeProcessorStep(crop_params_dict=CROPS, resize_size=(224, 224)))
        return pre, post
    trainer.make_pre_post_processors = processors
    original_preprocess = trainer._preprocess_dataset_batch
    def preprocess(batch, camera_keys, rename_map, preprocessor):
        if not all(batch[k].dtype == torch.float32 and tuple(batch[k].shape[-2:]) == (224, 224)
                   for k in camera_keys):
            raise ValueError('Dataset must return already letterboxed float32 RGB224')
        steps = preprocessor.steps
        preprocessor.steps = [s for s in steps if not isinstance(s, ImageCropResizeProcessorStep)]
        try:
            return original_preprocess(batch, camera_keys, rename_map, preprocessor)
        finally:
            preprocessor.steps = steps
    trainer._preprocess_dataset_batch = preprocess

    original_policy = trainer.make_policy
    first_tensor_name = None
    def checked_policy(*args, **kwargs):
        nonlocal first_tensor_name
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        policy, base_evidence = strict_base_load(lambda: original_policy(*args, **kwargs), PI05Policy)
        before = {n: p.detach().cpu().clone() for n, p in policy.named_parameters()
                  if any('.'+suffix+'.' in n for suffix in cfg.peft.full_training_modules)}
        if len(before) != 8:
            raise RuntimeError('Expected8 action/time projection tensors')
        policy = policy.wrap_with_peft(peft_cli_overrides=peft_overrides(cfg))
        named = dict(policy.named_parameters())
        b = [(n, p) for n, p in named.items() if '.lora_B.' in n]
        if len(b) != 252 or any(torch.count_nonzero(p) for _, p in b):
            raise RuntimeError('Fresh rank32 adapters must have252 zero LoRA B matrices')
        for name, expected in before.items():
            stem, parameter = name.rsplit('.', 1)
            if not torch.equal(named['base_model.model.'+stem+'.modules_to_save.default.'+parameter].detach().cpu(), expected):
                raise RuntimeError('Projection initialization differs from official base')
        trainable = [(n, p.numel()) for n, p in named.items() if p.requires_grad]
        if any('vision_tower' in n or ('lora_' not in n and 'modules_to_save' not in n) for n, _ in trainable):
            raise RuntimeError('Unexpected trainable backbone parameters')
        if policy.peft_config['default'].lora_dropout != 0.:
            raise RuntimeError('LoRA dropout differs from approved config')
        first_tensor_name = b[0][0]
        write(prefix+'initialization_verified.json', dict(base=str(BASE), fresh_adapter=True,
              lora_b_count=len(b), projections_match_base=True, vision_frozen=True,
              trainable_parameters=sum(n for _, n in trainable), lora_dropout=0., **base_evidence))
        return policy
    trainer.make_policy = checked_policy

    original_update = trainer.update_policy
    calls = 0
    started = time.monotonic()
    def update(*args, **kwargs):
        nonlocal calls
        result = original_update(*args, **kwargs)
        calls += 1
        loss = float(result[0].loss.val)
        if not math.isfinite(loss):
            raise RuntimeError('Nonfinite training loss')
        if calls % ACCUMULATION == 0:
            accelerator = kwargs['accelerator']
            if not accelerator.sync_gradients or accelerator.optimizer_step_was_skipped:
                raise RuntimeError('Optimizer step skipped or accumulation misaligned')
            actual = {int(s['step'].item()) for s in args[3].state.values() if 'step' in s}
            if actual != {calls//ACCUMULATION}:
                raise RuntimeError(f'Optimizer update count mismatch: {actual}')
            if kwargs['lr_scheduler'].state_dict()['last_epoch'] != calls:
                raise RuntimeError('Scheduler must advance in microsteps')
            evidence = dict(microsteps=calls, optimizer_updates=calls//ACCUMULATION, target_updates=cfg.steps//ACCUMULATION,
                loss=loss, learning_rate=args[3].param_groups[0]['lr'], elapsed_s=time.monotonic()-started,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
            if calls == ACCUMULATION:
                tensor = dict(accelerator.unwrap_model(args[1]).named_parameters())[first_tensor_name]
                if not torch.count_nonzero(tensor):
                    raise RuntimeError('LoRA B has not changed after first optimizer update')
                write(prefix+'first_update_verified.json', dict(evidence, adapter_changed=True, actual_optimizer_steps=[1]))
            if calls == ACCUMULATION or calls % 80 == 0:
                write(prefix+'live_training.json', evidence)
        return result
    trainer.update_policy = update

    class ValidationEvidence(logging.Handler):
        def emit(self, record):
            item = validation_evidence(record.getMessage())
            if item is not None:
                write(prefix+'validation_latest.json', item)
                with (RUN/(prefix+'validation_history.jsonl')).open('a') as stream:
                    stream.write(json.dumps(item)+'\n')
    original_logging = trainer.init_logging
    def init_logging(*args, **kwargs):
        original_logging(*args, **kwargs)
        logging.getLogger().addHandler(ValidationEvidence())
    trainer.init_logging = init_logging

    original_save = trainer.save_checkpoint
    def save(**kwargs):
        if not smoke:
            kwargs['cfg'].save_freq = 40000
        original_save(**kwargs)
        checkpoint = Path(kwargs['checkpoint_dir'])/'pretrained_model'
        tensors = load_file(str(checkpoint/'adapter_model.safetensors'))
        if not tensors or not all(torch.isfinite(x).all() for x in tensors.values()):
            raise RuntimeError('Invalid adapter checkpoint')
        pre = json.loads((checkpoint/'policy_preprocessor.json').read_text())
        if not any(s['registry_name'] == 'image_crop_resize_processor' for s in pre['steps']):
            raise RuntimeError('Deployment image crop not serialized')
        item = dict(checkpoint=str(checkpoint), microsteps=kwargs['step'], optimizer_updates=kwargs['step']//ACCUMULATION)
        write(prefix+'latest_checkpoint.json', item)
        if not smoke and (RUN/'validation_latest.json').exists():
            val = json.loads((RUN/'validation_latest.json').read_text())
            best = RUN/'best_validation_checkpoint.json'
            if val['microsteps'] == kwargs['step'] and (not best.exists() or val['heldout_loss'] < json.loads(best.read_text())['heldout_loss']):
                write('best_validation_checkpoint.json', dict(item, **{'heldout_loss': val['heldout_loss']}))
    trainer.save_checkpoint = save
    return trainer


def main():
    parser = argparse.ArgumentParser()
    flags = parser.add_mutually_exclusive_group(required=True)
    flags.add_argument('--smoke', action='store_true')
    flags.add_argument('--train', action='store_true')
    args = parser.parse_args()
    sys.argv = [sys.argv[0]]
    from pipeline import require_ready_for_training
    require_ready_for_training()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: refusing silent CPU training')
    cfg = make_config(smoke=args.smoke)
    if Path(cfg.output_dir).exists():
        raise RuntimeError(f'Output already exists, refusing overwrite: {cfg.output_dir}')
    from verify import verify_prepared
    verify_prepared()
    if args.train:
        evidence=json.loads((RUN/'checkpoint_reload_verified.json').read_text())
        if not evidence.get('passed') or evidence.get('preparation_sha256')!=sha(RUN/'preparation_complete.json'):
            raise RuntimeError('Smoke checkpoint reload gate not passed')

    write('smoke_config.json' if args.smoke else 'resolved_train_config.json', cfg.to_dict())
    trainer = install_hooks(cfg, args.smoke)
    trainer.train(cfg)
    write('smoke_completed.json' if args.smoke else 'completed.json',
          dict(microsteps=cfg.steps, optimizer_updates=cfg.steps//ACCUMULATION, output=str(cfg.output_dir)))


if __name__ == '__main__':
    main()
