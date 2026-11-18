"""Train native LeRobot PI05 LoRA, with a small memory preflight first."""
import argparse
import json
from pathlib import Path

from lerobot.configs.accelerator import AcceleratorConfig, GradientAccumulationConfig
from lerobot.configs.default import DatasetConfig, PeftConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.scripts.lerobot_train import train

WORK = Path(__file__).resolve().parent
OUTPUT = Path('/home/lp531/smolvla_training/pi05_bottle_lora_20261001')


def make_config(microbatch=2, smoke=False):
    # LeRobot counts micro-batches, not optimizer updates, in cfg.steps.
    accumulation = 8 // microbatch
    total_steps = 30000 * accumulation
    policy = PI05Config(
        pretrained_path=WORK / 'weights/pi05_base', device='cuda', push_to_hub=False,
        dtype='bfloat16', gradient_checkpointing=True, freeze_vision_encoder=True,
        chunk_size=50, n_action_steps=5, text_tokenizer_name=str(WORK / 'weights/tokenizer'),
        use_relative_actions=False, optimizer_lr=1e-4, optimizer_weight_decay=.01,
        scheduler_warmup_steps=1000 * accumulation,
        scheduler_decay_steps=total_steps, scheduler_decay_lr=1e-5,
        input_features={}, output_features={},
    )
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id='lp531/z1_bottle_pi05_lora',
                              root='/home/lp531/smolvla_data/z1_tabletop_bottle_pi05_lora_v1',
                              video_backend='pyav', eval_split=.1),
        policy=policy,
        peft=PeftConfig(r=32, lora_alpha=32,
                        target_modules=r'.*paligemma_with_expert\.(paligemma\.model\.language_model|gemma_expert\.model)\.layers\.\d+\.self_attn\.(q|v)_proj',
                        full_training_modules=['action_in_proj', 'action_out_proj', 'time_mlp_in', 'time_mlp_out']),
        accelerator=AcceleratorConfig(mixed_precision='bf16',
                                      gradient_accumulation=GradientAccumulationConfig(steps=accumulation)),
        batch_size=microbatch, num_workers=4, prefetch_factor=2,
        steps=total_steps, save_freq=3000 * accumulation,
        eval_steps=3000 * accumulation, max_eval_samples=0,
        env_eval_freq=0, log_freq=10 * accumulation,
        output_dir=OUTPUT, job_name='pi05_bottle_lora', seed=20261001,
        wandb=WandBConfig(enable=False),
    )
    if smoke:
        cfg.steps = accumulation
        cfg.eval_steps = 0
        cfg.save_checkpoint = False
        cfg.num_workers = 0
        cfg.log_freq = 1
        cfg.output_dir = WORK / f'smoke_batch{microbatch}'
    return cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--microbatch', type=int, choices=[1, 2], default=2)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    cfg = make_config(args.microbatch, args.smoke)
    # Clear custom script flags so LeRobot's CLI config resolver only sees our object.
    import sys
    sys.argv = [sys.argv[0]]
    cfg.validate()
    if not args.smoke:
        (WORK / 'resolved_train_config.json').write_text(json.dumps(cfg.to_dict(), indent=2, default=str) + '\n')
    train(cfg)
