"""Approved PI0.5 experiment; steps/save/scheduler use microsteps."""
from pathlib import Path

RUN = Path(__file__).resolve().parent
WORK = RUN.parent
OUTPUT = Path('/home/lp531/smolvla_training/pi05_current_round_20261111')
BASE = WORK / 'weights/pi05_base'
ACCUMULATION = 8


def peft_overrides(cfg):
    from dataclasses import asdict
    return dict(asdict(cfg.peft), lora_dropout=0.)


def make_config(smoke=False, validate=True):
    from lerobot.configs.accelerator import AcceleratorConfig, GradientAccumulationConfig
    from lerobot.configs.default import DatasetConfig, PeftConfig, WandBConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id='lp531/four_objects_current_pi05', root=str(RUN),
                              video_backend='pyav', eval_split=0.),
        policy=PI05Config(pretrained_path=BASE, device='cuda', push_to_hub=False,
            dtype='bfloat16', gradient_checkpointing=True, freeze_vision_encoder=True,
            chunk_size=30, n_action_steps=15, num_inference_steps=10,
            text_tokenizer_name=str(WORK/'weights/tokenizer'), use_relative_actions=False,
            optimizer_lr=2.5e-5, optimizer_weight_decay=.01, optimizer_betas=(.9, .95),
            optimizer_grad_clip_norm=1., scheduler_warmup_steps=20000,
            scheduler_decay_steps=800000, scheduler_decay_lr=2.5e-6,
            input_features={}, output_features={}, compile_model=False),
        peft=PeftConfig(r=32, lora_alpha=32,
            target_modules=(r'.*paligemma_with_expert\.(paligemma\.model\.language_model|gemma_expert\.model)'
                            r'\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)'),
            full_training_modules=['action_in_proj', 'action_out_proj', 'time_mlp_in', 'time_mlp_out']),
        accelerator=AcceleratorConfig(mixed_precision='bf16',
            gradient_accumulation=GradientAccumulationConfig(steps=ACCUMULATION)),
        batch_size=2, num_workers=2, prefetch_factor=2, steps=ACCUMULATION if smoke else 800000,
        save_freq=ACCUMULATION if smoke else 40000, eval_steps=0,
        max_eval_samples=0, env_eval_freq=0, log_freq=8 if smoke else 80,
        output_dir=RUN/'smoke_output' if smoke else OUTPUT,
        job_name='pi05_four_objects_lora32_100k', seed=20261029, wandb=WandBConfig(enable=False))
    cfg.dataset.image_transforms.enable = False
    cfg.dataloader_multiprocessing_context = 'spawn'
    if validate:
        cfg.validate()
    # LeRobot otherwise silently changes unavailable cuda to cpu at construction.
    cfg.policy.device = 'cuda'
    return cfg
