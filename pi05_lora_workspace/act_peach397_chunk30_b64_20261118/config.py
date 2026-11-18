"""The approved peach ACT recipe; one step is one actual AdamW update."""
from common import RUN, OUTPUT, SEED, read


def make_config(smoke=False, validate=True):
    from lerobot.configs.accelerator import AcceleratorConfig, GradientAccumulationConfig
    from lerobot.configs.default import DatasetConfig, WandBConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.optim.optimizers import AdamWConfig
    from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
    from lerobot.policies.act.configuration_act import ACTConfig
    steps=4 if smoke else 60000
    cfg=TrainPipelineConfig(
        dataset=DatasetConfig(repo_id='lp531/peach397_act_20261118',root=str(RUN),episodes=list(range(397)),
                              eval_split=0.,video_backend='pyav',use_imagenet_stats=True),
        policy=ACTConfig(device='cuda',push_to_hub=False,n_obs_steps=1,chunk_size=30,n_action_steps=1,
                         temporal_ensemble_coeff=.01,vision_backbone='resnet18',
                         pretrained_backbone_weights='ResNet18_Weights.IMAGENET1K_V1',
                         dim_model=512,n_heads=8,dim_feedforward=3200,n_encoder_layers=4,n_decoder_layers=1,
                         use_vae=True,latent_dim=32,n_vae_encoder_layers=4,dropout=.1,kl_weight=10.,
                         optimizer_lr=1e-5,optimizer_lr_backbone=1e-6,optimizer_weight_decay=1e-4),
        optimizer=AdamWConfig(lr=1e-5,weight_decay=1e-4,betas=(.9,.999),eps=1e-8,grad_clip_norm=1.),
        scheduler=CosineDecayWithWarmupSchedulerConfig(num_warmup_steps=1 if smoke else 1000,
                  num_decay_steps=steps,peak_lr=1e-5,decay_lr=1e-6),use_policy_training_preset=False,
        accelerator=AcceleratorConfig(mixed_precision='no',gradient_accumulation=GradientAccumulationConfig(steps=1)),
        batch_size=64,num_workers=4,prefetch_factor=2,persistent_workers=True,
        dataloader_multiprocessing_context='spawn',steps=steps,save_freq=4 if smoke else 5000,
        log_freq=1 if smoke else 100,eval_steps=0,env_eval_freq=0,max_eval_samples=0,
        output_dir=RUN/'smoke' if smoke else OUTPUT,job_name='act_peach397_chunk30_b64_60k',seed=SEED,
        wandb=WandBConfig(enable=False))
    cfg.dataset.image_transforms.enable=False
    if validate:
        cfg.validate()
    cfg.policy.device='cuda'
    return cfg
