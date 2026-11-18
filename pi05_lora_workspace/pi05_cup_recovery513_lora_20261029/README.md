# PI0.5 cup recovery LoRA, 2026-10-29

User authorized the current saved recovery subset and one fresh LoRA run, without ongoing monitoring.

Data: 400 basic +100 original corrections +13 accepted real-policy-push recoveries =513 episodes. Of14 saved recoveries, episode_000005 was rejected because its actual replay tilted before takeover. Original sources and original19-candidate completion gate remain unchanged. Selected data:461 training/52 heldout episodes, split by original spatial cells(1,1),(2,3). No source layout/seed crosses train and validation.

Training: fresh local official pi05_base and optimizer; rank32/alpha32/dropout0. Attention q/k/v/o and FFN gate/up/down in language and action expert; action_in/out_proj and time_mlp_in/out fully trained; vision frozen. BF16, gradient checkpointing, microbatch1/accumulation8/effective8. 30000 optimizer updates=240000 microsteps, warmup1000 updates=8000 microsteps, cosine5e-5→5e-6. AdamW betas(.9,.95), weight_decay.01, clip1. Chunk30, execution5, flow inference10. Seed20261029. No image augmentation.

Inputs: 3 RGB224 views with the same letterbox as ACT; 7D physical joint state; recorded cup task text. Outputs:6 absolute joint commands plus0/1 gripper. Train-only expert quantiles(q01/q99), never old bottle statistics. Base/correction total masses75/25; recovery anchors3x, transitions1.5x, overlap max. Full30-step expert windows only. Recovery anchors receive16.9531% of global sampling probability after normalization. Excluded nonexpert frames:2397.

The old500 verified float32 crops are read-only memory maps; new13 images are decoded on demand. Deployment crop is serialized, bypassed only for already resized training batches. Raw lowdim metadata, accepted image bytes, source identities, and prepared artifacts have hash evidence. All14 new recordings were independently checked using the original physical-acceptance function;13 passed, with all accepted JPEGs fully decoded. Numeric six-joint FK from original URDF matches recorded preclose roll; no ROS/xacro is needed for those literal joint elements.

Run on the workstation with CUDA access:

```bash
bash /home/lp531/pi05_lora_workspace/pi05_cup_recovery513_lora_20261029/launch.sh
```

The script checks CUDA/BF16, refuses CPU fallback or existing output, submits a user systemd service(or nohup if user services unavailable), then returns. The background workflow rechecks data/config, runs an isolated8-microstep smoke, reloads it for2 heldout predictions, and starts formal training from fresh base/adapters. It does not continue from smoke weights. Every8000 microsteps(1000 updates), validate16 fixed windows per heldout episode and save an adapter; retain best validation-loss marker and final. First-update evidence checks actual optimizer state and nonzero LoRA B. Smoke and final reload gates require finite(1,30,7) predictions.

Output: `/home/lp531/smolvla_training/pi05_cup_recovery513_lora_20261029`.
Status: `workflow_status.json`; actual first optimizer update: `first_update_verified.json`; formal log: `training.log`; best marker: `best_validation_checkpoint.json`; completion: `completed.json` plus `final_checkpoint_reload_verified.json`. `stage=training` alone means the trainer command was invoked; the first-update marker proves weights changed.

Launched2026-10-29 16:07 on the host RTX4090 after GPU access became available. CPU verification, actual smoke update and heldout checkpoint reload passed. Formal fresh-base training is active under pi05-cup-recovery513-lora-20261029.service, with first optimizer update verified. See launch_verified.json and first_update_verified.json. No ongoing monitoring is running.
