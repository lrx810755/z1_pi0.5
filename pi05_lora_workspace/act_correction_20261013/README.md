# ACT bottle baseline, 2026-10-13

Independent ACT run on the 211-episode RGB/state bottle dataset. The source dataset and previous models are not modified.

- Dataset: `/home/lp531/smolvla_data/z1_bottle_with_fragments_20261012`
- Split: 190 train / 21 validation; validation includes 16 original, 3 old correction and 2 new fragment episodes. Duplicate raw-recording hashes never cross the split.
- Source-balanced sampling: 70% original / 30% correction. Uniform eligible anchors within each pool, without fine-stage or loss weights.
- Complete 16-action windows remain inside each approved interval. Whole-episode data remains available for image/state context.
- Three cameras are padded to square without cropping away image content, then resized to 224x224. The built-in processor is serialized into every checkpoint.
- ACT: ResNet18 ImageNet initialization, all parameters trainable; hidden 512, heads 8, encoder 4, decoder 1; VAE latent 32, KL weight 10, dropout 0.1.
- AdamW: learning rate 1e-5 for both backbone and other parameters, weight decay 1e-4, gradient clipping 1.0; no LR scheduler.
- Batch 16, accumulation 1; 50,000 loop steps = 50,000 optimizer updates. The final short batch of an epoch may contain fewer than 16 samples.
- Float32; single-process video loading avoids a PyAV worker-shutdown abort observed in both fork and spawn smoke runs.
- Training-only vector statistics; ImageNet RGB statistics. Binary gripper action remains 0=open / 1=close.
- Checkpoints after update 20 and every 5,000 updates; held-out loss every 5,000 updates on 16 fixed windows per validation episode. No simulation rollout is launched by training.
- Deployment configuration: predicts 16 actions, executes 5 before querying again; no temporal ensemble. Deployment still needs to be wired into the existing ROS evaluation launcher after training.

Start command (already launched, do not execute twice):

```sh
bash /home/lp531/pi05_lora_workspace/act_correction_20261013/run_training.sh
```

Output: `/home/lp531/smolvla_training/act_correction_chunk16_20261013`
Log: `training.log`
Progress: `live_training.json`
Checkpoints: `latest_checkpoint.json`
Completion: `completed.json` (created only after all 50,000 updates return successfully)

Verification evidence: `sampling_audit.json`, `sampler_verified.json`, `smoke_first_update.json`, `checkpoint_reload_verified.json`, and `first_update_verified.json` for the formal run. A completed training run is not evidence of closed-loop task success.
