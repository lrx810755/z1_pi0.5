---
library_name: lerobot
pipeline_tag: robotics
tags:
- robotics
- vla
- lerobot
- pi05
- flow-matching
- libero-plus
license: gemma
---

# pi0.5 LIBERO-plus LoRA

LoRA fine-tuned weights of [pi0.5](https://www.physicalintelligence.company/research/pi05) (PaliGemma-2B + Action Expert 300M) on the [LIBERO-plus](https://huggingface.co/datasets/lerobot/libero_plus) dataset, targeting the Action Expert and projection layers.

---

## 1. Model Configuration

| Item | Value |
|---|---|
| Base Model | [`lerobot/pi05_libero_base`](https://huggingface.co/lerobot/pi05_libero_base) (commit [`a217bfd`](https://huggingface.co/lerobot/pi05_libero_base/commit/a217bfd3b14673cf2ce597e69997ab21866438dd)) |
| This Repo | [`nosuke113/pi0.5-libero-plus-lora`](https://huggingface.co/nosuke113/pi0.5-libero-plus-lora) (commit [`db2ccc2`](https://huggingface.co/nosuke113/pi0.5-libero-plus-lora/commit/db2ccc2a42e1f6ccab45f3fe3601a8d07a3ae2e2)) |

---

## 2. Training

### Dataset

The model was trained on [`nosuke113/libero-plus-deduplicated`](https://huggingface.co/datasets/nosuke113/libero-plus-deduplicated), which is a deduplicated version of the original [`lerobot/libero_plus`](https://huggingface.co/datasets/lerobot/libero_plus) dataset. 

### Data Split Methodology

The validation set was carefully constructed to prevent data leakage. Since the original dataset contains identical physical trajectories artificially augmented with different visual appearances, a simple random split would result in the model evaluating on physical trajectories it has already seen during training, leading to an artificially low validation loss.

To address this, we computed a SHA-1 hash over the full action sequence of each episode. Episodes sharing the same action hash (i.e., identical physical trajectories) were grouped together. The train/val split was then performed at the **trajectory level**, ensuring that all episodes belonging to a specific physical trajectory are placed entirely in either the training set or the validation set. 

This strict trajectory-level separation means the model's validation loss accurately reflects its ability to generalize to unseen physical demonstrations, rather than just memorizing seen trajectories with different textures or lighting.

### Training Details

| Item | Value |
|---|---|
| Dataset | [`nosuke113/libero-plus-deduplicated`](https://huggingface.co/datasets/nosuke113/libero-plus-deduplicated) (deduplicated: 8,395 episodes / 1,681 trajectories) |
| Data Split | Trajectory-level held-out (train: 1,376 traj / 6,873 ep, val: 154 traj / 767 ep, overlap: 0) |
| Fine-tuning | LoRA (Rank = 16, Alpha = 32, Action Expert + Projection Layers) |
| Checkpoint Path | `checkpoints/libero_plus_b8_r16_step35000` (b8: batch_size=8, r16: LoRA rank=16) |
| Steps | 35,000 |
| Optimizer | AdamW (lr=1e-4, batch_size=8, Cosine Scheduler) |
| Code | [`takashinnosuke/parc2026-submission-code`](https://github.com/takashinnosuke/parc2026-submission-code) (commit [`abdba5b`](https://github.com/takashinnosuke/parc2026-submission-code/commit/abdba5be999f454ba1bd81da5b598cd458e7e030)) |

![Training Loss](loss_curve.png)

---

## 3. Evaluation Dataset & Perturbation Axes

The evaluation dataset containing the 180 held-out tasks is available at [`nosuke113/libero-plus-evaluation`](https://huggingface.co/datasets/nosuke113/libero-plus-evaluation). These tasks are constructed along 4 perturbation axes from the LIBERO-plus BDDL scene definitions.

1. **Base** — Standard BDDL scenes. Camera `(0, 0, 100, 0, 0)`, fixed object poses, no noise.
2. **View** — Camera orbit angle (0-356 deg), zoom distance (100-191), and gaze tilt are varied.
3. **Noise** — Image corruptions: Gaussian noise, motion blur, fog, glass distortion (level 8-48).
4. **Moved** — Object initial positions shifted by several cm from default.

---

## 4. Evaluation Results (900 Rollouts)



Measured on 180 fully held-out tasks with n=5 seeds (900 total rollouts). Closed-loop control with max 600 steps per episode, bfloat16 mixed precision on NVIDIA L4 GPU.
*Inference used Receding Horizon Control (RTC) with an action chunk size of 5.*
**Note: The evaluation results below were measured using the checkpoint at 35,000 steps.**

| Metric | Value | Note |
|---|---|---|
| Total Episodes | 900 | 180 tasks x 5 seeds |
| **Overall Success Rate** | **46.00%** (414/900) | 95% Wilson CI: [42.7%, 49.3%] |
| Collision Rate | 32.00% (288/900) | 1mm displacement / link collision |
| Avg Steps | 362.9 | Max 600 steps |
| Action Jerk | 0.0221 | Mean step-to-step 6-DoF delta |

### Breakdown by Perturbation

| Category | N | **Success Rate** | Collision Rate | Avg Steps |
|---|---|---|---|---|
| **Base** | 150 | **78.67%** (118/150) | 15.33% | 247.9 |
| **View** | 300 | **55.00%** (165/300) | 32.67% | 286.6 |
| **Noise** | 300 | **30.67%** (92/300) | 29.33% | 441.0 |
| **Moved** | 150 | **26.00%** (39/150) | 52.67% | 474.2 |

<br/>
<div align="center">
  <h4>Robustness across Perturbations</h4>
  <p><em>Parallel playback comparing Base, View, Noise, and Position perturbed evaluation environments.</em></p>
  <table width="100%">
    <tr>
      <td width="50%" align="center" valign="top">
        <video src="https://huggingface.co/nosuke113/pi0.5-libero-plus-lora/resolve/main/videos/grid_0.mp4" controls autoplay loop muted width="100%"></video>
        <br/><b>Task: Put the black bowl on the plate</b>
      </td>
      <td width="50%" align="center" valign="top">
        <video src="https://huggingface.co/nosuke113/pi0.5-libero-plus-lora/resolve/main/videos/grid_1.mp4" controls autoplay loop muted width="100%"></video>
        <br/><b>Task: Turn on the stove bottom</b>
      </td>
    </tr>
    <tr>
      <td width="50%" align="center" valign="top">
        <video src="https://huggingface.co/nosuke113/pi0.5-libero-plus-lora/resolve/main/videos/grid_2.mp4" controls autoplay loop muted width="100%"></video>
        <br/><b>Task: Put the black bowl in the top drawer</b>
      </td>
      <td width="50%" align="center" valign="top">
      </td>
    </tr>
  </table>
</div>

---

## 5. Quickstart

This model is fully compatible with the Hugging Face [`lerobot`](https://github.com/huggingface/lerobot) library. You can instantiate it with a few lines of code:

```python
from lerobot.common.policies.factory import make_policy

# Load the policy with LoRA weights
policy = make_policy(repo_id="nosuke113/pi0.5-libero-plus-lora")
policy.eval()

# Example usage (assuming `obs` is provided by the environment)
# obs = {
#     "observation.images.agentview": ...,
#     "observation.images.robot0_eye_in_hand": ...,
#     "observation.state": ...,
# }
# action = policy.select_action(obs)
```

---

## 6. License & Attribution

| Component | License |
|---|---|
| Model weights (LoRA) | [Google Gemma Terms of Use](https://ai.google.dev/gemma/terms) |
| This repository (Code) | [MIT](https://github.com/takashinnosuke/parc2026-submission-code/blob/main/LICENSE) |
| Base model ([`lerobot/pi05_libero_base`](https://huggingface.co/lerobot/pi05_libero_base)) | [Google Gemma Terms of Use](https://ai.google.dev/gemma/terms) |
| [`lerobot`](https://github.com/huggingface/lerobot) library | Apache License 2.0 |
| [`libero`](https://github.com/Lifelong-Robot-Learning/LIBERO) package | MIT License |
| Training data ([`lerobot/libero_plus`](https://huggingface.co/datasets/lerobot/libero_plus)) | MIT License (Inherited from original LIBERO repository) |
