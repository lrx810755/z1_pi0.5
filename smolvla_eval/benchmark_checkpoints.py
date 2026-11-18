import gc
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


ROOT = Path("/home/lp531/smolvla_data/z1_tabletop_lerobot_4x100_v1")
TRAINING = Path("/home/lp531/smolvla_training/z1_4x100_smolvla_20260921_113511/checkpoints")
CHECKPOINTS = (20000, 40000, 60000, 80000, 100000)
EPISODE_INDEX = 26
FRAME_INDICES = (0, 90, 150, 190, 250)


def main() -> None:
    episode_rows = [json.loads(line) for line in (ROOT / "meta/episodes.jsonl").read_text().splitlines()]
    start = sum(row["length"] for row in episode_rows[:EPISODE_INDEX])
    dataset = LeRobotDataset(
        "lp531/z1_smolvla_tabletop_4x100", root=ROOT, video_backend="pyav"
    )
    samples = []
    for frame in FRAME_INDICES:
        item = dataset[start + frame]
        truth = np.asarray(
            [dataset[start + frame + step]["action"].numpy() for step in range(50)]
        )
        samples.append((frame, item, truth))

    report = []
    for step in CHECKPOINTS:
        checkpoint = TRAINING / f"{step:06d}/pretrained_model"
        policy = SmolVLAPolicy.from_pretrained(str(checkpoint), local_files_only=True)
        policy.eval()
        device = torch.device(policy.config.device)
        amp = (
            torch.autocast("cuda", dtype=torch.float16)
            if policy.config.use_amp and device.type == "cuda"
            else nullcontext()
        )
        rows = []
        for frame, item, truth in samples:
            batch = {
                key: value.unsqueeze(0).to(device)
                for key, value in item.items()
                if key.startswith("observation.")
            }
            batch["task"] = [item["task"]]
            policy.reset()
            torch.manual_seed(0)
            with torch.inference_mode(), amp:
                predicted = policy.predict_action_chunk(batch)[0].float().cpu().numpy()
            error = np.abs(predicted - truth)
            rows.append(
                {
                    "frame": frame,
                    "first_10_mae": float(error[:10].mean()),
                    "arm_50_mae": float(error[:, :6].mean()),
                    "gripper_50_mae": float(error[:, 6].mean()),
                }
            )
        report.append(
            {
                "checkpoint": step,
                "mean_first_10_mae": float(np.mean([row["first_10_mae"] for row in rows])),
                "mean_arm_50_mae": float(np.mean([row["arm_50_mae"] for row in rows])),
                "mean_gripper_50_mae": float(np.mean([row["gripper_50_mae"] for row in rows])),
                "frames": rows,
            }
        )
        del policy
        gc.collect()
        torch.cuda.empty_cache()
    path = Path("/home/lp531/smolvla_eval/checkpoint_benchmark.json")
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
