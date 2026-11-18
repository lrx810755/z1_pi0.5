import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


ROOT = Path("/home/lp531/smolvla_data/z1_tabletop_lerobot_4x100_v1")
CHECKPOINT = Path(
    "/home/lp531/smolvla_training/z1_4x100_smolvla_20260921_113511/"
    "checkpoints/100000/pretrained_model"
)
EPISODE_INDEX = 26
FRAME_INDICES = (0, 90, 150, 190, 250)


def first_closed(actions: np.ndarray) -> int | None:
    indices = np.flatnonzero(actions[:, 6] >= -0.75)
    return int(indices[0]) if len(indices) else None


def main() -> None:
    episode_rows = [
        json.loads(line)
        for line in (ROOT / "meta/episodes.jsonl").read_text().splitlines()
    ]
    start_index = sum(row["length"] for row in episode_rows[:EPISODE_INDEX])
    dataset = LeRobotDataset(
        "lp531/z1_smolvla_tabletop_4x100",
        root=ROOT,
        video_backend="pyav",
    )
    policy = SmolVLAPolicy.from_pretrained(str(CHECKPOINT), local_files_only=True)
    policy.eval()
    device = torch.device(policy.config.device)
    amp = (
        torch.autocast("cuda", dtype=torch.float16)
        if policy.config.use_amp and device.type == "cuda"
        else nullcontext()
    )
    results = []
    for frame_index in FRAME_INDICES:
        item = dataset[start_index + frame_index]
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
        available = min(50, episode_rows[EPISODE_INDEX]["length"] - frame_index)
        truth = np.asarray(
            [dataset[start_index + frame_index + step]["action"].numpy() for step in range(available)]
        )
        error = np.abs(predicted[:available] - truth)
        results.append(
            {
                "frame_index": frame_index,
                "mae_all_50": float(error.mean()),
                "mae_arm_50": float(error[:, :6].mean()),
                "mae_gripper_50": float(error[:, 6].mean()),
                "mae_first_10": float(error[:10].mean()),
                "predicted_first_closed_step": first_closed(predicted[:available]),
                "truth_first_closed_step": first_closed(truth),
            }
        )
    path = Path("/home/lp531/smolvla_eval/converted_training_probe.json")
    path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
