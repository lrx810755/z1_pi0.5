import json
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


CHECKPOINT = Path(
    "/home/lp531/smolvla_training/z1_4x100_smolvla_20260921_113511/"
    "checkpoints/100000/pretrained_model"
)
ROOT = Path("/home/lp531/smolvla_data/z1_tabletop_raw_dual_wrist")
EPISODES = (0, 3, 14, 26, 34, 48, 64, 77, 91, 97)


def image(path: Path, device: torch.device) -> torch.Tensor:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return (
        torch.from_numpy(np.ascontiguousarray(rgb))
        .to(device=device, dtype=torch.float32)
        .permute(2, 0, 1)
        .unsqueeze(0)
        / 255.0
    )


def main() -> None:
    policy = SmolVLAPolicy.from_pretrained(str(CHECKPOINT), local_files_only=True)
    policy.eval()
    device = torch.device(policy.config.device)
    amp = (
        torch.autocast("cuda", dtype=torch.float16)
        if policy.config.use_amp and device.type == "cuda"
        else nullcontext()
    )
    predictions, truths, positions, per_episode = [], [], [], []
    for index in EPISODES:
        directory = ROOT / f"episode_{index:06d}"
        rows = [json.loads(line) for line in (directory / "frames.jsonl").read_text().splitlines()]
        start = json.loads((directory / "episode_start.json").read_text())
        row = rows[0]
        batch = {
            "observation.state": torch.tensor(
                row["observation.state"], dtype=torch.float32, device=device
            ).unsqueeze(0),
            "task": [row["task"]],
        }
        for camera in ("wrist", "vla_wrist", "global"):
            batch[f"observation.images.{camera}"] = image(
                directory / row[f"observation.images.{camera}"], device
            )
        policy.reset()
        torch.manual_seed(2468)
        with torch.inference_mode(), amp:
            predicted = policy.predict_action_chunk(batch)[0].float().cpu().numpy()
        truth = np.asarray([item["action"] for item in rows[:50]], dtype=np.float32)
        predictions.append(predicted)
        truths.append(truth)
        position = start["scene_randomization"]["object_positions_m"]["apple"][:2]
        positions.append(position)
        per_episode.append(
            {
                "episode": index,
                "apple_xy": position,
                "mae": float(np.mean(np.abs(predicted - truth))),
                "arm_mae": float(np.mean(np.abs(predicted[:, :6] - truth[:, :6]))),
            }
        )
    predictions = np.asarray(predictions)
    truths = np.asarray(truths)
    # Mean standard deviation over scenes for every time/joint coordinate.
    predicted_scene_std = float(predictions[:, :, :6].std(axis=0).mean())
    truth_scene_std = float(truths[:, :, :6].std(axis=0).mean())
    report = {
        "episodes": per_episode,
        "predicted_arm_scene_std": predicted_scene_std,
        "truth_arm_scene_std": truth_scene_std,
        "visual_variation_ratio": predicted_scene_std / truth_scene_std,
        "mean_arm_mae": float(np.mean(np.abs(predictions[:, :, :6] - truths[:, :, :6]))),
    }
    path = Path("/home/lp531/smolvla_eval/visual_conditioning_probe.json")
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
