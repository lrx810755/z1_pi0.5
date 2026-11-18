import json
from contextlib import nullcontext
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import torch

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


CHECKPOINT = Path(
    "/home/lp531/smolvla_training/z1_4x100_smolvla_20260921_113511/"
    "checkpoints/100000/pretrained_model"
)
EPISODE = Path(
    "/home/lp531/smolvla_data/z1_tabletop_raw_dual_wrist/episode_000026"
)
TASKS = {
    "apple": "Physically grasp the apple, lift it clear, and place it in the box.",
    "cup": "Physically grasp the cup, lift it clear, and place it in the box.",
    "peach": "Physically grasp the peach, lift it clear, and place it in the box.",
    "bottle": "Physically grasp the bottle, lift it clear, and place it in the box.",
}


def load_image(path: Path, device: torch.device) -> torch.Tensor:
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
    rows = [json.loads(line) for line in (EPISODE / "frames.jsonl").read_text().splitlines()]
    policy = SmolVLAPolicy.from_pretrained(str(CHECKPOINT), local_files_only=True)
    policy.eval()
    device = torch.device(policy.config.device)
    amp = (
        torch.autocast("cuda", dtype=torch.float16)
        if policy.config.use_amp and device.type == "cuda"
        else nullcontext()
    )
    report = {}
    for frame_index in (0, 90, 150):
        row = rows[frame_index]
        common = {
            "observation.state": torch.tensor(
                row["observation.state"], dtype=torch.float32, device=device
            ).unsqueeze(0)
        }
        for camera in ("wrist", "vla_wrist", "global"):
            common[f"observation.images.{camera}"] = load_image(
                EPISODE / row[f"observation.images.{camera}"], device
            )
        chunks = {}
        for name, task in TASKS.items():
            policy.reset()
            torch.manual_seed(1234)
            batch = {**common, "task": [task]}
            with torch.inference_mode(), amp:
                chunks[name] = (
                    policy.predict_action_chunk(batch)[0].float().cpu().numpy()
                )
        pairs = {}
        for left, right in combinations(TASKS, 2):
            delta = np.abs(chunks[left] - chunks[right])
            pairs[f"{left}_vs_{right}"] = {
                "all_50_mae": float(delta.mean()),
                "arm_50_mae": float(delta[:, :6].mean()),
                "first_10_mae": float(delta[:10].mean()),
                "max_abs": float(delta.max()),
            }
        report[str(frame_index)] = pairs
    path = Path("/home/lp531/smolvla_eval/language_conditioning_probe.json")
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
