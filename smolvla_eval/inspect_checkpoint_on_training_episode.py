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
EPISODE = Path(
    "/home/lp531/smolvla_data/z1_tabletop_raw_dual_wrist/episode_000026"
)
CAMERAS = ("wrist", "vla_wrist", "global")
FRAME_INDICES = (0, 90, 150, 190, 250)


def image_tensor(path: Path, device: torch.device) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"could not read {path}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return (
        torch.from_numpy(np.ascontiguousarray(rgb))
        .to(device=device, dtype=torch.float32)
        .permute(2, 0, 1)
        .unsqueeze(0)
        / 255.0
    )


def first_closed(actions: np.ndarray) -> int | None:
    indices = np.flatnonzero(actions[:, 6] >= -0.75)
    return int(indices[0]) if len(indices) else None


def main() -> None:
    rows = [json.loads(line) for line in (EPISODE / "frames.jsonl").read_text().splitlines()]
    policy = SmolVLAPolicy.from_pretrained(
        str(CHECKPOINT), local_files_only=True
    )
    policy.eval()
    device = torch.device(policy.config.device)
    amp = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if policy.config.use_amp and device.type == "cuda"
        else nullcontext()
    )
    results = []
    for frame_index in FRAME_INDICES:
        row = rows[frame_index]
        batch = {
            "observation.state": torch.tensor(
                row["observation.state"], dtype=torch.float32, device=device
            ).unsqueeze(0),
            "task": [row["task"]],
        }
        for camera in CAMERAS:
            batch[f"observation.images.{camera}"] = image_tensor(
                EPISODE / row[f"observation.images.{camera}"], device
            )
        policy.reset()
        torch.manual_seed(0)
        with torch.inference_mode(), amp:
            predicted = policy.predict_action_chunk(batch)[0].float().cpu().numpy()
        available = min(50, len(rows) - frame_index)
        truth = np.asarray(
            [item["action"] for item in rows[frame_index : frame_index + available]],
            dtype=np.float32,
        )
        predicted = predicted[:available]
        absolute_error = np.abs(predicted - truth)
        window_scores = {}
        for offset in range(-30, 31):
            start = frame_index + offset
            if start < 0 or start + available > len(rows):
                continue
            candidate = np.asarray(
                [item["action"] for item in rows[start : start + available]],
                dtype=np.float32,
            )
            window_scores[offset] = float(np.mean(np.abs(predicted - candidate)))
        best_offset = min(window_scores, key=window_scores.get)
        results.append(
            {
                "frame_index": frame_index,
                "timestamp_s": row["timestamp"],
                "phase": row["phase"],
                "mae_all_50": float(absolute_error.mean()),
                "mae_arm_50": float(absolute_error[:, :6].mean()),
                "mae_gripper_50": float(absolute_error[:, 6].mean()),
                "mae_first_10": float(absolute_error[:10].mean()),
                "predicted_first_closed_step": first_closed(predicted),
                "truth_first_closed_step": first_closed(truth),
                "best_matching_truth_offset_steps": int(best_offset),
                "best_offset_mae": window_scores[best_offset],
                "predicted_first_action": predicted[0].tolist(),
                "truth_first_action": truth[0].tolist(),
            }
        )
    output = {
        "checkpoint": str(CHECKPOINT),
        "episode": str(EPISODE),
        "results": results,
    }
    path = Path("/home/lp531/smolvla_eval/checkpoint_training_probe.json")
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
