#!/usr/bin/env python3
"""Convert recoverable Z1 raw episodes to a LeRobot v3 dataset.

Run in the existing environment that contains LeRobot 0.3.4:

    conda run -n unifolm-vla python export_z1_lerobot_dataset.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


EXPECTED_FORMAT = "z1_smolvla_raw_v1"
DEFAULT_CAMERA_KEYS = (
    "observation.images.wrist",
    "observation.images.global",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/home/lp531/smolvla_data/z1_tabletop_raw_16x9"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/lp531/smolvla_data/z1_tabletop_lerobot_16x9"),
    )
    parser.add_argument("--repo-id", default="lp531/z1_smolvla_tabletop")
    parser.add_argument(
        "--images",
        action="store_true",
        help="Keep PNG images instead of encoding MP4 videos.",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include episodes whose summary is not task_success=true.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_frames(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON in {path}:{line_number}: {exc}"
                ) from exc


def load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"cannot decode image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def select_episodes(root: Path, include_failed: bool) -> list[Path]:
    selected = []
    for episode_dir in sorted(root.glob("episode_[0-9][0-9][0-9][0-9][0-9][0-9]")):
        summary_path = episode_dir / "episode_summary.json"
        frames_path = episode_dir / "frames.jsonl"
        if not summary_path.is_file() or not frames_path.is_file():
            raise FileNotFoundError(f"incomplete episode: {episode_dir}")
        summary = read_json(summary_path)
        if include_failed or bool(summary.get("task_success", False)):
            selected.append(episode_dir)
    return selected


def inspect_first_frame(
    episode_dir: Path, camera_keys: Tuple[str, ...]
) -> Dict[str, Tuple[int, int, int]]:
    first = next(iter(read_frames(episode_dir / "frames.jsonl")), None)
    if first is None:
        raise ValueError(f"episode has no frames: {episode_dir}")
    shapes = {}
    for key in camera_keys:
        image = load_rgb(episode_dir / first[key])
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(f"invalid camera image shape for {key}: {image.shape}")
        shapes[key] = tuple(int(value) for value in image.shape)
    return shapes


def features(
    joint_names,
    image_shapes: Dict[str, Tuple[int, int, int]],
    camera_keys: Tuple[str, ...],
    use_videos: bool,
) -> Dict:
    image_dtype = "video" if use_videos else "image"
    result = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(joint_names),),
            "names": list(joint_names),
        },
        "action": {
            "dtype": "float32",
            "shape": (len(joint_names),),
            "names": list(joint_names),
        },
    }
    for key in camera_keys:
        result[key] = {
            "dtype": image_dtype,
            "shape": image_shapes[key],
            "names": ["height", "width", "channels"],
        }
    return result


def convert(args: argparse.Namespace) -> None:
    input_root = args.input.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    info_path = input_root / "dataset_info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"raw dataset metadata not found: {info_path}")
    info = read_json(info_path)
    if info.get("format") != EXPECTED_FORMAT:
        raise ValueError(
            f"unsupported raw format {info.get('format')!r}; "
            f"expected {EXPECTED_FORMAT!r}"
        )
    if output_root.exists():
        raise FileExistsError(
            f"output already exists: {output_root}; choose a new --output path"
        )
    episode_dirs = select_episodes(input_root, args.include_failed)
    if not episode_dirs:
        raise ValueError("no eligible saved episodes found")

    camera_keys = tuple(
        key
        for key in info.get("observation_keys", DEFAULT_CAMERA_KEYS)
        if str(key).startswith("observation.images.")
    )
    if not camera_keys:
        raise ValueError("dataset metadata contains no camera observation keys")
    image_shapes = inspect_first_frame(episode_dirs[0], camera_keys)
    fps = int(round(float(info["fps"])))
    joint_names = list(info["joint_names"])
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=fps,
        root=output_root,
        robot_type="unitree_z1",
        features=features(joint_names, image_shapes, camera_keys, not args.images),
        use_videos=not args.images,
        image_writer_threads=4,
    )

    for output_index, episode_dir in enumerate(episode_dirs):
        summary = read_json(episode_dir / "episode_summary.json")
        expected_index = 0
        for frame in read_frames(episode_dir / "frames.jsonl"):
            frame_index = int(frame["frame_index"])
            if frame_index != expected_index:
                raise ValueError(
                    f"non-contiguous frame index in {episode_dir}: "
                    f"{frame_index} != {expected_index}"
                )
            state = np.asarray(frame["observation.state"], dtype=np.float32)
            action = np.asarray(frame["action"], dtype=np.float32)
            if state.shape != (len(joint_names),) or action.shape != (
                len(joint_names),
            ):
                raise ValueError(
                    f"invalid state/action shape in {episode_dir} frame {frame_index}"
                )
            frame_payload = {
                "observation.state": state,
                "action": action,
            }
            for camera_key in camera_keys:
                frame_payload[camera_key] = load_rgb(
                    episode_dir / frame[camera_key]
                )
            dataset.add_frame(
                frame_payload,
                task=str(summary["task"]),
                timestamp=frame_index / fps,
            )
            expected_index += 1
        if expected_index != int(summary["frame_count"]):
            raise ValueError(
                f"frame count mismatch in {episode_dir}: "
                f"{expected_index} != {summary['frame_count']}"
            )
        dataset.save_episode()
        print(
            f"Converted raw {episode_dir.name} -> LeRobot episode "
            f"{output_index:06d} ({expected_index} frames)",
            flush=True,
        )

    dataset.stop_image_writer()
    reopened = LeRobotDataset(args.repo_id, root=output_root)
    print(
        f"LeRobot dataset ready: {output_root}\n"
        f"episodes={reopened.num_episodes}, frames={reopened.num_frames}, "
        f"fps={reopened.fps}",
        flush=True,
    )


def main() -> None:
    convert(parse_args())


if __name__ == "__main__":
    main()
