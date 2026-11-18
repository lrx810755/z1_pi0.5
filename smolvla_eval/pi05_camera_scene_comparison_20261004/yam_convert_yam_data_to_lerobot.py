"""Convert timestamped Gello YAM episodes to fixed-rate LeRobot datasets.

The source MP4 files declare 30 FPS even though the control-loop timestamps do
not. This converter uses ``trajectory.npz`` as the timing authority and writes
separate train/eval datasets at 15 Hz.
"""

from __future__ import annotations

from collections.abc import Iterable
import dataclasses
import json
from pathlib import Path
import shutil

import cv2
from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.video_utils import encode_video_frames
import numpy as np
import torch
import tqdm
import tyro

CAMERAS = ("middle", "left", "right")
JOINT_NAMES = (
    "left_joint_0",
    "left_joint_1",
    "left_joint_2",
    "left_joint_3",
    "left_joint_4",
    "left_joint_5",
    "left_gripper",
)
TASK = "pick up the red cap and place it in the black box"
DEFAULT_EVAL_EPISODES = (6, 16, 26, 36, 46, 56, 66, 76, 86, 96)


@dataclasses.dataclass(frozen=True)
class ResampledEpisode:
    source_id: int
    source_frames: int
    source_rate_hz: float
    timestamps: np.ndarray
    source_indices: np.ndarray
    state: np.ndarray
    action: np.ndarray


class YamLeRobotDataset(LeRobotDataset):
    """LeRobot writer using H.264 for lower randomized-read CPU cost."""

    def encode_episode_videos(self, episode_index: int) -> dict:
        video_paths = {}
        for key in self.meta.video_keys:
            video_path = self.root / self.meta.get_video_file_path(episode_index, key)
            video_paths[key] = str(video_path)
            if video_path.is_file():
                continue
            image_dir = self._get_image_file_path(
                episode_index=episode_index,
                image_key=key,
                frame_index=0,
            ).parent
            encode_video_frames(image_dir, video_path, self.fps, vcodec="h264", overwrite=True)
        return video_paths


def _nearest_indices(source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    right = np.searchsorted(source_times, target_times, side="left")
    right = np.clip(right, 0, len(source_times) - 1)
    left = np.clip(right - 1, 0, len(source_times) - 1)
    choose_left = np.abs(target_times - source_times[left]) <= np.abs(source_times[right] - target_times)
    return np.where(choose_left, left, right).astype(np.int64)


def _interpolate_joints(source_times: np.ndarray, values: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    output = np.empty((len(target_times), 7), dtype=np.float32)
    for dim in range(6):
        output[:, dim] = np.interp(target_times, source_times, values[:, dim])
    nearest = _nearest_indices(source_times, target_times)
    output[:, 6] = values[nearest, 6]
    return output


def load_resampled_episode(episode_dir: Path, fps: int) -> ResampledEpisode:
    source_id = int(episode_dir.name.rsplit("_", 1)[1])
    with np.load(episode_dir / "trajectory.npz") as trajectory:
        source_times = np.asarray(trajectory["timestamp"], dtype=np.float64)
        state = np.asarray(trajectory["follower_joints"], dtype=np.float32)
        action = np.asarray(trajectory["leader_actions"], dtype=np.float32)

    if source_times.ndim != 1 or len(source_times) < 2:
        raise ValueError(f"{episode_dir}: timestamps must contain at least two samples")
    if state.shape != (len(source_times), 14) or action.shape != state.shape:
        raise ValueError(
            f"{episode_dir}: expected timestamps (N,), state/action (N, 14); "
            f"got {source_times.shape}, {state.shape}, {action.shape}"
        )
    if not np.isfinite(source_times).all() or not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError(f"{episode_dir}: trajectory contains non-finite values")
    if np.any(np.diff(source_times) <= 0):
        raise ValueError(f"{episode_dir}: timestamps are not strictly increasing")

    duration = float(source_times[-1] - source_times[0])
    target_count = int(np.floor(duration * fps)) + 1
    target_times = source_times[0] + np.arange(target_count, dtype=np.float64) / fps
    source_indices = _nearest_indices(source_times, target_times)
    return ResampledEpisode(
        source_id=source_id,
        source_frames=len(source_times),
        source_rate_hz=float((len(source_times) - 1) / duration),
        timestamps=target_times,
        source_indices=source_indices,
        state=_interpolate_joints(source_times, state[:, :7], target_times),
        action=_interpolate_joints(source_times, action[:, :7], target_times),
    )


def _features() -> dict:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": [list(JOINT_NAMES)],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": [list(JOINT_NAMES)],
        },
    }
    for camera in CAMERAS:
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        }
    return features


def _create_dataset(repo_id: str, fps: int, *, force: bool) -> LeRobotDataset:
    output = HF_LEROBOT_HOME / repo_id
    if output.exists():
        if not force:
            raise FileExistsError(f"dataset already exists: {output}; pass --force to replace it")
        shutil.rmtree(output)
    return YamLeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type="yam_left_arm",
        features=_features(),
        use_videos=True,
        image_writer_processes=8,
        image_writer_threads=4,
    )


def _emit_episode(dataset: LeRobotDataset, episode_dir: Path, episode: ResampledEpisode) -> None:
    captures = {camera: cv2.VideoCapture(str(episode_dir / f"{camera}.mp4")) for camera in CAMERAS}
    try:
        for camera, capture in captures.items():
            if not capture.isOpened():
                raise ValueError(f"{episode_dir}: could not open {camera}.mp4")
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count != episode.source_frames:
                raise ValueError(
                    f"{episode_dir}: {camera}.mp4 has {frame_count} frames, trajectory has {episode.source_frames}"
                )

        target_index = 0
        for source_index in range(episode.source_frames):
            images: dict[str, np.ndarray] = {}
            for camera, capture in captures.items():
                ok, bgr = capture.read()
                if not ok:
                    raise ValueError(f"{episode_dir}: failed reading {camera}.mp4 frame {source_index}")
                images[camera] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            while target_index < len(episode.source_indices) and episode.source_indices[target_index] == source_index:
                frame = {
                    "observation.state": torch.from_numpy(episode.state[target_index]),
                    "action": torch.from_numpy(episode.action[target_index]),
                    "task": TASK,
                }
                for camera, image in images.items():
                    frame[f"observation.images.{camera}"] = image
                dataset.add_frame(frame)
                target_index += 1

        if target_index != len(episode.source_indices):
            raise ValueError(f"{episode_dir}: emitted {target_index} of {len(episode.source_indices)} target frames")
        dataset.save_episode()
    finally:
        for capture in captures.values():
            capture.release()


def _episode_dirs(raw_dir: Path) -> list[Path]:
    directories = sorted(raw_dir.glob("episode_[0-9][0-9][0-9][0-9]"))
    if len(directories) != 100:
        raise ValueError(f"expected 100 episode directories under {raw_dir}, found {len(directories)}")
    return directories


def _convert_split(
    repo_id: str,
    episode_dirs: Iterable[Path],
    fps: int,
    *,
    force: bool,
    split: str,
) -> list[dict]:
    dataset = _create_dataset(repo_id, fps, force=force)
    records = []
    for episode_dir in tqdm.tqdm(list(episode_dirs), desc=f"Converting {split}"):
        episode = load_resampled_episode(episode_dir, fps)
        _emit_episode(dataset, episode_dir, episode)
        records.append(
            {
                "source_episode_id": episode.source_id,
                "source_frames": episode.source_frames,
                "converted_frames": len(episode.timestamps),
                "source_rate_hz": episode.source_rate_hz,
            }
        )
    dataset.stop_image_writer()
    manifest = {
        "schema": "yam-red-cap-conversion.v1",
        "split": split,
        "repo_id": repo_id,
        "fps": fps,
        "task": TASK,
        "gripper_semantics": {"native_zero": "closed", "native_one": "open"},
        "episodes": records,
        "total_frames": sum(record["converted_frames"] for record in records),
    }
    output_path = HF_LEROBOT_HOME / repo_id / "conversion_manifest.json"
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return records


def convert_yam_data(
    raw_dir: Path,
    train_repo_id: str = "npow/yam-gello-red-cap-100-15hz-train",
    eval_repo_id: str = "npow/yam-gello-red-cap-100-15hz-eval",
    fps: int = 15,
    eval_episode_ids: tuple[int, ...] = DEFAULT_EVAL_EPISODES,
    *,
    force: bool = False,
) -> None:
    """Convert raw episodes into deterministic train and held-out datasets."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    directories = _episode_dirs(raw_dir)
    by_id = {int(directory.name.rsplit("_", 1)[1]): directory for directory in directories}
    eval_ids = set(eval_episode_ids)
    missing = eval_ids - set(by_id)
    if missing:
        raise ValueError(f"eval episode IDs do not exist: {sorted(missing)}")
    if not eval_ids or len(eval_ids) >= len(directories):
        raise ValueError("eval split must contain between 1 and 99 episodes")

    train_dirs = [directory for episode_id, directory in by_id.items() if episode_id not in eval_ids]
    eval_dirs = [directory for episode_id, directory in by_id.items() if episode_id in eval_ids]
    _convert_split(train_repo_id, train_dirs, fps, force=force, split="train")
    _convert_split(eval_repo_id, eval_dirs, fps, force=force, split="eval")


if __name__ == "__main__":
    tyro.cli(convert_yam_data)
