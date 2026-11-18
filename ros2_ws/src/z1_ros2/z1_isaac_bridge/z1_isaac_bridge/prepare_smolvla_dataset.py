#!/usr/bin/env python3
"""Prepare raw SmolVLA episodes by trimming static prefix duplicates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REMOVED_CAMERA_KEY = "observation.images.vla_wrist"
PHASE_ORDER = {
    "READY": 0,
    "GRASPING": 1,
    "LIFTING": 2,
    "PHYSICAL_TEST_LIFT": 2,
    "TRANSPORT": 3,
    "PLACING": 4,
    "SUCCESS": 5,
}
CANONICAL_PHASES = (
    "READY",
    "GRASPING",
    "PHYSICAL_TEST_LIFT",
    "TRANSPORT",
    "PLACING",
    "SUCCESS",
)


def _max_difference(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("state/action dimensions changed within an episode")
    return max((abs(float(a) - float(b)) for a, b in zip(left, right)), default=0.0)


def clean_frames(
    frames: list[dict[str, Any]],
    static_threshold: float = 0.002,
    fps: float = 10.0,
    leading_context_frames: int = 3,
    max_action_step_rad: float = 0.1,
    terminal_hold_frames: int = 10,
    max_identical_run_frames: int = 3,
) -> tuple[list[dict[str, Any]], int]:
    if not frames:
        raise ValueError("episode has no frames")
    first_state = frames[0]["observation.state"]
    first_moving = len(frames)
    for index, frame in enumerate(frames[1:], start=1):
        # Gripper commands can change before the arm starts.  Only the six arm
        # joints define the useful visual-to-motion start of an episode.
        difference = _max_difference(
            frame["observation.state"][:6], first_state[:6]
        )
        if difference >= static_threshold:
            first_moving = index
            break

    start = max(0, first_moving - leading_context_frames)
    kept = frames[start:]
    first_success = next(
        (index for index, frame in enumerate(kept) if frame.get("phase") == "SUCCESS"),
        None,
    )
    if first_success is not None:
        kept = kept[: first_success + max(1, terminal_hold_frames)]

    cleaned = []
    highest_phase = 0
    previous_signature = None
    identical_run = 0
    for index, original in enumerate(kept):
        frame = dict(original)
        phase = str(frame.get("phase", "READY")).strip().upper()
        highest_phase = max(highest_phase, PHASE_ORDER.get(phase, 0))
        frame["phase"] = CANONICAL_PHASES[highest_phase]
        state = [float(value) for value in frame["observation.state"]]
        action = [float(value) for value in frame["action"]]
        frame["action"] = [
            current
            + max(-max_action_step_rad, min(max_action_step_rad, target - current))
            for current, target in zip(state, action)
        ]
        signature = (tuple(state), tuple(frame["action"]), frame["phase"])
        if signature == previous_signature:
            identical_run += 1
        else:
            previous_signature = signature
            identical_run = 1
        if frame["phase"] != "SUCCESS" and identical_run > max_identical_run_frames:
            continue
        frame["frame_index"] = len(cleaned)
        frame["timestamp"] = len(cleaned) / fps
        frame.pop(REMOVED_CAMERA_KEY, None)
        frame.pop("vla_wrist_image_stamp", None)
        cleaned.append(frame)
    return cleaned, start


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def prepare(
    source: Path,
    output: Path,
    static_threshold: float,
    leading_context_frames: int,
    max_action_step_rad: float,
    terminal_hold_frames: int,
    max_identical_run_frames: int,
) -> None:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    info = _read_json(source / "dataset_info.json")
    fps = float(info["fps"])
    episodes = sorted(source.glob("episode_[0-9][0-9][0-9][0-9][0-9][0-9]"))
    if not episodes:
        raise ValueError(f"no episodes found in {source}")

    output.mkdir(parents=True)
    try:
        info["observation_keys"] = [
            key for key in info["observation_keys"] if key != REMOVED_CAMERA_KEY
        ]
        info.get("image_shapes", {}).pop("vla_wrist", None)
        info["preprocessing"] = {
            "removed_camera": REMOVED_CAMERA_KEY,
            "leading_static_threshold_rad": static_threshold,
            "leading_context_frames": leading_context_frames,
            "max_action_step_rad": max_action_step_rad,
            "terminal_hold_frames": terminal_hold_frames,
            "max_identical_run_frames": max_identical_run_frames,
            "monotonic_phase_labels": True,
            "first_frame_preserved": False,
        }
        _write_json(output / "dataset_info.json", info)

        removals: dict[str, int] = {}
        total_frames = 0
        for episode in episodes:
            resolved = episode.resolve()
            frames = [
                json.loads(line)
                for line in (resolved / "frames.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            summary = _read_json(resolved / "episode_summary.json")
            cleaned, removed = clean_frames(
                frames,
                static_threshold,
                fps,
                leading_context_frames,
                max_action_step_rad,
                terminal_hold_frames,
                max_identical_run_frames,
            )
            destination = output / episode.name
            (destination / "images").mkdir(parents=True)
            for camera in ("wrist", "global"):
                (destination / "images" / camera).symlink_to(
                    resolved / "images" / camera,
                    target_is_directory=True,
                )
            with (destination / "frames.jsonl").open("w", encoding="utf-8") as stream:
                for frame in cleaned:
                    stream.write(json.dumps(frame, ensure_ascii=False) + "\n")
            summary["frame_count"] = len(cleaned)
            summary["duration_s"] = len(cleaned) / fps
            summary["preprocessing"] = {
                "leading_static_frames_removed": removed,
                "removed_camera": REMOVED_CAMERA_KEY,
                "leading_context_frames": leading_context_frames,
                "max_action_step_rad": max_action_step_rad,
                "terminal_hold_frames": terminal_hold_frames,
                "max_identical_run_frames": max_identical_run_frames,
                "monotonic_phase_labels": True,
            }
            _write_json(destination / "episode_summary.json", summary)
            removals[episode.name] = removed
            total_frames += len(cleaned)

        selection = source / "selection_manifest.json"
        if selection.is_file():
            manifest = _read_json(selection)
            manifest["prepared_source"] = str(source)
            _write_json(output / "selection_manifest.json", manifest)
        _write_json(
            output / "preprocessing_manifest.json",
            {
                "source": str(source),
                "episode_count": len(episodes),
                "total_frames": total_frames,
                "total_leading_static_frames_removed": sum(removals.values()),
                "leading_static_frames_removed": removals,
                "removed_camera": REMOVED_CAMERA_KEY,
                "leading_static_threshold_rad": static_threshold,
                "leading_context_frames": leading_context_frames,
                "max_action_step_rad": max_action_step_rad,
                "terminal_hold_frames": terminal_hold_frames,
                "max_identical_run_frames": max_identical_run_frames,
                "monotonic_phase_labels": True,
            },
        )
    except Exception:
        import shutil

        shutil.rmtree(output, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--static-threshold", type=float, default=0.002)
    parser.add_argument("--leading-context-frames", type=int, default=3)
    parser.add_argument("--max-action-step-rad", type=float, default=0.1)
    parser.add_argument("--terminal-hold-frames", type=int, default=10)
    parser.add_argument("--max-identical-run-frames", type=int, default=3)
    args = parser.parse_args()
    if args.static_threshold <= 0:
        parser.error("--static-threshold must be positive")
    if args.leading_context_frames < 0:
        parser.error("--leading-context-frames cannot be negative")
    if args.max_action_step_rad <= 0:
        parser.error("--max-action-step-rad must be positive")
    if args.terminal_hold_frames <= 0:
        parser.error("--terminal-hold-frames must be positive")
    if args.max_identical_run_frames <= 0:
        parser.error("--max-identical-run-frames must be positive")
    prepare(
        args.source,
        args.output,
        args.static_threshold,
        args.leading_context_frames,
        args.max_action_step_rad,
        args.terminal_hold_frames,
        args.max_identical_run_frames,
    )
    print(f"prepared dataset: {args.output}", flush=True)


if __name__ == "__main__":
    main()
