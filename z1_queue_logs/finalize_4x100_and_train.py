#!/usr/bin/env python3
"""Wait for a balanced raw dataset, convert it, then train SmolVLA."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen


RAW = Path("/home/lp531/smolvla_data/z1_tabletop_raw_dual_wrist")
REVIEWED = Path("/home/lp531/smolvla_data/z1_tabletop_reviewed_4x100_v1")
LEROBOT = Path("/home/lp531/smolvla_data/z1_tabletop_lerobot_4x100_v1")
EXPORTER = Path(
    "/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/scripts/"
    "export_z1_lerobot_dataset.py"
)
RUNS = Path("/home/lp531/smolvla_training")
CONDA = "/home/lp531/miniconda3/bin/conda"
STATE = Path("/home/lp531/z1_queue_logs/finalize_4x100_state.json")
TRAIN_PID = Path("/home/lp531/z1_queue_logs/smolvla_train.pid")
TRAIN_LOG = Path("/home/lp531/z1_queue_logs/smolvla_train.log")
OBJECTS = ("apple", "peach", "beverage_bottle", "cup")
PER_OBJECT = 100
EPISODE_PATTERN = re.compile(r"^episode_\d{6}$")


def log(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def save_state(stage: str, **extra) -> None:
    payload = {"stage": stage, "updated_at": datetime.now().isoformat(), **extra}
    STATE.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def successful_episodes() -> dict[str, list[Path]]:
    episodes: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(RAW.iterdir()):
        if not path.is_dir() or not EPISODE_PATTERN.match(path.name):
            continue
        summary_path = path / "episode_summary.json"
        frames_path = path / "frames.jsonl"
        if not summary_path.is_file() or not frames_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        target = summary.get("target_object")
        if target in OBJECTS and summary.get("task_success") is True:
            episodes[target].append(path)
    return episodes


def post(path: str) -> None:
    request = Request(
        f"http://127.0.0.1:8767{path}",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=30):
            pass
    except Exception as exc:
        log(f"best-effort POST {path} failed: {exc}")


def matching_pids(needle: str) -> list[int]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,args="], text=True, capture_output=True, check=True
    )
    found = []
    for line in result.stdout.splitlines():
        pid_text, _, command = line.strip().partition(" ")
        if needle in command and int(pid_text) != os.getpid():
            found.append(int(pid_text))
    return found


def stop_collection() -> None:
    post("/api/force_stop")
    post("/api/recording/discard")
    for pid in matching_pids(
        "/z1_isaac_bridge/continuous_training --target cup"
    ):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(5)


def validate_episode(path: Path, expected_object: str) -> dict:
    summary = json.loads((path / "episode_summary.json").read_text(encoding="utf-8"))
    if summary.get("target_object") != expected_object or summary.get("task_success") is not True:
        raise ValueError(f"invalid target/success metadata: {path}")
    info = json.loads((RAW / "dataset_info.json").read_text(encoding="utf-8"))
    camera_keys = [
        key for key in info["observation_keys"] if key.startswith("observation.images.")
    ]
    frame_count = 0
    with (path / "frames.jsonl").open(encoding="utf-8") as stream:
        for frame_count, line in enumerate(stream, start=1):
            frame = json.loads(line)
            if int(frame["frame_index"]) != frame_count - 1:
                raise ValueError(f"non-contiguous frames: {path}")
            if len(frame["observation.state"]) != len(info["joint_names"]):
                raise ValueError(f"invalid state shape: {path}")
            if len(frame["action"]) != len(info["joint_names"]):
                raise ValueError(f"invalid action shape: {path}")
            for key in camera_keys:
                if not (path / frame[key]).is_file():
                    raise FileNotFoundError(path / frame[key])
    if frame_count != int(summary["frame_count"]):
        raise ValueError(f"frame count mismatch: {path}")
    return summary


def prepare_reviewed() -> dict[str, list[str]]:
    episodes = successful_episodes()
    missing = {name: len(episodes[name]) for name in OBJECTS if len(episodes[name]) < PER_OBJECT}
    if missing:
        raise RuntimeError(f"insufficient successful episodes: {missing}")
    REVIEWED.mkdir(parents=True, exist_ok=True)
    info_link = REVIEWED / "dataset_info.json"
    if not info_link.exists():
        info_link.symlink_to(RAW / "dataset_info.json")
    selected: dict[str, list[str]] = {}
    task_counts = Counter()
    for target in OBJECTS:
        chosen = episodes[target][:PER_OBJECT]
        selected[target] = [path.name for path in chosen]
        for path in chosen:
            summary = validate_episode(path, target)
            task_counts[summary["task"]] += 1
            link = REVIEWED / path.name
            if not link.exists():
                link.symlink_to(path, target_is_directory=True)
    expected_names = {name for names in selected.values() for name in names}
    for path in REVIEWED.iterdir():
        if EPISODE_PATTERN.match(path.name) and path.name not in expected_names:
            path.unlink()
    manifest = {
        "source": str(RAW),
        "per_object": PER_OBJECT,
        "objects": selected,
        "task_counts": dict(task_counts),
    }
    (REVIEWED / "selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return selected


def run_conversion() -> None:
    if LEROBOT.exists():
        log(f"using existing LeRobot dataset: {LEROBOT}")
        return
    command = [
        CONDA, "run", "--no-capture-output", "-n", "unifolm-vla", "python",
        str(EXPORTER), "--input", str(REVIEWED), "--output", str(LEROBOT),
        "--repo-id", "lp531/z1_smolvla_tabletop_4x100",
    ]
    subprocess.run(command, check=True)


def verify_conversion() -> None:
    code = (
        "from lerobot.datasets.lerobot_dataset import LeRobotDataset;"
        f"d=LeRobotDataset('lp531/z1_smolvla_tabletop_4x100',root=r'{LEROBOT}');"
        "assert d.num_episodes==400,d.num_episodes;"
        "assert d.num_frames>0;"
        "print(f'episodes={d.num_episodes} frames={d.num_frames} fps={d.fps}')"
    )
    subprocess.run(
        [CONDA, "run", "--no-capture-output", "-n", "unifolm-vla", "python", "-c", code],
        check=True,
    )


def stop_simulation() -> None:
    for pid in matching_pids(
        "/opt/ros/humble/bin/ros2 launch z1_isaac_bridge z1_isaac.launch.py"
    ):
        try:
            os.kill(pid, signal.SIGINT)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 90
    while matching_pids("run_z1_ros2_scene.py") and time.monotonic() < deadline:
        time.sleep(2)
    for pid in matching_pids("run_z1_ros2_scene.py"):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def run_training() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    output = RUNS / f"z1_4x100_smolvla_{datetime.now():%Y%m%d_%H%M%S}"
    command = [
        CONDA, "run", "--no-capture-output", "-n", "unifolm-vla", "lerobot-train",
        "--policy.path=lerobot/smolvla_base",
        "--policy.device=cuda",
        "--policy.use_amp=true",
        "--policy.push_to_hub=false",
        "--dataset.repo_id=lp531/z1_smolvla_tabletop_4x100",
        f"--dataset.root={LEROBOT}",
        "--dataset.video_backend=pyav",
        f"--output_dir={output}",
        "--job_name=z1_tabletop_4x100_smolvla",
        "--batch_size=4",
        "--num_workers=4",
        "--steps=100000",
        "--log_freq=200",
        "--save_freq=10000",
        "--wandb.enable=false",
    ]
    with TRAIN_LOG.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "PYTHONUNBUFFERED": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            },
            start_new_session=True,
        )
        TRAIN_PID.write_text(f"{process.pid}\n", encoding="utf-8")
        save_state("training", pid=process.pid, output_dir=str(output), command=command)
        return_code = process.wait()
    if return_code != 0:
        save_state("training_failed", return_code=return_code, output_dir=str(output))
        raise RuntimeError(f"SmolVLA training exited with code {return_code}")
    save_state("complete", return_code=0, output_dir=str(output))
    log(f"SmolVLA training complete: {output}")


def main() -> int:
    save_state("waiting_for_cup")
    last_count = -1
    while True:
        episodes = successful_episodes()
        count = len(episodes["cup"])
        if count != last_count:
            log(f"cup successful episodes: {count}/{PER_OBJECT}")
            last_count = count
        if count >= PER_OBJECT:
            break
        time.sleep(15)
    save_state("stopping_collection", cup_count=last_count)
    stop_collection()
    selected = prepare_reviewed()
    save_state("converting", counts={name: len(paths) for name, paths in selected.items()})
    run_conversion()
    verify_conversion()
    save_state("converted", dataset=str(LEROBOT))
    stop_simulation()
    run_training()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        save_state("failed", error=repr(exc))
        log(f"fatal: {exc!r}")
        raise
