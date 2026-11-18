"""Select equal successful episode quotas from every target-position bin."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

from .grasp_config import GRASP_CONFIG


DEFAULT_OBJECTS = ("apple", "peach", "beverage_bottle", "cup")


def grid_bin(position, workspace, shape: tuple[int, int]) -> tuple[int, int]:
    (x_min, x_max), (y_min, y_max) = workspace
    x, y = (float(position[0]), float(position[1]))
    if not x_min <= x <= x_max or not y_min <= y <= y_max:
        raise ValueError(f"target position {(x, y)} is outside workspace {workspace}")
    columns, rows = shape
    column = min(columns - 1, int((x - x_min) / (x_max - x_min) * columns))
    row = min(rows - 1, int((y - y_min) / (y_max - y_min) * rows))
    return column, row


def quota_per_bin(episodes: int, shape: tuple[int, int]) -> int:
    bins = shape[0] * shape[1]
    if episodes <= 0 or episodes % bins:
        raise ValueError(f"episode count must be positive and divisible by {bins}")
    return episodes // bins


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def collection_progress(
    source: Path,
    quota_per_cell: int,
    objects: Sequence[str] = DEFAULT_OBJECTS,
) -> dict:
    config = GRASP_CONFIG["scene_randomization"]
    workspace = (tuple(config["workspace_x_m"]), tuple(config["workspace_y_m"]))
    shape = tuple(config["levels"]["full"]["target_grid_shape"])
    counts = Counter()
    for episode in sorted(source.glob("episode_[0-9][0-9][0-9][0-9][0-9][0-9]")):
        summary_path = episode / "episode_summary.json"
        if not summary_path.is_file():
            continue
        summary = _read(summary_path)
        target = summary.get("target_object")
        if target not in objects or summary.get("task_success") is not True:
            continue
        randomization = summary.get("scene_randomization", {})
        cell = randomization.get("target_grid")
        if not isinstance(cell, (list, tuple)) or len(cell) != 2:
            position = randomization.get("object_positions_m", {}).get(target)
            if position is None:
                continue
            cell = grid_bin(position, workspace, shape)
        column, row = (int(cell[0]), int(cell[1]))
        if not 0 <= column < shape[0] or not 0 <= row < shape[1]:
            continue
        counts[(target, column, row)] += 1

    cells = {}
    totals = dict.fromkeys(objects, 0)
    for row in range(shape[1]):
        for column in range(shape[0]):
            key = f"{column},{row}"
            cells[key] = {}
            for target in objects:
                count = counts[(target, column, row)]
                cells[key][target] = count
                totals[target] += count
    return {
        "source": str(source.resolve()),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "target_grid_shape": list(shape),
        "quota_per_cell": int(quota_per_cell),
        "cells": cells,
        "totals": totals,
    }


def write_collection_progress(
    source: Path,
    state_file: Path,
    quota_per_cell: int,
) -> dict:
    progress = collection_progress(source, quota_per_cell)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = state_file.with_suffix(state_file.suffix + ".tmp")
    temporary.write_text(
        json.dumps(progress, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(state_file)
    return progress


def select_balanced(
    source: Path,
    output: Path,
    episodes_per_object: int,
    objects: Sequence[str] = DEFAULT_OBJECTS,
) -> dict:
    config = GRASP_CONFIG["scene_randomization"]
    workspace = (tuple(config["workspace_x_m"]), tuple(config["workspace_y_m"]))
    shape = tuple(config["levels"]["full"]["target_grid_shape"])
    quota = quota_per_bin(episodes_per_object, shape)
    grouped = {name: defaultdict(list) for name in objects}
    for episode in sorted(source.glob("episode_[0-9][0-9][0-9][0-9][0-9][0-9]")):
        summary_path = episode / "episode_summary.json"
        if not summary_path.is_file():
            continue
        summary = _read(summary_path)
        target = summary.get("target_object")
        if target not in grouped or summary.get("task_success") is not True:
            continue
        position = summary.get("scene_randomization", {}).get("object_positions_m", {}).get(target)
        if position is None:
            continue
        grouped[target][grid_bin(position, workspace, shape)].append(episode)

    missing = {}
    selected = {}
    for target in objects:
        chosen = []
        for row in range(shape[1]):
            for column in range(shape[0]):
                available = grouped[target][(column, row)]
                if len(available) < quota:
                    missing[f"{target}:{column},{row}"] = {"need": quota, "have": len(available)}
                chosen.extend(available[:quota])
        selected[target] = chosen
    if missing:
        raise ValueError("insufficient successful episodes per spatial bin: " + json.dumps(missing, sort_keys=True))
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)
    (output / "dataset_info.json").symlink_to((source / "dataset_info.json").resolve())
    manifest_objects = {}
    task_counts = Counter()
    for target, episodes in selected.items():
        manifest_objects[target] = [episode.name for episode in episodes]
        for episode in episodes:
            summary = _read(episode / "episode_summary.json")
            task_counts[summary["task"]] += 1
            (output / episode.name).symlink_to(episode.resolve(), target_is_directory=True)
    manifest = {
        "source": str(source.resolve()),
        "per_object": episodes_per_object,
        "workspace_m": {"x": list(workspace[0]), "y": list(workspace[1])},
        "target_grid_shape": list(shape),
        "quota_per_bin": quota,
        "objects": manifest_objects,
        "task_counts": dict(task_counts),
    }
    (output / "selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-object", type=int, default=200)
    args = parser.parse_args()
    manifest = select_balanced(args.source.expanduser(), args.output.expanduser(), args.per_object)
    print(f"balanced dataset ready: {args.output} ({sum(len(v) for v in manifest['objects'].values())} episodes)")


def progress_main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan saved episodes and update resumable collection progress."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--quota", type=int, required=True)
    parser.add_argument("--target", choices=DEFAULT_OBJECTS, required=True)
    parser.add_argument("--column", type=int, required=True)
    parser.add_argument("--row", type=int, required=True)
    args = parser.parse_args()
    progress = write_collection_progress(
        args.source.expanduser(), args.state_file.expanduser(), args.quota
    )
    key = f"{args.column},{args.row}"
    try:
        count = progress["cells"][key][args.target]
    except KeyError as exc:
        raise SystemExit(f"grid cell {key} is outside configured shape") from exc
    print(count, flush=True)


if __name__ == "__main__":
    main()
