#!/usr/bin/env python3
"""Create a state-augmented LeRobot dataset without re-encoding videos."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "z1_isaac_bridge"))

from z1_isaac_bridge.smolvla_state import (  # noqa: E402
    AUGMENTED_STATE_NAMES,
    augmented_state,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--repo-id", default="lp531/z1_smolvla_tabletop_4x100_state13"
    )
    return parser.parse_args()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def raw_episode_directories(raw: Path) -> list[Path]:
    episodes = sorted(raw.glob("episode_[0-9][0-9][0-9][0-9][0-9][0-9]"))
    if not episodes:
        raise ValueError(f"no raw episodes in {raw}")
    return episodes


def read_augmented_states(episode: Path) -> np.ndarray:
    values = []
    with (episode / "frames.jsonl").open(encoding="utf-8") as stream:
        for expected_index, line in enumerate(stream):
            frame = json.loads(line)
            if int(frame["frame_index"]) != expected_index:
                raise ValueError(f"non-contiguous frames in {episode}")
            values.append(
                augmented_state(frame["observation.state"], frame)
            )
    return np.asarray(values, dtype=np.float32)


def state_stats(values: np.ndarray) -> dict[str, list]:
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0, dtype=np.float64).tolist(),
        "std": values.std(axis=0, dtype=np.float64).tolist(),
        "count": [int(len(values))],
    }


def replace_state_column(path: Path, values: np.ndarray) -> None:
    table = pq.read_table(path)
    if len(table) != len(values):
        raise ValueError(f"frame count mismatch for {path}: {len(table)} != {len(values)}")
    flat = pa.array(values.reshape(-1), type=pa.float32())
    state_column = pa.FixedSizeListArray.from_arrays(flat, values.shape[1])
    table = table.set_column(
        table.schema.get_field_index("observation.state"),
        "observation.state",
        state_column,
    )
    metadata = dict(table.schema.metadata or {})
    huggingface = json.loads(metadata[b"huggingface"])
    huggingface["info"]["features"]["observation.state"]["length"] = values.shape[1]
    metadata[b"huggingface"] = json.dumps(huggingface).encode()
    table = table.replace_schema_metadata(metadata)
    temporary = path.with_suffix(".parquet.tmp")
    pq.write_table(table, temporary)
    temporary.replace(path)


def main() -> None:
    args = arguments()
    source = args.source.expanduser().resolve()
    raw = args.raw.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    episodes = raw_episode_directories(raw)
    source_info = read_json(source / "meta/info.json")
    if len(episodes) != int(source_info["total_episodes"]):
        raise ValueError(
            f"episode count mismatch: {len(episodes)} != {source_info['total_episodes']}"
        )

    shutil.copytree(source, output, copy_function=os.link)
    stats_path = output / "meta/episodes_stats.jsonl"
    stats_rows = [json.loads(line) for line in stats_path.read_text().splitlines()]
    try:
        for episode_index, episode in enumerate(episodes):
            values = read_augmented_states(episode)
            parquet = output / "data/chunk-000" / f"episode_{episode_index:06d}.parquet"
            replace_state_column(parquet, values)
            if int(stats_rows[episode_index]["episode_index"]) != episode_index:
                raise ValueError("episode stats are not ordered by episode index")
            stats_rows[episode_index]["stats"]["observation.state"] = state_stats(values)
            if episode_index % 25 == 0:
                print(f"augmented {episode_index}/{len(episodes)}", flush=True)

        temporary_stats = stats_path.with_suffix(".jsonl.tmp")
        with temporary_stats.open("w", encoding="utf-8") as stream:
            for row in stats_rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary_stats.replace(stats_path)

        info_path = output / "meta/info.json"
        info = read_json(info_path)
        info["features"]["observation.state"]["shape"] = [
            len(AUGMENTED_STATE_NAMES)
        ]
        info["features"]["observation.state"]["names"] = list(
            AUGMENTED_STATE_NAMES
        )
        write_json(info_path, info)
        write_json(
            output / "meta/state_schema.json",
            {
                "version": 1,
                "repo_id": args.repo_id,
                "observation_state_names": list(AUGMENTED_STATE_NAMES),
                "source_dataset": str(source),
                "raw_dataset": str(raw),
            },
        )
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
    print(f"state-augmented dataset ready: {output}", flush=True)


if __name__ == "__main__":
    main()
