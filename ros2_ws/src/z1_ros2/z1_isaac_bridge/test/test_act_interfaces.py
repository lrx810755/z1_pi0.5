import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


REPO = Path(__file__).resolve().parents[2]
EXPORTER = REPO / "isaac_assets/z1/scripts/export_z1_lerobot_dataset.py"
TRAINER = REPO / "scripts/train_z1_act.sh"


@pytest.fixture
def exporter(monkeypatch):
    monkeypatch.syspath_prepend(str(EXPORTER.parent))
    spec = importlib.util.spec_from_file_location("act_exporter", EXPORTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_act_export_selects_only_successful_target_episodes(tmp_path, exporter):
    for index, target, success in (
        (0, "apple", True),
        (1, "peach", True),
        (2, "apple", False),
    ):
        episode = tmp_path / f"episode_{index:06d}"
        episode.mkdir()
        (episode / "episode_summary.json").write_text(
            json.dumps({"target_object": target, "task_success": success})
        )
        (episode / "frames.jsonl").write_text("{}\n")

    selected = exporter.select_episodes(tmp_path, False, "apple")

    assert [path.name for path in selected] == ["episode_000000"]


def test_act_export_uses_only_requested_camera(exporter):
    info = {
        "observation_keys": [
            "observation.images.wrist",
            "observation.images.vla_wrist",
            "observation.images.global",
            "observation.state",
        ]
    }

    assert exporter.camera_keys_from_metadata(info, ("wrist",)) == (
        "observation.images.wrist",
    )
    with pytest.raises(ValueError, match="camera.*missing"):
        exporter.camera_keys_from_metadata(info, ("overhead",))


def test_act_train_dry_run_targets_one_dataset_without_starting_training(tmp_path):
    dataset_base = tmp_path / "datasets"
    dataset_root = dataset_base / "z1_tabletop_lerobot_act_apple_v1"
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "meta/info.json").write_text("{}")
    (dataset_root / "meta/episodes.jsonl").write_text(
        '{"episode_index":0,"tasks":["apple"],"length":10}\n'
    )
    env = os.environ.copy()
    env["ACT_DATASET_BASE"] = str(dataset_base)
    env["ACT_TRAINING_ROOT"] = str(tmp_path / "training")

    result = subprocess.run(
        ["bash", str(TRAINER), "apple", "--dry-run"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--policy.type=act" in result.stdout
    assert f"--dataset.root={dataset_root}" in result.stdout
    assert "--dataset.repo_id=lp531/z1_act_apple_v1" in result.stdout
    assert "--policy.chunk_size=30" in result.stdout
    assert "--policy.n_action_steps=10" in result.stdout
    assert "--batch_size=2" in result.stdout
    assert not (tmp_path / "training").exists()
