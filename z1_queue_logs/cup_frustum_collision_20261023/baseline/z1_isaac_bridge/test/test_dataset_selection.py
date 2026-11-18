import json

from z1_isaac_bridge.dataset_selection import (
    collection_progress,
    grid_bin,
    quota_per_bin,
    select_balanced,
    write_collection_progress,
)


def test_grid_bin_covers_shared_workspace_edges():
    workspace = ((0.36, 0.69), (-0.27, 0.16))

    assert grid_bin((0.36, -0.27), workspace, (5, 5)) == (0, 0)
    assert grid_bin((0.69, 0.16), workspace, (5, 5)) == (4, 4)
    assert grid_bin((0.525, -0.055), workspace, (5, 5)) == (2, 2)


def test_quota_is_even_and_rejects_non_divisible_episode_count():
    assert quota_per_bin(100, (5, 5)) == 4

    try:
        quota_per_bin(101, (5, 5))
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("uneven spatial quota must be rejected")


def test_balanced_selection_requires_every_object_in_every_bin(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    (source / "dataset_info.json").write_text("{}")
    objects = ("apple", "peach", "beverage_bottle", "cup")
    index = 0
    for target in objects:
        for row in range(5):
            for column in range(5):
                episode = source / f"episode_{index:06d}"
                episode.mkdir()
                x = 0.36 + (column + 0.5) * (0.69 - 0.36) / 5
                y = -0.27 + (row + 0.5) * (0.16 + 0.27) / 5
                summary = {
                    "target_object": target,
                    "task_success": True,
                    "task": f"grasp {target}",
                    "scene_randomization": {"object_positions_m": {target: [x, y, 0.0]}},
                }
                (episode / "episode_summary.json").write_text(json.dumps(summary))
                index += 1

    output = tmp_path / "selected"
    manifest = select_balanced(source, output, 25)

    assert manifest["quota_per_bin"] == 1
    assert all(len(episodes) == 25 for episodes in manifest["objects"].values())
    assert len(list(output.glob("episode_*"))) == 100


def test_collection_progress_resumes_from_saved_successes(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    summaries = [
        {
            "target_object": "apple",
            "task_success": True,
            "scene_randomization": {"target_grid": [2, 3]},
        },
        {
            "target_object": "apple",
            "task_success": True,
            "scene_randomization": {
                "object_positions_m": {"apple": [0.525, 0.031, 0.0]}
            },
        },
        {
            "target_object": "apple",
            "task_success": False,
            "scene_randomization": {"target_grid": [2, 3]},
        },
    ]
    for index, summary in enumerate(summaries):
        episode = source / f"episode_{index:06d}"
        episode.mkdir()
        (episode / "episode_summary.json").write_text(json.dumps(summary))

    progress = collection_progress(source, quota_per_cell=8)
    state = tmp_path / "progress.json"
    write_collection_progress(source, state, quota_per_cell=8)

    assert progress["cells"]["2,3"]["apple"] == 2
    assert progress["totals"]["apple"] == 2
    assert json.loads(state.read_text())["cells"]["2,3"]["apple"] == 2
