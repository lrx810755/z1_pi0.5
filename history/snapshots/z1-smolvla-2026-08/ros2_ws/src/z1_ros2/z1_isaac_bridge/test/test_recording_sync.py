import pytest

from z1_isaac_bridge.recording_sync import RecordingSynchronizer


def populated():
    sync = RecordingSynchronizer(("wrist", "global", "vla_wrist"))
    for name, stamp in (("wrist", .99), ("global", 1.01), ("vla_wrist", 1.0)):
        sync.push(name, stamp, name + " image")
        sync.push(name, 1.2, "newest but wrong time")
    sync.push("state", .98, [0.0] * 7)
    sync.push("state", 1.02, [4.0] * 7)
    sync.push("action", .99, [3.0] * 7)
    sync.push("action", 1.01, [99.0] * 7)
    sync.push("status", .98, {"phase": "READY"})
    sync.push("status", 1.1, {"phase": "GRASPING"})
    return sync


def test_fixed_time_interpolation_and_causal_action():
    sample = populated().select(1.0)
    assert sample["state"] == pytest.approx([2.0] * 7)
    assert sample["action"] == (.99, [3.0] * 7)
    assert sample["images"]["wrist"] == (.99, "wrist image")
    assert sample["status"]["phase"] == "READY"


def test_reject_skew_and_duplicate_images():
    sync = populated()
    sync.buffers["global"].clear()
    sync.push("global", 1.383, "bad")
    with pytest.raises(ValueError, match="global"):
        sync.select(1.0)
    sync = populated()
    sync.commit(sync.select(1.0))
    with pytest.raises(ValueError, match="fresh image"):
        sync.select(1.0)


def test_reject_missing_state_bracket_and_stale_action():
    sync = populated()
    sync.buffers["state"].pop()
    with pytest.raises(ValueError, match="bracketing"):
        sync.select(1.0)
    sync = populated()
    sync.buffers["action"].clear()
    sync.push("action", .8, [3.0] * 7)
    with pytest.raises(ValueError, match="causal command"):
        sync.select(1.0)


def test_reset_allows_clock_restart_and_buffers_are_bounded():
    sync = populated()
    sync.reset()
    sync.push("wrist", 0.0, "new simulation")
    sync.push("wrist", 0.0, "duplicate")
    assert list(sync.buffers["wrist"]) == [(0.0, "new simulation")]
    for i in range(1, 400):
        sync.push("wrist", i / 30, i)
    assert len(sync.buffers["wrist"]) == 64


def test_episode_sync_failure_blocks_success_save():
    from types import SimpleNamespace
    import threading
    from z1_isaac_bridge.episode_recorder import EpisodeRecorder
    recorder = object.__new__(EpisodeRecorder)
    recorder._lock = threading.RLock()
    recorder._sync_failure = "missing image at fixed grid time"
    response = SimpleNamespace(success=True, message="")
    recorder._save_callback(None, response)
    assert not response.success
    assert "时间对齐失败" in response.message


def test_recorder_writes_fixed_grid_and_latches_missing_sample(tmp_path):
    import json
    import threading
    from types import SimpleNamespace
    import numpy as np
    from z1_isaac_bridge.episode_recorder import EpisodeRecorder

    node = object.__new__(EpisodeRecorder)
    node._lock = threading.RLock()
    node._sync = RecordingSynchronizer(("wrist", "global"))
    node.camera_names = ("wrist", "global")
    node.sync_delay = .2
    node.sync_tolerance = .025
    node.fps = 10
    node._recording = True
    node._sync_failure = ""
    node._last_sync_clock = 0
    node._start_sim_time = 0
    node._frame_count = 0
    node._dropped_frames = 0
    node._target_object = "apple"
    node._instruction = "pick apple"
    node._task_status = {"target_object": "apple", "phase": "READY"}
    node._episode_dir = tmp_path
    node.jpeg_quality = 95
    node._missing_streams = lambda: []
    node._publish_status_locked = lambda: None
    node.get_logger = lambda: SimpleNamespace(error=lambda message: None)
    clock = SimpleNamespace(nanoseconds=300000000)
    node.get_clock = lambda: SimpleNamespace(now=lambda: clock)
    for name in node.camera_names:
        (tmp_path / "images" / name).mkdir(parents=True)
    with (tmp_path / "frames.jsonl").open("w") as stream:
        node._frames_file = stream
        for stamp in (.08, .1, .12, .18, .2, .22):
            for name in node.camera_names:
                node._sync.push(name, stamp, np.ones((2, 2, 3), dtype=np.uint8))
            node._sync.push("state", stamp, [stamp] * 7)
            node._sync.push("action", stamp, [stamp + 1] * 7)
            node._sync.push("status", stamp, dict(node._task_status))
        node._sample_callback()
        clock.nanoseconds = 400000000
        node._sample_callback()
        assert node._frame_count == 2
        clock.nanoseconds = 500000000
        node._sample_callback()  # no images at .3, must not compress time
        assert node._sync_failure
        assert node._frame_count == 2
        clock.nanoseconds = 600000000
        node._sample_callback()
        assert node._frame_count == 2
    records = [json.loads(line) for line in (tmp_path / "frames.jsonl").read_text().splitlines()]
    assert [record["simulation_time"] for record in records] == pytest.approx([.1, .2])
    assert [record["timestamp"] for record in records] == pytest.approx([0, .1])
    assert records[0]["joint_state_source_stamps"] == [.1, .1]
    assert records[0]["action"] == pytest.approx([1.1] * 7)
