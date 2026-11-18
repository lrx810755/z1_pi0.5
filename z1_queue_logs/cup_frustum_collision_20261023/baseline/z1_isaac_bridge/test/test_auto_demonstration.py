import threading
import time
from pathlib import Path

import pytest

from z1_isaac_bridge.grasp_gui_bridge import (
    GUIDED_GRASP_TIMEOUT_S,
    GUIDED_PLACE_TIMEOUT_S,
    LAYOUT_RESAMPLE_ATTEMPTS,
    GRID_LAYOUT_RESAMPLE_ATTEMPTS,
    GraspGuiBridge,
    _Handler,
)


def make_bridge(recording=False):
    bridge = object.__new__(GraspGuiBridge)
    bridge._lock = threading.Lock()
    bridge._auto_lock = threading.Lock()
    bridge._recording_status = {"recording": recording}
    bridge._recording_status["ready"] = True
    bridge._episode_reset_status = {"ready": True}
    bridge._collision_scene_status = {"ready": True}
    bridge._object_collision_status = {"ready": True}
    bridge._last_status_monotonic = time.monotonic()
    bridge.status_timeout = 1.0
    bridge._auto_status = {}
    bridge._force_stop_event = threading.Event()
    bridge._force_stop_status = {}
    bridge.calls = []
    bridge._stop_motion_client = object()

    def record(name, result=None):
        def operation(*_args):
            bridge.calls.append(name)
            return result or {"success": True, "message": name}

        return operation

    bridge._call_trigger = lambda *args, **kwargs: record("record_hold")()
    bridge.set_target = record("target")
    bridge.reset_episode = record("reset")
    bridge.randomize_episode = record("randomize")
    bridge._generate_pregrasp_for_auto = record("generate")
    bridge.start_recording = record("record_start")
    bridge._execute_pregrasp_for_auto = record("execute")
    bridge._guided_grasp_for_auto = record("grasp")
    bridge._guided_place_for_auto = record("place")
    bridge.save_recording = record(
        "save", {"success": True, "message": "episode saved"}
    )
    bridge.discard_recording = record("discard")
    return bridge


def test_auto_demonstration_runs_and_saves_in_order():
    bridge = make_bridge()

    result = bridge.auto_demonstration("apple")

    assert result["success"] is True
    assert bridge.calls == [
        "target",
        "randomize",
        "generate",
        "record_hold",
        "record_start",
        "execute",
        "grasp",
        "place",
        "save",
    ]
    assert bridge._auto_status["phase"] == "SUCCESS"
    assert bridge._auto_status["running"] is False


def test_auto_demonstration_forwards_curriculum_level():
    bridge = make_bridge()
    selected_levels = []

    def randomize(level, target_grid=None):
        bridge.calls.append("randomize")
        selected_levels.append(level)
        return {"success": True, "message": level}

    bridge.randomize_episode = randomize
    bridge.auto_demonstration("apple", "medium")

    assert selected_levels == ["medium"]


def test_auto_demonstration_forwards_same_grid_to_every_layout_resample():
    bridge = make_bridge()
    selected = []
    attempts = 0

    def randomize(level, target_grid=None):
        bridge.calls.append("randomize")
        selected.append((level, target_grid))
        return {"success": True, "message": level}

    def generate():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("blocked")
        return {"success": True, "message": "clear"}

    bridge.randomize_episode = randomize
    bridge._generate_pregrasp_for_auto = generate
    bridge.auto_demonstration("apple", "full", (2, 3))

    assert selected == [("full", (2, 3)), ("full", (2, 3))]


def test_auto_demonstration_resamples_blocked_layout_before_recording():
    bridge = make_bridge()
    attempts = 0

    def generate():
        nonlocal attempts
        attempts += 1
        bridge.calls.append("generate")
        if attempts < LAYOUT_RESAMPLE_ATTEMPTS:
            raise RuntimeError("夹爪扫掠通道被占用")
        return {"success": True, "message": "clear"}

    bridge._generate_pregrasp_for_auto = generate
    result = bridge.auto_demonstration("apple", "easy")

    assert result["success"] is True
    assert bridge.calls[:7] == [
        "target",
        "randomize",
        "generate",
        "randomize",
        "generate",
        "randomize",
        "generate",
    ]
    assert bridge.calls[7:9] == ["record_hold", "record_start"]


def test_auto_demonstration_keeps_resampling_until_force_stop():
    bridge = make_bridge()
    attempts = 0

    def generate():
        nonlocal attempts
        attempts += 1
        bridge.calls.append("generate")
        if attempts >= 6:
            bridge._force_stop_event.set()
        raise RuntimeError("夹爪扫掠通道被占用：mouse(夹爪接近扫掠通道)")

    bridge._generate_pregrasp_for_auto = generate
    with pytest.raises(RuntimeError, match="强制停止"):
        bridge.auto_demonstration("cup", "full")

    assert attempts == 6


def test_grid_locked_demonstration_keeps_resampling_until_force_stop():
    bridge = make_bridge()
    attempts = 0

    def generate():
        nonlocal attempts
        attempts += 1
        if attempts >= 6:
            bridge._force_stop_event.set()
        raise RuntimeError("unreachable")

    bridge._generate_pregrasp_for_auto = generate
    with pytest.raises(RuntimeError, match="强制停止"):
        bridge.auto_demonstration("cup", "full", (1, 4))

    assert attempts == 6


def test_grid_locked_demonstration_marks_persistently_unreachable_grid():
    bridge = make_bridge()
    attempts = 0

    def generate():
        nonlocal attempts
        attempts += 1
        raise RuntimeError("MoveIt error_code=99999")

    bridge._generate_pregrasp_for_auto = generate
    with pytest.raises(RuntimeError, match="GRID_UNREACHABLE"):
        bridge.auto_demonstration("beverage_bottle", "full", (0, 0))

    assert attempts == GRID_LAYOUT_RESAMPLE_ATTEMPTS


def test_curriculum_level_rejects_unknown_value():
    assert GraspGuiBridge._normalize_curriculum_level(" FULL ") == "full"
    with pytest.raises(ValueError, match="unknown curriculum_level"):
        GraspGuiBridge._normalize_curriculum_level("extreme")


def test_auto_demonstration_discards_but_does_not_reset_after_failure():
    bridge = make_bridge()

    def fail_grasp():
        bridge.calls.append("grasp")
        raise RuntimeError("grasp failed")

    bridge._guided_grasp_for_auto = fail_grasp

    with pytest.raises(RuntimeError, match="grasp failed"):
        bridge.auto_demonstration("apple")

    assert bridge.calls == [
        "target",
        "randomize",
        "generate",
        "record_hold",
        "record_start",
        "execute",
        "grasp",
        "discard",
    ]
    assert bridge._auto_status["phase"] == "ERROR"
    assert bridge._auto_status["running"] is False
    assert "手动点击“一键重置 Episode”" in bridge._auto_status["message"]


def test_auto_demonstration_does_not_destroy_existing_recording():
    bridge = make_bridge(recording=True)

    with pytest.raises(RuntimeError, match="已有 Episode"):
        bridge.auto_demonstration("apple")

    assert bridge.calls == []
    assert bridge._auto_status["phase"] == "ERROR"


def test_guided_timeouts_cover_slow_physical_lift_and_place():
    # The latest verified apple lift took about 88 wall-clock seconds.
    assert GUIDED_GRASP_TIMEOUT_S >= 600.0
    assert GUIDED_PLACE_TIMEOUT_S >= 600.0


def test_force_stop_cancels_assistant_and_latches_motion_controller():
    bridge = make_bridge()
    bridge._stop_assistant_client = object()
    bridge._stop_motion_client = object()
    services = []

    def call_trigger(
        _client,
        service_name,
        timeout_s,
        ignore_force_stop=False,
    ):
        services.append((service_name, timeout_s, ignore_force_stop))
        return {"success": True, "message": f"stopped {service_name}"}

    bridge._call_trigger = call_trigger
    result = bridge.force_stop()

    assert result["success"] is True
    assert bridge._force_stop_event.is_set()
    assert [service[0] for service in services] == [
        "/smolvla/stop_assistant",
        "/smolvla/stop_motion",
    ]
    assert all(service[2] is True for service in services)
    assert bridge._force_stop_status["phase"] == "STOPPED"
    assert bridge._auto_status["phase"] == "STOPPED"


def test_staged_pregrasp_and_fine_adjustment_are_not_web_features():
    package_root = Path(__file__).resolve().parents[1]
    frontend = (package_root / "web" / "smolvla_task_gui.html").read_text()
    backend = (package_root / "z1_isaac_bridge" / "grasp_gui_bridge.py").read_text()
    assistant = (package_root / "z1_isaac_bridge" / "grasp_assistant.py").read_text()
    reset = (package_root / "z1_isaac_bridge" / "episode_reset.py").read_text()
    handler = _Handler.do_POST

    assert "半自动预抓取" not in frontend
    assert "末端精调" not in frontend
    assert 'id="generate"' not in frontend
    assert 'data-jog=' not in frontend
    assert "/api/manual" not in backend
    assert "/api/pregrasp/" not in backend
    assert "/api/guided/" not in backend
    assert "/api/demonstration/start" not in backend
    assert 'id="forceStop"' in frontend
    assert "/api/force_stop" in backend
    assert "/smolvla/stop_assistant" in backend
    assert "/smolvla/stop_motion" in backend
    assert "/smolvla/manual_command" not in backend
    assert "/smolvla/manual_command" not in assistant
    assert "/smolvla/manual_command" not in reset
    assert '"/smolvla/generate_pregrasp"' not in assistant
    assert '"/smolvla/execute_pregrasp"' not in assistant
    assert '"/smolvla/execute_guided_pregrasp"' not in assistant
    assert not hasattr(GraspGuiBridge, "manual_command")
    assert handler is not None


def test_web_gui_exposes_curriculum_randomization_levels():
    package_root = Path(__file__).resolve().parents[1]
    frontend = (package_root / "web" / "smolvla_task_gui.html").read_text()

    assert 'id="curriculumLevel"' in frontend
    for level in ("easy", "medium", "full"):
        assert f'value="{level}"' in frontend
    assert "curriculum_level" in frontend


def test_web_gui_preserves_unsubmitted_target_selection_during_refresh():
    package_root = Path(__file__).resolve().parents[1]
    frontend = (package_root / "web" / "smolvla_task_gui.html").read_text()

    assert "let targetSelectionDirty = false;" in frontend
    assert '$("targetSelect").onchange=()=>{targetSelectionDirty=true;};' in frontend
    assert "if(!targetSelectionDirty" in frontend
    assert frontend.count("targetSelectionDirty=false;") == 2
