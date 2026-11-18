import json

import pytest

from z1_isaac_bridge.smolvla_voice_core import (
    InvalidTransition,
    VoiceCommand,
    VoiceSession,
    decode_command,
    match_target,
)


@pytest.mark.parametrize(
    ("text", "parsed_object", "expected"),
    [
        ("抓取杯子", "cup", "cup"),
        ("把苹果抓起来", "apple", "apple"),
        ("请抓桃子", "peach", "peach"),
        ("抓取饮料瓶", "bottle", "beverage_bottle"),
        ("pick up the beverage bottle", "beverage bottle", "beverage_bottle"),
        ("pick up the drinking bottle", "drinking bottle", "beverage_bottle"),
    ],
)
def test_match_target_maps_voice_aliases_to_training_targets(
    text, parsed_object, expected
):
    assert match_target(text, parsed_object) == expected


def test_match_target_rejects_unsupported_and_ambiguous_objects():
    assert match_target("抓取鼠标", "mouse") is None
    assert match_target("pick the cupboard", "cupboard") is None
    assert match_target("抓苹果还是杯子", "none") is None


def test_decode_command_accepts_only_the_backend_protocol():
    assert decode_command('{"command":"start"}') == VoiceCommand("start")
    assert decode_command(json.dumps({
        "command": "grasp", "target": "beverage_bottle"
    })) == VoiceCommand("grasp", "beverage_bottle")
    assert decode_command({"command": "stop"}) == VoiceCommand("stop")
    for payload in (
        "not json",
        "[]",
        "null",
        '{"command":"grasp","target":"mouse"}',
        '{"command":"move","target":"cup"}',
    ):
        with pytest.raises(ValueError):
            decode_command(payload)


def test_voice_session_requires_start_before_one_grasp_attempt():
    session = VoiceSession()
    assert session.state == "IDLE"
    assert session.accept(VoiceCommand("start")) == "randomize"
    assert session.state == "RANDOMIZING"
    session.randomized(123)
    assert session.state == "LISTENING"
    assert session.accept(VoiceCommand("grasp", "cup")) == "execute"
    assert session.state == "EXECUTING"
    session.episode_done()
    assert session.state == "IDLE"
    assert session.seed is None
    with pytest.raises(InvalidTransition):
        session.accept(VoiceCommand("grasp", "apple"))


def test_voice_session_stop_cancels_waiting_or_requests_execution_stop():
    waiting = VoiceSession()
    waiting.accept(VoiceCommand("start"))
    waiting.randomized(456)
    assert waiting.accept(VoiceCommand("stop")) == "cancel"
    assert waiting.state == "IDLE"

    executing = VoiceSession()
    executing.accept(VoiceCommand("start"))
    executing.randomized(789)
    executing.accept(VoiceCommand("grasp", "peach"))
    assert executing.accept(VoiceCommand("stop")) == "stop"
    assert executing.stop_requested
    assert executing.state == "EXECUTING"


def test_voice_session_recovers_from_randomization_failure():
    session = VoiceSession()
    session.accept(VoiceCommand("start"))
    session.randomization_failed()
    assert session.state == "IDLE"
    assert session.seed is None


def test_voice_backend_exposes_ros_entrypoint_and_immediate_stop_path():
    import inspect
    from pathlib import Path

    from z1_isaac_bridge.smolvla_voice_backend import SmolVLAVoiceBackend

    source = inspect.getsource(SmolVLAVoiceBackend._command_callback)
    assert "self._stop_event.set()" in source
    assert source.index("self._stop_event.set()") < source.index(
        "self._commands.put(command)"
    )
    setup = (Path(__file__).resolve().parents[1] / "setup.py").read_text(
        encoding="utf-8"
    )
    assert (
        'smolvla_voice_backend = '
        'z1_isaac_bridge.smolvla_voice_backend:main'
    ) in setup


def test_voice_target_uses_the_exact_training_instruction():
    from z1_isaac_bridge.smolvla_rollout_core import TASK_INSTRUCTIONS

    target = match_target("请抓取饮料瓶", "bottle")
    assert TASK_INSTRUCTIONS[target] == (
        "Physically grasp the bottle, lift it clear, and place it in the box."
    )


def test_voice_commands_are_volatile_and_start_waits_for_backend_ack():
    import inspect
    from pathlib import Path

    from z1_isaac_bridge.smolvla_voice_backend import SmolVLAVoiceBackend

    backend = inspect.getsource(SmolVLAVoiceBackend.__init__)
    frontend = Path("/home/lp531/robot/smolvla_voice_frontend.py").read_text(
        encoding="utf-8"
    )
    assert "command_qos.durability = DurabilityPolicy.VOLATILE" in backend
    assert "command_qos.durability = DurabilityPolicy.VOLATILE" in frontend
    assert "revision = frontend.revision" in frontend
    assert "frontend.wait_for_update(revision" in frontend


def test_voice_processes_shutdown_without_publishing_on_closed_context():
    import inspect
    from pathlib import Path

    from z1_isaac_bridge import smolvla_voice_backend

    backend_main = inspect.getsource(smolvla_voice_backend.main)
    frontend = Path("/home/lp531/robot/smolvla_voice_frontend.py").read_text(
        encoding="utf-8"
    )
    assert "if rclpy.ok():\n            node._set_active(False)" in backend_main
    assert 'if frontend.state in {"LISTENING", "EXECUTING"}' in frontend
    assert "if rclpy.ok():\n            rclpy.shutdown()" in frontend
