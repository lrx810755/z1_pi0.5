import math
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from z1_isaac_bridge.grasp_assistant import (
    GraspAssistant,
    _bottle_recovery_pregrasp_position,
)


def test_recovery_waypoint_is_bounded_and_lateral_to_approach():
    position = (0.4, -0.1, 0.2)
    for azimuth in (0.0, 45.0, 90.0, 180.0):
        left = _bottle_recovery_pregrasp_position(position, azimuth, "left")
        right = _bottle_recovery_pregrasp_position(position, azimuth, "right")
        for shifted in (left, right):
            delta = tuple(a - b for a, b in zip(shifted, position))
            assert math.hypot(*delta[:2]) == pytest.approx(0.008)
            assert delta[2] == 0
            heading = math.radians(azimuth)
            assert delta[0] * math.cos(heading) + delta[1] * math.sin(heading) == pytest.approx(0)
        assert tuple(a + b for a, b in zip(left, right)) == pytest.approx(
            tuple(2 * value for value in position)
        )
    assert _bottle_recovery_pregrasp_position(position, 45, "high") == pytest.approx(
        (0.4, -0.1, 0.206)
    )
    assert _bottle_recovery_pregrasp_position(position, 45, "baseline") == position
    with pytest.raises(ValueError):
        _bottle_recovery_pregrasp_position(position, 0, "unknown")


def test_recovery_launcher_rejects_unknown_mode_before_starting_scene():
    launcher = Path("/home/lp531/start_smolvla_bottle_recovery_collection.sh")
    result = subprocess.run(
        ["bash", str(launcher), "unknown"], capture_output=True, text=True
    )
    assert result.returncode == 2
    assert "left|right|high" in result.stderr


def test_physics_ready_satisfies_unfreeze_acknowledgement():
    node = SimpleNamespace(
        _lock=threading.Lock(),
        _scene_grasp_status={"phase": "PHYSICS_READY"},
        _raise_if_stop_requested=Mock(),
    )
    assert GraspAssistant._wait_scene_grasp(node, "UNFROZEN", "", 0.01)
