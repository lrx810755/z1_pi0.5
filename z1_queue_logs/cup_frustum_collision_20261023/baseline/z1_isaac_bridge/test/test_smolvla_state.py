import pytest

from z1_isaac_bridge.smolvla_state import (
    AUGMENTED_STATE_NAMES,
    PHASES,
    augmented_state,
)


def test_augmented_state_has_stable_13_value_schema():
    state = augmented_state(
        [0.0, 0.1, -0.2, 0.3, -0.4, 0.5, -0.9],
        {
            "phase": "PHYSICAL_TEST_LIFT",
            "grasp_success": True,
            "physical_grasp_ok": True,
            "object_touching_table": False,
            "task_success": False,
        },
    )
    assert len(AUGMENTED_STATE_NAMES) == len(state) == 13
    assert sum(state[7 : 7 + len(PHASES)]) == 1.0
    assert state[7 + PHASES.index("PHYSICAL_TEST_LIFT")] == 1.0
    assert not any(name.startswith("status.") for name in AUGMENTED_STATE_NAMES)


def test_augmented_state_rejects_unknown_phase():
    with pytest.raises(ValueError, match="unknown SmolVLA phase"):
        augmented_state([0.0] * 7, {"phase": "UNKNOWN"})
