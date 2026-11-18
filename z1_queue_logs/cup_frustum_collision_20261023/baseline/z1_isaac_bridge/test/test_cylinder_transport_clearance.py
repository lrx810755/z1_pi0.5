import math
from types import SimpleNamespace

import pytest

from z1_isaac_bridge.grasp_assistant import (
    GraspAssistant,
    TransportHeightError,
)
from z1_isaac_bridge.grasp_config import OBJECT_GRASP_PROFILES


class HeldObject:
    box_center = (.32, .14)
    box_approach_object_z = .14
    max_joint_delta = 2.4
    physical_test_lift_velocity_scale = .12

    def __init__(self, *, never_reaches=False):
        self.position = (.32, .14, .141)
        self.moves = []
        self.move_options = []
        self.probes = 0
        self.never_reaches = never_reaches

    def _current_object_position(self, target):
        return self.position

    def _require_loaded_physical_grasp(self, *args, **kwargs):
        self.probes += 1
        # Reproduce the failure: height passes before the contact probe,
        # then the bottle settles below clearance while still held.
        if self.probes == 1 or self.never_reaches:
            self.position = (.32, .14, .1345)
        return {"object_touching_table": False, "bilateral_contact": True}

    def _move_object_to(self, destination, description, **kwargs):
        self.moves.append(destination)
        self.move_options.append(kwargs)
        self.position = destination

    def get_logger(self):
        return SimpleNamespace(warning=lambda message: None)


@pytest.mark.parametrize(
    ("target", "expect_move"),
    [("apple", False), ("peach", False), ("cup", False),
     ("beverage_bottle", True)],
)
def test_non_bottle_transport_accepts_55mm_box_approach(target, expect_move):
    class HeldAtEdge(HeldObject):
        def _require_loaded_physical_grasp(self, *args, **kwargs):
            return {"object_touching_table": False, "bilateral_contact": True}

    node = HeldAtEdge()
    node.position = (0.3654, 0.14, 0.1487)
    GraspAssistant._carry_loaded_object_toward_box(
        node, target, OBJECT_GRASP_PROFILES.get(target, {}).get("loaded_envelope", {}),
        0.130, "clearance test", require_box_approach=True,
    )
    assert bool(node.moves) is expect_move


@pytest.mark.parametrize("target", ["beverage_bottle", "cup"])
def test_bottle_and_cup_remeasure_after_contact_check_and_correct_height(target):
    profile = OBJECT_GRASP_PROFILES[target]
    node = HeldObject()
    result = GraspAssistant._carry_loaded_object_toward_box(
        node, target, profile["loaded_envelope"],
        profile["physical_test_lift_safe_completion_z_m"], "clearance test",
        require_box_approach=True,
        preferred_object_z=profile["transport_object_z_m"],
    )
    assert result["bilateral_contact"]
    assert node.moves  # must not accept the pre-settling 0.141 m snapshot
    assert node.moves[0][2] == pytest.approx(.15)
    assert node.position[2] >= .14


@pytest.mark.parametrize("target", ["beverage_bottle", "cup"])
def test_cylinder_persistent_height_shortfall_is_not_retried(target):
    node = HeldObject(never_reaches=True)
    with pytest.raises(TransportHeightError, match="任务安全位"):
        GraspAssistant._carry_loaded_object_toward_box(
            node, target,
            OBJECT_GRASP_PROFILES[target]["loaded_envelope"],
            .14, "clearance test", require_box_approach=True,
            preferred_object_z=.15,
        )
    assert len(node.moves) == 1
    assert node.position[2] == pytest.approx(.1345)


@pytest.mark.parametrize(
    ("target", "expected_scale"),
    (("beverage_bottle", 0.08), ("cup", 0.08)),
)
def test_cylinders_use_configured_loaded_transport_speed(target, expected_scale):
    node = HeldObject(never_reaches=True)
    with pytest.raises(TransportHeightError):
        GraspAssistant._carry_loaded_object_toward_box(
            node,
            target,
            OBJECT_GRASP_PROFILES[target]["loaded_envelope"],
            .14,
            "clearance test",
            require_box_approach=True,
            preferred_object_z=.15,
        )
    assert node.move_options
    assert node.move_options[0]["velocity_scale"] == pytest.approx(
        expected_scale
    )


def test_bottle_transport_uses_bounded_dynamics_from_target_profile():
    node = HeldObject(never_reaches=True)

    with pytest.raises(TransportHeightError):
        GraspAssistant._carry_loaded_object_toward_box(
            node,
            "beverage_bottle",
            OBJECT_GRASP_PROFILES["beverage_bottle"]["loaded_envelope"],
            .14,
            "clearance test",
            require_box_approach=True,
            preferred_object_z=.15,
        )

    assert node.move_options
    options = node.move_options[0]
    assert options["orientation_tolerance"] == pytest.approx(.20)
    assert options["max_joint_delta"] == pytest.approx(1.70)
    assert options["velocity_scale"] == pytest.approx(.08)
    assert options["acceleration_scale"] == pytest.approx(.04)
