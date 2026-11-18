import threading
from types import SimpleNamespace

import pytest

from z1_isaac_bridge.grasp_assistant import GraspAssistant


class AtReleasePosition(Exception):
    """Stop before the real gripper and simulator release operations."""


class HeldCylinder:
    box_center = (.32, .14)
    box_approach_object_z = .14
    box_release_object_z = .115
    max_joint_delta = 2.4
    physical_test_lift_velocity_scale = .12

    def __init__(self, target, position, alignment_residual=None):
        self._lock = threading.RLock()
        self._task_status = {"physical_grasp_ok": True}
        self._verified_grasp_sequence = True
        self.target_object = target
        self.position = position
        self.moves = []
        self.alignment_residual = alignment_residual

    def _current_object_position(self, _target):
        return self.position

    def _move_object_to(self, destination, description, **kwargs):
        self.moves.append((destination, description, kwargs))
        self.position = destination
        if "水平对中" in description and self.alignment_residual is not None:
            self.position = self.alignment_residual

    def _require_loaded_physical_grasp(self, _target, _envelope, stage, **_kwargs):
        if stage in ("到达蓝盒释放位后", "杯子对中后直接释放前"):
            raise AtReleasePosition
        return {"bilateral_contact": True, "object_touching_table": False}

    def _carry_loaded_object_toward_box(self, *args, **kwargs):
        return GraspAssistant._carry_loaded_object_toward_box(self, *args, **kwargs)

    def get_logger(self):
        return SimpleNamespace(warning=lambda _message: None)


class ReleasedCylinder(HeldCylinder):
    def __init__(self, *, stable):
        super().__init__("beverage_bottle", (.32, .14, .14))
        self.stable = stable
        self._scene_grasp_status = {}
        self.control_updates = []
        self.warnings = []

    def _require_loaded_physical_grasp(self, *_args, **_kwargs):
        return {"bilateral_contact": True, "object_touching_table": False}

    def _set_status(self, *_args, **_kwargs):
        return None

    def _set_grasp_control_state(self, **kwargs):
        self.control_updates.append(kwargs)

    def _open_gripper_for_box_release(self, _target):
        return -0.2

    def _current_ee_pose(self):
        return ((.30, .14, .10), (0.0, 0.0, 0.0, 1.0))

    def _execute_manual_pose(self, *_args, **_kwargs):
        raise RuntimeError("retreat blocked")

    def _execute_gripper(self, *_args, **_kwargs):
        return 0.0

    def _command_scene_grasp(self, *_args, **_kwargs):
        return None

    def _wait_scene_grasp(self, *_args, **_kwargs):
        return True

    def _wait_task_flag(self, *_args, **_kwargs):
        return self.stable

    def get_logger(self):
        return SimpleNamespace(warning=self.warnings.append)


@pytest.mark.parametrize("target", ["beverage_bottle", "cup"])
def test_centered_cylinder_lowers_vertically_without_extra_place_moves(target):
    # A 19 mm offset keeps the configured 26 mm cylinder footprint inside
    # the 45 mm box half-extent, so another horizontal plan adds no value.
    node = HeldCylinder(target, (.339, .14, .15))

    with pytest.raises(AtReleasePosition):
        GraspAssistant._guided_place(node)

    if target == "cup":
        assert node.moves == []  # Direct release at the measured height.
    else:
        assert len(node.moves) == 1
        assert node.moves[0][0] == pytest.approx((.339, .14, .115))
        assert node.moves[0][2]["linear"] is True


@pytest.mark.parametrize("target", ["beverage_bottle", "cup"])
def test_cylinder_off_center_uses_one_direct_alignment_at_measured_height(
    target,
):
    # Every single-pass cylinder uses one measured correction to the box
    # centre instead of three 8 mm motions.
    node = HeldCylinder(target, (.354, .14, .15))

    with pytest.raises(AtReleasePosition):
        GraspAssistant._guided_place(node)

    assert len(node.moves) == (1 if target == "cup" else 2)
    assert node.moves[0][0] == pytest.approx((.32, .14, .15))
    assert node.moves[0][2]["linear"] is True
    assert node.moves[0][2]["orientation_tolerance"] == pytest.approx(.025)
    if target != "cup":
        assert node.moves[1][0] == pytest.approx((.32, .14, .115))


def test_bottle_accepts_alignment_height_settle_within_lift_tolerance():
    node = HeldCylinder(
        "beverage_bottle",
        (.354, .14, .14),
        alignment_residual=(.32, .14, .1397),
    )

    with pytest.raises(AtReleasePosition):
        GraspAssistant._guided_place(node)

    assert len(node.moves) == 2
    assert node.moves[0][0] == pytest.approx((.32, .14, .14))
    assert node.moves[1][0] == pytest.approx((.32, .14, .115))


def test_bottle_rejects_alignment_height_below_lift_tolerance():
    node = HeldCylinder(
        "beverage_bottle",
        (.354, .14, .14),
        alignment_residual=(.32, .14, .1299),
    )

    with pytest.raises(RuntimeError, match="最低验收高度=0.1300 m"):
        GraspAssistant._guided_place(node)

    assert len(node.moves) == 1


@pytest.mark.parametrize("target", ["beverage_bottle", "cup"])
def test_cylinder_far_from_box_uses_existing_loaded_transport_then_lowers(target):
    node = HeldCylinder(target, (.45, -.14, .15))

    with pytest.raises(AtReleasePosition):
        GraspAssistant._guided_place(node)

    assert len(node.moves) == (1 if target == "cup" else 2)
    assert node.moves[0][0] == pytest.approx((.32, .14, .14))
    assert node.moves[0][2]["linear"] is False
    if target != "cup":
        assert node.moves[1][0] == pytest.approx((.32, .14, .115))
        assert node.moves[1][2]["linear"] is True


@pytest.mark.parametrize("target", ["beverage_bottle", "cup"])
def test_cylinder_rejects_unsafe_residual_before_lowering(target):
    node = HeldCylinder(
        target, (.341, .14, .15), alignment_residual=(.341, .14, .15)
    )

    with pytest.raises(RuntimeError, match="蓝盒对中仍超差"):
        GraspAssistant._guided_place(node)

    assert len(node.moves) == 1  # no descent toward the wall


def test_stable_release_succeeds_when_only_post_release_retreat_fails():
    node = ReleasedCylinder(stable=True)

    result = GraspAssistant._guided_place(node)

    assert "任务判定成功" in result
    assert any(
        update.get("release_sequence_complete") is True
        for update in node.control_updates
    )
    assert any("retreat blocked" in warning for warning in node.warnings)


def test_retreat_failure_is_reported_when_release_never_becomes_stable():
    node = ReleasedCylinder(stable=False)

    with pytest.raises(RuntimeError, match="retreat blocked"):
        GraspAssistant._guided_place(node)
