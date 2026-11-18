import threading
from types import SimpleNamespace

import pytest

from z1_isaac_bridge.grasp_assistant import GraspAssistant


class AtReleasePosition(Exception):
    """Stop before real gripper and simulator release operations."""


class HeldFruit:
    box_center = (.32, .14)
    box_approach_object_z = .14
    box_release_object_z = .115
    max_joint_delta = 2.4
    physical_test_lift_velocity_scale = .12

    def __init__(
        self,
        target,
        position,
        *,
        reject_direct_alignment=False,
        stall_alignment=False,
        descent_error=None,
        descent_shift=(0.0, 0.0, 0.0),
    ):
        self._lock = threading.RLock()
        self._task_status = {"physical_grasp_ok": True}
        self._verified_grasp_sequence = True
        self.target_object = target
        self.position = position
        self.moves = []
        self.attempts = []
        self.probe_stages = []
        self.reject_direct_alignment = reject_direct_alignment
        self.stall_alignment = stall_alignment
        self.descent_error = descent_error
        self.descent_shift = descent_shift

    def _current_object_position(self, _target):
        return self.position

    def _move_object_to(self, destination, description, **kwargs):
        self.attempts.append((destination, description, kwargs))
        if "水平对中" in description and self.reject_direct_alignment:
            raise RuntimeError("MoveIt IK rejected direct alignment")
        if description == "垂直放入蓝盒" and self.descent_error:
            self.position = tuple(a + b for a, b in zip(self.position, self.descent_shift))
            raise RuntimeError(self.descent_error)
        self.moves.append((destination, description, kwargs))
        if "水平对中" in description and self.stall_alignment:
            # The commanded correction never reaches the object.
            return
        self.position = destination

    def _require_loaded_physical_grasp(self, _target, _envelope, stage, **_kwargs):
        self.probe_stages.append(stage)
        if stage in ("到达蓝盒释放位后", "高位回退释放前"):
            raise AtReleasePosition
        return {"bilateral_contact": True, "object_touching_table": False}

    def _carry_loaded_object_toward_box(self, *args, **kwargs):
        return GraspAssistant._carry_loaded_object_toward_box(self, *args, **kwargs)

    def get_logger(self):
        return SimpleNamespace(warning=lambda _message: None)


@pytest.mark.parametrize("target", ["peach"])
def test_fruit_inside_25mm_lowers_without_redundant_segments(target):
    node = HeldFruit(target, (.344, .14, .14))

    with pytest.raises(AtReleasePosition):
        GraspAssistant._guided_place(node)

    assert len(node.moves) == 1
    assert node.moves[0][0] == pytest.approx((.344, .14, .115))
    assert node.moves[0][2]["linear"] is True


@pytest.mark.parametrize("target", ["peach"])
def test_fruit_outside_25mm_steps_toward_center_at_measured_height(target):
    node = HeldFruit(target, (.346, .14, .15))

    with pytest.raises(AtReleasePosition):
        GraspAssistant._guided_place(node)

    assert len(node.moves) == 2
    assert node.moves[0][0] == pytest.approx((.338, .14, .15))
    assert node.moves[0][2]["linear"] is True
    assert node.moves[0][2]["orientation_tolerance"] == pytest.approx(.025)
    assert node.moves[1][0] == pytest.approx((.338, .14, .115))
    assert any("对中第 1 步后" in stage for stage in node.probe_stages)


@pytest.mark.parametrize("target", ["peach"])
def test_fruit_alignment_steps_are_bounded_and_converge(target):
    # 34 mm is inside the 35 mm box-approach corridor but outside the 25 mm
    # final acceptance, so the bounded loop needs more than one 8 mm step.
    node = HeldFruit(target, (.354, .14, .14))

    with pytest.raises(AtReleasePosition):
        GraspAssistant._guided_place(node)

    assert [
        tuple(round(value, 6) for value in move[0]) for move in node.moves
    ] == [
        (.346, .14, .14),
        (.338, .14, .14),
        (.338, .14, .115),
    ]
    assert all(move[2]["linear"] is True for move in node.moves)
    assert len(node.probe_stages) >= 3


@pytest.mark.parametrize("target", ["peach"])
def test_fruit_stops_when_alignment_stops_making_progress(target):
    node = HeldFruit(target, (.354, .14, .14), stall_alignment=True)

    with pytest.raises(RuntimeError, match="蓝盒对中仍超差"):
        GraspAssistant._guided_place(node)

    alignment_moves = [move for move in node.moves if "对中" in move[1]]
    assert len(alignment_moves) == 1
    assert not any("垂直放入蓝盒" in move[1] for move in node.moves)


@pytest.mark.parametrize("target", ["peach"])
def test_fruit_far_from_box_uses_existing_loaded_transport_only(target):
    node = HeldFruit(target, (.45, -.14, .14))

    with pytest.raises(AtReleasePosition):
        GraspAssistant._guided_place(node)

    assert len(node.moves) == 2
    assert node.moves[0][0] == pytest.approx((.32, .14, .14))
    assert node.moves[1][0] == pytest.approx((.32, .14, .115))


@pytest.mark.parametrize("target", ["peach"])
def test_fruit_stops_when_single_alignment_fails(target):
    node = HeldFruit(target, (.346, .14, .15), reject_direct_alignment=True)

    with pytest.raises(RuntimeError, match="IK rejected direct alignment"):
        GraspAssistant._guided_place(node)

    assert any(
        "水平对中" in description
        for _, description, _ in node.attempts
    )
    assert node.moves == []
    assert not any(
        "水平搬运" in description for _, description, _ in node.attempts
    )
    assert not any("水平搬运" in stage for stage in node.probe_stages)


@pytest.mark.parametrize(
    ("target", "height"),
    [("peach", .18), ("cup", .15), ("beverage_bottle", .18)],
)
def test_all_objects_can_release_after_descent_ik_failure(target, height):
    node = HeldFruit(target, (.32, .14, height), descent_error="垂直放入蓝盒失败：MoveIt error_code=-31")

    with pytest.raises(AtReleasePosition):
        GraspAssistant._guided_place(node)

    assert node.position == pytest.approx((.32, .14, height))
    assert "高位回退释放前" in node.probe_stages


def test_fallback_release_rejects_unrelated_descent_failure():
    node = HeldFruit("peach", (.32, .14, .14), descent_error="垂直放入蓝盒失败：MoveIt error_code=-2")

    with pytest.raises(RuntimeError, match="error_code=-2"):
        GraspAssistant._guided_place(node)

    assert "高位回退释放前" not in node.probe_stages


def test_fallback_release_rejects_excessive_drop_height():
    node = HeldFruit("peach", (.32, .14, .30), descent_error="垂直放入蓝盒失败：MoveIt error_code=-31")

    with pytest.raises(RuntimeError, match="回退释放高度超限"):
        GraspAssistant._guided_place(node)

    assert "高位回退释放前" not in node.probe_stages


def test_fallback_release_rejects_object_that_moved_outside_box_alignment():
    node = HeldFruit(
        "cup", (.32, .14, .15),
        descent_error="垂直放入蓝盒失败：MoveIt error_code=-31",
        descent_shift=(.03, 0.0, 0.0),
    )

    with pytest.raises(RuntimeError, match="回退释放位置已不安全"):
        GraspAssistant._guided_place(node)

    assert "高位回退释放前" not in node.probe_stages


def test_apple_dispatches_preplanned_route_before_shared_release():
    node = HeldFruit("apple", (.45, -.14, .14))
    calls = []
    def preplanned(envelope):
        calls.append(envelope)
        node.position = (.32, .14, .115)
    node._place_apple_preplanned = preplanned
    with pytest.raises(AtReleasePosition):
        GraspAssistant._guided_place(node)
    assert len(calls) == 1
    assert node.moves == []
    assert "高位回退释放前" not in node.probe_stages
