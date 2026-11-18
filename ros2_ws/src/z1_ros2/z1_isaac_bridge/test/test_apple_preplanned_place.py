"""Apple placement must find a complete route before moving a held object."""
from types import SimpleNamespace as NS
import threading
import pytest
from z1_isaac_bridge.grasp_assistant import GraspAssistant


class Planner:
    box_center = (.32, .14)
    box_approach_object_z = .14
    box_release_object_z = .115
    manual_minimum_z = .02
    max_joint_delta = 2.4
    physical_test_lift_velocity_scale = .12
    _state_after_trajectory = staticmethod(lambda result: result.end)
    _trajectory_score = staticmethod(lambda trajectory: (1., .2))

    def __init__(self, reject=0, drift=0., stop=False):
        self._lock = threading.RLock()
        self._joint_positions = {"joint1": 0.}
        self.position = (.5, -.1, .14)
        self.ee = (.5, -.1, .24)
        self.events = []
        self.reject = reject
        self.drift = drift
        self.stop = stop
        self._move_group = NS(wait_for_server=lambda **kw: True)

    def _current_object_position(self, target):
        return self.position

    def _current_ee_pose(self):
        return self.ee, (0., 0., 0., 1.)

    def _set_status(self, *args, **kw):
        pass

    def _raise_if_stop_requested(self):
        pass

    def _plan_candidate(self, candidate, **kw):
        self.events.append(('plan', candidate, kw))
        if candidate['position'][2] < .23 and self.reject:
            self.reject -= 1
            raise RuntimeError('MoveIt error_code=-31')
        return NS(end=candidate['position'], planned_trajectory=NS(
            joint_trajectory=NS(joint_names=["joint1"], points=[NS(positions=[0.])]), destination=candidate['position']))

    def _send_arm_trajectory(self, trajectory):
        self.events.append(('execute', trajectory.destination))
        self.ee = trajectory.destination
        self.position = (self.ee[0] + self.drift, self.ee[1], self.ee[2] - .1)
        return self.stop

    def _require_loaded_physical_grasp(self, *args, **kw):
        self.events.append(('probe',))
        return {}


def test_complete_carry_and_descent_planned_before_first_execution():
    node = Planner()
    GraspAssistant._place_apple_preplanned(node, {})
    assert [e[0] for e in node.events][:3] == ['plan', 'plan', 'probe']
    plans = [e for e in node.events if e[0] == 'plan']
    assert plans[1][2]['start_state'] == plans[0][1]['position']
    assert all(e[2]['linear'] for e in plans)
    assert len([e for e in node.events if e[0] == 'execute']) == 2
    assert node.position == pytest.approx((.32, .14, .115))


def test_descent_rejected_retries_entire_route_without_moving():
    node = Planner(reject=1)
    GraspAssistant._place_apple_preplanned(node, {})
    first_execution = next(i for i,e in enumerate(node.events) if e[0] == 'execute')
    assert len([e for e in node.events[:first_execution] if e[0] == 'plan']) == 5


def test_all_descent_candidates_fail_no_movement_or_high_release():
    node = Planner(reject=99)
    with pytest.raises(RuntimeError, match='完整放置路径'):
        GraspAssistant._place_apple_preplanned(node, {})
    assert not any(e[0] == 'execute' for e in node.events)
    assert len([e for e in node.events if e[0] == 'plan']) == 8


def test_object_position_difference_does_not_abort_descent():
    node = Planner(drift=.012)
    GraspAssistant._place_apple_preplanned(node, {})
    assert len([e for e in node.events if e[0] == 'execute']) == 2


def test_execution_stop_never_replans_or_descends():
    node = Planner(stop=True)
    with pytest.raises(RuntimeError, match='停止'):
        GraspAssistant._place_apple_preplanned(node, {})
    assert len([e for e in node.events if e[0] == 'execute']) == 1


def test_changed_joint_branch_rejects_cached_path_before_motion():
    node = Planner()
    node._joint_positions["joint1"] = .2
    with pytest.raises(RuntimeError, match="关节起点偏差"):
        GraspAssistant._place_apple_preplanned(node, {})
    assert not any(e[0] == 'execute' for e in node.events)


def test_near_table_start_plans_vertical_lift_before_horizontal_carry():
    node = Planner()
    node.position = (.5, -.1, .08)
    node.ee = (.5, -.1, .18)
    GraspAssistant._place_apple_preplanned(node, {})
    moves = [e[1] for e in node.events if e[0] == 'execute']
    assert moves[0] == pytest.approx((.5, -.1, .24))
    assert moves[1] == pytest.approx((.32, .14, .24))
    assert moves[2] == pytest.approx((.32, .14, .215))
