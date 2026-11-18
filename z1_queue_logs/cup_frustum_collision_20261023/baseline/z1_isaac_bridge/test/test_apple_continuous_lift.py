from types import SimpleNamespace as NS
import pytest
from test_apple_preplanned_place import Planner
from z1_isaac_bridge.grasp_assistant import GraspAssistant


class Lift(Planner):
    _plan_apple_transfer = GraspAssistant._plan_apple_transfer
    _execute_apple_transfer_leg = GraspAssistant._execute_apple_transfer_leg
    def __init__(self, failures=0, execution_error=None):
        super().__init__()
        self.position = (.5, -.1, .02)
        self.ee = (.5, -.1, .12)
        self.failures = failures
        self.execution_error = execution_error

    def _plan_candidate(self, candidate, **kw):
        self.events.append(('plan', candidate, kw))
        if self.failures:
            self.failures -= 1
            raise RuntimeError('MoveIt error_code=-31')
        return NS(end=candidate['position'], planned_trajectory=NS(joint_trajectory=NS(
            joint_names=['joint1'], points=[NS(positions=[0.])]), destination=candidate['position']))

    def _send_arm_trajectory(self, trajectory):
        if self.execution_error:
            self.events.append(('execute',))
            raise RuntimeError(self.execution_error)
        return super()._send_arm_trajectory(trajectory)


def test_lift_is_one_full_linear_trajectory():
    n=Lift()
    GraspAssistant._lift_apple_preplanned(n,.14,{})
    assert [e[0] for e in n.events].count('execute') == 1
    plans=[e for e in n.events if e[0]=='plan']
    assert len(plans)==3 and all(p[2]['linear'] for p in plans)
    assert n.position==pytest.approx((.5,-.1,.14))


def test_rejected_full_lift_only_retries_before_moving():
    n=Lift(failures=2)
    GraspAssistant._lift_apple_preplanned(n,.14,{})
    assert [e[0] for e in n.events][:3]==['plan']*3
    assert [e[0] for e in n.events].count('execute')==2


def test_unreachable_lift_never_executes_partial_or_free_motion():
    n=Lift(failures=99)
    with pytest.raises(RuntimeError,match='完整搬运'):
        GraspAssistant._lift_apple_preplanned(n,.14,{})
    assert not any(e[0]=='execute' for e in n.events)
    assert len(n.events)==6


def test_execution_error_cannot_trigger_planning_retry():
    n=Lift(execution_error='MoveIt error_code=-31')
    with pytest.raises(RuntimeError):
        GraspAssistant._lift_apple_preplanned(n,.14,{})
    assert sum(e[0]=='plan' for e in n.events)==3
    assert sum(e[0]=='execute' for e in n.events)==1
