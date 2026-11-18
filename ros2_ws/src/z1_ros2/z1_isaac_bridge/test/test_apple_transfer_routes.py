from types import SimpleNamespace as NS
import pytest
from test_apple_continuous_lift import Lift
from z1_isaac_bridge.grasp_assistant import GraspAssistant

class Routes(Lift):
    _plan_apple_transfer = GraspAssistant.__dict__.get('_plan_apple_transfer')
    _execute_apple_transfer_leg = GraspAssistant.__dict__.get('_execute_apple_transfer_leg')
    def __init__(self, reject_vertical=False, reject_descent=False):
        super().__init__()
        self.reject_vertical=reject_vertical
        self.reject_descent=reject_descent
    def _plan_candidate(self,candidate,**kw):
        self.events.append(('plan',candidate,kw))
        x,y,z=candidate['position']
        if self.reject_vertical and abs(x-.5)<1e-6 and z>.2:
            raise RuntimeError('MoveIt error_code=-31')
        if self.reject_descent and abs(x-.32)<1e-6 and z<.23:
            raise RuntimeError('MoveIt error_code=-31')
        return NS(end=candidate['position'],planned_trajectory=NS(
            joint_trajectory=NS(joint_names=['joint1'],points=[NS(positions=[0.])]),destination=candidate['position']))


def test_route_can_leave_vertical_branch_without_trial_motion():
    n=Routes(reject_vertical=True)
    route=GraspAssistant._plan_apple_transfer(n,n.position,n.ee,(0,0,0,1),.14)
    assert [leg['phase'] for leg in route]==['clearance','lift','carry','descent']
    assert route[0]['object'][0]==.5
    assert .32<route[1]['object'][0]<.5
    assert all(e[0]=='plan' for e in n.events)


def test_descent_unreachable_rejects_all_routes_without_execution():
    n=Routes(reject_descent=True)
    with pytest.raises(RuntimeError,match='完整搬运'):
        GraspAssistant._plan_apple_transfer(n,n.position,n.ee,(0,0,0,1),.14)
    assert all(e[0]=='plan' for e in n.events)


def test_lift_preplans_and_caches_place_without_replanning():
    n=Routes(reject_vertical=True)
    GraspAssistant._lift_apple_preplanned(n,.14,{})
    assert [leg['phase'] for leg in n._apple_place_route]==['carry','descent']
    count=sum(e[0]=='plan' for e in n.events)
    GraspAssistant._place_apple_preplanned(n,{})
    assert sum(e[0]=='plan' for e in n.events)==count
    assert n._apple_place_route is None
    assert n.position==pytest.approx((.32,.14,.115))


def test_precheck_chains_from_candidate_end_state():
    n=Routes()
    GraspAssistant._plan_apple_transfer(n,n.position,n.ee,(0,0,0,1),.14,start_state='candidate_end')
    plans=[e for e in n.events if e[0]=='plan']
    assert plans[0][2]['start_state']=='candidate_end'
    assert plans[1][2]['start_state']==plans[0][1]['position']


def test_cached_route_allows_position_difference_over_six_mm():
    n=Routes()
    n.drift=.012
    GraspAssistant._lift_apple_preplanned(n,.14,{})
    GraspAssistant._place_apple_preplanned(n,{})
    assert sum(e[0]=='execute' for e in n.events)==3
    assert n.position==pytest.approx((.332,.14,.115))
