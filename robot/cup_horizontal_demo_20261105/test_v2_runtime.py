import unittest,threading,time,inspect
from unittest.mock import Mock,patch
from types import SimpleNamespace as N
import numpy as np
import tempfile
from pathlib import Path
from core import Settings,route_candidates
import assistant
from moveit_msgs.msg import Constraints,RobotTrajectory
class RuntimeTests(unittest.TestCase):
 def bare(self):
  n=object.__new__(assistant.HorizontalAssistant);n.gripper_open=-.9;n.group_name='z1_arm';n.planning_time=4.;n.settings=Settings();n._raise_if_stop_requested=Mock();n._pose_constraints=Mock(return_value=Constraints());n.sequence=Mock();return n
 def test_three_segment_request(self):
  n=self.bare();trajectory=RobotTrajectory();n._wait_future=Mock(return_value=N(response=N(error_code=N(val=1),planned_trajectories=[trajectory])))
  route=route_candidates([.4,0,.08],[.55,0,0],[1,0,0],n.settings)[0]
  self.assertIs(n._plan_route([0.]*6,route,np.array([.4,0,.08]),np.array([0.,0.,0.,1.]),time.monotonic()+8),trajectory)
  request=n.sequence.call_async.call_args.args[0];items=request.request.items
  self.assertEqual([x.req.planner_id for x in items],['PTP','LIN','LIN']);self.assertEqual([x.blend_radius for x in items],[.003,.003,0.]);self.assertEqual([x.req.max_velocity_scaling_factor for x in items],[.08,.08,.064])
  self.assertLessEqual(sum(x.req.allowed_planning_time for x in items),7.)
  self.assertEqual(len(items[0].req.start_state.joint_state.position),7)
 def test_direct_low_request(self):
  n=self.bare();trajectory=RobotTrajectory();n._wait_future=Mock(return_value=N(response=N(error_code=N(val=1),planned_trajectories=[trajectory])))
  route=route_candidates([.4,0,.08],[.55,0,0],[1,0,0],n.settings)[0];route=dict(route,strategy='direct_low')
  n._plan_route([0.]*6,route,np.array([.4,0,.08]),np.array([0.,0.,0.,1.]),time.monotonic()+8)
  items=n.sequence.call_async.call_args.args[0].request.items
  self.assertEqual([x.req.planner_id for x in items],['PTP','LIN']);self.assertEqual([x.blend_radius for x in items],[.003,0.]);self.assertEqual([x.req.max_velocity_scaling_factor for x in items],[.08,.064])
 def selection_node(self):
  n=self.bare();n._lock=threading.RLock();n._joint_positions={f'joint{i}':(-.1 if i==3 else 0.) for i in range(1,7)}
  n._set_bottle_finger_collision=Mock();n._command_scene_grasp=Mock();n._wait_scene_grasp=Mock(return_value=True);n._current_object_position=Mock(return_value=[.55,0,0]);n._set_status=Mock()
  n._calibrated_candidates=Mock(return_value=[dict(position=[.4,0,.08],orientation=[0.,0.,0.,1.],final_grasp_target=True)])
  n.sequence.wait_for_service.return_value=True;n._audit_trajectory=Mock(return_value={})
  return n
 def test_successful_direct_never_tries_high(self):
  n=self.selection_node();n._plan_route=Mock(return_value=RobotTrajectory())
  with tempfile.TemporaryDirectory() as td,patch.object(assistant,'Path',return_value=Path(td)):
   n._generate_plan()
  self.assertEqual(n._plan_route.call_count,1);self.assertEqual(n._plan_route.call_args.args[1]['strategy'],'direct_low')
 def test_high_only_after_direct_rejections(self):
  n=self.selection_node();seen=[]
  def plan(start,route,*rest):
   seen.append(route['strategy'])
   if route['strategy']=='direct_low':raise RuntimeError('unreachable direct')
   return RobotTrajectory()
  n._plan_route=Mock(side_effect=plan)
  with tempfile.TemporaryDirectory() as td,patch.object(assistant,'Path',return_value=Path(td)):
   n._generate_plan()
  self.assertEqual(seen,['direct_low']*4+['via_high'])
 def test_expired_budget_never_sends(self):
  n=self.bare()
  with self.assertRaises(RuntimeError):n._plan_route([0.]*6,{},[],[],time.monotonic()-1)
  n.sequence.call_async.assert_not_called()
 def test_slow_direct_search_exhausts_every_candidate_before_high(self):
  n=self.selection_node();candidate=n._calibrated_candidates.return_value[0]
  n._calibrated_candidates.return_value=[dict(candidate),dict(candidate)]
  seen=[];clock=[100.]
  def plan(start,route,*rest):
   seen.append((route['strategy'],route['retreat_m']))
   if route['strategy']=='direct_low':
    clock[0]+=40.
    raise RuntimeError('unreachable direct')
   return RobotTrajectory()
  n._plan_route=Mock(side_effect=plan)
  with tempfile.TemporaryDirectory() as td,patch.object(assistant,'Path',return_value=Path(td)),patch.object(assistant.time,'monotonic',side_effect=lambda:clock[0]):
   n._generate_plan()
  self.assertEqual(seen,[( 'direct_low',distance) for distance in n.settings.retreat_candidates_m for _ in range(2)]+[('via_high',.05)])
 def test_legacy_path_not_prerequisite(self):
  self.assertNotIn('super()._generate_plan()',inspect.getsource(assistant.HorizontalAssistant._generate_plan))
 def test_stale_plan_never_resurrects(self):
  n=self.bare();n._lock=threading.RLock();n.new_plan=object();n._approved_trajectory=None;n._approved_target='';n.target_object='cup'
  with self.assertRaises(RuntimeError):n._execute_plan()
  self.assertIsNone(n.new_plan)
if __name__=='__main__':unittest.main()
