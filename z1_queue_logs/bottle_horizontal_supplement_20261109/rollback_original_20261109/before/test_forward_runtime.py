import time
import threading,unittest
from unittest.mock import Mock
from types import SimpleNamespace as N
import numpy as np
from forward_transport import ForwardTransport
from forward_core import ForwardGuard,GuardViolation
class Parent:
 def _send_action_goal(self,*args,**kwargs):return 'unchanged'
class Node(ForwardTransport,Parent):
 def __init__(self):pass
class Runtime(unittest.TestCase):
 def test_bad_paths_are_never_executed_even_after_planning_fallback(self):
  n=Node();n._current_object_position=Mock(return_value=[.55,0,0]);n._current_ee_pose=Mock(return_value=([.4,0,.1],[0,0,0,1]));n._raise_if_stop_requested=Mock();n._set_status=Mock();n._plan_candidate=Mock(return_value=N(planned_trajectory=object()));n._audit_forward_trajectory=Mock(side_effect=GuardViolation('reverse progress .093m'));n._send_arm_trajectory=Mock();n.get_logger=Mock(return_value=N(warning=Mock()))
  with self.assertRaises(RuntimeError):n._forward_segment([.55,0,.18],'lift')
  n._send_arm_trajectory.assert_not_called()
  self.assertEqual([c.kwargs['linear'] for c in n._plan_candidate.call_args_list],[True,True,False])
 def test_execution_violation_cancels_the_active_arm_action(self):
  n=Node();n._lock=threading.RLock();n._trajectory_client=object();n._loaded_guard={'test':True};n._raise_if_stop_requested=Mock();n._forward_live_check=Mock(side_effect=GuardViolation('reverse progress .01m'))
  handle=Mock(accepted=True);handle.get_result_async.return_value=Mock(done=Mock(return_value=False));client=n._trajectory_client
  n._trajectory_client=Mock();client=n._trajectory_client;n._wait_future=Mock(return_value=handle)
  with self.assertRaisesRegex(GuardViolation,'reverse'):n._send_action_goal(client,object(),90)
  handle.cancel_goal_async.assert_called_once();self.assertIsNone(n._active_goal_handle)
 def test_gripper_actions_keep_the_existing_execution_path(self):
  n=Node();n._loaded_guard={'test':True};n._trajectory_client=object()
  self.assertEqual(n._send_action_goal(object(),object(),90),'unchanged')
if __name__=='__main__':unittest.main()

class Stages(unittest.TestCase):
 def test_real_carry_override_lifts_before_horizontal_transfer(self):
  n=Node();position=[[.55,0.,0.]];n.box_center=(.32,.14);n.box_approach_object_z=.14;n.transport_loaded_grasp_guard_margin=.03
  n._current_object_position=Mock(side_effect=lambda *_:position[0]);n._require_loaded_physical_grasp=Mock(return_value={'ok':True})
  def execute(goal,phase):position[0]=list(goal)
  n._forward_segment=Mock(side_effect=execute)
  n._carry_loaded_object_toward_box('beverage_bottle',{},.15,'verified',require_box_approach=True)
  self.assertEqual([c.args[1] for c in n._forward_segment.call_args_list],['lift','horizontal'])
  self.assertEqual(n._forward_segment.call_args_list[0].args[0],[.55,0.,.18])
  self.assertEqual(n._forward_segment.call_args_list[1].args[0],[.32,.14,.18])
 def test_already_arrived_pose_does_not_add_a_move(self):
  n=Node();n.box_center=(.32,.14);n.transport_loaded_grasp_guard_margin=.03;n._current_object_position=Mock(return_value=[.32,.14,.16]);n._require_loaded_physical_grasp=Mock(return_value={});n._forward_segment=Mock()
  n._carry_loaded_object_toward_box('beverage_bottle',{},.15,'verified',require_box_approach=True,preferred_object_z=.18)
  n._forward_segment.assert_not_called()

class Alignment(unittest.TestCase):
 def test_box_alignment_uses_same_guarded_forward_executor(self):
  n=Node();n.target_object='beverage_bottle';n._forward_segment=Mock()
  n._move_object_to((.32,.14,.18),'蓝盒上方瓶子水平对中（第1/1步）',linear=True)
  n._forward_segment.assert_called_once_with((.32,.14,.18),'horizontal')

class LiveChecks(unittest.TestCase):
 def node(self):
  n=Node();n._lock=threading.RLock();now=time.monotonic();n._forward_joint_at=now;n._forward_contact_at=now
  n._loaded_guard={'guard':ForwardGuard([.5,0,.18],[.32,.14,.18],np.eye(3),'horizontal'),'offset':np.zeros(3),'last_contact_ok':now}
  n._current_object_position=Mock(return_value=[.5,0,.18]);n._forward_pose=Mock(return_value=(np.array([.5,0,.18]),np.eye(3)));n.physical_test_lift_relative_position_tolerance=.04
  n._scene_grasp_contact_status={'contact_report_available':True,'left_finger_contact_object':'beverage_bottle','right_finger_contact_object':'beverage_bottle'}
  return n
 def test_fresh_dual_contact_is_accepted(self):
  self.node()._forward_live_check()
 def test_stale_joint_feedback_is_rejected(self):
  n=self.node();n._forward_joint_at-=2
  with self.assertRaisesRegex(GuardViolation,'joint feedback stale'):n._forward_live_check()
 def test_missing_contact_has_grace_then_is_rejected(self):
  n=self.node();n._scene_grasp_contact_status={};n._forward_live_check();n._loaded_guard['last_contact_ok']-=1
  with self.assertRaisesRegex(GuardViolation,'bilateral contact'):n._forward_live_check()

class WristDirection(unittest.TestCase):
 def test_wrist_reversal_rejected_even_with_bottle_position_unchanged(self):
  n=LiveChecks.node(self);n._loaded_guard['wrist_guard']=ForwardGuard([.5,0,.18],[.32,.14,.25],np.eye(3),'carry')
  n._forward_pose.return_value=(np.array([.52,0,.18]),np.eye(3))
  with self.assertRaisesRegex(GuardViolation,'away'):n._forward_live_check()
