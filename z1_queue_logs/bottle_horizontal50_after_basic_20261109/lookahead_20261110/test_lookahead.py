import sys,unittest,threading
from unittest.mock import Mock
from types import SimpleNamespace as N
import numpy as np
sys.path.insert(0,'/home/lp531/robot/bottle_horizontal_supplement_20261109')
from moveit_msgs.msg import RobotState,RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
try:
 from bottle_lookahead import BottleLookahead
except ImportError:
 class BottleLookahead:pass
class Rig(BottleLookahead):
 def __init__(self):
  self._lock=threading.RLock();self._joint_positions=dict(zip([f'joint{i}' for i in range(1,7)]+['jointGripper'],[.5,0,0,0,0,0,-.6]));self.position=np.array([.5,0,0.]);self.target_object='beverage_bottle';self.box_center=(.32,.14);self.box_release_object_z=.115;self.max_joint_delta=3.;self.manual_minimum_z=.04;self.transport_loaded_grasp_guard_margin=.03
  self.geometry=N(link_fk=lambda q:(np.array(q[:3]),np.eye(3)));self.events=[];self._require_loaded_physical_grasp=Mock(return_value={});self._raise_if_stop_requested=Mock();self._set_status=Mock();self.get_logger=Mock(return_value=N(info=Mock(),warning=Mock()));self._plan_candidate=Mock(side_effect=self.plan);self._send_arm_trajectory=Mock(side_effect=self.send);self._trajectory_score=Mock(return_value=(0,1.))
 def _current_object_position(self,*_):return self.position.copy()
 def plan(self,candidate,**kwargs):
  self.events.append('plan');state=kwargs['start_state'];q=list(state.joint_state.position[:6]);dest=list(candidate['position'])
  tr=RobotTrajectory();tr.joint_trajectory.joint_names=[f'joint{i}' for i in range(1,7)]
  for v in [q,dest+[0.,0.,0.]]:
   p=JointTrajectoryPoint();p.positions=[float(x) for x in v];tr.joint_trajectory.points.append(p)
  return N(planned_trajectory=tr)
 def send(self,tr):
  self.events.append('send');q=tr.joint_trajectory.points[-1].positions;self._joint_positions.update(zip(tr.joint_trajectory.joint_names,q));self.position=np.array(q[:3]);return False
 def carry(self):self._carry_loaded_object_toward_box('beverage_bottle',{},.15,'test',True,.18)
class Lookahead(unittest.TestCase):
 def test_both_plans_before_carry_and_no_replanning_for_descent(self):
  n=Rig();n.carry();self.assertEqual(n.events,['plan','plan','send']);n._move_object_to((.32,.14,.115),'垂直放入蓝盒',True);self.assertEqual(n.events,['plan','plan','send','send']);self.assertAlmostEqual(n.position[2],.115)
 def test_descent_planning_failure_never_executes_carry(self):
  n=Rig();real=n.plan
  def planner(c,**kw):
   if c['position'][2]<.15:raise RuntimeError('no descent IK')
   return real(c,**kw)
  n._plan_candidate.side_effect=planner
  with self.assertRaisesRegex(RuntimeError,'no descent IK'):n.carry()
  n._send_arm_trajectory.assert_not_called()
 def test_stale_cached_descent_is_rejected_without_new_planning(self):
  n=Rig();n.carry();n._joint_positions['joint1']+=.03
  with self.assertRaisesRegex(RuntimeError,'起点'):n._move_object_to((.32,.14,.115),'垂直放入蓝盒',True)
  self.assertEqual(n._plan_candidate.call_count,2);self.assertEqual(n._send_arm_trajectory.call_count,1)
 def test_carry_retains_original_three_radian_limit(self):
  n=Rig();n.max_joint_delta=2.4;n._trajectory_score.side_effect=[(0,2.7),(0,1.)];n.carry();self.assertEqual(n._send_arm_trajectory.call_count,1)
 def test_bottle_shift_invalidates_cached_descent_even_with_unchanged_joints(self):
  n=Rig();n.carry();n.position[0]+=.004
  with self.assertRaisesRegex(RuntimeError,'瓶子起点'):n._move_object_to((.32,.14,.115),'垂直放入蓝盒',True)
  self.assertEqual(n._plan_candidate.call_count,2);self.assertEqual(n._send_arm_trajectory.call_count,1)
if __name__=='__main__':unittest.main()
