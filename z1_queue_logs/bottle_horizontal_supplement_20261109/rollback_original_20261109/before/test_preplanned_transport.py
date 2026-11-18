import threading,time,unittest
from types import SimpleNamespace as N
from unittest.mock import Mock
import numpy as np
import assistant
from moveit_msgs.msg import RobotState,RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
try:
 from preplanned_transport import PreplannedBottleTransport
except ImportError:
 from forward_transport import ForwardTransport as PreplannedBottleTransport

class Rig(PreplannedBottleTransport):
 def __init__(self):
  self._lock=threading.RLock();self._joint_positions={f'joint{i}':v for i,v in enumerate([.55,0,0,0,0,0],1)};self._joint_positions['jointGripper']=-.63
  self.position=np.array([.55,0,0.]);self.box_center=(.32,.14);self.box_approach_object_z=.14;self.transport_loaded_grasp_guard_margin=.03
  self.geometry=N(link_fk=lambda q:(np.array(q[:3]),np.eye(3)),height=lambda q:.03)
  self._raise_if_stop_requested=Mock();self._set_status=Mock();self.get_logger=Mock(return_value=N(warning=Mock(),info=Mock()))
  self._require_loaded_physical_grasp=Mock(return_value={});self._plan_candidate=Mock(side_effect=self.plan);self._send_arm_trajectory=Mock(side_effect=self.send);self.events=[]
 def _current_object_position(self,*_):return self.position.copy()
 def _current_ee_pose(self):return self.position.copy(),[0,0,0,1]
 def plan(self,candidate,**kwargs):
  self.events.append('plan');state=kwargs['start_state'];names=list(state.joint_state.name);start=list(state.joint_state.position);arm=[start[names.index(f'joint{i}')] for i in range(1,7)]
  trajectory=RobotTrajectory();trajectory.joint_trajectory.joint_names=[f'joint{i}' for i in range(1,7)]
  for q in [arm,list(candidate['position'])+[0.,0.,0.]]:
   point=JointTrajectoryPoint();point.positions=[float(v) for v in q];trajectory.joint_trajectory.points.append(point)
  return N(planned_trajectory=trajectory)
 def send(self,trajectory):
  self.events.append('send');q=trajectory.joint_trajectory.points[-1].positions
  self._joint_positions.update(zip(trajectory.joint_trajectory.joint_names,q));self.position=np.array(q[:3]);return False
 def run_carry(self):return self._carry_loaded_object_toward_box('beverage_bottle',{},.15,'test',require_box_approach=True)

class Preplanning(unittest.TestCase):
 def test_second_leg_failure_never_executes_lift(self):
  n=Rig();real=n.plan
  def plan(candidate,**kwargs):
   if candidate['position'][0]<.5:raise RuntimeError('no box IK')
   return real(candidate,**kwargs)
  n._plan_candidate.side_effect=plan
  with self.assertRaisesRegex(RuntimeError,'no box IK'):n.run_carry()
  n._send_arm_trajectory.assert_not_called()
 def test_both_plans_precede_first_motion_with_predicted_start_and_closed_gripper(self):
  n=Rig();n.run_carry();self.assertEqual(n.events,['plan','plan','send','send'])
  second=n._plan_candidate.call_args_list[1].kwargs;state=second['start_state'];values=dict(zip(state.joint_state.name,state.joint_state.position))
  self.assertAlmostEqual(values['joint3'],.18);self.assertAlmostEqual(values['jointGripper'],-.63)
  np.testing.assert_allclose(n.position,[.32,.14,.18])
 def test_stale_second_start_is_rejected_without_replanning(self):
  n=Rig();send=n.send
  def drift(trajectory):
   result=send(trajectory)
   if len(n.events)==3:n._joint_positions['joint1']+=.02
   return result
  n._send_arm_trajectory.side_effect=drift
  with self.assertRaisesRegex(RuntimeError,'stale'):n.run_carry()
  self.assertEqual(n._send_arm_trajectory.call_count,1);self.assertEqual(n._plan_candidate.call_count,2)
