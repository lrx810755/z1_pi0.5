import unittest
from unittest.mock import Mock
from types import SimpleNamespace as N
import assistant
from moveit_msgs.msg import Constraints,RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
class PlanningPipeline(unittest.TestCase):
 def request(self,linear):
  trajectory=RobotTrajectory();trajectory.joint_trajectory.points=[JointTrajectoryPoint()]
  node=N(group_name='z1_arm',planning_time=4.,speed_scale=.35,_move_group=Mock(),_pose_constraints=Mock(return_value=Constraints()))
  handle=Mock(accepted=True);node._wait_future=Mock(side_effect=[handle,N(result=N(error_code=N(val=1),planned_trajectory=trajectory))])
  assistant.HorizontalAssistant._plan_candidate(node,{},linear=linear)
  return node._move_group.send_goal_async.call_args.args[0].request
 def test_free_space_explicitly_selects_ompl_despite_default_pilz(self):
  self.assertEqual(self.request(False).pipeline_id,'ompl')
 def test_linear_motion_still_explicitly_selects_pilz_lin(self):
  request=self.request(True)
  self.assertEqual(request.pipeline_id,'pilz_industrial_motion_planner');self.assertEqual(request.planner_id,'LIN')
if __name__=='__main__':unittest.main()
