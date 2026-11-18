import sys,unittest
sys.path.insert(0,'/home/lp531/robot/bottle_horizontal_supplement_20261109')
from types import SimpleNamespace as N
from unittest.mock import Mock
import assistant
from moveit_msgs.msg import Constraints,RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
class Pipeline(unittest.TestCase):
 def request(self,linear):
  tr=RobotTrajectory();tr.joint_trajectory.points=[JointTrajectoryPoint()]
  n=N(group_name='z1_arm',planning_time=4.,speed_scale=.35,_move_group=Mock(),_pose_constraints=Mock(return_value=Constraints()))
  n._wait_future=Mock(side_effect=[Mock(accepted=True),N(result=N(error_code=N(val=1),planned_trajectory=tr))])
  assistant.OriginalBottleAssistant._plan_candidate(n,{},linear=linear)
  return n._move_group.send_goal_async.call_args.args[0].request
 def test_free_space_preserves_ompl_when_sequence_default_is_pilz(self):self.assertEqual(self.request(False).pipeline_id,'ompl')
 def test_linear_preserves_pilz(self):
  r=self.request(True);self.assertEqual((r.pipeline_id,r.planner_id),('pilz_industrial_motion_planner','LIN'))
if __name__=='__main__':unittest.main()
