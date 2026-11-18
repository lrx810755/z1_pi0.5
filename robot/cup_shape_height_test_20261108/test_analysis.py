import unittest
from analysis import failure_kind
class Analysis(unittest.TestCase):
 def test_preclose_failure_is_not_counted_as_closed_grasp(self):
  self.assertEqual(failure_kind({'failed_stage':'grasp_and_lift','error':'目标物体闭爪前最终位置验收失败'}),'preclose')
  self.assertEqual(failure_kind({'failed_stage':'grasp_and_lift','error':'目标物体一次闭爪失败：双指未接触'}),'closure')
  self.assertEqual(failure_kind({'failed_stage':'grasp_and_lift','error':'目标物体纯物理抬升失败：带载荷'}),'lift_or_transport')
  self.assertEqual(failure_kind({'failed_stage':'planning'}),'planning')
  self.assertEqual(failure_kind({'passed':True}),'passed')
