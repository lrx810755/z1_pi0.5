import sys,unittest,threading,tempfile
from pathlib import Path
from types import SimpleNamespace as N
from unittest.mock import Mock,patch
sys.path.insert(0,'/home/lp531/robot/bottle_horizontal_supplement_20261109')
import horizontal50 as h
from horizontal50_core import Settings
from gui_complete_search import CompleteSearchBridge
from continuous_training import CompleteSearchClient,OriginalClient
class CompleteSearch(unittest.TestCase):
 def test_every_angle_is_reached_for_each_distance_when_all_fail(self):
  candidates=[dict(position=[.5,0,.1],orientation=[0.,0.,0.,1.],azimuth_deg=float(i)) for i in range(17)]
  n=N(_lock=threading.RLock(),_joint_positions={f'joint{i}':0. for i in range(1,7)},_collection_column=Mock(return_value=1),_set_bottle_finger_collision=Mock(),_command_scene_grasp=Mock(),_wait_scene_grasp=Mock(return_value=True),_current_object_position=Mock(return_value=[.6,0,0]),_calibrated_candidates=Mock(return_value=candidates),_set_status=Mock(),sequence=N(wait_for_service=Mock(return_value=True)),settings=Settings(retreat_candidates_m=(.05,.07,.08,.09),maximum_axis_pitch_deg=8),planning_time=4.,_raise_if_stop_requested=Mock(),get_logger=Mock(return_value=N(info=Mock())),_plan_route=Mock(side_effect=RuntimeError('no IK')))
  with tempfile.TemporaryDirectory() as d,patch.object(h,'Path',return_value=Path(d)):
   with self.assertRaises(RuntimeError):h.HorizontalAssistant._generate_plan(n)
  self.assertEqual(n._plan_route.call_count,68)
  self.assertEqual([c.args[1]['retreat_m'] for c in n._plan_route.call_args_list],[.05]*17+[.07]*17+[.08]*17+[.09]*17)
 def test_bridge_wait_covers_full_search_without_sleeping(self):
  n=N(_generate_client=object(),_call_trigger=Mock(return_value={}))
  CompleteSearchBridge._generate_pregrasp_for_auto(n)
  self.assertGreaterEqual(n._call_trigger.call_args.kwargs['timeout_s'],1624.)
 def test_http_outer_wait_covers_ten_layouts(self):
  with patch.object(OriginalClient,'post',return_value={}) as post:
   CompleteSearchClient('http://localhost').post('/api/demonstration/auto',{},7200.)
  self.assertGreaterEqual(post.call_args.args[2],19720.)
if __name__=='__main__':unittest.main()
