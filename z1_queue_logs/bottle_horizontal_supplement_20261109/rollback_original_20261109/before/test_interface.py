import json,subprocess,unittest
from pathlib import Path
R=Path(__file__).resolve().parent
ENTRY=Path('/home/lp531/SH/start_bottle_horizontal_supplement_20261109.sh')
class Interface(unittest.TestCase):
 def test_preview_names_new_target_paths_and_quota(self):
  result=subprocess.run(['bash',str(ENTRY),'--dry-run','3'],capture_output=True,text=True)
  self.assertEqual(result.returncode,0,result.stderr)
  self.assertIn('target_object:=beverage_bottle',result.stdout)
  self.assertIn('bottle_horizontal_supplement_20261109',result.stdout)
  self.assertIn('8839',result.stdout);self.assertIn('60',result.stdout)
  self.assertNotIn('target_object:=cup',result.stdout)
 def test_preview_never_creates_dataset(self):
  p=Path('/home/lp531/smolvla_data/bottle_horizontal_supplement_20261109');before=p.exists()
  subprocess.run(['bash',str(ENTRY),'--dry-run'],check=True,capture_output=True)
  self.assertEqual(p.exists(),before)
 def test_invalid_quota_rejected(self):
  for value in ['0','bad','-1']:
   result=subprocess.run(['bash',str(ENTRY),'--dry-run',value],capture_output=True)
   self.assertNotEqual(result.returncode,0)
 def test_bottle_physics_profile_preserved_except_authorized_preclose_10mm(self):
  original=json.loads(Path('/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge/config/grasp_physics.json').read_text())
  private=json.loads((R/'grasp_physics.json').read_text())
  actual=private['object_grasp_profiles']['beverage_bottle'];expected=original['object_grasp_profiles']['beverage_bottle']
  self.assertEqual(actual['calibrated_grasp_reference']['position_tolerance_m'],[.01]*3)
  actual['calibrated_grasp_reference']['position_tolerance_m']=expected['calibrated_grasp_reference']['position_tolerance_m']
  for key in ['preclose_envelope','preclose_capture_envelope']:
   self.assertEqual(actual[key],dict(x_min_m=.11583045,x_max_m=.13583045,y_abs_max_m=.032,z_min_m=.010,z_max_m=.0287797))
   actual[key]=expected[key]
  self.assertEqual(actual['preclose_no_contact_x_max_m'],.13583045)
  actual['preclose_no_contact_x_max_m']=expected['preclose_no_contact_x_max_m']
  self.assertEqual(actual,expected)
 def test_column_setter_is_bottle_specific_and_bounded(self):
  s=(R/'set_collection_column.py').read_text()
  self.assertIn('bottle_collection_column',s);self.assertNotIn('cup_collection_column',s)
  self.assertIn('timeout_sec=timeout',s)
if __name__=='__main__':unittest.main()
