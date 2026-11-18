"""Contract checks for isolated peach synchronization; never launch a scene."""
import ast,json,sys,unittest
from pathlib import Path
ROOT=Path('/home/lp531/robot')
class PeachSync(unittest.TestCase):
 def test_peach_configuration_preserves_own_geometry_and_mass(self):
  p=ROOT/'peach_scale120_demo_20261106/grasp_physics.json';self.assertTrue(p.exists())
  d=json.loads(p.read_text());self.assertAlmostEqual(d['collision_proxy']['peach']['visual_scale'],1.2)
  self.assertAlmostEqual(d['collision_proxy']['peach']['mass_kg'],.1728)
  self.assertEqual(d['collision_proxy']['peach']['radii_m'],[.01854,.018179999999999998,.0174])
  self.assertAlmostEqual(d['object_center_z_offsets_m']['peach'],.0174)
 def test_basic_transports_peach_without_querying_apple(self):
  p=ROOT/'peach_scale120_demo_20261106/base_assistant_snapshot.py';self.assertTrue(p.exists())
  t=ast.parse(p.read_text());methods={n.name:n for n in ast.walk(t) if isinstance(n,ast.FunctionDef)}
  for name in ['_execute_apple_transfer_leg','_lift_apple_preplanned','_place_apple_preplanned']:
   calls=[n for n in ast.walk(methods[name]) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ['_current_object_position','_require_loaded_physical_grasp']]
   self.assertFalse(any(n.args and isinstance(n.args[0],ast.Constant) and n.args[0].value=='apple' for n in calls),name)
 def test_correction_uses_peach_target_and_explicit_close_events(self):
  p=ROOT/'peach_scale120_correction60_20261106/nodes.py';self.assertTrue(p.exists())
  s=p.read_text();self.assertIn('import base_assistant_snapshot as module',s)
  self.assertIn("module.GRASP_PROFILES['peach']",s);self.assertIn('close_request_sim_time',s)
  self.assertIn('single_position_world_clearance_lift_m=.005',s)
  self.assertNotIn("'apple'",s)
 def test_entrypoints_are_previewable(self):
  for p in ['/home/lp531/SH/start_peach_scale120_demo.sh','/home/lp531/SH/start_peach_scale120_correction60.sh']:
   self.assertTrue(Path(p).is_file())
if __name__=='__main__':unittest.main()
