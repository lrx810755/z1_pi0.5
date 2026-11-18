import importlib.util,json,sys,unittest
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
class Protocol(unittest.TestCase):
 def test_plan_covers_cells_without_heldout_leakage(self):
  import core
  tasks=core.make_tasks()
  self.assertEqual(len(tasks),48);self.assertEqual(len({t['id'] for t in tasks}),48)
  self.assertEqual(sum(t['kind']=='approach_recovery' for t in tasks),36)
  self.assertEqual({tuple(t['grid']) for t in tasks},{(c,r) for r in range(4) for c in range(5)})
  for t in tasks:
   self.assertEqual(t['split']=='validation',tuple(t['grid']) in {(1,1),(2,3)})
   self.assertEqual(t['freeze_policy'],'pre_approach_only')
 def test_dynamic_constraint_latches_transient_freeze(self):
  from fragment_core import DynamicEvidence
  e=DynamicEvidence();s=dict(frozen=False,target_collisions_enabled=True)
  e.observe(s,1.);e.observe(dict(s,frozen=True),1.1);e.observe(s,1.2)
  self.assertFalse(e.valid());self.assertTrue(e.violations)
 def test_missing_dynamic_status_cannot_pass(self):
  from fragment_core import DynamicEvidence
  e=DynamicEvidence();e.observe({},1.);self.assertFalse(e.valid())
 def test_rotated_recovery_preserves_world_vertical(self):
  from preflight import routes
  t={'kind':'approach_recovery','offset_approach_m':[-.025,.02,.01],'rotation_tool_deg':[0,4,0]}
  r=next(routes([.4,0,.12],[0,0,2**-.5,2**-.5],[.5,0,0],t))
  self.assertAlmostEqual(r['bad_position'][0],.38);self.assertAlmostEqual(r['bad_position'][2],.13)
  self.assertEqual(r['recovery'][-1]['position'],[.4,0,.12])
 def test_shift_changes_stale_target_not_goal(self):
  from preflight import routes
  t={'kind':'post_shift_realign','actual_shift_world_m':[.012,-.008,0],'offset_approach_m':[0,0,0],'rotation_tool_deg':[0,0,0]}
  r=next(routes([.4,0,.12],[0,0,0,1],[.5,0,0],t))
  self.assertAlmostEqual(r['bad_position'][0],.4-.012-.012)
  self.assertAlmostEqual(r['bad_position'][1],.008)
  self.assertEqual(r['recovery'][-1]['position'],[.4,0,.12])
 def test_freeze_command_rejected_after_approach(self):
  import ast
  tree=ast.parse((HERE/'nodes.py').read_text())
  method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_command_scene_grasp')
  class Base:
   def _command_scene_grasp(self,*a,**kw):return a[0]
  klass=ast.ClassDef(name='Subject',bases=[ast.Name(id='Base',ctx=ast.Load())],keywords=[],body=[method],decorator_list=[])
  env={'Base':Base};exec(compile(ast.fix_missing_locations(ast.Module(body=[klass],type_ignores=[])),'guard','exec'),env)
  x=env['Subject']();x.approach_started=False;self.assertEqual(x._command_scene_grasp('freeze'),'freeze')
  x.approach_started=True
  with self.assertRaises(RuntimeError):x._command_scene_grasp('freeze')
  self.assertEqual(x._command_scene_grasp('unfreeze'),'unfreeze')
 def test_scene_extension_compiles_against_current_scene(self):
  import ast
  tree=ast.parse((HERE/'scene.py').read_text());v={}
  for n in tree.body:
   if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('needle','replacement'):v[n.targets[0].id]=ast.literal_eval(n.value)
  source=HERE.parents[1]/'isaac_assets/z1/scripts/run_z1_ros2_scene.py';code=source.read_text()
  self.assertEqual(code.count(v['needle']),1);compile(code.replace(v['needle'],v['replacement']),str(source),'exec')
if __name__=='__main__':unittest.main()
