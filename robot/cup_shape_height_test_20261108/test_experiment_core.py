import unittest
from experiment_core import cases, geometry, passed
class Contract(unittest.TestCase):
 def test_matrix(self):
  a=cases();self.assertEqual(len(a),120);self.assertEqual(len({c['id'] for c in a}),120)
  self.assertEqual({tuple(c['cup_position_m']) for c in a},{(.393,-.141,0.),(.525,-.141,0.),(.657,-.141,0.)})
  from collections import Counter
  self.assertEqual(set(Counter((c['shape'],c['height_mm'],c['position']) for c in a).values()),{10})
 def test_geometry(self):
  self.assertEqual(geometry('cylinder')['radius_m'],.0252)
  self.assertEqual(geometry('frustum')['bottom_radius_m'],.018)
  for s in ['cylinder','frustum']:self.assertEqual(geometry(s)['height_m'],.061)
  with self.assertRaises(ValueError):geometry('unknown')
 def test_physical_success(self):
  good=dict(service_success=True,ever_physical_grasp=True,maximum_rise_m=.04,final_status=dict(physical_grasp_ok=True,object_touching_table=False))
  self.assertTrue(passed(good))
  for k,v in [('service_success',False),('ever_physical_grasp',False),('maximum_rise_m',.01),('final_status',{})]:
   self.assertFalse(passed(dict(good,**{k:v})))
if __name__=='__main__':unittest.main()

class FixedPoseRuntime(unittest.TestCase):
 def test_real_scene_sampler_uses_fixed_pose_and_keeps_other_objects(self):
  import ast,json,os,tempfile,types,numpy as np
  from pathlib import Path
  tree=ast.parse(Path(__file__).with_name('scene.py').read_text())
  sampler=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_sample_randomized_object_poses')
  ns=dict(np=np,json=json,os=os,OBJECT_NAME_TO_PRIM={'cup':'cup','apple':'apple'})
  for n in tree.body:
   if isinstance(n,ast.ImportFrom) and n.module=='pathlib':exec(compile(ast.Module(body=[n],type_ignores=[]),'imports','exec'),ns)
  exec(compile(ast.Module(body=[sampler],type_ignores=[]),'sampler','exec'),ns)
  with tempfile.TemporaryDirectory() as directory:
   p=Path(directory)/'case.json';p.write_text(json.dumps(dict(cup_position_m=[.393,-.141,0])))
   old=os.environ.get('CUP_EXPERIMENT_CASE');os.environ['CUP_EXPERIMENT_CASE']=str(p)
   try:
    fake=types.SimpleNamespace(_initial_object_poses={'cup':([0,0,0],[1,0,0,0]),'apple':([.5,.1,0],[1,0,0,0])})
    out=ns['_sample_randomized_object_poses'](fake,123,'full')
    np.testing.assert_allclose(out['cup'][0],[.393,-.141,0],atol=1e-7);np.testing.assert_allclose(out['apple'][0],[.43,.12,0])
   finally:
    if old is None:os.environ.pop('CUP_EXPERIMENT_CASE',None)
    else:os.environ['CUP_EXPERIMENT_CASE']=old

class Accounting(unittest.TestCase):
 def test_setup_and_unfinished_service_are_not_robot_trials(self):
  from experiment_core import valid_trial
  self.assertFalse(valid_trial(dict(failed_stage='reset')))
  self.assertFalse(valid_trial(dict(failed_stage='planning',error='Service timeout: auto_generate_pregrasp')))
  self.assertFalse(valid_trial(dict(failed_stage='approach',cleanup_error='connection lost')))
  self.assertTrue(valid_trial(dict(failed_stage='planning',error='MoveIt error_code=99999')))
  self.assertTrue(valid_trial(dict(service_success=True)))

class DistractorPlacement(unittest.TestCase):
 def test_distractors_clear_all_fixed_cup_positions(self):
  from experiment_core import DISTRACTOR_XY
  import math
  self.assertEqual(set(DISTRACTOR_XY),{'apple','peach','mouse','beverage_bottle'})
  for case in cases():
   for xy in DISTRACTOR_XY.values():self.assertGreater(math.dist(xy,case['cup_position_m'][:2]),.15)
