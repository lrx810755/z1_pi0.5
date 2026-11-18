import json,math,unittest
from pathlib import Path
from collections import Counter
from protocol import build_tasks,validate_category,training_mask
from route_helpers import routes

class CoverageTests(unittest.TestCase):
 def test_exact_quotas_and_spatial_coverage(self):
  tasks=build_tasks();self.assertEqual(Counter(t['kind'] for t in tasks),{'approach':20,'low_clearance':10});self.assertEqual(len({t['id'] for t in tasks}),30)
  self.assertEqual(len({tuple(t['grid']) for t in tasks if t['kind']=='approach'}),20)
  self.assertEqual(len({tuple(t['grid']) for t in tasks if t['kind']=='low_clearance'}),10)
  for t in tasks:
   ref=json.loads((Path(t['reference_basic_episode'])/'episode_summary.json').read_text())
   self.assertTrue(ref['task_success']);self.assertEqual(t['grid'],ref['scene_randomization']['target_grid']);self.assertEqual(t['source_pool'],'correction')
 def test_nominal_pose_cannot_count_as_correction(self):
  task=dict(kind='approach');meta=dict(recovery_complete=True,measured_start_deviation_m=.02,actual_start_position_m=[.3,0,.11],reference_position_m=[.3,0,.09])
  with self.assertRaisesRegex(ValueError,'horizontal error'):validate_category(task,meta)
  meta['actual_start_position_m']=[.31,0,.09];validate_category(task,meta)
  with self.assertRaises(ValueError):validate_category(dict(kind='low_clearance'),dict(recovery_complete=True,measured_world_z_deviation_m=-.0001))
  validate_category(dict(kind='low_clearance'),dict(recovery_complete=True,measured_world_z_deviation_m=-.006))
 def test_height_route_preserves_intended_negative_error(self):
  for t in build_tasks():
   if t['kind']!='low_clearance':continue
   route=next(routes([.3,0,.11],[0,0,0,1],[.45,0,.09],t))
   self.assertAlmostEqual(route['bad_position'][2]-.11,t['offset_approach_m'][2])
   self.assertLessEqual(route['bad_position'][2],.107)
 def test_30_frame_windows_do_not_cross_interventions(self):
  rows=[dict(simulation_time=i/10) for i in range(90)];events=[dict(event='intervention_start',time=4.),dict(event='intervention_end',time=4.5)]
  mask,anchors=training_mask(rows,events,horizon=30)
  self.assertFalse(any(anchors[11:46]));self.assertTrue(anchors[0]);self.assertFalse(any(anchors[-29:]))

if __name__=='__main__':unittest.main()
