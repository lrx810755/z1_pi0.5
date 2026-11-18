import unittest
from schedule import pending_cells
class ScheduleTests(unittest.TestCase):
 def test_only_missing_cells_are_scheduled(self):
  c={(x,y):20 for y in range(4) for x in range(5)};c[0,2]=9;c[4,3]=0
  self.assertEqual(pending_cells(c),[(0,2),(4,3)])
 def test_failed_cell_remains_for_next_round(self):
  c={(0,2):9,(1,2):19};round1=pending_cells(c);self.assertEqual(round1,[(0,2),(1,2)])
  c[1,2]=20;self.assertEqual(pending_cells(c),[(0,2)])
 def test_over_quota_rejected(self):
  with self.assertRaises(ValueError):pending_cells({(0,0):21})
if __name__=='__main__':unittest.main()

class ReferenceSelectionTests(unittest.TestCase):
 def test_known_success_is_tried_first(self):
  from schedule import reference_index
  self.assertEqual(reference_index(1,14,0,20,7),7)
 def test_retry_still_rotates_other_layouts(self):
  from schedule import reference_index
  self.assertNotEqual(reference_index(1,14,1,20,7),reference_index(1,15,1,20,7))
 def test_without_preferred_preserves_original_rotation(self):
  from schedule import reference_index
  self.assertEqual(reference_index(1,14,0,20),14)

class RouteSelectionTests(unittest.TestCase):
 def test_preferred_existing_route_goes_first_without_removing_alternatives(self):
  from schedule import prioritize_routes
  routes=[{'strategy':'side_entry_0.06'},{'strategy':'direct'},{'strategy':'raised'}]
  self.assertEqual([x['strategy'] for x in prioritize_routes(routes,'raised')],['raised','side_entry_0.06','direct'])
 def test_unknown_preference_keeps_existing_order(self):
  from schedule import prioritize_routes
  routes=[{'strategy':'direct'},{'strategy':'raised'}]
  self.assertEqual(prioritize_routes(routes,'absent'),routes)

class CrossRoundRotationTests(unittest.TestCase):
 def test_physical_failures_rotate_all_same_cell_layouts_across_rounds(self):
  from schedule import reference_index
  tried={reference_index(1,r,0,20,7) for r in range(1,41)}
  self.assertEqual(tried,set(range(20)))
