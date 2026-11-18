import unittest
import numpy as np
from windows import build_window, anchor_mask


def episode(n=100, close=20, stable_start=40):
    return [dict(frame_index=i, simulation_time=i/10, phase='READY' if i<close else 'GRASPING',
                 action=[0]*6+[-.9 if i<close else -.64], physical_grasp_ok=i>=stable_start,
                 object_touching_table=i<stable_start, grasp_success=i>=stable_start) for i in range(n)]


class WindowTests(unittest.TestCase):
    def test_complete_prediction_ends_before_release(self):
        rows=episode();meta={'stages':[dict(start_sim_time=1.0,end_sim_time=1.8)]}
        w=build_window(rows,meta,close=20,opening=52)
        self.assertEqual(w['frame_start'],5)
        self.assertEqual(w['frame_end_inclusive'],51)
        self.assertEqual(w['anchor_end_inclusive'],37)
        self.assertTrue(w['hold_truncated'])
        for start in range(w['anchor_start'],w['anchor_end_inclusive']+1):
            self.assertLess(start+14,52)

    def test_transient_lift_is_not_stability_confirmation(self):
        rows=episode(stable_start=60)
        for i in range(25,29):rows[i].update(physical_grasp_ok=True,object_touching_table=False,grasp_success=True)
        w=build_window(rows,{'stages':[]},close=20)
        self.assertEqual(w['stable_confirmed_frame'],64)
        self.assertEqual(w['frame_end_inclusive'],79)
        self.assertEqual(w['frame_start'],10)

    def test_validation_is_excluded_and_original_range_is_preserved(self):
        records=[dict(episode_index=0,split='train',anchor_start=0,anchor_end_inclusive=9),
                 dict(episode_index=1,split='train',anchor_start=2,anchor_end_inclusive=4),
                 dict(episode_index=2,split='validation',anchor_start=0,anchor_end_inclusive=9)]
        ep=np.repeat(np.arange(3),10);fi=np.tile(np.arange(10),3)
        mask=anchor_mask(ep,fi,records)
        np.testing.assert_array_equal(np.flatnonzero(mask),list(range(10))+[12,13,14])

    def test_missing_stable_grasp_is_rejected(self):
        with self.assertRaises(ValueError):build_window(episode(stable_start=100),{'stages':[]},close=20)

if __name__=='__main__':unittest.main()
