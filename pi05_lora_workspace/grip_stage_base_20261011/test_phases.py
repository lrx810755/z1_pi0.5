import unittest
import numpy as np
from phases import build_phase_weights, probabilities

class PhaseWeightTests(unittest.TestCase):
    def rows(self):
        return [{'physical_grasp_ok': i >= 25, 'object_touching_table': i < 25} for i in range(50)]

    def test_close_preparation_hold_and_release_boundaries(self):
        weights, event = build_phase_weights(self.rows(), 20, 40)
        np.testing.assert_array_equal(weights[:10], 1)
        np.testing.assert_array_equal(weights[10:20], 1.5)
        np.testing.assert_array_equal(weights[20:30], 2)
        np.testing.assert_array_equal(weights[30:], 1)
        self.assertEqual(event['stable_lift_confirmed_frame'], 29)
        self.assertEqual(weights[40], 1)  # Correct release must remain learnable.

    def test_one_contact_free_frame_does_not_end_hold_window(self):
        rows = self.rows()
        rows[22] = {'physical_grasp_ok': True, 'object_touching_table': False}
        weights, event = build_phase_weights(rows, 20, 40)
        self.assertEqual(event['stable_lift_confirmed_frame'], 29)
        self.assertEqual(weights[24], 2)

    def test_no_stable_lift_before_release_is_rejected(self):
        rows = self.rows()
        for row in rows:
            row['physical_grasp_ok'] = False
        with self.assertRaises(ValueError):
            build_phase_weights(rows, 20, 40)

    def test_phase_windows_are_clipped_to_episode_start(self):
        weights, event = build_phase_weights(self.rows(), 3, 40)
        np.testing.assert_array_equal(weights[:3], 1.5)
        self.assertEqual(event['preclose_start_frame'], 0)

    def test_source_masses_and_relative_phase_weights(self):
        correction = np.array([False]*3 + [True]*3)
        weights = np.array([1, 1.5, 2, 2, 1.5, 1])
        p = probabilities(correction, weights)
        self.assertAlmostEqual(p[~correction].sum(), .7)
        self.assertAlmostEqual(p[correction].sum(), .3)
        self.assertAlmostEqual(p[1]/p[0], 1.5)
        self.assertAlmostEqual(p[2]/p[0], 2)
        self.assertAlmostEqual(p[3]/p[5], 2)

    def test_invalid_pool_or_weights_are_rejected(self):
        for correction, weights in [([False, False], [1, 2]), ([False, True], [1, float('nan')]), ([False, True], [1, 0])]:
            with self.assertRaises(ValueError):
                probabilities(correction, weights)

if __name__ == '__main__':
    unittest.main()
