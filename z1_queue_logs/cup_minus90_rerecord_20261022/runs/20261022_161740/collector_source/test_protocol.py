import copy
import unittest
from protocol import check_scene, full_labels, validate_saved


class ProtocolTests(unittest.TestCase):
    def test_scene_rejects_same_seed_with_different_cup_position(self):
        expected = {'seed': 1, 'grid': [0, 0], 'scene_randomization': {
            'object_positions_m': {'cup': [0.4, 0, 0]},
            'object_orientations_xyzw': {'cup': [0, 0, 0, 1]}}}
        actual = {'seed': 1, 'target_grid': [0, 0], 'object_positions_m': {'cup': [0.41, 0, 0]},
                  'object_orientations_xyzw': {'cup': [0, 0, 0, 1]}}
        with self.assertRaises(ValueError):
            check_scene(expected, actual)
        actual['object_positions_m']['cup'][0] = 0.400001
        self.assertLess(check_scene(expected, actual)['maximum_position_error_m'], 0.002)

    def test_labels_include_real_release_instead_of_closed_tail(self):
        angles = [-.9]*5 + [-.85, -.8, -.74] + [-.72]*20 + [-.78, -.84, -.9] + [-.9]*20
        rows = [{'action': [0]*6+[a]} for a in angles]
        labels, close, release = full_labels(rows)
        self.assertEqual(close, 5)
        self.assertEqual(release, 28)
        self.assertEqual(labels[:close], [0]*close)
        self.assertTrue(all(x == 1 for x in labels[close:release]))
        self.assertTrue(all(x == 0 for x in labels[release:]))
        with self.assertRaises(ValueError):
            full_labels(rows[:28])

    def test_saved_acceptance_requires_release_and_minus90(self):
        status = {'target_object': 'cup', 'task_success': True, 'physical_grasp_ok': True,
                  'release_sequence_complete': True, 'placement_stable': True,
                  'released': True, 'holding_observed': False, 'frozen': False,
                  'target_collisions_enabled': True}
        validate_saved(status, -89.7, False, 30)
        for key in ['task_success', 'release_sequence_complete', 'placement_stable', 'released']:
            bad = copy.deepcopy(status); bad[key] = False
            with self.assertRaises(ValueError):
                validate_saved(bad, -90, False, 30)
        with self.assertRaises(ValueError):
            validate_saved(status, 90, False, 30)
        with self.assertRaises(ValueError):
            validate_saved(status, -90, True, 30)
        with self.assertRaises(ValueError):
            validate_saved(status, -90, False, 3)


if __name__ == '__main__':
    unittest.main()
