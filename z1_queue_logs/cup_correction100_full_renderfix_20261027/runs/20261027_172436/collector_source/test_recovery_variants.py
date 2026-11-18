import importlib.util
from pathlib import Path
import unittest


class ElevatedApproachTests(unittest.TestCase):
    def test_horizontal_error_is_preserved_and_recovery_reaches_original_reference(self):
        path = Path(__file__).with_name('recovery_variants.py')
        self.assertTrue(path.exists(), 'Missing safe-height approach recovery variants')
        spec = importlib.util.spec_from_file_location('recovery_variants', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        original = dict(bad_position=[.396, .112, .08], bad_orientation=[0., 0., 0., 1.],
                        preparation=[], recovery=[])
        reference = [.4, .1, .08]
        routes = list(module.elevated_approach_routes(original, reference, [0., 0., 0., 1.]))
        self.assertEqual(len(routes), 3)
        for route, added in zip(routes, [.055, .04, .025]):
            self.assertEqual(route['bad_position'][:2], original['bad_position'][:2])
            self.assertAlmostEqual(route['bad_position'][2], .08 + added)
            self.assertEqual(route['recovery'][-1]['position'], reference)
            self.assertEqual(route['preparation'][-1]['position'], route['bad_position'])
            self.assertEqual(route['recovery'][0]['position'][2], route['bad_position'][2])
            # A frustum's wide rim must not be traversed by vertical descent
            # at the cup centre; lower behind it, then enter horizontally.
            before_final = route['recovery'][-2]['position']
            self.assertEqual(before_final[2], reference[2])
            self.assertAlmostEqual(before_final[0], reference[0] - .06)
            self.assertEqual(before_final[1], reference[1])
        self.assertEqual(original['bad_position'], [.396, .112, .08])


if __name__ == '__main__':
    unittest.main()
