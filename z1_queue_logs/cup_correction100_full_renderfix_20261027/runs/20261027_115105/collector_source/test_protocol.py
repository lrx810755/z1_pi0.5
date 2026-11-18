import importlib.util
from pathlib import Path
import unittest


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).with_name('protocol.py')
        self.assertTrue(path.exists(), 'Missing correction100 acceptance protocol')
        spec = importlib.util.spec_from_file_location('correction100_protocol', path)
        self.p = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.p)

    def test_quota_plan_covers_all_categories_cells_and_holdouts(self):
        from collections import Counter
        tasks = self.p.build_tasks()
        self.assertEqual(len(tasks), 100)
        self.assertEqual(len({t['id'] for t in tasks}), 100)
        self.assertEqual(Counter(t['kind'] for t in tasks), {
            'approach': 20, 'close_transition': 15, 'shifted_target': 20,
            'low_clearance': 10, 'regrasp': 10, 'transport': 10, 'placement': 15})
        self.assertEqual(set(Counter(tuple(t['grid']) for t in tasks).values()), {5})
        self.assertEqual(sum(t['split'] == 'validation' for t in tasks), 10)
        self.assertTrue(all(t['source_type'] == 'synthetic_failure_informed' for t in tasks))

    def test_gripper_events_represent_retry_and_final_release(self):
        rows = [{'simulation_time': i / 10, 'action': [0] * 6 + [a]}
                for i, a in enumerate([-.9, -.9, -.82, -.75, -.9, -.9, -.8, -.65, -.65, -.7, -.9, -.9])]
        events = [{'event': 'gripper_intent', 'time': .2, 'state': 'close'},
                  {'event': 'gripper_intent', 'time': .4, 'state': 'open'},
                  {'event': 'gripper_intent', 'time': .6, 'state': 'close'},
                  {'event': 'gripper_intent', 'time': .9, 'state': 'open'}]
        labels, close, release = self.p.command_labels(rows, events)
        self.assertEqual(labels, [0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0])
        self.assertEqual((close, release), (6, 9))

    def test_intervention_masks_exclude_every_intersecting_action_window(self):
        rows = [{'simulation_time': i / 10} for i in range(40)]
        events = [{'event': 'intervention_start', 'time': 1.0},
                  {'event': 'intervention_end', 'time': 1.4}]
        mask, anchors = self.p.training_mask(rows, events, horizon=16)
        self.assertFalse(any(mask[10:15]))
        self.assertFalse(any(anchors[:15]))
        self.assertTrue(anchors[15])
        self.assertFalse(any(anchors[25:]))

    def test_requested_category_without_measured_evidence_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'measured'):
            self.p.validate_category({'kind': 'transport'}, {'recovery_complete': True})
        self.p.validate_category({'kind': 'transport'}, {
            'recovery_complete': True, 'measured_disturbance_m': .02,
            'recovery_after_disturbance': True})
        with self.assertRaisesRegex(ValueError, 'contact'):
            self.p.validate_category({'kind': 'regrasp'}, {
                'recovery_complete': True, 'retry_performed': True, 'retry_contact_seen': False})

    def test_healthy_bilateral_clamp_cannot_fill_regrasp_quota(self):
        with self.assertRaisesRegex(ValueError, 'unilateral'):
            self.p.validate_category({'kind': 'regrasp'}, {
                'recovery_complete': True, 'retry_performed': True, 'retry_contact_seen': True,
                'retry_unilateral_contact': False})

    def test_placement_state_is_high_and_centered_distinct_from_transport(self):
        tasks = self.p.build_tasks()
        self.assertTrue(all(t['loaded_offset_world_m'][:2] == [0., 0.]
                            and t['loaded_offset_world_m'][2] >= .025
                            for t in tasks if t['kind'] == 'placement'))
        with self.assertRaisesRegex(ValueError, 'hover'):
            self.p.validate_category({'kind': 'placement'}, {
                'recovery_complete': True, 'measured_disturbance_m': .03,
                'recovery_after_disturbance': True})

    def test_actual_preparation_clearance_and_contact_are_mandatory(self):
        self.assertTrue(callable(getattr(self.p, 'validate_preparation', None)), 'Missing preparation acceptance')
        for clearance, contact in [(-.001, False), (.002, False), (.01, True)]:
            with self.assertRaises(ValueError):
                self.p.validate_preparation(clearance, contact)
        self.p.validate_preparation(.004, False)

    def test_final_flags_do_not_substitute_for_measured_withdrawal(self):
        self.assertTrue(callable(getattr(self.p, 'validate_withdrawal', None)), 'Missing withdrawal acceptance')
        with self.assertRaisesRegex(ValueError, 'withdrawal'):
            self.p.validate_withdrawal({'withdrawal_rise_m': .005})
        self.p.validate_withdrawal({'withdrawal_rise_m': .035})


if __name__ == '__main__':
    unittest.main()
