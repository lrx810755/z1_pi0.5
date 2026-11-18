import unittest
from analysis import episode_metrics


class MetricsTests(unittest.TestCase):
    def test_contact_before_first_close_is_not_counted_as_closed_contact(self):
        result = {'seed': 101, 'success': False, 'failure_category': 'grasp_failed',
                  'simulation_duration_s': 5, 'physical_predicates': {}, 'final_status': {}}
        trace = [
            {'command': [0]*6+[-.9], 'policy_action': [0]*6+[-.9], 'simulation_elapsed_s': 0,
             'contact_status': {}},
            {'command': [0]*6+[-.7], 'policy_action': [0]*6+[-.7], 'simulation_elapsed_s': 1,
             'contact_status': {'left_finger_contact_object': 'beverage_bottle'}},
        ]
        m = episode_metrics(result, trace)
        self.assertEqual(m['first_close_sim_s'], 1)
        self.assertTrue(m['preclose_target_contact'])
        self.assertTrue(m['closed_once'])

    def test_never_closed_and_final_table_contact(self):
        result = {'seed': 102, 'success': False, 'failure_category': 'finger_table_contact',
                  'simulation_duration_s': 5, 'physical_predicates': {},
                  'final_status': {'left_finger_touching_table': True}}
        trace = [{'command': [0]*6+[-.9], 'policy_action': [0]*6+[-.9],
                  'simulation_elapsed_s': 0, 'contact_status': {}}]
        m = episode_metrics(result, trace)
        self.assertFalse(m['closed_once'])
        self.assertIsNone(m['first_close_sim_s'])
        self.assertTrue(m['table_contact'])
        self.assertFalse(m['preclose_target_contact'])


if __name__ == '__main__':
    unittest.main()
