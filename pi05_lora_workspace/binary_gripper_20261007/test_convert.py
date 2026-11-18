import unittest
import convert

class LabelsTest(unittest.TestCase):
    def test_commands_define_intent_even_when_state_is_obstructed(self):
        phases = ['READY', 'GRASPING', 'GRASPING', 'PHYSICAL_TEST_LIFT', 'TRANSPORT', 'TRANSPORT', 'TRANSPORT', 'TRANSPORT', 'PLACING']
        angles = [-.9, -.84, -.64, -.64, -.64, -.71, -.80, -.9, -.9]
        rows = [{'phase': phase, 'action': [0]*6+[angle], 'observation.state': [0]*6+[-.71]} for phase, angle in zip(phases, angles)]
        labels, event = convert.label_episode(rows)
        self.assertEqual(labels.tolist(), [0,1,1,1,1,0,0,0,0])
        self.assertEqual(event['close_frame'], 1)
        self.assertEqual(event['open_frame'], 5)
        self.assertEqual(rows[5]['observation.state'][-1], -.71)

    def test_ambiguous_release_is_rejected(self):
        rows = [{'phase': p, 'action': [0]*6+[a]} for p,a in [('READY',-.9),('GRASPING',-.84),('TRANSPORT',-.64),('PLACING',-.9)]]
        with self.assertRaises(ValueError): convert.label_episode(rows)

if __name__ == '__main__': unittest.main()
