import json
from pathlib import Path
import unittest
import protocol

class RecoveryTests(unittest.TestCase):
    def test_shift_compares_same_phase_of_archived_scene(self):
        plan=json.loads((Path(__file__).parent.parent/'collection_plan.json').read_text())
        for task in plan['tasks']:
            if task['id'] in ('new_correction_episode_000001','new_correction_episode_000011'):
                old=task['original_correction']
                expected=protocol.archived_shift_position(old)
                self.assertEqual(expected,old['prepared_cup_shift']['after'])
                self.assertNotEqual(expected,old['actual_cup_position_m'])
