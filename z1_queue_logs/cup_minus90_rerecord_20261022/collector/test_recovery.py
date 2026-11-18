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

class CorridorTests(unittest.TestCase):
    def test_side_entry_preserves_deviation_and_recovery_endpoint(self):
        from recovery_routes import side_entry_routes
        from source_old_preflight import routes
        import numpy as np
        from scipy.spatial.transform import Rotation
        q=Rotation.from_euler('xyz',[-90,6,30],degrees=True).as_quat().tolist()
        p=[.4,-.1,.08]
        task=dict(kind='near',offset_approach_m=[.0015,0,0],rotation_tool_deg=[0,0,0])
        original=next(routes(p,q,[.55,0,0],task))
        route=next(side_entry_routes(original))
        self.assertEqual(route['bad_position'],original['bad_position'])
        self.assertEqual(route['bad_orientation'],original['bad_orientation'])
        self.assertEqual(route['preparation'][-1]['position'],original['bad_position'])
        self.assertEqual(route['recovery'][-1]['position'],p)
        # Final preparation travels horizontally toward the cup, not down
        # through its sidewall; preceding point is at least 6cm behind.
        before=route['preparation'][-2]['position'];bad=route['bad_position']
        self.assertAlmostEqual(before[2],bad[2])
        self.assertGreaterEqual(np.linalg.norm(np.array(before[:2])-bad[:2]),.059)
