import unittest
import numpy as np
from preflight import routes, choose_route, trajectory_points

class PreflightTests(unittest.TestCase):
    def test_corridors_preserve_requested_low_and_tilt(self):
        task={'offset_approach_m':[0,0,-.018],'rotation_tool_deg':[14,0,0]}
        candidates=list(routes([.5,0,.10],[0,0,0,1],[.65,0,0],task))
        self.assertGreater(len(candidates),1)
        self.assertGreater(len({c['lift_m'] for c in candidates}),1)
        for c in candidates:
            self.assertAlmostEqual(c['bad_position'][2]-c['anchor_position'][2],-.018)
            self.assertEqual(c['recovery'][0]['orientation'],c['bad_orientation'])
            self.assertGreater(c['recovery'][0]['position'][2],c['bad_position'][2])
            self.assertEqual(c['recovery'][-1]['position'],[.5,0,.1])
    def test_failed_whole_route_never_selected(self):
        seen=[]
        def plan(route):
            seen.append(route)
            if route==1:raise RuntimeError('recovery unreachable after valid preparation')
            return ['prep','recovery']
        chosen,plans,rejected=choose_route([1,2],plan)
        self.assertEqual(chosen,2);self.assertEqual(plans,['prep','recovery'])
        self.assertEqual(seen,[1,2]);self.assertEqual(len(rejected),1)
    def test_all_fail_closed(self):
        with self.assertRaisesRegex(RuntimeError,'预检'):
            choose_route([1,2],lambda r: (_ for _ in ()).throw(RuntimeError('no IK')))
    def test_joint_mapping_keeps_gripper(self):
        from types import SimpleNamespace as NS
        trajectory=NS(joint_names=['joint3','joint1'],points=[NS(positions=[-.5,.2])])
        points=trajectory_points(trajectory,[0,1,-1,0,0,0,-.9])
        np.testing.assert_allclose(points[-1],[.2,1,-.5,0,0,0,-.9])

if __name__=='__main__':unittest.main()
