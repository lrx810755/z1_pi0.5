"""Exercise collection motion routing without starting ROS or Isaac."""
import sys
import types
import unittest
from unittest.mock import patch
import nodes


class MotionBackend:
    def _plan_candidate(self, candidate, linear=False, **kwargs):
        self.routes.append(linear)
        if linear and self.failure:
            raise RuntimeError(self.failure)
        return candidate
    def _move_object_to(self, destination, description, linear, **kwargs):
        self.routes.append(linear)
        if linear and self.failure:
            raise RuntimeError(self.failure)
        self.position = destination
        return destination

    def get_logger(self):
        return types.SimpleNamespace(warning=lambda message:None)


def assistant(failure=None):
    module = types.ModuleType('z1_isaac_bridge.grasp_assistant')
    module.GraspAssistant = MotionBackend
    module.GRASP_PROFILES = {'cup':{'candidate_azimuth_offsets_deg':(0.,)}}
    module.OBJECT_GRASP_PROFILES = {'cup':{}}
    module.BOX_TRANSPORT_XY_TOLERANCE_M = {}
    module.GRASP_CONFIG = {'physical_test_lift':{}}
    module._is_recoverable_lift_planning_error = lambda exc:'MoveIt error_code=' in str(exc)
    module.main = lambda:None
    package = types.ModuleType('z1_isaac_bridge'); package.grasp_assistant = module
    with patch.dict(sys.modules,{'z1_isaac_bridge':package,'z1_isaac_bridge.grasp_assistant':module}):
        nodes.assistant_main()
    obj = object.__new__(module.GraspAssistant)
    obj.routes = []; obj.failure = failure; obj.position = None
    return obj


class MotionTests(unittest.TestCase):
    def test_open_gripper_final_approach_can_route_around_blocker(self):
        obj=assistant('MoveIt error_code=-2')
        obj.generating_grasp=True
        candidate={'position':(.4,0,.08)}
        self.assertEqual(obj._plan_candidate(candidate,linear=True),candidate)
        self.assertEqual(obj.routes,[True,False])

    def test_reference_search_retains_successful_final_candidates(self):
        obj=assistant()
        obj.generating_grasp=True
        obj.reference_candidates=[]
        candidate={'position':(.4,0,.08),'final_grasp_target':True}
        obj._plan_candidate(candidate,linear=True)
        self.assertEqual(len(obj.reference_candidates),1)
        self.assertEqual(obj.reference_candidates[0][0],candidate)

    def test_loaded_carry_uses_straight_motion_when_available(self):
        obj = assistant()
        destination = (.32,.14,.162)
        obj._move_object_to(destination,'带载荷自由规划到蓝盒上方安全位',False)
        self.assertEqual(obj.routes,[True])
        self.assertEqual(obj.position,destination)

    def test_planning_failure_falls_back_to_original_free_planner(self):
        obj = assistant('MoveIt error_code=-31')
        destination = (.32,.14,.162)
        obj._move_object_to(destination,'带载荷自由规划到蓝盒上方安全位',False)
        self.assertEqual(obj.routes,[True,False])
        self.assertEqual(obj.position,destination)

    def test_contact_stop_never_falls_back(self):
        obj = assistant('Finger/table contact detected')
        with self.assertRaises(RuntimeError):
            obj._move_object_to((.32,.14,.162),'带载荷自由规划到蓝盒上方安全位',False)
        self.assertEqual(obj.routes,[True])


if __name__=='__main__':
    unittest.main()
