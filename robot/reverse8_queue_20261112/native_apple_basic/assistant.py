"""Original 16-candidate apple expert, with guarded reuse of plans/checks."""
import importlib.util
from pathlib import Path
import sys

# Preserve the ROS package context required by the frozen assistant's relative imports.
spec = importlib.util.spec_from_file_location(
    'z1_isaac_bridge.reverse8_apple_snapshot', Path(__file__).with_name('base_assistant_snapshot.py'))
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
from wait_optimization import AppleWaitOptimization

base.GRASP_PROFILES['apple']['candidate_count'] = 16
# Restore the bounded preload seen in the original successful source (~0.08rad).
base.GRASP_PROFILES['apple']['contact_hold_minimum_preload_rad'] = .08

class AppleAssistant(AppleWaitOptimization, base.GraspAssistant):
    def _plan_apple_transfer(self, *args, **kwargs):
        previous=self.physical_test_lift_velocity_scale
        self.physical_test_lift_velocity_scale=min(previous,.06)
        try:
            return super()._plan_apple_transfer(*args,**kwargs)
        finally:
            self.physical_test_lift_velocity_scale=previous

if __name__ == '__main__':
    base.GraspAssistant = AppleAssistant
    base.main()
