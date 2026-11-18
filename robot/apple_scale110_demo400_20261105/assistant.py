"""Original 16-candidate apple expert, with guarded reuse of plans/checks."""
from z1_isaac_bridge import grasp_assistant as base
from wait_optimization import AppleWaitOptimization

base.GRASP_PROFILES['apple']['candidate_count'] = 16

class AppleAssistant(AppleWaitOptimization, base.GraspAssistant):
    pass

if __name__ == '__main__':
    base.GraspAssistant = AppleAssistant
    base.main()
