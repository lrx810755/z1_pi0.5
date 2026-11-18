"""Original apple expert with an additional low-angle fallback candidate bank."""
from z1_isaac_bridge import grasp_assistant as base

OriginalAssistant = base.GraspAssistant
PRIMARY_COUNT = 16
base.GRASP_PROFILES['apple']['candidate_count'] = 32

class AppleAssistant(OriginalAssistant):
    def _sample_candidate(self, object_position, target_object, candidate_index=0, avoid_box=False):
        if target_object == 'apple' and candidate_index >= PRIMARY_COUNT:
            return super()._sample_candidate(object_position, target_object,
                candidate_index - PRIMARY_COUNT, avoid_box,
                elevation_range_override=(60., 70.))
        return super()._sample_candidate(object_position, target_object, candidate_index, avoid_box)

if __name__ == '__main__':
    base.GraspAssistant = AppleAssistant
    base.main()
