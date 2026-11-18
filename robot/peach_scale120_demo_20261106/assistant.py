"""Peach expert with its own geometry and guarded wait reductions."""
import base_assistant_snapshot as base
from wait_optimization import PeachWaitOptimization
base.GRASP_PROFILES['peach']['candidate_count']=16
class PeachAssistant(PeachWaitOptimization,base.GraspAssistant):
    def _generate_plan(self):
        if self.target_object!='peach':raise RuntimeError('此入口只用于桃子采集')
        return super()._generate_plan()
if __name__=='__main__':
    base.GraspAssistant=PeachAssistant
    base.main()
