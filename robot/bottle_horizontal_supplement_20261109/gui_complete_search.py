"""Private GUI wrapper: let each complete17x4 candidate search finish."""
import z1_isaac_bridge.grasp_gui_bridge as original
OriginalBridge=original.GraspGuiBridge
class CompleteSearchBridge(OriginalBridge):
 def _generate_pregrasp_for_auto(self):
  return self._call_trigger(self._generate_client,'/smolvla/auto_generate_pregrasp',timeout_s=1800.)
original.GraspGuiBridge=CompleteSearchBridge
if __name__=='__main__':original.main()
