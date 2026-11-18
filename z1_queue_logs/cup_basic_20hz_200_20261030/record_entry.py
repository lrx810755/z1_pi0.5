"""Original recorder with persistent synchronization-fault evidence only."""
from pathlib import Path
from fault_latch import latch_fault
import z1_isaac_bridge.episode_recorder as recorder
Original=recorder.EpisodeRecorder
class AuditedRecorder(Original):
 def _sample_callback(self):
  # Keep the same recorder lock through failure capture so discard cannot clear it.
  with self._lock:
   try:super()._sample_callback()
   finally:latch_fault(Path(__file__).with_name('sync_fault.json'),self._sync_failure,self._episode_index)
recorder.EpisodeRecorder=AuditedRecorder
if __name__=='__main__':recorder.main()
