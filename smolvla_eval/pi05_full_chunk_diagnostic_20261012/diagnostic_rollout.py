"""Use the normal ROS rollout with full-chunk logging; execution behavior is identical."""
from contextlib import nullcontext
from pathlib import Path
import time
import torch
from z1_isaac_bridge import smolvla_sim_rollout as rollout
from chunk_logging import record_prediction

class DiagnosticRollout(rollout.SmolVLASimRollout):
 def _predict(self,batch):
  _,joints,status,evidence=self._snapshot()
  with self._lock:sim_stamp=self._joint_sim_stamp
  metadata=dict(prediction=self._last_input_sync['prediction'],input_sync=dict(self._last_input_sync),
    before_prediction_sim_stamp=sim_stamp,observed_joint_state=joints.tolist(),contact_status=status,physical_evidence=evidence)
  context=torch.autocast(device_type='cuda',dtype=torch.float16) if self.use_amp else nullcontext()
  started=time.monotonic()
  with torch.inference_mode(),context:chunk=self.policy.predict_action_chunk(batch)
  metadata['inference_wall_s']=time.monotonic()-started
  with self._lock:metadata['after_prediction_sim_stamp']=self._joint_sim_stamp
  return record_prediction(chunk.detach().float().cpu().numpy(),self.args.execute_steps,
    Path(self._input_sync_path).parent/'predictions.jsonl',metadata)

if __name__=='__main__':
 rollout.SmolVLASimRollout=DiagnosticRollout
 rollout.main()
