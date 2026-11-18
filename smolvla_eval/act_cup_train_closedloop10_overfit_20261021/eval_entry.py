"""Training-scene diagnostics only; production seed-exclusion logic is unchanged."""
import json,os,time
from pathlib import Path
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1')
from std_msgs.msg import String
from z1_isaac_bridge import smolvla_sim_rollout as module
from scene_checks import check_scene
R=Path(__file__).resolve().parent
PLAN=json.loads((R/'selected_scenes.json').read_text())
SCENES={s['seed']:s for s in PLAN['scenes']}
Base=module.SmolVLASimRollout

class TrainingSceneDiagnostic(Base):
 def _randomize_scene(self,target,fixed_seed=None):
  if target!='cup' or fixed_seed not in SCENES:raise ValueError('Only the ten selected training scenes are permitted')
  expected=SCENES[fixed_seed];self._set_active(False)
  self._call_trigger(self._reset_episode,120.)
  with self._lock:
   self._randomization_status={};self._evidence={};self._status={}
  msg=String();msg.data=json.dumps(dict(seed=fixed_seed,level='full',target_grid=expected['grid'],settle_duration_s=2.,non_target_max_angular_speed_rad_s=.35))
  self._randomization_command_pub.publish(msg)
  deadline=time.monotonic()+120
  while time.monotonic()<deadline:
   with self._lock:
    status=dict(self._randomization_status);joints=[self._joints.get(n) for n in module.JOINT_NAMES]
   if status.get('phase')=='ERROR':raise RuntimeError(f'Training scene reset failed: {status}')
   if status.get('seed')==fixed_seed and status.get('phase')=='RANDOMIZED':
    evidence=check_scene(expected,status,joints,PLAN['position_tolerance_m'],PLAN['orientation_tolerance_deg'],PLAN['initial_joint_tolerance_rad'])
    with (R/'scene_reproduction.jsonl').open('a') as f:f.write(json.dumps(dict(seed=fixed_seed,training_episode=expected['training_episode'],grid=expected['grid'],verified=True,**evidence))+'\n')
    return fixed_seed
   time.sleep(.02)
  raise TimeoutError(f'Training-scene reset timed out for {fixed_seed}')

module.SmolVLASimRollout=TrainingSceneDiagnostic
if __name__=='__main__':module.main()
