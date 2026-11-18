"""Exercise the installed rollout loader and process bridge; never initialize ROS or publish."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from z1_isaac_bridge import smolvla_sim_rollout as rollout
from z1_isaac_bridge.pi05_policy_bridge import decode_batch
from z1_isaac_bridge.smolvla_rollout_core import binary_gripper_target
run=Path('/home/lp531/pi05_lora_workspace/act_correction_20261013')
cp=Path('/home/lp531/smolvla_training/act_correction_chunk16_20261013/checkpoints/050000/pretrained_model')
node=SimpleNamespace(args=SimpleNamespace(policy_type='act',execute_steps=5),
                     get_logger=lambda:SimpleNamespace(info=print))
policy=rollout.SmolVLASimRollout._load_policy(node,cp)
node.policy=policy
checks=[]
try:
 for entry in json.loads((run/'eval_bridge_inputs.json').read_text()):
  policy.reset()
  actions=rollout.SmolVLASimRollout._predict(node,decode_batch(entry['batch']))
  reference=np.asarray(entry['reference'])[0,:5]
  assert actions.shape==(5,7) and np.isfinite(actions).all()
  error=float(np.max(np.abs(actions-reference)))
  assert error<1e-5,(entry['source'],error)
  mapped=np.asarray([binary_gripper_target(a,-.75,'binary') for a in actions])
  assert np.isin(mapped[:,-1],[-.9,0]).all()
  checks.append(dict(source=entry['source'],episode=entry['episode'],max_abs_error=error,execution_shape=list(actions.shape)))
 report=dict(checkpoint=str(cp),rollout_module=rollout.__file__,camera_keys=list(node.camera_keys),
             state_dim=node.state_dim,device=str(node.device),binary_gripper_checked=True,checks=checks,
             robot_commands_sent=False)
 (run/'eval_integration_verified.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2))
finally:
 policy.close()
