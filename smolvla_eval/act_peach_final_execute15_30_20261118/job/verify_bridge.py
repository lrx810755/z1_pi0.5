"""Exercise real ROS-side ACT loader and 15-step truncation without publishing."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from entry import runtime,CHECKPOINT,HERE
from z1_isaac_bridge.pi05_policy_bridge import decode_batch
from z1_isaac_bridge.smolvla_rollout_core import binary_gripper_target
node=SimpleNamespace(args=SimpleNamespace(policy_type='act',execute_steps=15),get_logger=lambda:SimpleNamespace(info=print))
policy=runtime.SmolVLASimRollout._load_policy(node,CHECKPOINT);node.policy=policy
checks=[]
try:
    for sample in json.loads((HERE/'reference_inputs.json').read_text()):
        policy.reset()
        batch=decode_batch(sample['batch']);result=runtime.SmolVLASimRollout._predict(node,batch)
        expected=np.asarray(sample['reference'])[0,:15]
        error=float(np.abs(expected-result).max())
        assert result.shape==(15,7) and np.isfinite(result).all() and error<1e-5,error
        mapped=np.asarray([binary_gripper_target(x,-.75,'binary') for x in result])
        assert np.isin(mapped[:,-1],[-.9,0]).all()
        checks.append(dict(source=sample['source'],max_absolute_error=error,executed_shape=list(result.shape)))
finally:policy.close()
report=dict(passed=True,checkpoint=str(CHECKPOINT),tests=checks,cameras=node.camera_keys,state_dim=node.state_dim,temporal_ensemble_used=False,robot_commands_sent=False)
(HERE/'bridge_verified.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
