"""Load the deployed worker and infer one real three-camera frame; no ROS commands."""
from pathlib import Path
import json
import sys
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, '/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge')
from z1_isaac_bridge.pi05_policy_bridge import PI05ProcessPolicy
from z1_isaac_bridge.smolvla_rollout_core import binary_gripper_target, TASK_INSTRUCTIONS
run = Path(__file__).resolve().parent
label = json.loads((run / 'episode_labels.json').read_text())[0]
root = Path(label['raw_root'])
frame = label['close_frame'] - 1
row = json.loads((root / 'frames.jsonl').read_text().splitlines()[frame])
batch = {'observation.state': torch.tensor([row['observation.state']], dtype=torch.float32),
         'task': [TASK_INSTRUCTIONS['beverage_bottle']]}
for camera in ('wrist', 'global', 'global_right'):
    key = 'observation.images.' + camera
    array = np.array(Image.open(root / row[key]).convert('RGB'))
    batch[key] = torch.from_numpy(array).permute(2,0,1).unsqueeze(0).float()/255
checkpoint = Path('/home/lp531/smolvla_training/pi05_bottle_binary_chunk15_attn_ffn_20261007/checkpoints/080000/pretrained_model')
policy = PI05ProcessPolicy(checkpoint)
try:
    policy.reset()
    actions = policy.predict_action_chunk(batch)[0].numpy()
    assert actions.shape == (15,7) and np.isfinite(actions).all()
    targets = np.array([binary_gripper_target(action, encoding='binary') for action in actions])
    assert np.array_equal(targets[:,:6], actions[:,:6])
    result = dict(passed=True, checkpoint=str(checkpoint), scope='offline saved three-camera frame; no commands published',
                  frame=frame, shape=list(actions.shape), gripper_predictions=actions[:,-1].tolist(),
                  gripper_targets_rad=targets[:,-1].tolist())
    (run/'eval_interface_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
finally:
    policy.close()
