import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,'/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge')
from z1_isaac_bridge.pi05_policy_bridge import PI05ProcessPolicy
from z1_isaac_bridge.smolvla_rollout_core import camera_keys_for_features,TASK_INSTRUCTIONS
from PIL import Image
checkpoint=Path('/home/lp531/smolvla_training/pi05_bottle_three_camera_20261006/best')
raw=Path('/home/lp531/smolvla_data/z1_tabletop_raw_left_wrist_three_camera_v1')
episode=sorted(raw.glob('episode_[0-9][0-9][0-9][0-9][0-9][0-9]'))[0]
frame=json.loads((episode/'frames.jsonl').read_text().splitlines()[0])
policy=PI05ProcessPolicy(checkpoint)
try:
    cameras=camera_keys_for_features(policy.config.input_features)
    assert cameras==('wrist','global','global_right'),cameras
    batch={'observation.state':torch.tensor([frame['observation.state']],dtype=torch.float32),
           'task':[TASK_INSTRUCTIONS['beverage_bottle']]}
    for name in cameras:
        key=f'observation.images.{name}'
        rgb=np.array(Image.open(episode/frame[key]).convert('RGB'))
        batch[key]=torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).float()/255
    policy.reset()
    actions=policy.predict_action_chunk(batch)
    assert tuple(actions.shape)==(1,50,7) and torch.isfinite(actions).all()
    result={'checkpoint':str(checkpoint.resolve()),'cameras':list(cameras),
            'input_shapes':{k:list(v.shape) for k,v in batch.items() if hasattr(v,'shape')},
            'action_shape':list(actions.shape),'finite':True,'source_episode':str(episode),
            'scope':'Real checkpoint prediction from saved synchronized RGB/state; no robot commands'}
    Path(__file__).with_name('result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
finally:policy.close()
