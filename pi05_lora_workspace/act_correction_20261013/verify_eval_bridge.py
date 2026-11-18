"""Build direct inference references for the ROS/Python process boundary, without robot commands."""
import json,sys
import numpy as np
import torch
from torch.utils.data import default_collate
from train import make_config,datasets
from prepare import RUN
sys.path.insert(0,'/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge')
from z1_isaac_bridge.pi05_policy_bridge import encode_batch
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
cp='/home/lp531/smolvla_training/act_correction_chunk16_20261013/checkpoints/050000/pretrained_model'
policy=ACTPolicy.from_pretrained(cp,local_files_only=True,strict=True).eval()
pre,post=make_pre_post_processors(policy.config,pretrained_path=cp)
split=json.loads((RUN/'split_manifest.json').read_text())
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig
cfg=TrainPipelineConfig(dataset=DatasetConfig(repo_id='lp531/z1_bottle_with_fragments_20261012',root='/home/lp531/smolvla_data/z1_bottle_with_fragments_20261012',episodes=split['train_episodes'],video_backend='pyav'),policy=policy.config)
cfg.dataset.image_transforms.enable=False
_,val=datasets(cfg)
checks=[]
with np.load(RUN/'sampling_arrays.npz') as a:
 for source in ['original','old_correction','new_fragment']:
  ep=next(r['episode_index'] for r in split['source_records'] if r['source']==source and r['episode_index'] in split['validation_episodes'])
  ids=np.flatnonzero((a['episode']==ep)&a['eligible_validation'])
  anchor=int(ids[len(ids)//2])
  sample=default_collate([val[val.absolute_to_relative_idx[anchor]]])
  batch={'observation.state':sample['observation.state']}
  for cam in val.meta.camera_keys:batch[cam]=sample[cam].float()/255
  policy.reset()
  with torch.inference_mode():reference=post(policy.predict_action_chunk(pre(batch)))
  checks.append(dict(source=source,episode=ep,batch=encode_batch(batch),reference=reference.tolist()))
(RUN/'eval_bridge_inputs.json').write_text(json.dumps(checks))
print('Prepared direct ACT references for three held-out source types.')
