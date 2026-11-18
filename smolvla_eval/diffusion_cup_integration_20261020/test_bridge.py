import sys,json,time
from pathlib import Path
import torch
from torch.utils.data import default_collate
sys.path.insert(0,'/home/lp531/pi05_lora_workspace/diffusion_cup_gentle_cached_20261020')
sys.path.insert(0,'/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge')
from cache_data import CupDataset
from z1_isaac_bridge.diffusion_policy_bridge import DiffusionProcessPolicy
r=Path('/home/lp531/pi05_lora_workspace/diffusion_cup_gentle_cached_20261020');split=json.loads((r/'split_manifest.json').read_text());ds=CupDataset(split['validation_episodes'],raw_images=True)
policy=DiffusionProcessPolicy('/home/lp531/smolvla_training/diffusion_cup_gentle_cached_h16_20261020/checkpoints/050000/pretrained_model_ema');checks=[]
try:
 for source in ['original','old_correction','new_correction']:
  rec=next(x for x in split['source_records'] if x['split']=='validation' and x['source']==source);anchor=rec['dataset_from_index']+max(1,rec['close_frame']-10);b=default_collate([ds[ds.absolute_to_relative_idx[anchor]]]);b={k:v for k,v in b.items() if k.startswith('observation.')}
  for k in ds.meta.camera_keys:b[k]=b[k].float()/255
  policy.reset();start=time.monotonic();a=policy.predict_action_chunk(b);elapsed=time.monotonic()-start
  assert a.shape==(1,15,7) and torch.isfinite(a).all()
  policy.reset();again=policy.predict_action_chunk(b);assert torch.allclose(a,again,atol=1e-6)
  checks.append(dict(source=source,shape=list(a.shape),wall_s=elapsed,reset_deterministic=True))
finally:policy.close()
Path(__file__).with_name('bridge_verified.json').write_text(json.dumps(checks,indent=2));print(checks)
