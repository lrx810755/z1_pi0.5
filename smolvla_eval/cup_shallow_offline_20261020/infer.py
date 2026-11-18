import sys,json,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import default_collate
R=Path('/home/lp531/pi05_lora_workspace/act_cup_recovery_cached_20261020');sys.path.insert(0,str(R))
from cache_data import CupDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies import make_pre_post_processors
P=Path(__file__).resolve().parent
CK='/home/lp531/smolvla_training/act_cup_recovery_gentle_cached_chunk16_20261020/checkpoints/050000/pretrained_model'
torch.set_num_threads(4);torch.manual_seed(0)
probes=json.loads((P/'probes.json').read_text());ds=CupDataset(sorted(set(x['episode_index'] for x in probes['samples'])),raw_images=True)
policy=ACTPolicy.from_pretrained(CK,local_files_only=True);policy.eval();pre,post=make_pre_post_processors(policy_cfg=policy.config,pretrained_path=CK)
preds=[];truth=[];started=time.monotonic()
for offset in range(0,len(probes['samples']),8):
 ids=probes['absolute_indices'][offset:offset+8];batch=default_collate([ds[ds.absolute_to_relative_idx[i]] for i in ids]);assert not batch['action_is_pad'].any()
 truth.append(batch['action'].numpy())
 obs={k:v for k,v in batch.items() if k=='observation.state' or k.startswith('observation.images.')}
 for cam in ds.meta.camera_keys:obs[cam]=obs[cam].float()/255
 processed=pre(obs);policy.reset()
 with torch.inference_mode():pred=post(policy.predict_action_chunk(processed))
 assert pred.shape==(len(ids),16,7) and torch.isfinite(pred).all();preds.append(pred.cpu().numpy())
 if offset==0:
  with torch.inference_mode():repeated=post(policy.predict_action_chunk(processed))
  assert torch.allclose(pred,repeated,atol=1e-6)
  print('verified raw JPG / saved processors / finite deterministic 16-step output',flush=True)
 print(f'{offset+len(ids)}/{len(probes["samples"])} elapsed {time.monotonic()-started:.1f}s',flush=True)
np.savez_compressed(P/'predictions.npz',predicted=np.concatenate(preds),expert=np.concatenate(truth),absolute_indices=probes['absolute_indices'])
(P/'inference_completed.json').write_text(json.dumps(dict(checkpoint=CK,samples=len(probes['samples']),elapsed_s=time.monotonic()-started,only_observation_inputs=True,raw_images_saved_preprocessors=True,deterministic_verified=True),indent=2))
