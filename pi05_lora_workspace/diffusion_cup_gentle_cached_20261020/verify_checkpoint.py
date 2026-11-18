import json,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import default_collate
from prepare import RUN,write
from cache_data import CupDataset
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies import make_pre_post_processors
from lerobot.processor import ImageCropResizeProcessorStep

def main():
 torch.set_num_threads(4);split=json.loads((RUN/'split_manifest.json').read_text());raw=CupDataset(split['validation_episodes'],raw_images=True);cached=CupDataset(split['validation_episodes'])
 root=RUN/'smoke/checkpoints/000020';checks=[]
 for model_name in ['pretrained_model','pretrained_model_ema']:
  ck=root/model_name;policy=DiffusionPolicy.from_pretrained(ck,local_files_only=True);policy.eval();pre,post=make_pre_post_processors(policy_cfg=policy.config,pretrained_path=ck)
  assert (ck/'policy_preprocessor.json').is_file() and policy.config.horizon==16 and policy.config.n_obs_steps==2
  for source in ['original','old_correction','new_correction']:
   rec=next(r for r in split['source_records'] if r['source']==source and r['split']=='validation');absolute=rec['dataset_from_index']+max(1,rec['close_frame']-16)
   b=default_collate([raw[raw.absolute_to_relative_idx[absolute]]]);c=default_collate([cached[cached.absolute_to_relative_idx[absolute]]])
   # Saved preprocessing is applied per observation BEFORE online history queues.
   obs={k:v for k,v in b.items() if k.startswith('observation.')}
   for cam in raw.meta.camera_keys:obs[cam]=obs[cam].float()/255
   flat={k:v.flatten(0,1) for k,v in obs.items()};processed=pre(flat);processed={k:v.reshape(1,2,*v.shape[1:]) for k,v in processed.items() if k.startswith('observation.')}
   steps=pre.steps;pre.steps=[s for s in steps if not isinstance(s,ImageCropResizeProcessorStep)]
   try:cache_processed=pre(c)
   finally:pre.steps=steps
   for k,v in processed.items():
    error=float((v-cache_processed[k]).abs().max());print(k,'maxerror',error,flush=True);assert error<=1e-4,(k,error)
   restored=post(cache_processed['action']);assert torch.allclose(restored.cpu(),c['action'],atol=1e-6)
   policy.reset()
   with torch.inference_mode():prediction=post(policy.predict_action_chunk(processed))
   assert prediction.shape==(1,15,7) and torch.isfinite(prediction).all()
   checks.append(dict(model=model_name,source=source,output_shape=list(prediction.shape),raw_cached_processors_match=True,action_roundtrip=True))
  del policy;torch.cuda.empty_cache()
 state=torch.load(root/'training_state/ema_state.pt',map_location='cpu',weights_only=True)
 assert state['optimization_step']==20
 write('checkpoint_reload_verified.json',dict(tests=checks,ema_updates=state['optimization_step'],deploy_preprocessing='per-frame crop/resize then collect two observation steps'))
 print('live and EMA checkpoints reload; all three source predictions and preprocessing verified',flush=True)
if __name__=='__main__':main()
