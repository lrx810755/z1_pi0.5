"""Small full-episode PI05 fitting experiment; uses native episode filtering."""
import argparse,importlib.util,json,sys
from pathlib import Path
RUN=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('binary_train',RUN.parent/'binary_gripper_20261007/train.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)

def make_config():
 cfg=base.make_config()
 selected=json.loads((RUN/'selected.json').read_text())
 cfg.dataset.episodes=sorted(x['episode_index'] for x in selected)
 cfg.dataset.eval_split=0.0
 cfg.steps=1000*cfg.accelerator.gradient_accumulation.steps
 cfg.save_freq=250*cfg.accelerator.gradient_accumulation.steps
 cfg.output_dir=Path('/home/lp531/smolvla_training/pi05_bottle_small4_chunk15_attn_ffn_20261007')
 cfg.job_name='pi05_bottle_small4_chunk15_attn_ffn'
 # Keep full-dataset normalization. The native scheduler proportionally
 # compresses the inherited schedule to 50 warmup / 1000 total updates.
 return cfg

def verify(cfg):
 import numpy as np
 from lerobot.datasets.factory import make_dataset
 from lerobot.configs.types import FeatureType
 from lerobot.utils.feature_utils import dataset_to_policy_features
 selected=json.loads((RUN/'selected.json').read_text())
 assert len(selected)==4 and len(cfg.dataset.episodes)==4
 assert cfg.policy.chunk_size==15 and cfg.policy.use_relative_actions is False
 assert cfg.policy.pretrained_path==RUN.parent/'weights/pi05_base'
 assert not cfg.resume and cfg.dataset.eval_split==0 and not cfg.dataset.image_transforms.enable
 ds=make_dataset(cfg)
 features=dataset_to_policy_features(ds.meta.features)
 cfg.policy.output_features={k:v for k,v in features.items() if v.type is FeatureType.ACTION}
 cfg.policy.input_features={k:v for k,v in features.items() if v.type is not FeatureType.ACTION}
 actual=sorted(set(int(x) for x in ds.hf_dataset['episode_index']))
 assert actual==cfg.dataset.episodes,(actual,cfg.dataset.episodes)
 assert len(ds)==sum(x['frames'] for x in selected),(len(ds),sum(x['frames'] for x in selected))
 assert ds.num_episodes==4
 checked=[]
 for ep in actual:
  ids=np.flatnonzero(np.array(ds.hf_dataset['episode_index'])==ep)
  frames=np.array(ds.hf_dataset['frame_index'])[ids]
  assert np.array_equal(frames,np.arange(len(ids)))
  sample=ds[int(ids[len(ids)//2])]
  assert tuple(sample['action'].shape)==(15,7)
  assert set(sample['action'][:,-1].tolist())<={0.,1.}
  for cam in ['wrist','global','global_right']:
   image=sample['observation.images.'+cam]
   assert image.shape[0]==3 and image.float().std()>1
  checked.append(dict(episode_index=ep,frames=len(ids),decoded_index=int(ids[len(ids)//2])))
 result=dict(passed=True,episodes=4,frames=len(ds),episode_indices=actual,checked=checked,frame_removal=False,normalization='unchanged full160 training statistics',effective_batch=cfg.batch_size*cfg.accelerator.gradient_accumulation.steps,microsteps=cfg.steps,optimizer_updates=1000,save_optimizer_updates=[250,500,750,1000],base_weights=str(cfg.policy.pretrained_path),resume=False)
 (RUN/'dataset_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--train',action='store_true');args=p.parse_args();sys.argv=[sys.argv[0]]
 cfg=make_config();cfg.validate();verify(cfg)
 (RUN/'resolved_train_config.json').write_text(json.dumps(cfg.to_dict(),indent=2,default=str))
 if args.train:
  results=json.loads((RUN/'replay_results.json').read_text())
  selected=json.loads((RUN/'selected.json').read_text())
  assert len(results)==4 and all(x['success'] for x in results),'Reference playback must pass before training'
  assert {x['source_episode'] for x in results}=={x['raw_episode'] for x in selected}
  from lerobot.scripts.lerobot_train import train
  train(cfg)
if __name__=='__main__':main()
