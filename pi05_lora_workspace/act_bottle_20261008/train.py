"""Native ACT on the same 160 bottle episodes, with serialized image resizing."""
import argparse
import json
import sys
from pathlib import Path

RUN = Path(__file__).resolve().parent
DATA = Path('/home/lp531/smolvla_data/z1_bottle_three_camera_pi05_binary_20261007')
OUTPUT = Path('/home/lp531/smolvla_training/act_bottle_binary_chunk15_20261008')

def make_config(smoke=False):
    from lerobot.configs.accelerator import AcceleratorConfig, GradientAccumulationConfig
    from lerobot.configs.default import DatasetConfig, WandBConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.act.configuration_act import ACTConfig
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id='lp531/z1_bottle_three_camera_pi05_binary_20261007',root=str(DATA),video_backend='pyav',eval_split=0.0),
        policy=ACTConfig(device='cuda',push_to_hub=False,chunk_size=15,n_action_steps=5,
                         optimizer_lr=1e-4,optimizer_lr_backbone=1e-5,kl_weight=10.0),
        accelerator=AcceleratorConfig(mixed_precision='no',gradient_accumulation=GradientAccumulationConfig(steps=4)),
        batch_size=4,num_workers=4,prefetch_factor=2,steps=4 if smoke else 120000,
        save_freq=8000,log_freq=80,eval_steps=0,env_eval_freq=0,
        output_dir=RUN/'smoke' if smoke else OUTPUT,job_name='act_bottle_binary_chunk15',
        seed=20261008,wandb=WandBConfig(enable=False))
    return cfg

def verify(cfg):
    import numpy as np
    from lerobot.datasets.factory import make_dataset
    from lerobot.configs.types import FeatureType
    from lerobot.utils.feature_utils import dataset_to_policy_features
    ds=make_dataset(cfg)
    assert len(ds)==37557 and ds.num_episodes==160
    assert cfg.dataset.episodes is None and cfg.dataset.eval_split==0
    assert ds.meta.fps==10 and len(ds.meta.camera_keys)==3
    features=dataset_to_policy_features(ds.meta.features)
    cfg.policy.output_features={k:v for k,v in features.items() if v.type is FeatureType.ACTION}
    cfg.policy.input_features={k:v for k,v in features.items() if v.type is not FeatureType.ACTION}
    actions=np.asarray(ds.hf_dataset['action'])
    assert actions.shape==(37557,7) and np.isfinite(actions).all()
    assert set(np.unique(actions[:,-1]))=={0.,1.}
    decoded=[]
    for index in [0,10000,20000,37556]:
        sample=ds[index]
        assert tuple(sample['action'].shape)==(15,7)
        for key in ds.meta.camera_keys:
            assert sample[key].shape[0]==3 and sample[key].float().std()>1
        decoded.append(index)
    return dict(episodes=160,frames=len(ds),fps=10,cameras=ds.meta.camera_keys,
                binary_actions=True,decoded_indices=decoded,frame_removal=False,validation_split=False)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--train',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();sys.argv=[sys.argv[0]]
    cfg=make_config(args.smoke);cfg.validate()
    result=verify(cfg)
    (RUN/'dataset_verification.json').write_text(json.dumps(result,indent=2))
    name='smoke_config.json' if args.smoke else 'resolved_train_config.json'
    (RUN/name).write_text(json.dumps(cfg.to_dict(),indent=2,default=str))
    print('DATA_VERIFIED',json.dumps(result),flush=True)
    if not (args.train or args.smoke):return
    import torch
    import lerobot.scripts.lerobot_train as trainer
    from lerobot.processor import ImageCropResizeProcessorStep
    from check_actions import check_actions
    original_processors=trainer.make_pre_post_processors
    def resized_processors(*a,**kw):
        pre,post=original_processors(*a,**kw)
        # Persist the built-in step so checkpoint inference also resizes before normalization.
        pre.steps.insert(3,ImageCropResizeProcessorStep(resize_size=(224,224)))
        return pre,post
    trainer.make_pre_post_processors=resized_processors
    original_save=trainer.save_checkpoint
    def save_and_check(**kw):
        kw['cfg'].save_freq=8000
        original_save(**kw)
        check_actions(**kw)
        memory=dict(step=kw['step'],allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        (RUN/('smoke_memory.json' if args.smoke else 'training_memory.json')).write_text(json.dumps(memory,indent=2))
        print('GPU_MEMORY',json.dumps(memory),flush=True)
    trainer.save_checkpoint=save_and_check
    cfg.save_freq=4  # Verify save + offline inference after the first complete optimizer update.
    trainer.train(cfg)
    (RUN/('smoke_passed.json' if args.smoke else 'completed.json')).write_text(json.dumps(dict(output=str(cfg.output_dir),optimizer_updates=cfg.steps//4)))

if __name__=='__main__':main()
