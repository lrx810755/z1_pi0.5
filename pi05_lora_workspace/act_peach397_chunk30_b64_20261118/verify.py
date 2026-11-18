"""Real cache/processor checks and actual GPU checkpoint reload; no robot actions."""
import argparse
import itertools
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import default_collate
from common import RUN, CACHE, CAMERAS, CROPS, read, write, sha, verify_prepared, checkpoint_fingerprints, verify_files
from dataset import PeachDataset
from config import make_config
from sampling import BalancedSampler


def cpu():
    from torchvision.transforms import functional as F
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.processor import ImageCropResizeProcessorStep
    from lerobot.utils.feature_utils import dataset_to_policy_features
    from lerobot.configs.types import FeatureType
    torch.set_num_threads(2)
    prepared=verify_prepared();cache=read(CACHE/'completed.json')
    verify_files(read(RUN/'code_snapshot.json')['files'])
    if not cache.get('complete') or cache['prepared_sha256']!=sha(RUN/'prepared.json'):
        raise ValueError('Current decoded cache is incomplete')
    raw=PeachDataset(raw_images=True);decoded=PeachDataset()
    cfg=make_config(validate=False);cfg.policy.device='cpu'
    features=dataset_to_policy_features(decoded.meta.features)
    cfg.policy.input_features={k:v for k,v in features.items() if v.type is not FeatureType.ACTION}
    cfg.policy.output_features={k:v for k,v in features.items() if v.type is FeatureType.ACTION}
    pre,post=make_pre_post_processors(cfg.policy,dataset_stats=decoded.meta.stats)
    pre.steps.insert(3,ImageCropResizeProcessorStep(crop_params_dict=CROPS,resize_size=(224,224)))
    pre.save_pretrained(RUN/'processor_verification');post.save_pretrained(RUN/'processor_verification')
    pre,post=make_pre_post_processors(cfg.policy,pretrained_path=RUN/'processor_verification',
        preprocessor_overrides={'device_processor':{'device':'cpu'}})
    no_crop,_=make_pre_post_processors(cfg.policy,pretrained_path=RUN/'processor_verification',
        preprocessor_overrides={'device_processor':{'device':'cpu'}})
    no_crop.steps=[s for s in no_crop.steps if not isinstance(s,ImageCropResizeProcessorStep)]
    started=time.monotonic();tests=[];cache_files=0
    with np.load(RUN/'sampling_arrays.npz') as archive:
        eligible,episodes,pool=archive['eligible_train'],archive['episode'],archive['source_pool']
    for record in decoded.records:
        ep=record['episode_index'];start=record['dataset_from_index'];end=record['dataset_to_index']
        for cam in CAMERAS:
            data=CACHE/cam/f'episode_{ep:06d}.bin';index=data.with_suffix('.npz');audit=read(data.with_suffix('.json'))
            if sha(data)!=audit['data_sha256'] or sha(index)!=audit['index_sha256']:
                raise ValueError('Cache digest differs: '+str(data))
            with np.load(index) as archive:offsets=archive['offsets']
            if len(offsets)!=record['frames']+1 or offsets[0]!=0 or offsets[-1]!=data.stat().st_size or not np.all(np.diff(offsets)>0):
                raise ValueError('Invalid compressed frame index')
            cache_files+=1
        valid=np.flatnonzero(eligible[start:end]);frames={int(valid[0]),int(valid[-1])}
        transitions=np.flatnonzero(np.diff(decoded.arrays['actions'][start:end,-1])!=0)+1
        for transition in transitions:frames.add(int(valid[np.argmin(abs(valid-transition))]))
        for frame in sorted(frames):
            absolute=start+frame;x=raw[raw.absolute_to_relative_idx[absolute]];y=decoded[decoded.absolute_to_relative_idx[absolute]]
            for key in ['observation.state','action','action_is_pad']:
                if not torch.equal(x[key],y[key]):raise ValueError('Cached lowdim/labels differ')
            if y['action_is_pad'].any():raise ValueError('Fake padded actions are not allowed')
            for cam in CAMERAS:
                expected=F.resize(F.crop(x[cam],*CROPS[cam]),[224,224],antialias=True).clamp(0,1)
                if not torch.equal(expected,y[cam]):raise ValueError('Lossless cache differs from raw deployed pixels')
            batch=default_collate([x]);processed=pre(batch)
            if not torch.allclose(post(processed['action']),batch['action'],atol=1e-6):
                raise ValueError('Per-peach action normalization roundtrip failed')
            cached_processed=no_crop(default_collate([y]))
            for cam in CAMERAS:
                if processed[cam].shape!=(1,3,224,224) or not torch.equal(processed[cam],cached_processed[cam]):
                    raise ValueError('Raw and cached normalized images differ')
            tests.append(dict(episode=ep,frame=frame,source=record['source']))
        if (ep+1)%10==0:
            write('verification_progress.json',dict(phase='cache_audit',episodes=ep+1,target=397,paired_windows=len(tests),elapsed_s=time.monotonic()-started))
    sampler=BalancedSampler(decoded.meta.episodes['dataset_from_index'],decoded.meta.episodes['dataset_to_index'],
               episode_indices_to_use=decoded.episodes,seed=20261118,absolute_to_relative_idx=decoded.absolute_to_relative_idx)
    first=list(itertools.islice(iter(sampler),64*20));anchors=np.array([decoded.indices[i] for i in first])
    for block in anchors.reshape(-1,64):
        if np.bincount(pool[block],minlength=2).tolist()!=[48,16] or len(set(block))!=64 or not eligible[block].all():
            raise ValueError('Actual sampler fails48/16/no-duplicate constraints')
    sampler.load_state_dict({'epoch':0,'start_index':77})
    if list(itertools.islice(iter(sampler),100))!=first[77:177]:raise ValueError('Sampler resume is not deterministic')
    write('cpu_verified.json',dict(passed=True,episodes=397,images=352611,cache_files=cache_files,paired_windows=len(tests),
          max_pixel_error=0.,normalization_roundtrip=True,source_mask_preserved=True,all_train=True,
          actual_batch_pools=[48,16],sampler_resume_exact=True,prepared_sha256=sha(RUN/'prepared.json'),
          cache_manifest_sha256=sha(CACHE/'completed.json'),elapsed_s=time.monotonic()-started,tests=tests))
    marker=read(RUN/'cpu_verified.json');marker['code_snapshot_sha256']=sha(RUN/'code_snapshot.json');write('cpu_verified.json',marker)
    print('CPU/cache audit passed: '+str(len(tests))+' real paired windows',flush=True)


def reload_checkpoint(smoke=True):
    if not torch.cuda.is_available():raise RuntimeError('GPU required for actual checkpoint reload')
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    path=Path(read(RUN/('smoke_latest_checkpoint.json' if smoke else 'latest_checkpoint.json'))['checkpoint'])
    cfg=PreTrainedConfig.from_pretrained(path);cfg.device='cuda';cfg.pretrained_backbone_weights=None
    model=ACTPolicy.from_pretrained(path,config=cfg,local_files_only=True,strict=True);model.eval()
    pre,post=make_pre_post_processors(cfg,pretrained_path=path,preprocessor_overrides={'device_processor':{'device':'cuda'}})
    raw=PeachDataset(raw_images=True);results=[]
    with np.load(RUN/'sampling_arrays.npz') as a:
        for source in ['basic','correction']:
            record=next(r for r in raw.records if r['source']==source)
            ep=record['episode_index'];anchor=int(np.flatnonzero(a['eligible_train']&(a['episode']==ep))[0])
            batch=default_collate([raw[raw.absolute_to_relative_idx[anchor]]]);processed=pre(batch)
            with torch.inference_mode():
                model.reset();prediction=post(model.predict_action_chunk(processed))
                model.reset();single=post(model.select_action(processed))
            if prediction.shape!=(1,30,7) or single.shape!=(1,7) or not torch.isfinite(prediction).all() or not torch.isfinite(single).all():
                raise ValueError('Saved ACT inference shape/values invalid')
            if not torch.allclose(post(processed['action']),batch['action'],atol=1e-6):raise ValueError('Saved stats do not invert')
            results.append(dict(source=source,episode=ep,prediction_shape=list(prediction.shape),temporal_action_shape=list(single.shape),finite=True))
    write('smoke_reload_verified.json' if smoke else 'final_checkpoint_reload_verified.json',dict(passed=True,checkpoint=str(path),tests=results,normalization_roundtrip=True,
          prepared_sha256=sha(RUN/'prepared.json'),cache_manifest_sha256=sha(CACHE/'completed.json'),not_closed_loop_evaluation=True))
    marker_name='smoke_reload_verified.json' if smoke else 'final_checkpoint_reload_verified.json'
    marker=read(RUN/marker_name);marker.update(checkpoint_files=checkpoint_fingerprints(path),code_snapshot_sha256=sha(RUN/'code_snapshot.json'));write(marker_name,marker)
    print('Actual smoke checkpoint raw RGB inference and normalization passed',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cpu',action='store_true');parser.add_argument('--reload-smoke',action='store_true');parser.add_argument('--reload-final',action='store_true');args=parser.parse_args()
    if args.reload_smoke or args.reload_final:reload_checkpoint(smoke=args.reload_smoke)
    else:cpu()
