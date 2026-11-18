"""CPU preparation/processor/model-shape checks and actual GPU smoke reload."""
import argparse
from collections import Counter
import copy
import json
import re
from pathlib import Path

import numpy as np
import torch

from config import RUN, BASE, make_config, peft_overrides
from dataset_prepare import write, sha, assert_pins, CAMERAS, CROPS
from dataset import CupDataset


def verify_prepared():
    from pipeline import verify_frozen
    verify_frozen()
    prepared=json.loads((RUN/'preparation_complete.json').read_text())
    assert prepared['fps']==10 and prepared['chunk_size']==30 and prepared['episodes']>0
    for path,digest in prepared['files'].items():
        if sha(path)!=digest:raise ValueError('Prepared artifact changed: '+path)
    split=json.loads((RUN/'split_manifest.json').read_text());assert_pins(split['source_records'])
    assert len(split['train_episodes'])==prepared['episodes'] and not split['validation_episodes']
    return prepared


def cpu():
    torch.set_num_threads(2)
    prepared = verify_prepared()
    split = json.loads((RUN/'split_manifest.json').read_text())
    data = CupDataset(split['train_episodes'])
    from sampling import BalancedSampler
    import itertools
    sampler=BalancedSampler(data.meta.episodes['dataset_from_index'],data.meta.episodes['dataset_to_index'],episode_indices_to_use=data.episodes,seed=20261029,absolute_to_relative_idx=data.absolute_to_relative_idx)
    sampled=np.array([int(data.indices[i]) for i in itertools.islice(iter(sampler),16)])
    with np.load(RUN/'sampling_arrays.npz') as arrays:
        pools=arrays['source_pool'][sampled]
        if np.bincount(pools,minlength=8).tolist()!=[3,1]*4:raise ValueError('Actual dataset update is not3 basic+1 correction per object')
        pairs=(pools//2).reshape(8,2)
        if not np.all(pairs[:,0]!=pairs[:,1]) or not arrays['eligible_train'][sampled].all():raise ValueError('Invalid actual data microbatches')
    if any(r['task']!=r['original_task'] for r in split['source_records']):raise ValueError('Task text was changed')
    cfg = make_config()
    write('resolved_train_config.json', cfg.to_dict())
    write('peft_overrides.json', peft_overrides(cfg))
    from lerobot.utils.feature_utils import dataset_to_policy_features
    from lerobot.configs.types import FeatureType
    features = dataset_to_policy_features(data.meta.features)
    cfg.policy.input_features = {k: v for k, v in features.items() if v.type is not FeatureType.ACTION}
    cfg.policy.output_features = {k: v for k, v in features.items() if v.type is FeatureType.ACTION}
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    model_cfg = copy.deepcopy(cfg.policy)
    model_cfg.device = 'meta'
    with torch.device('meta'):
        policy = PI05Policy(model_cfg)
        names = [n for n, m in policy.named_modules() if isinstance(m, torch.nn.Linear)
                 and re.fullmatch(cfg.peft.target_modules, n)]
        coverage = Counter(('language' if '.language_model.' in n else 'expert',
                            'attention' if '.self_attn.' in n else 'ffn') for n in names)
        if dict(coverage) != {('language', 'attention'): 72, ('language', 'ffn'): 54,
                              ('expert', 'attention'): 72, ('expert', 'ffn'): 54}:
            raise ValueError(f'Adapter target coverage differs: {coverage}')
        policy = policy.wrap_with_peft(peft_cli_overrides=peft_overrides(cfg))
        trainable = [(n, p.numel()) for n, p in policy.named_parameters() if p.requires_grad]
        if any('vision_tower' in n or ('lora_' not in n and 'modules_to_save' not in n) for n, _ in trainable):
            raise ValueError('Unexpected trainable parameters')
    del policy
    # Process real batches on CPU, including serialization and inverse action normalization.
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.processor import ImageCropResizeProcessorStep
    from torch.utils.data import default_collate
    cfg.policy.device = 'cpu'
    pre, post = make_pre_post_processors(policy_cfg=cfg.policy, dataset_stats=data.meta.stats)
    pre.steps.insert(3, ImageCropResizeProcessorStep(crop_params_dict=CROPS, resize_size=(224, 224)))
    path = RUN/'processor_verification'
    pre.save_pretrained(path)
    post.save_pretrained(path)
    pre, post = make_pre_post_processors(policy_cfg=cfg.policy, pretrained_path=path)
    crop_steps = [s for s in pre.steps if isinstance(s, ImageCropResizeProcessorStep)]
    if len(crop_steps) != 1:
        raise ValueError('Image crop serialization lost')
    pre.steps = [s for s in pre.steps if not isinstance(s, ImageCropResizeProcessorStep)]
    with np.load(RUN/'sampling_arrays.npz') as a:
        anchors = [int(np.flatnonzero(a['eligible_train'] & (a['episode'] == ep))[0])
                   for ep in [next(x['episode_index'] for x in split['source_records'] if x['source']==source and x['split']=='train') for source in [o+'_'+p for o in ['apple','peach','cup','bottle'] for p in ['basic','correction']]]]
    tests = []
    for anchor in anchors:
        item = data[data.absolute_to_relative_idx[anchor]]
        batch = default_collate([item])
        processed = pre(batch)
        restored = post(processed['action'])
        if not torch.allclose(restored, batch['action'], atol=2e-6):
            raise ValueError('Action quantile normalization roundtrip failed')
        if not all(tuple(processed[k].shape) == (1, 3, 224, 224) for k in CAMERAS):
            raise ValueError('Image preprocessing shape mismatch')
        tests.append(dict(episode=int(item['episode_index']), action_shape=list(restored.shape)))
    report = dict(passed=True, episodes=prepared['episodes'], accepted_new=prepared['accepted_new'],
        coverage={f'{k[0]}_{k[1]}': v for k, v in coverage.items()},
        trainable_parameters=sum(n for _, n in trainable), train_only_quantiles=True,
        normalization_roundtrip=True, serialized_letterbox=True, source_raw10hz=True,
        original_task_preserved=True, real_update_pool_counts=np.bincount(pools,minlength=8).tolist(),different_objects_per_microbatch=True,
        tested_batches=tests, training_started=False)
    write('cpu_verification.json', report)
    print(json.dumps(report, indent=2))


def reload_checkpoint(smoke):
    if not torch.cuda.is_available():
        raise RuntimeError('Checkpoint inference verification needs CUDA')
    verify_prepared()
    marker = 'smoke_latest_checkpoint.json' if smoke else 'latest_checkpoint.json'
    path = Path(json.loads((RUN/marker).read_text())['checkpoint'])
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.processor import ImageCropResizeProcessorStep
    from torch.utils.data import default_collate
    split = json.loads((RUN/'split_manifest.json').read_text())
    data = CupDataset(split['train_episodes'])
    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.device = 'cuda'
    cfg.pretrained_path = path
    cfg.use_peft = True
    cfg.num_inference_steps = 10
    from guards import strict_base_load
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    policy, base_evidence = strict_base_load(lambda: make_policy(cfg=cfg, ds_meta=data.meta), PI05Policy)
    policy.eval()
    pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=path,
        preprocessor_overrides={'device_processor': {'device': 'cuda'}})
    pre.steps = [s for s in pre.steps if not isinstance(s, ImageCropResizeProcessorStep)]
    results = []
    with np.load(RUN/'sampling_arrays.npz') as a:
        for source in [o+'_'+p for o in ['apple','peach','cup','bottle'] for p in ['basic','correction']]:
            eligible = [r for r in data.records if r['split'] == 'train' and r['source'] == source]
            if not eligible:
                raise ValueError(f'Missing source for reload sanity check (not evaluation): {source}')
            ep = eligible[0]['episode_index']
            anchor = int(np.flatnonzero(a['eligible_train'] & (a['episode'] == ep))[0])
            batch = default_collate([data[data.absolute_to_relative_idx[anchor]]])
            processed = pre(batch)
            with torch.inference_mode():
                policy.reset()
                prediction = post(policy.predict_action_chunk(processed))
            if prediction.shape != (1, 30, 7) or not torch.isfinite(prediction).all():
                raise ValueError('Reloaded checkpoint prediction invalid')
            if not torch.allclose(post(processed['action']), batch['action'], atol=2e-6):
                raise ValueError('Saved quantile stats do not invert')
            results.append(dict(source=source, episode=ep, prediction_shape=list(prediction.shape), finite=True))
    report = dict(passed=True, preparation_sha256=sha(RUN/'preparation_complete.json'), checkpoint=str(path), tests=results, normalization_roundtrip=True, **base_evidence)
    write('checkpoint_reload_verified.json' if smoke else 'final_checkpoint_reload_verified.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reload-smoke', action='store_true')
    parser.add_argument('--reload-final', action='store_true')
    args = parser.parse_args()
    if args.reload_smoke or args.reload_final:
        reload_checkpoint(args.reload_smoke)
    else:
        cpu()
