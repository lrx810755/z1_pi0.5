"""Reload the saved ACT and processors; verify real observations and action round-trip."""
import json,os
import numpy as np
import torch
from torch.utils.data import default_collate
from prepare import RUN,write
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies import make_pre_post_processors

def main():
    checkpoint=os.environ.get('Z1_ACT_VERIFY_CHECKPOINT') or json.loads((RUN/'smoke_checkpoint.json').read_text())['checkpoint']
    policy=ACTPolicy.from_pretrained(checkpoint,local_files_only=True)
    policy.eval()
    pre,post=make_pre_post_processors(policy_cfg=policy.config,pretrained_path=checkpoint)
    from cache_data import CupDataset
    val=CupDataset(json.loads((RUN/"split_manifest.json").read_text())["validation_episodes"],raw_images=True)
    split=json.loads((RUN/'split_manifest.json').read_text())
    tests=[]
    with np.load(RUN/'sampling_arrays.npz') as a:
        for source in ['original','old_correction','new_correction']:
            ep=next(r['episode_index'] for r in split['source_records'] if r['source']==source and r['episode_index'] in split['validation_episodes'])
            anchor=int(np.flatnonzero((a['episode']==ep)&a['eligible_validation'])[0])
            batch=default_collate([val[val.absolute_to_relative_idx[anchor]]])
            assert not batch['action_is_pad'].any()
            for cam in val.meta.camera_keys:
                batch[cam]=batch[cam].float()/255
            processed=pre(batch)
            for cam in val.meta.camera_keys:assert processed[cam].shape==(1,3,224,224)
            restored=post(processed['action'])
            assert torch.allclose(restored,batch['action'],atol=1e-6)
            policy.reset()
            with torch.inference_mode():pred=post(policy.predict_action_chunk(processed))
            assert pred.shape==(1,16,7) and torch.isfinite(pred).all()
            tests.append(dict(source=source,episode=ep,output_shape=list(pred.shape),finite=True))
    report='final_checkpoint_reload_verified.json' if os.environ.get('Z1_ACT_VERIFY_CHECKPOINT') else 'checkpoint_reload_verified.json'
    write(report,dict(checkpoint=checkpoint,tests=tests,
        serialized_letterbox=True,normalization_roundtrip=True))
    print((RUN/report).read_text())
if __name__=='__main__':main()
