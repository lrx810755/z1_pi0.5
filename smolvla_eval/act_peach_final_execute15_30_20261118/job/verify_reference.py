"""Generate real raw-image ACT reference predictions without robot actions."""
import json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,'/home/lp531/pi05_lora_workspace/act_peach397_chunk30_b64_20261118')
sys.path.insert(0,'/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge')
import numpy as np
import torch
from torch.utils.data import default_collate
from dataset import PeachDataset
from common import RUN,read,checkpoint_fingerprints,verify_files
from z1_isaac_bridge.pi05_policy_bridge import encode_batch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
cp=Path(read(RUN/'latest_checkpoint.json')['checkpoint'])
evidence=read(RUN/'final_checkpoint_reload_verified.json')
assert evidence['passed'] and evidence['checkpoint']==str(cp) and evidence['checkpoint_files']==checkpoint_fingerprints(cp)
verify_files(read(RUN/'code_snapshot.json')['files'])
assert read(RUN/'completed.json')['optimizer_updates']==60000
torch.set_num_threads(8)
cfg=PreTrainedConfig.from_pretrained(cp);cfg.device='cuda';cfg.pretrained_backbone_weights=None
model=ACTPolicy.from_pretrained(cp,config=cfg,local_files_only=True,strict=True);model.eval()
pre,post=make_pre_post_processors(cfg,pretrained_path=cp,preprocessor_overrides={'device_processor':{'device':'cuda'}})
dataset=PeachDataset(raw_images=True);samples=[]
for source in ['basic','correction']:
    record=next(r for r in dataset.records if r['source']==source)
    i=record['dataset_from_index'];anchor=i+int(np.flatnonzero(dataset.approved[i:record['dataset_to_index']])[0])
    batch=default_collate([dataset[dataset.absolute_to_relative_idx[anchor]]])
    batch={k:v for k,v in batch.items() if k.startswith('observation.')}
    with torch.inference_mode():reference=post(model.predict_action_chunk(pre(batch))).cpu().numpy()
    samples.append(dict(source=source,batch=encode_batch(batch),reference=reference.tolist()))
(HERE/'reference_inputs.json').write_text(json.dumps(samples)+'\n')
print('Two real peach raw input references saved',flush=True)
