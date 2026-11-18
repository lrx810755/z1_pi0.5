"""Read real RGB/state/15-action batches through LeRobot without policy inference."""
from pathlib import Path
import json
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
W=Path(__file__).resolve().parent
config=json.loads((W/'training_input.json').read_text());records=json.loads((W/'mapped_windows.json').read_text())['records']
ds=LeRobotDataset(repo_id='lp531/z1_bottle_with_fragments_20261012',root=config['dataset_root'],
                  delta_timestamps={'action':[i/10 for i in range(15)]},video_backend='pyav')
checks=[]
for i in [160,192,210]:
    r=records[i];absolute=r['absolute_anchor_end_inclusive'];sample=ds[absolute]
    assert sample['action'].shape==(15,7)
    assert torch.isfinite(sample['action']).all()
    assert not sample['action_is_pad'].any()
    shapes={}
    for camera in ['observation.images.wrist','observation.images.global','observation.images.global_right']:
        image=sample[camera];assert image.ndim==3 and image.shape[0]==3 and torch.isfinite(image).all()
        shapes[camera]=list(image.shape)
    checks.append(dict(episode_index=i,absolute_anchor=absolute,action_shape=list(sample['action'].shape),camera_shapes=shapes))
report=dict(passed=True,real_lerobot_dataset_read=True,three_cameras_decoded=True,action_chunk_size=15,
            no_padding_at_sampled_correction_boundaries=True,policy_inference_run=False,checks=checks)
(W/'training_read_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='checks'},ensure_ascii=False,indent=2))
