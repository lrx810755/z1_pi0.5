import json
from pathlib import Path
import numpy as np
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset
OUT=Path(__file__).resolve().parent;labels=json.loads((OUT/'selected.json').read_text())
ds=LeRobotDataset('lp531/z1_bottle_three_camera_pi05_binary_20261007',root='/home/lp531/smolvla_data/z1_bottle_three_camera_pi05_binary_20261007',episodes=sorted(x['episode_index'] for x in labels),video_backend='pyav',return_uint8=True)
eps=np.array(ds.hf_dataset['episode_index']);fr=np.array(ds.hf_dataset['frame_index']);out=[]
for lab in labels:
 root=Path(lab['raw_root']);rows=[json.loads(x) for x in (root/'frames.jsonl').read_text().splitlines()]
 for frame in [0,lab['close_frame']-1]:
  ids=np.flatnonzero((eps==lab['episode_index'])&(fr==frame));assert len(ids)==1
  sample=ds[int(ids[0])];row=rows[frame];record=dict(episode=lab['raw_episode'],frame=frame,cameras={},state_max_error=float(np.max(abs(sample['observation.state'].numpy()-row['observation.state']))))
  for cam in ['wrist','global','global_right']:
   key='observation.images.'+cam;a=np.array(Image.open(root/row[key]).convert('RGB'));b=sample[key].numpy().transpose(1,2,0);assert a.shape==b.shape
   record['cameras'][cam]=dict(decoded_dtype=str(b.dtype),mean_absolute_pixel_error=float(abs(a.astype(float)-b.astype(float)).mean()))
  out.append(record)
(OUT/'training_video_comparison.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
