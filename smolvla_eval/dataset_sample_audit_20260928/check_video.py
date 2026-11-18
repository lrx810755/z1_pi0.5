import json,subprocess
from pathlib import Path
import cv2,numpy as np
O=Path(__file__).parent;R=Path('/home/lp531/smolvla_data/z1_tabletop_left_wrist_prepared_all_v1');L=Path('/home/lp531/smolvla_data/z1_tabletop_lerobot_left_wrist_2cam_all_v1')
res=[]
for s in json.load(open(O/'results.json'))['samples']:
 f=[json.loads(z) for z in (R/s['episode']/'frames.jsonl').read_text().splitlines()];i=s['first_close_frame']
 for cam in ['global','wrist']:
  v=L/'videos/chunk-000'/f'observation.images.{cam}'/f"episode_{s['training_episode']:06d}.mp4"
  p=O/'video_check.png'
  subprocess.run(['ffmpeg','-v','error','-threads','1','-ss',str(i/10),'-i',str(v),'-frames:v','1','-threads','1','-y',str(p)],check=True)
  a=cv2.imread(str(p));b=cv2.imread(str(R/s['episode']/f[i][f'observation.images.{cam}']))
  res.append(dict(episode=s['episode'],camera=cam,mean_absolute_pixel_difference=float(np.abs(a.astype(float)-b).mean())))
json.dump(res,open(O/'video_check.json','w'),indent=2);print('checked',len(res),'max MAE',max(s['mean_absolute_pixel_difference'] for s in res))
