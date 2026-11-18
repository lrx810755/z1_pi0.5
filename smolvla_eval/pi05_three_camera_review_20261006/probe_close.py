from pathlib import Path
import json,sys,os
import numpy as np,torch
from PIL import Image
sys.path.insert(0,'/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge')
from z1_isaac_bridge.pi05_policy_bridge import PI05ProcessPolicy
os.environ['Z1_PI05_INFERENCE_SEED']='123'
manifest=json.loads(Path('/home/lp531/pi05_lora_workspace/three_camera_20261006/split_manifest.json').read_text())
selected=[]
for split in ['train','validation']:
 selected += [e for e in manifest['episodes'] if e['split']==split][:4]
policy=PI05ProcessPolicy(Path('/home/lp531/smolvla_training/pi05_bottle_three_camera_20261006/best'))
results=[]
try:
 for e in selected:
  p=Path(e['source']);rows=[json.loads(l) for l in (p/'frames.jsonl').read_text().splitlines()];c=next(i for i,r in enumerate(rows) if r['action'][-1]>=-.89)
  for offset in [-6,-1]:
   i=max(0,c+offset);r=rows[i];batch={'observation.state':torch.tensor([r['observation.state']],dtype=torch.float32),'task':[e['task']]}
   for name in ['wrist','global','global_right']:
    k='observation.images.'+name;im=np.array(Image.open(p/r[k]).convert('RGB'));batch[k]=torch.from_numpy(im).permute(2,0,1).unsqueeze(0).float()/255
   policy.reset();a=policy.predict_action_chunk(batch)[0].numpy();cl=np.flatnonzero(a[:,-1]>=-.75)
   item=dict(episode=p.name,split=e['split'],offset_frames=offset,input_gripper=r['observation.state'][-1],reference_next5_max=max(x['action'][-1] for x in rows[i:i+5]),pred_next5_max=float(a[:5,-1].max()),pred_chunk_max=float(a[:,-1].max()),first_close_chunk_index=int(cl[0]) if len(cl) else None)
   results.append(item);print(json.dumps(item),flush=True)
finally:policy.close()
Path(__file__).with_name('close_probe.json').write_text(json.dumps(results,indent=2)+'\n')
