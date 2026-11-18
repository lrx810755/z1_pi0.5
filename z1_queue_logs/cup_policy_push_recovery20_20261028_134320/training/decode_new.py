"""Decode only newly accepted policy recovery episodes with the prior exact crop."""
import hashlib,io,json,time
import numpy as np
from prepare import RUN,DATA,CAMERAS,CROPS,write,sha,assert_pins
def job(record):
 import torch
 from PIL import Image
 from torchvision.transforms import functional as F
 torch.set_num_threads(1)
 idx=record['episode_index'];root=Path(record['source_path']);n=record['frames'];assert_pins([record])
 rows=[json.loads(line) for line in (RUN/'source_pins'/f'episode_{idx:06d}'/'frames.jsonl').read_text().splitlines()]
 result=[]
 for cam in CAMERAS:
  dest=DATA/cam/f'episode_{idx:06d}.npy';dest.parent.mkdir(parents=True,exist_ok=True);audit=dest.with_suffix('.json')
  if dest.exists() and audit.exists():
   m=json.loads(audit.read_text());a=np.load(dest,mmap_mode='r')
   if a.shape!=(n,3,224,224) or a.dtype!=np.float32 or m['cache_sha256']!=sha(dest):raise ValueError(f'Invalid existing cache {dest}')
   del a
   # Source-image bytes pinned during the first decode must still match.
   for item in m['source_images']:
    if sha(root/item['path'])!=item['sha256']:raise ValueError(f'Source image changed: {root/item["path"]}')
   result.append(m);continue
  temp=dest.with_suffix('.building.npy');out=np.lib.format.open_memmap(temp,mode='w+',dtype=np.float32,shape=(n,3,224,224));images=[]
  for begin in range(0,n,8):
   batch=[]
   for row in rows[begin:begin+8]:
    raw=(root/row[cam]).read_bytes();images.append(dict(path=row[cam],sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)))
    with Image.open(io.BytesIO(raw)) as image:
     pixels=np.array(image.convert('RGB'));expected=(360 if cam.endswith('wrist') else 480,640,3)
     if pixels.shape!=expected:raise ValueError(f'Native image shape mismatch {root/row[cam]}')
     batch.append(pixels)
   pixels=torch.from_numpy(np.stack(batch)).permute(0,3,1,2).float()/255
   resized=F.resize(F.crop(pixels,*CROPS[cam]),[224,224]).clamp(0,1).numpy()
   if not np.isfinite(resized).all():raise ValueError('Nonfinite pixels')
   out[begin:begin+len(batch)]=resized
   if begin%64==0:write(f'cache_workers/episode_{idx:06d}.json',dict(episode=idx,camera=cam,decoded_frames=min(begin+len(batch),n),frames=n))
  out.flush();del out;temp.replace(dest)
  m=dict(episode_index=idx,camera=cam,frames=n,shape=[n,3,224,224],dtype='float32',cache_sha256=sha(dest),source_images=images,crop=CROPS[cam],resize=[224,224],antialias=True)
  t=audit.with_suffix('.tmp');t.write_text(json.dumps(m));t.replace(audit);result.append(m)
 write(f'cache_workers/episode_{idx:06d}.json',dict(episode=idx,complete=True,cameras=3,frames=n))
 return idx,n

