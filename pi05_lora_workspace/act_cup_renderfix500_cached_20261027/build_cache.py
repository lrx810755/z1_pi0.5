"""Stream every original three-camera JPEG through the deployed exact float32 crop.
Raw JPEGs remain the native-resolution source; only224 crops are persisted.
"""
import argparse,hashlib,io,json,multiprocessing as mp,shutil,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from prepare import RUN,DATA,CAMERAS,CROPS,write,sha,collection_gate,assert_pins

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

def main():
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=2);args=p.parse_args()
 if not 1<=args.workers<=4:raise ValueError('Use1-4 bounded decoder workers')
 collection_gate();records=json.loads((RUN/'split_manifest.json').read_text())['source_records'];assert_pins(records)
 needed=sum(r['frames'] for r in records)*3*3*224*224*4
 existing=sum(p.stat().st_size for p in DATA.glob('observation.images.*/*.npy'))
 if shutil.disk_usage(RUN).free < max(0,needed-existing)+80*2**30:raise RuntimeError('Insufficient disk: crop cache plus80GiB checkpoint reserve required')
 start=time.monotonic();results=[]
 with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool:
  for f in as_completed([pool.submit(job,r) for r in records]):
   results.append(f.result());status=dict(episodes=len(results),target=500,frames=sum(n for _,n in results),elapsed_s=time.monotonic()-start,workers=args.workers)
   write('cache_progress.json',status);print(json.dumps(status),flush=True)
 import torch,torchvision,PIL
 manifest=dict(complete=True,episodes=500,frames=sum(n for _,n in results),shape=[3,224,224],dtype='float32',cameras=CAMERAS,crops=CROPS,
  raw_format='native JPEG image sequences; every frame in all3 cameras decoded, byte-hashed and shape-checked; raw sources retained',
  stored_format='out-of-core float32 exact deployed224 crop; native-resolution decoded duplicates intentionally omitted',
  torch=torch.__version__,torchvision=torchvision.__version__,pillow=PIL.__version__,elapsed_s=time.monotonic()-start,manifest_sha256=sha(RUN/'split_manifest.json'))
 write('cache/completed.json',manifest);print(json.dumps(manifest),flush=True)
if __name__=='__main__':main()
