"""Decode source JPEG frames once; preserve exact deployed letterbox/resize."""
import json,os,time,hashlib,multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from prepare import RUN,DATA,CAMERAS,CROPS,write

def job(record):
 import torch
 from PIL import Image
 from torchvision.transforms import functional as F
 torch.set_num_threads(1)
 root=Path(record['source_path']);raw=root/'frames.jsonl'
 assert hashlib.sha256(raw.read_bytes()).hexdigest()==record['source_frames_sha256']
 rows=[json.loads(x) for x in raw.read_text().splitlines()];n=len(rows);idx=record['episode_index']
 for cam in CAMERAS:
  dest=DATA/cam/f'episode_{idx:06d}.npy';dest.parent.mkdir(exist_ok=True)
  if dest.exists():
   a=np.load(dest,mmap_mode='r');assert a.shape==(n,3,224,224) and a.dtype==np.float32;continue
  # Prior cache is immutable and source hashes are verified before linking.
  previous=Path('/home/lp531/pi05_lora_workspace/act_cup_recovery_cached_20261020/split_manifest.json')
  old_records=json.loads(previous.read_text())['source_records']
  match=next((r for r in old_records if r['source_path']==record['source_path'] and r['source_frames_sha256']==record['source_frames_sha256']),None)
  if match:
   old=Path('/home/lp531/smolvla_data/act_cup_recovery_rgb224_cache_20261020')/cam/f'episode_{match["episode_index"]:06d}.npy'
   if old.exists():
    a=np.load(old,mmap_mode='r');assert a.shape==(n,3,224,224) and a.dtype==np.float32
    del a;os.link(old,dest);continue
  temp=dest.with_suffix('.building.npy');out=np.lib.format.open_memmap(temp,mode='w+',dtype=np.float32,shape=(n,3,224,224))
  for begin in range(0,n,16):
   batch=[]
   for row in rows[begin:begin+16]:
    with Image.open(root/row[cam]) as image:batch.append(np.array(image.convert('RGB')))
   pixels=torch.from_numpy(np.stack(batch)).permute(0,3,1,2).float()/255
   resized=F.resize(F.crop(pixels,*CROPS[cam]),[224,224]).clamp(0,1).numpy()
   out[begin:begin+len(batch)]=resized
  out.flush();del out;temp.replace(dest)
 return idx,n

def main():
 records=json.loads((RUN/'split_manifest.json').read_text())['source_records'];start=time.monotonic();results=[]
 with ProcessPoolExecutor(max_workers=6,mp_context=mp.get_context('spawn')) as pool:
  for f in as_completed([pool.submit(job,r) for r in records]):
   results.append(f.result());status=dict(episodes=len(results),target=len(records),frames=sum(n for _,n in results),elapsed_s=time.monotonic()-start)
   write('cache_progress.json',status)
   if len(results)%10==0:print(json.dumps(status),flush=True)
 manifest=dict(complete=True,episodes=len(records),frames=sum(n for _,n in results),shape=[3,224,224],dtype='float32',source_format='raw JPEG, decoded directly without intermediate video compression',cameras=CAMERAS,crops=CROPS,elapsed_s=time.monotonic()-start,manifest_sha256=hashlib.sha256((RUN/'split_manifest.json').read_bytes()).hexdigest())
 temp=DATA/'completed.tmp';temp.write_text(json.dumps(manifest,indent=2));temp.replace(DATA/'completed.json');print('CACHE COMPLETE',json.dumps(manifest),flush=True)
if __name__=='__main__':main()
