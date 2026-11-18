"""Decode each of nine source videos sequentially once, preserving timestamp alignment."""
import json,time,os,hashlib
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
import numpy as np
from cache_data import CACHE
DATA=Path('/home/lp531/smolvla_data/z1_bottle_with_fragments_20261012');RUN=Path(__file__).resolve().parent
CROPS={'observation.images.wrist':(-140,0,640,640),'observation.images.global':(-80,0,640,640),'observation.images.global_right':(-80,0,640,640)}
def job(task):
 import av,torch
 from torchvision.transforms import functional as F
 torch.set_num_threads(1)
 video,key,records=task
 requests=sorted((e['from_timestamp']+i/10,e['episode_index'],i) for e in records for i in range(e['length']))
 maps={};temps={}
 for e in records:
  final=CACHE/key/f"episode_{e['episode_index']:06d}.npy";final.parent.mkdir(parents=True,exist_ok=True)
  tmp=final.with_suffix('.building.npy');temps[e['episode_index']]=(tmp,final)
  maps[e['episode_index']]=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=(e['length'],3,224,224))
 started=time.monotonic();batch=[];targets=[];maxerror=0
 def flush():
  if not batch:return
  pixels=torch.from_numpy(np.stack(batch)).permute(0,3,1,2).float()/255
  images=F.resize(F.crop(pixels,*CROPS[key]),[224,224]).clamp(0,1).numpy()
  for a,(ep,i) in zip(images,targets):maps[ep][i]=a
  batch.clear();targets.clear()
 with av.open(video) as container:
  stream=container.streams.video[0];stream.codec_context.thread_count=2
  iterator=iter(container.decode(stream));previous=None;current=None
  def next_frame():
   for frame in iterator:
    if frame.pts is not None:return (float(frame.pts*stream.time_base),frame)
   return None
  current=next_frame()
  for number,(stamp,ep,i) in enumerate(requests):
   while current is not None and current[0]<stamp-1e-7:previous=current;current=next_frame()
   choices=[x for x in [previous,current] if x is not None];chosen=min(choices,key=lambda x:abs(x[0]-stamp))
   error=abs(chosen[0]-stamp);maxerror=max(error,maxerror)
   if error>.001:raise RuntimeError(f'Video/frame mismatch {video} {stamp} {chosen[0]}')
   batch.append(chosen[1].to_ndarray(format='rgb24'));targets.append((ep,i))
   if len(batch)==32:flush()
   if number and number%5000==0:print(f'{key} {Path(video).name}: {number}/{len(requests)}',flush=True)
  flush()
 for a in maps.values():a.flush()
 maps.clear()
 for tmp,final in temps.values():tmp.replace(final)
 digest=hashlib.file_digest(Path(video).open('rb'),'sha256').hexdigest()
 return dict(video=video,camera=key,frames=len(requests),episodes=len(records),elapsed_s=time.monotonic()-started,max_timestamp_error_s=maxerror,source_sha256=digest)
def main():
 from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
 CACHE.mkdir(exist_ok=True,parents=True)
 if (CACHE/'completed.json').exists():print('Existing complete cache; use verification before training',flush=True);return
 meta=LeRobotDatasetMetadata('lp531/z1_bottle_with_fragments_20261012',root=DATA);groups=defaultdict(list)
 for ep in range(meta.total_episodes):
  e=meta.episodes[ep]
  for key in meta.video_keys:
   path=str(DATA/meta.get_video_file_path(ep,key));groups[path,key].append(dict(episode_index=ep,length=e['length'],from_timestamp=e[f'videos/{key}/from_timestamp']))
 print('Building',len(groups),'video streams;',meta.total_frames,'frames x 3; float32 cache ~85GB',flush=True)
 results=[];start=time.monotonic()
 with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn')) as pool:
  futures=[pool.submit(job,(path,key,records)) for (path,key),records in groups.items()]
  for f in as_completed(futures):
   result=f.result();results.append(result);print('Completed video',json.dumps(result),flush=True)
   (CACHE/'progress.json').write_text(json.dumps(dict(completed_videos=len(results),target_videos=len(groups),elapsed_s=time.monotonic()-start),indent=2))
 expected=meta.total_frames*len(meta.video_keys)
 assert sum(x['frames'] for x in results)==expected
 manifest=dict(complete=True,dataset=str(DATA),episodes=meta.total_episodes,frames=meta.total_frames,cameras=meta.video_keys,dtype='float32',shape=[3,224,224],crop_params=CROPS,normalization='RGB /255 only; ImageNet normalization remains in training processor',elapsed_s=time.monotonic()-start,videos=results)
 (CACHE/'completed.json').write_text(json.dumps(manifest,indent=2));print('CACHE COMPLETE',manifest['elapsed_s'],flush=True)
if __name__=='__main__':main()
