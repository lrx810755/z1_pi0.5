"""Reuse500 verified immutable crops and decode only newly accepted episodes."""
import argparse,json,multiprocessing as mp,time,shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from prepare import RUN,DATA,OLD,CAMERAS,CROPS,load,sha,write,collection_gate,assert_pins
from decode_new import job

def verify_source_image(path,digest,decoded_at):
 stat=path.stat()
 if max(stat.st_mtime_ns,stat.st_ctime_ns)>decoded_at and sha(path)!=digest:raise ValueError(f'Raw source image changed since original decode: {path}')


def inherited_cache(records):
 evidence=load(OLD/'cache_verified.json');verified_at=(OLD/'cache_verified.json').stat().st_mtime_ns
 if not evidence.get('passed') or not evidence.get('digests_verified') or evidence['manifest_sha256']!=sha(OLD/'split_manifest.json'):raise ValueError('Old cache does not have current digest evidence')
 old_records=load(OLD/'split_manifest.json')['source_records']
 inherited=[]
 for r,old in zip(records[:500],old_records):
  for key in ['episode_index','source_path','frames','source_hashes','dataset_from_index','dataset_to_index']:
   if r[key]!=old[key]:raise ValueError('Existing500 record ordering/source changed')
  for cam in CAMERAS:
   source=OLD/'cache'/cam/f'episode_{r["episode_index"]:06d}.npy';audit=source.with_suffix('.json')
   for p in [source,audit]:
    stat=p.stat()
    if max(stat.st_mtime_ns,stat.st_ctime_ns)>verified_at:raise ValueError(f'Inherited cache changed after verification: {p}')
   a=np.load(source,mmap_mode='r')
   if a.shape!=(r['frames'],3,224,224) or a.dtype!=np.float32:raise ValueError('Inherited cache shape mismatch')
   meta=load(audit)
   for image in meta['source_images']:verify_source_image(Path(r['source_path'])/image['path'],image['sha256'],audit.stat().st_mtime_ns)
   if meta['crop']!=list(CROPS[cam]) or meta['resize']!=[224,224] or meta.get('antialias') is not True:raise ValueError('Inherited crop parameters differ')
   for src in [source,audit]:
    dest=DATA/cam/src.name;dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() or dest.is_symlink():
     if not dest.is_symlink() or dest.resolve()!=src.resolve():raise ValueError('Refuse overwrite cache')
    else:dest.symlink_to(src.resolve())
   inherited.append(str(source.resolve()))
 low=OLD/'cache'/'lowdim.npz';stat=low.stat()
 if max(stat.st_mtime_ns,stat.st_ctime_ns)>verified_at:raise ValueError('Inherited lowdim changed since verification')
 return dict(cache_files=len(inherited),digest_evidence=str(OLD/'cache_verified.json'),digest_evidence_sha256=sha(OLD/'cache_verified.json'),unchanged_metadata_verified=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=2);args=p.parse_args()
 if not 1<=args.workers<=4:raise ValueError('Use1..4 bounded decoder workers')
 collection_gate();records=load(RUN/'split_manifest.json')['source_records'];assert_pins(records)
 if load(RUN/'split_manifest.json')['completion_evidence_sha256']!=sha(__import__('prepare').NEW/'completion_verified.json'):raise ValueError('Acceptance evidence changed after preparation')
 new=records[500:];needed=sum(r['frames'] for r in new)*3*3*224*224*4
 if shutil.disk_usage(RUN).free<needed+80*2**30:raise ValueError('Need new crop cache plus80GiB reserve')
 inherited=inherited_cache(records);write('cache_reuse_provenance.json',inherited)
 with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as executor:
  for n,future in enumerate(as_completed([executor.submit(job,r) for r in new]),1):
   future.result();write('cache_progress.json',dict(new_episodes=n,total_new=len(new),inherited_episodes=500))
 write('cache/completed.json',dict(complete=True,episodes=len(records),frames=sum(r['frames'] for r in records),shape=[3,224,224],dtype='float32',cameras=CAMERAS,crops=CROPS,inherited500=inherited,decoded_new_only=len(new),manifest_sha256=sha(RUN/'split_manifest.json')))
if __name__=='__main__':main()
