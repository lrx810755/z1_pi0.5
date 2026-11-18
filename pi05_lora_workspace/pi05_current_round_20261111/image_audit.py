"""Decode every source JPEG once, then reuse content-pinned per-episode caches."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
from io import BytesIO
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image

CAMERAS=['observation.images.'+c for c in ['wrist','global','global_right']]
KEYS=['image_sizes','image_mtime_ns','image_ctime_ns','image_hashes']

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def atomic_json(path,value):
    p=Path(path);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)

def audit_episode(item,cache_dir):
    root=Path(item['path']).resolve();frames=root/'frames.jsonl'
    if digest(frames)!=item['frames_sha256']:raise ValueError('Frame manifest changed: '+str(root))
    rows=[json.loads(x) for x in frames.read_text().splitlines()];n=len(rows)
    cache_dir=Path(cache_dir);cache_dir.mkdir(parents=True,exist_ok=True)
    key=hashlib.sha256((str(root)+'\n'+item['frames_sha256']).encode()).hexdigest();cache=cache_dir/(key+'.npz')
    reused=cache.exists();changed=False
    if reused:
        with np.load(cache,allow_pickle=False) as values:
            if str(values['source_path'])!=str(root) or str(values['frames_sha256'])!=item['frames_sha256']:raise ValueError('Image cache identity mismatch')
            arrays={k:values[k].copy() for k in KEYS}
        if any(a.shape!=(n,3) for a in arrays.values()):raise ValueError('Image cache shape mismatch')
    else:
        arrays={k:np.empty((n,3),dtype='S64' if k=='image_hashes' else np.int64) for k in KEYS}
    for i,row in enumerate(rows):
        for j,camera in enumerate(CAMERAS):
            path=(root/row[camera]).resolve()
            if not path.is_relative_to(root):raise ValueError('Image escapes episode: '+str(path))
            st=path.stat();actual=(st.st_size,st.st_mtime_ns,st.st_ctime_ns)
            if reused:
                expected=tuple(int(arrays[k][i,j]) for k in KEYS[:3])
                if actual==expected:continue
            raw=path.read_bytes();image_hash=hashlib.sha256(raw).hexdigest()
            if reused and image_hash!=arrays['image_hashes'][i,j].decode():raise ValueError('Previously audited JPEG changed: '+str(path))
            with Image.open(BytesIO(raw)) as im:
                im.load()
                if im.mode!='RGB' or im.size!=(640,360 if camera.endswith('wrist') else 480):raise ValueError('Image RGB shape/mode mismatch: '+str(path))
            after=path.stat()
            if actual!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise ValueError('JPEG modified while being audited')
            for k,value in zip(KEYS[:3],actual):arrays[k][i,j]=value
            arrays['image_hashes'][i,j]=image_hash.encode();changed=True
    if not reused or changed:
        tmp=cache.with_suffix('.npz.tmp')
        with tmp.open('wb') as f:np.savez_compressed(f,source_path=np.asarray(str(root)),frames_sha256=np.asarray(item['frames_sha256']),**arrays)
        tmp.replace(cache)
    return arrays,dict(path=str(root),frames=n,images=n*3,cache=str(cache),reused=reused)

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    from config import RUN
    from pipeline import verify_frozen
    frozen=verify_frozen();started=time.monotonic();results=[];errors=[]
    with ThreadPoolExecutor(max_workers=a.workers) as executor:
        jobs={executor.submit(audit_episode,item,RUN/'image_audit_cache'):item for item in frozen['sources']}
        for future in as_completed(jobs):
            try:_,result=future.result();results.append(result)
            except Exception as exc:errors.append(dict(path=jobs[future]['path'],error=str(exc)))
            completed=len(results)+len(errors)
            if completed%20==0 or completed==len(jobs):
                state=dict(stage='auditing',completed=completed,target=len(jobs),images=sum(r['images'] for r in results),errors=len(errors),elapsed_s=round(time.monotonic()-started,1))
                atomic_json(RUN/'image_audit_progress.json',state);print(json.dumps(state),flush=True)
    verify_frozen()
    report=dict(passed=not errors,episodes=len(results),images=sum(r['images'] for r in results),elapsed_s=time.monotonic()-started,errors=errors,records=results,workers=a.workers)
    atomic_json(RUN/'image_audit_complete.json',report)
    if errors:raise RuntimeError('Source JPEG audit failed: '+str(errors[:3]))
    print('ALL_SOURCE_JPEGS_PASSED',report['images'],flush=True)
if __name__=='__main__':main()
