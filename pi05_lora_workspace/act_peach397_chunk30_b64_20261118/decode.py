"""Decode and pin all352611 source JPEGs into lossless indexed float32 RGB cache."""
import hashlib
import io
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import shutil
import time
import numpy as np
from common import RUN, CACHE, CAMERAS, CROPS, read, write, sha, assert_sources, verify_prepared
from codec import pack_frame

IMAGE_HASHES = None


def required_free_bytes(total_frames,existing_bytes):
    #240kB/image is above the sampled199kB; completed files count toward the projection on resume.
    return max(0,total_frames*3*240000-existing_bytes)+20*2**30


def init_worker():
    global IMAGE_HASHES
    import torch
    torch.set_num_threads(1)
    with np.load(RUN/'image_pins.npz') as archive:
        IMAGE_HASHES = archive['image_hashes']


def decode_record(record):
    import torch
    from PIL import Image
    from torchvision.transforms import functional as F
    ep, frames = record['episode_index'], record['frames']
    root = Path(record['source_path'])
    assert_sources([record])
    pinned = RUN/'source_pins'/f'episode_{ep:06d}'/'frames.jsonl'
    if sha(pinned) != record['source_hashes']['frames.jsonl']:
        raise ValueError('Copied frame index changed')
    rows = [json.loads(line) for line in pinned.read_text().splitlines()]
    if len(rows) != frames:
        raise ValueError('Frame count differs')
    camera_reports = []
    for camera_index, cam in enumerate(CAMERAS):
        directory = CACHE/cam; directory.mkdir(parents=True,exist_ok=True)
        dest = directory/f'episode_{ep:06d}.bin'
        index = dest.with_suffix('.npz'); audit = dest.with_suffix('.json')
        if all(p.is_file() for p in [dest,index,audit]):
            report = read(audit)
            if report['data_sha256'] != sha(dest) or report['index_sha256'] != sha(index):
                raise ValueError('Existing cache content changed: '+str(dest))
            camera_reports.append(report); continue
        temp = dest.with_suffix('.building.bin')
        offsets = [0]; digest = hashlib.sha256()
        with temp.open('wb') as stream:
            for begin in range(0,frames,8):
                if shutil.disk_usage(RUN).free < 20*2**30:
                    raise RuntimeError('Cache stopped to retain20GiB for checkpoints and system')
                pixels = []
                for frame, row in enumerate(rows[begin:begin+8],begin):
                    path = root/row[cam]
                    encoded = path.read_bytes()
                    actual = hashlib.sha256(encoded).hexdigest()
                    expected = IMAGE_HASHES[record['dataset_from_index']+frame,camera_index].decode()
                    if actual != expected:
                        raise ValueError('Frozen JPEG bytes changed: '+str(path))
                    with Image.open(io.BytesIO(encoded)) as image:
                        image.load()
                        if image.mode!='RGB' or image.size!=(640,360 if cam.endswith('wrist') else 480):
                            raise ValueError('Raw camera format changed: '+str(path))
                        pixels.append(np.asarray(image).copy())
                tensor = torch.from_numpy(np.stack(pixels)).permute(0,3,1,2).float()/255
                resized = F.resize(F.crop(tensor,*CROPS[cam]),[224,224],antialias=True).clamp(0,1).contiguous().numpy()
                for array in resized:
                    payload = pack_frame(array)
                    stream.write(payload);digest.update(payload);offsets.append(offsets[-1]+len(payload))
                if begin%64==0:
                    write(f'cache_workers/episode_{ep:06d}.json',dict(episode=ep,camera=cam,frames_decoded=min(begin+8,frames),target_frames=frames))
        temp.replace(dest)
        index_temp=index.with_suffix('.building.npz')
        np.savez(index_temp, offsets=np.array(offsets,dtype=np.int64))
        index_temp.replace(index)
        report=dict(episode=ep,camera=cam,frames=frames,shape=[frames,3,224,224],dtype='float32',
                    codec='zlib-level1-per-frame',data=str(dest),index=str(index),
                    compressed_bytes=dest.stat().st_size,data_sha256=digest.hexdigest(),index_sha256=sha(index),
                    source_jpeg_sha256_verified=True,source_frames_sha256=record['source_hashes']['frames.jsonl'],
                    crop=CROPS[cam],resize=[224,224],antialias=True)
        tmp=audit.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(audit)
        camera_reports.append(report)
    write(f'cache_workers/episode_{ep:06d}.json',dict(episode=ep,complete=True,cameras=3,frames=frames))
    return dict(episode=ep,frames=frames,bytes=sum(r['compressed_bytes'] for r in camera_reports),cameras=camera_reports)


def main():
    if not (RUN/'prepared.json').exists():
        import prepare
        prepare.main()
    prepared=verify_prepared()
    if (CACHE/'completed.json').exists():
        marker=read(CACHE/'completed.json')
        if marker['prepared_sha256'] != sha(RUN/'prepared.json'):
            raise ValueError('Cache preparation hash differs')
        print('Existing completed cache marker found; full audit remains required',flush=True);return
    existing=sum(p.stat().st_size for p in CACHE.glob('observation.images.*/*.bin') if p.with_suffix('.json').exists() and p.with_suffix('.npz').exists())
    if shutil.disk_usage(RUN).free < required_free_bytes(117537,existing):
        raise RuntimeError('Insufficient remaining cache capacity plus20GiB reserve')
    records=read(RUN/'split_manifest.json')['source_records']
    start=time.monotonic();reports=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=init_worker) as pool:
        for future in as_completed([pool.submit(decode_record,r) for r in records]):
            reports.append(future.result())
            state=dict(phase='decoding',episodes=len(reports),target_episodes=397,decoded_images=sum(r['frames'] for r in reports)*3,
                       total_images=117537*3,compressed_gib=sum(r['bytes'] for r in reports)/2**30,elapsed_s=time.monotonic()-start)
            write('decode_progress.json',state);print(json.dumps(state),flush=True)
    assert len(reports)==397 and sum(r['frames'] for r in reports)==117537
    write('cache/completed.json',dict(complete=True,episodes=397,frames=117537,images=352611,cameras=CAMERAS,
          dtype='float32',codec='lossless-indexed-zlib',source_jpeg_sha256_verified=True,prepared_sha256=sha(RUN/'prepared.json'),
          elapsed_s=time.monotonic()-start,compressed_bytes=sum(r['bytes'] for r in reports),records=sorted(reports,key=lambda r:r['episode'])))
    write('decode_progress.json',dict(phase='decoded',episodes=397,images=352611,elapsed_s=time.monotonic()-start))
    print('All352611 JPEGs decoded and pinned',flush=True)


if __name__=='__main__':main()
