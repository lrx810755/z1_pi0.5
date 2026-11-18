"""Verify encoded correction cameras, metadata ranges, and reused video bytes."""
from pathlib import Path
import hashlib,json
from concurrent.futures import ThreadPoolExecutor
import av
import pyarrow.parquet as pq
W=Path(__file__).resolve().parent
DATA=Path('/home/lp531/smolvla_data/z1_bottle_with_fragments_20261012')
BASE=Path('/home/lp531/smolvla_data/z1_bottle_gentle_correction_20261010')
info=json.loads((DATA/'meta/info.json').read_text());episodes=pq.read_table(DATA/'meta/episodes').sort_by([('episode_index','ascending')]).to_pylist()
cameras=[k for k in info['features'] if k.startswith('observation.images.')]
files={}
for e in episodes[160:]:
    for camera in cameras:
        prefix='videos/'+camera+'/'
        p=DATA/info['video_path'].format(video_key=camera,chunk_index=e[prefix+'chunk_index'],file_index=e[prefix+'file_index'])
        files.setdefault(p,[]).append(e)
        assert abs((e[prefix+'to_timestamp']-e[prefix+'from_timestamp'])*10-e['length'])<1e-4

def check(item):
    path,es=item;expected=sum(e['length'] for e in es);count=0;previous=None
    with av.open(str(path)) as container:
        stream=container.streams.video[0];stream.thread_type='AUTO';stream.thread_count=2
        assert float(stream.average_rate)==10
        for frame in container.decode(video=0):
            pts=float(frame.pts*frame.time_base)
            if previous is not None:assert abs(pts-previous-.1)<1e-5
            previous=pts;count+=1
    assert count==expected,(str(path),count,expected)
    return dict(video=str(path),decoded_frames=count,episodes=len(es),fps=10)

with ThreadPoolExecutor(max_workers=3) as pool:verified=list(pool.map(check,files.items()))
reused=[]
for p in (BASE/'videos').rglob('*.mp4'):
    dst=DATA/p.relative_to(BASE)
    def digest(path):
        with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    assert digest(p)==digest(dst);reused.append(str(dst))
report=dict(passed=True,camera_count=len(cameras),correction_episodes=51,
            decoded_video_files=len(verified),decoded_camera_frames=sum(r['decoded_frames'] for r in verified),
            all_correction_episode_camera_time_ranges_verified=True,reused_video_files_bytes_unchanged=len(reused),
            videos=verified)
(W/'media_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='videos'},ensure_ascii=False,indent=2))
