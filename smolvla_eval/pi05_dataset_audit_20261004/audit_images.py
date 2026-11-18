"""Check waiting observations in every episode; verify converted video samples."""
import json
from pathlib import Path
import cv2
import numpy as np
import pyarrow.parquet as pq
import av
ROOT=Path(__file__).resolve().parent
RAW=Path('/home/lp531/smolvla_data/z1_tabletop_bottle_recovery_boundary_only_prepared_v1')
TRAIN=Path('/home/lp531/smolvla_data/z1_tabletop_bottle_pi05_lora_v1')
episodes=json.loads((ROOT/'local_audit.json').read_text())['episodes']
rows=[]
panels=[]
selected=[0,89,187,229]
for e in episodes:
    folder=Path(e['raw_source']);rr=[json.loads(s) for s in (folder/'frames.jsonl').read_text().splitlines()]
    c=e['first_close'];i,j=c-25,c-5
    item={'episode':e['original_episode_index'],'indices':[i,j],
          'state_max_difference_rad':max(abs(a-b) for a,b in zip(rr[i]['observation.state'],rr[j]['observation.state'])),
          'future_close_offsets':[c-i,c-j],'cameras':{}}
    for cam in ['global','wrist']:
        key='observation.images.'+cam
        a,b=[cv2.imread(str(folder/rr[k][key])) for k in [i,j]]
        assert a is not None and b is not None
        diff=np.abs(a.astype(np.float32)-b.astype(np.float32))
        item['cameras'][cam]={'mae_255':float(diff.mean()),'fraction_pixels_change_gt5':float((diff.max(axis=2)>5).mean()),
            'std_255':[float(a.std()),float(b.std())]}
    rows.append(item)
for old in selected:
    e=next(e for e in episodes if e['original_episode_index']==old)
    folder=Path(e['raw_source']);rr=[json.loads(s) for s in (folder/'frames.jsonl').read_text().splitlines()];c=e['first_close']
    for cam in ['global','wrist']:
        tiles=[]
        for k in [c-25,c-5,c]:
            im=cv2.imread(str(folder/rr[k]['observation.images.'+cam]));im=cv2.resize(im,(384,216))
            label=f"ep{old:03d} {cam} f{k} jaw={rr[k]['observation.state'][-1]:.3f} a={rr[k]['action'][-1]:.3f}"
            cv2.putText(im,label,(5,16),cv2.FONT_HERSHEY_SIMPLEX,.38,(0,0,255),1)
            tiles.append(im)
        panels.append(np.hstack(tiles))
cv2.imwrite(str(ROOT/'waiting_contactsheet.jpg'),np.vstack(panels))
meta={x['episode_index']:x for x in pq.read_table(TRAIN/'meta/episodes').to_pylist()}
checks=[]
for old in selected:
    e=next(e for e in episodes if e['original_episode_index']==old);ep=meta[e['episode_index']];folder=Path(e['raw_source'])
    rr=[json.loads(s) for s in (folder/'frames.jsonl').read_text().splitlines()];k=e['first_close']-5
    for cam in ['global','wrist']:
        key='observation.images.'+cam;prefix='videos/'+key
        path=TRAIN/f"videos/{key}/chunk-{ep[prefix+'/chunk_index']:03d}/file-{ep[prefix+'/file_index']:03d}.mp4"
        wanted=ep[prefix+'/from_timestamp']+k/10
        with av.open(str(path)) as container:
            stream=container.streams.video[0]
            container.seek(int(max(0,wanted-1)/stream.time_base),stream=stream)
            frames=[]
            for frame in container.decode(stream):
                if float(frame.time)>=wanted-.051: frames.append(frame)
                if float(frame.time)>wanted+.051:break
            frame=min(frames,key=lambda f:abs(float(f.time)-wanted))
            raw=cv2.imread(str(folder/rr[k][key]));decoded=frame.to_ndarray(format='bgr24')
            checks.append({'original_episode':old,'camera':cam,'requested_s':wanted,'decoded_s':frame.time,
                'mean_absolute_pixel_error_255':float(np.abs(raw.astype(float)-decoded).mean())})
summary={cam:{'waiting_mae_median_255':float(np.median([r['cameras'][cam]['mae_255'] for r in rows])),
    'waiting_mae_max_255':max(r['cameras'][cam]['mae_255'] for r in rows),
    'waiting_frames_std_below5':sum(min(r['cameras'][cam]['std_255'])<5 for r in rows)} for cam in ['global','wrist']}
(ROOT/'image_audit.json').write_text(json.dumps({'scope':'2 waiting images per camera per episode; 8 converted video frames; 4 episodes visualized. Not full visual quality annotation.',
    'summary':summary,'waiting_pairs':rows,'conversion_checks':checks},indent=2))
print(json.dumps(summary,indent=2));print('video checks',checks)
