from pathlib import Path
import json,csv
import numpy as np
import pyarrow.parquet as pq
import cv2
P=Path(__file__).resolve().parent
source=Path('/home/lp531/smolvla_data/z1_tabletop_raw_left_wrist_three_camera_v1/episode_000058')
rows=[json.loads(s) for s in (source/'frames.jsonl').read_text().splitlines()]
t=pq.read_table('/home/lp531/smolvla_data/z1_bottle_gentle_correction_20261010/data',filters=[('episode_index','=',0)]).sort_by([('frame_index','ascending')]);actions=t['action'].to_pylist();assert len(rows)==len(actions)==246
labels=[a[-1] for a in actions];assert set(labels)=={0.,1.};switches=[i for i in range(1,len(labels)) if labels[i]!=labels[i-1]];assert switches==[100,216]
records=[dict(frame=i,time_s=i/10,label=labels[i],raw_command_rad=r['action'][-1],actual_joint_rad=r['observation.state'][-1],phase=r['phase']) for i,r in enumerate(rows)]
(P/'timeline.json').write_text(json.dumps(dict(source=str(source),merged_episode=0,fps=10,frames=246,switch_frames=switches,records=records),ensure_ascii=False,indent=2))
with (P/'timeline.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
writer=cv2.VideoWriter(str(P/'demo_raw.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(1440,440));assert writer.isOpened()
for i,r in enumerate(rows):
 frame=np.full((440,1440,3),245,np.uint8)
 for col,cam in enumerate(['global','global_right','wrist']):
  img=cv2.imread(str(source/r['observation.images.'+cam]));assert img is not None
  h,w=img.shape[:2];img=cv2.resize(img,(480,round(h*480/w)));h=img.shape[0];frame[40:40+h,col*480:(col+1)*480]=img
 cv2.putText(frame,f't={i/10:04.1f}s   frame={i:03d}   TRAIN LABEL: {int(labels[i])} ({"CLOSE" if labels[i] else "OPEN"})   {r["phase"]}',(15,28),cv2.FONT_HERSHEY_SIMPLEX,.65,(30,30,30),2)
 cv2.putText(frame,f'Raw demonstration command: {r["action"][-1]:.3f} rad    Actual joint: {r["observation.state"][-1]:.3f} rad',(15,425),cv2.FONT_HERSHEY_SIMPLEX,.65,(30,30,30),1)
 writer.write(frame)
writer.release()
for i in [0,99,100,121,215,216,245]:print(records[i])
