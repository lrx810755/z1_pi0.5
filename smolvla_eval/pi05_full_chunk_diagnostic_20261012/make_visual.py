import json
from pathlib import Path
import cv2
from PIL import Image,ImageDraw
P=Path(__file__).resolve().parent;ep=next((P/'new_120000').glob('*/episodes/*_276137783'))
rows=[json.loads(x) for x in (ep/'trace.jsonl').read_text().splitlines()];events=[]
for r in rows:
 if r['policy_action_index']==0 and r['execution_substep']==0:events.append(r['simulation_elapsed_s'])
 if r['execution_substep']==5:events.append(r['simulation_elapsed_s'])
im=Image.new('RGB',(1440,1200),'white');draw=ImageDraw.Draw(im)
for k,time in enumerate([20,27,32]):
 j=min(range(len(events)),key=lambda i:abs(events[i]-time));draw.text((5,k*400+5),f'new120 near sim~{events[j]:.2f}s',fill='black')
 for c,cam in enumerate(['global_third_person','global_right','wrist']):
  cap=cv2.VideoCapture(str(ep/'videos'/f'{cam}.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,j);ok,frame=cap.read();cap.release();assert ok
  tile=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));tile.thumbnail((480,360));im.paste(tile,(c*480,k*400+30))
im.save(P/'new120_near_lift_release.jpg')
