from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
P=Path(__file__).resolve().parent
a=json.loads((P/'audit.json').read_text());font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',18)
def sheet(seed,model,times,name):
 d=next(x for x in a['episodes'] if x['seed']==seed and x['model']==model);ep=Path(d['path']);rows=[json.loads(x) for x in (ep/'trace.jsonl').read_text().splitlines()]
 events=[]
 for r in rows:
  if r['policy_action_index']==0 and r['execution_substep']==0:events.append(r['simulation_elapsed_s'])
  if r['execution_substep']==5:events.append(r['simulation_elapsed_s'])
 canvas=Image.new('RGB',(1440,len(times)*405),'white');draw=ImageDraw.Draw(canvas)
 checks=[]
 for k,time in enumerate(times):
  j=min(range(len(events)),key=lambda i:abs(events[i]-time));r=min(rows,key=lambda r:abs(r['simulation_elapsed_s']-events[j]));draw.text((5,k*405),f'{model} seed={seed} sim≈{events[j]:.2f}s raw={r["policy_action"][-1]:.4f} rel={r["contact_status"]["relative_position_gripper_m"]}',font=font,fill='black')
  for c,cam in enumerate(['global_third_person','global_right','wrist']):
   cap=cv2.VideoCapture(str(ep/'videos'/f'{cam}.mp4'));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));checks.append(dict(camera=cam,video_frames=count,reconstructed_events=len(events),difference=count-len(events)));cap.set(cv2.CAP_PROP_POS_FRAMES,min(j,count-1));ok,frame=cap.read();cap.release();assert ok
   im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));im.thumbnail((480,360));canvas.paste(im,(c*480,k*405+40));draw.text((c*480,k*405+378),cam,font=font,fill='black')
 canvas.save(P/name);return checks
r=[]
for model,times in [('old_080000',[8.8,10.4,13.1,25]),('new_080000',[10.4,15.7,17.2,59]),('new_096000',[10.4,30,41.2,59]),('new_120000',[10.4,18,29.9,59])]:
 r+=sheet(276137783,model,times,f'{model}_near_sequence.jpg')
r+=sheet(1174105379,'new_120000',[23.2,30.6,32,59],'new120_far_empty_close.jpg')
(P/'video_alignment_check.json').write_text(json.dumps(r,indent=2))
print('sheets=5 max framecount diff',max(abs(x['difference']) for x in r))
