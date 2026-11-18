import json,cv2
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parent;D=json.loads((R/'comparison.json').read_text());font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
def read_video(p,t,trace=None):
 cap=cv2.VideoCapture(str(p));assert cap.isOpened(),p
 fps=cap.get(cv2.CAP_PROP_FPS);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));duration=n/fps
 if trace:
  times=[]
  for row in trace:
   if row['execution_substep']==0 and row['policy_action_index']==0:times.append(row['simulation_elapsed_s'])
   if row['execution_substep']==5:times.append(row['simulation_elapsed_s'])
  idx=min(range(len(times)),key=lambda i:abs(times[i]-t));idx=min(n-1,idx)
 else:idx=min(n-1,int(t*fps))
 cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,frame=cap.read();cap.release();assert ok,(p,idx)
 return Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)),dict(frame=idx,fps=fps,frames=n,duration_s=duration)
def tile(im,label):
 out=Image.new('RGB',(320,280),'#111111');im=im.copy();im.thumbnail((320,240));out.paste(im,((320-im.width)//2,30));ImageDraw.Draw(out).text((5,5),label,font=font,fill='white');return out
images=[];video_checks=[]
for e in D['episodes']:
 d=e['nearest_demo'];src=Path(d['path']);rows=[json.loads(l) for l in (src/'frames.jsonl').read_text().splitlines()]
 demo=Image.open(src/rows[max(0,d['close_frame']-1)]['observation.images.global']).convert('RGB')
 event=e['first_close'] or e['best_preclose'];t=event['time'];trace=[json.loads(l) for l in (Path(e['path'])/'trace.jsonl').read_text().splitlines()]
 global_im,check=read_video(e['video_paths']['global'],t,trace);wrist,_=read_video(e['video_paths']['wrist'],t,trace);final,_=read_video(e['video_paths']['global'],trace[-1]['simulation_elapsed_s'],trace)
 row=Image.new('RGB',(1280,315),'#202020');draw=ImageDraw.Draw(row)
 title=f'EVAL {e["episode"]:02d} | {e["category"]} | close count {e["close_count"]} | t={t:.2f}s | nearest demo {d["episode"]:03d}, XY {e["nearest_demo_xy_distance_m"]*1000:.1f}mm'
 draw.text((5,3),title,font=font,fill='#ffff80')
 for i,(im,label) in enumerate([(demo,'DEMO before close'),(global_im,'EVAL first close' if e['first_close'] else 'EVAL nearest approach'),(wrist,'EVAL wrist at same event'),(final,'EVAL final')]):row.paste(tile(im,label),(i*320,30))
 images.append(row);video_checks.append(dict(episode=e['episode'],**check))
for start in range(0,30,3):
 page=Image.new('RGB',(1280,315*3),'black')
 for i,row in enumerate(images[start:start+3]):page.paste(row,(0,i*315))
 page.save(R/f'comparison_{start+1:02d}_{start+3:02d}.jpg',quality=92)
(R/'video_frames_verified.json').write_text(json.dumps(video_checks,indent=2))
print('Created 10 comparison sheets covering all 30 evaluation videos.')
