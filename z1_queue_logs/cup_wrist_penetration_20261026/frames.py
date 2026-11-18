from pathlib import Path
import json,cv2
from PIL import Image,ImageDraw
r=Path('/home/lp531/smolvla_eval/act_cup_only_frustum_steps15_paired30_20261025/failure_analysis/never_close19')
a=json.loads((r/'metrics.json').read_text());out=Path('/home/lp531/z1_queue_logs/cup_wrist_penetration_20261026')
for n in [2,8,12,23]:
 e=next(x for x in a if x['episode']==n);ep=Path(e['path']);tr=[json.loads(l) for l in (ep/'trace.jsonl').read_text().splitlines()];ts=[]
 for t in tr:
  if t['execution_substep']==0 and t['policy_action_index']==0:ts.append(t['simulation_elapsed_s'])
  if t['execution_substep']==5:ts.append(t['simulation_elapsed_s'])
 time=e['closest_capture']['time'];idx=min(range(len(ts)),key=lambda i:abs(ts[i]-time));page=Image.new('RGB',(1920,510),'#111')
 for i,name in enumerate(['global_third_person.mp4','global_right.mp4','wrist.mp4']):
  cap=cv2.VideoCapture(str(ep/'videos'/name));cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,im=cap.read();assert ok;cap.release();im=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));page.paste(im,(i*640,30+(480-im.height)//2));ImageDraw.Draw(page).text((i*640+5,5),f'Trial {n}, t={time:.2f}s, {name}',fill='white')
 page.save(out/f'trial{n:02d}.jpg',quality=97)
