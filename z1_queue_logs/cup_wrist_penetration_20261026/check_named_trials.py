from pathlib import Path
import cv2,json
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).parent/'named_trials';R.mkdir(exist_ok=True)
base=Path('/home/lp531/smolvla_eval/act_cup_only_frustum_steps15_paired30_20261025/batch_01');font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18);info=[]
for n in [1,2,5,8,12]:
 ep=next(base.glob(f'*/episodes/cup_{n:03d}_*'));tr=[json.loads(l) for l in (ep/'trace.jsonl').read_text().splitlines()];times=[]
 for t in tr:
  if t['execution_substep']==0 and t['policy_action_index']==0:times.append(t['simulation_elapsed_s'])
  if t['execution_substep']==5:times.append(t['simulation_elapsed_s'])
 pg=Image.new('RGB',(1920,510*5),'#111');d=ImageDraw.Draw(pg)
 for row,f in enumerate([.55,.65,.75,.85,.95]):
  idx=int(len(times)*f);when=times[idx];t=min(tr,key=lambda t:abs(t['simulation_elapsed_s']-when));info.append(dict(trial=n,video_frame=idx,time=when,path=str(ep),trace=t));d.text((6,510*row+5),f'Trial {n:02d} | sim={when:.3f}s | frame={idx} | penetration={t["contact_status"].get("target_finger_max_penetration_m")}',font=font,fill='white')
  for col,name in enumerate(['global_third_person.mp4','global_right.mp4','wrist.mp4']):
   cap=cv2.VideoCapture(str(ep/'videos'/name));cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,im=cap.read();cap.release();assert ok;im=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));pg.paste(im,(640*col,510*row+30+(480-im.height)//2))
 pg.save(R/f'trial{n:02d}_second_half.jpg',quality=97)
(R/'frames.json').write_text(json.dumps(info,indent=2));print(R)
