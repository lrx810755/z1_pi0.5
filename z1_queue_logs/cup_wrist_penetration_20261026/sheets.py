from pathlib import Path
import json,cv2
from PIL import Image,ImageDraw
R=Path(__file__).parent;a=json.loads(Path('/home/lp531/smolvla_eval/act_cup_only_frustum_steps15_paired30_20261025/failure_analysis/never_close19/metrics.json').read_text())
for start in range(0,len(a),5):
 page=Image.new('RGB',(1920,210*len(a[start:start+5])),'#111');draw=ImageDraw.Draw(page)
 for row,e in enumerate(a[start:start+5]):
  cap=cv2.VideoCapture(str(Path(e['path'])/'videos/wrist.mp4'));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  for col,f in enumerate([.1,.25,.4,.55,.75,.95]):
   index=int(count*f);cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,im=cap.read();assert ok;im=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));im=im.resize((320,180));page.paste(im,(320*col,210*row+25));draw.text((320*col+5,210*row+5),f'Trial {e["episode"]:02d} Frame {index}',fill='white')
  cap.release()
 page.save(R/f'wrist_timeline_{start:02d}.jpg',quality=95)
