import json,cv2
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parent;TRAIN=Path('/home/lp531/pi05_lora_workspace/act_cup_minus90_full_cached_20261023')
a=json.loads((TRAIN/'review/basic160_5x_manifest.json').read_text());records=json.loads((TRAIN/'split_manifest.json').read_text())['source_records'][:160];cap=cv2.VideoCapture(a['file']);assert cap.isOpened()
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12);pages=[]
for first in range(0,160,40):
 page=Image.new('RGB',(1600,850),'#111111')
 for j in range(40):
  rec=records[first+j];chapter=a['chapters'][first+j]
  idx=round(chapter['start_s']*10)+max(0,(rec['close_frame']-1)//5)
  cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,img=cap.read();assert ok
  im=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB));im=im.resize((200,150))
  x=(j%8)*200;y=(j//8)*170;page.paste(im,(x,y+20));ImageDraw.Draw(page).text((x+3,y+2),f'Demo {first+j:03d} grid {rec["grid"]}',font=font,fill='white')
 out=R/f'demos_{first:03d}_{first+39:03d}.jpg';page.save(out,quality=90);pages.append(str(out))
cap.release();print('Decoded preclose frames from all 160 demonstrations in the actual review video.')
