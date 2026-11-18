from pathlib import Path
import json,cv2,numpy as np
out=Path(__file__).resolve().parent
rows=json.loads((out/'metrics.json').read_text())
for r in rows:
 time=r['run'].split('_')[-1]
 if time not in ['095822','100451','103451','103806','102951']:continue
 panels=[]
 for fraction in [.25,.55,.9]:
  images=[]
  for camera in ['global_third_person','wrist','global_right']:
   p=Path(r['path'])/'videos'/f'{camera}.mp4';cap=cv2.VideoCapture(str(p));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));idx=int((count-1)*fraction);cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,image=cap.read();cap.release()
   if not ok:image=np.zeros((240,320,3),np.uint8)
   image=cv2.resize(image,(400,300));cv2.putText(image,f'{camera} frame {idx}/{count}',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1);images.append(image)
  panels.append(np.hstack(images))
 canvas=np.vstack(panels);cv2.imwrite(str(out/f'{r["model"]}_{time}.jpg'),canvas)
