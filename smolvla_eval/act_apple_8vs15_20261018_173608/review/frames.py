import json
from pathlib import Path
import cv2,numpy as np
R=Path(__file__).resolve().parent
metrics=json.loads((R/'metrics.json').read_text())
def make(steps,n,seconds):
 m=next(x for x in metrics if x['steps']==steps and x['episode']==n);p=Path(m['path']);rows=[json.loads(l) for l in (p/'trace.jsonl').read_text().splitlines()]
 frames=[]
 for row in rows:
  if row['policy_action_index']==0 and row['execution_substep']==0:frames.append(row['simulation_elapsed_s'])
  if row['execution_substep']==5:frames.append(row['simulation_elapsed_s'])
 sheet=np.zeros((len(seconds)*205,3*320,3),np.uint8)
 for c,name in enumerate(['global_third_person','global_right','wrist']):
  cap=cv2.VideoCapture(str(p/'videos'/f'{name}.mp4'));actual=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));print(steps,n,name,'frames',actual,'map',len(frames))
  for row,t in enumerate(seconds):
   idx=int(np.argmin(abs(np.array(frames)-t)));cap.set(cv2.CAP_PROP_POS_FRAMES,min(idx,actual-1));ok,image=cap.read()
   if not ok:continue
   image=cv2.resize(image,(320,180));sheet[row*205+25:row*205+205,c*320:c*320+320]=image
   cv2.putText(sheet,f'{steps} steps ep{n} t={t:.1f}s {name}',(c*320+4,row*205+17),cv2.FONT_HERSHEY_SIMPLEX,.40,(255,255,255),1)
  cap.release()
 dest=R/f'frames_s{steps}_ep{n:02d}.jpg';cv2.imwrite(str(dest),sheet)
if __name__=='__main__':
 for steps,n,seconds in [(8,1,[17,19,35]),(8,27,[12,25,55]),(15,4,[12,25,55]),(15,5,[13,30,55]),(15,10,[20,40,58]),(15,11,[15,35,58]),(15,28,[20,40,58]),(15,8,[23,24.3,24.8,30]),(15,19,[37,39,39.8,45]),(15,14,[20,40,58]),(15,18,[2,4,6])]:make(steps,n,seconds)
