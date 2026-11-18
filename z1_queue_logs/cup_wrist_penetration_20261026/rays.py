import json,numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path
R=Path(__file__).parent;g=np.load(R/'geometry.npz');meta=json.loads((R/'geometry.json').read_text());verts=g['stator_0'];tri=verts[g['stator_faces']];v0=tri[:,0];e1=tri[:,1]-v0;e2=tri[:,2]-v0;origin=np.array([.02,-.08,.04]);tvec=origin-v0;cross=np.cross(tvec,e1);cm=np.array(meta['camera']['matrix_row_vector']);inv=np.linalg.inv(cm)
def hit(target):
 ray=target-origin;dist=np.linalg.norm(ray);dr=ray/dist;p=np.cross(np.broadcast_to(dr,e2.shape),e2);det=np.einsum('ij,ij->i',e1,p);nz=abs(det)>1e-12;iv=np.zeros_like(det);iv[nz]=1/det[nz];u=np.einsum('ij,ij->i',tvec,p)*iv;v=cross@dr*iv;t=np.einsum('ij,ij->i',e2,cross)*iv;mask=nz&(u>=0)&(v>=0)&(u+v<=1)&(t>1e-6)&(t<dist-.0002)
 if not mask.any():return None
 k=np.where(mask)[0][np.argmin(t[mask])];pos=origin+dr*t[k];depth=-np.append(pos,1).dot(inv)[2]
 return dict(distance=float(t[k]),depth=float(depth),backface=bool(det[k]<0),point=pos.tolist())
a=json.loads(Path('/home/lp531/smolvla_eval/act_cup_only_frustum_steps15_paired30_20261025/failure_analysis/never_close19/metrics.json').read_text());out=[]
for n in [2,5,8,9,12,23,24]:
 ep=next(e for e in a if e['episode']==n);tr=[json.loads(l) for l in (Path(ep['path'])/'trace.jsonl').read_text().splitlines()];t=tr[ep['closest_capture']['trace_index']];s=t['contact_status'];rg=Rotation.from_quat(s['gripper_world_orientation_xyzw']);rc=Rotation.from_quat(s['object_orientation_xyzw']);offset=rg.inv().apply(np.array(s['object_position_m'])-s['gripper_world_position_m']);pts=rg.inv().apply(rc.apply(g[next(k for k,v in meta.items() if isinstance(v,dict) and v.get('path','').endswith('/Object_2'))][::100]))+offset
 hits=[hit(v) for v in pts];hh=[h for h in hits if h];row=dict(episode=n,time=t['simulation_elapsed_s'],rays=len(pts),blocked_by_stator=len(hh),blocked_by_backface=sum(h['backface'] for h in hh),blocked_inside_near_clip=sum(h['depth']<.02 for h in hh),sample=hh[:2]);out.append(row)
print(json.dumps(out,indent=2));(R/'rays.json').write_text(json.dumps(out,indent=2))
import cv2
from PIL import Image,ImageDraw
for n in [5,8,9,23]:
 ep=next(e for e in a if e['episode']==n);tr=[json.loads(l) for l in (Path(ep['path'])/'trace.jsonl').read_text().splitlines()];t=tr[ep['closest_capture']['trace_index']];s=t['contact_status'];rg=Rotation.from_quat(s['gripper_world_orientation_xyzw']);rc=Rotation.from_quat(s['object_orientation_xyzw']);offset=rg.inv().apply(np.array(s['object_position_m'])-s['gripper_world_position_m']);pts=rg.inv().apply(rc.apply(g[next(k for k,v in meta.items() if isinstance(v,dict) and v.get('path','').endswith('/Object_2'))][::100]))+offset
 ts=[]
 for tt in tr:
  if tt['execution_substep']==0 and tt['policy_action_index']==0:ts.append(tt['simulation_elapsed_s'])
  if tt['execution_substep']==5:ts.append(tt['simulation_elapsed_s'])
 idx=min(range(len(ts)),key=lambda i:abs(ts[i]-t['simulation_elapsed_s']));cap=cv2.VideoCapture(str(Path(ep['path'])/'videos/wrist.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,img=cap.read();cap.release();assert ok
 img=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB));draw=ImageDraw.Draw(img)
 for p in pts:
  q=np.append(p,1).dot(inv);dep=-q[2]
  if dep<.02:continue
  u=320+250*q[0]/dep;v=180-250*q[1]/dep
  h=hit(p)
  if h:draw.ellipse((u-2,v-2,u+2,v+2),fill='red')
 img.save(R/f'projected_blocked_{n:02d}.png')
