exec(open('/home/lp531/z1_queue_logs/cup_wrist_penetration_20261026/rays.py').read().split('\na=json.loads')[0])
import cv2
from PIL import Image,ImageDraw

def closest_triangles(ray,triangles):
 d=ray/np.linalg.norm(ray);v0=triangles[:,0];e1=triangles[:,1]-v0;e2=triangles[:,2]-v0;p=np.cross(d,e2);det=np.einsum('ij,ij->i',e1,p);iv=np.zeros_like(det);mask=abs(det)>1e-12;iv[mask]=1/det[mask];tv=origin-v0;u=np.einsum('ij,ij->i',tv,p)*iv;cross=np.cross(tv,e1);v=cross@d*iv;t=np.einsum('ij,ij->i',e2,cross)*iv;mask&=(u>=0)&(v>=0)&(u+v<=1)&(t>1e-6)
 return t[mask].min() if mask.any() else float('inf')
a=json.loads(Path('/home/lp531/smolvla_eval/act_cup_only_frustum_steps15_paired30_20261025/failure_analysis/never_close19/metrics.json').read_text());results=[]
for n in [5,8,9,23]:
 ep=next(e for e in a if e['episode']==n);tr=[json.loads(l) for l in (Path(ep['path'])/'trace.jsonl').read_text().splitlines()];t=tr[ep['closest_capture']['trace_index']];s=t['contact_status'];rg=Rotation.from_quat(s['gripper_world_orientation_xyzw']);rc=Rotation.from_quat(s['object_orientation_xyzw']);offset=rg.inv().apply(np.array(s['object_position_m'])-s['gripper_world_position_m']);cup=rg.inv().apply(rc.apply(g[next(k for k,v in meta.items() if isinstance(v,dict) and v.get('path','').endswith('/Object_2'))]))+offset;ctri=cup[g['cup_faces']];pts=cup[::100]
 ts=[]
 for tt in tr:
  if tt['execution_substep']==0 and tt['policy_action_index']==0:ts.append(tt['simulation_elapsed_s'])
  if tt['execution_substep']==5:ts.append(tt['simulation_elapsed_s'])
 idx=min(range(len(ts)),key=lambda i:abs(ts[i]-t['simulation_elapsed_s']));cap=cv2.VideoCapture(str(Path(ep['path'])/'videos/wrist.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,img=cap.read();cap.release();assert ok;img=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB));draw=ImageDraw.Draw(img);blocked=0;before=[]
 for p in pts:
  h=hit(p)
  if not h:continue
  cd=closest_triangles(p-origin,ctri)
  if h['distance']>cd-.0005:continue
  q=np.append(p,1).dot(inv);dep=-q[2]
  if dep<.02:continue
  u=320+250*q[0]/dep;v=180-250*q[1]/dep
  if 0<=u<640 and 0<=v<360:
   blocked+=1;before.append(dict(pixel=[float(u),float(v)],robot_distance=h['distance'],cup_distance=float(cd),camera_depth=h['depth']));draw.ellipse((u-2,v-2,u+2,v+2),fill='red')
 img.save(R/f'occluded_{n:02d}.png');results.append(dict(episode=n,blocked=blocked,points=before))
(R/'occlusion.json').write_text(json.dumps(results,indent=2));print([(r['episode'],r['blocked']) for r in results])
