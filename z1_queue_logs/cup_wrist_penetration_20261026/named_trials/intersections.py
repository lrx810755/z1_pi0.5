import json,numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
R=Path(__file__).parent;G=R.parent;data=np.load(G/'geometry.npz');meta=json.loads((G/'geometry.json').read_text());cupkey=next(k for k,v in meta.items() if isinstance(v,dict) and v.get('path','').endswith('/Object_2'));cup=data[cupkey];stator=data['stator_0'];mover=data['mover_2'];frames=json.loads((R/'frames.json').read_text());out=[]

def inside_points(points,vertices,faces):
 lo=vertices.min(0);hi=vertices.max(0);cand=np.where(((points>lo)&(points<hi)).all(1))[0];tri=vertices[faces];a=tri[:,0];e1=tri[:,1]-a;e2=tri[:,2]-a;direction=np.array([.934128,.281793,.218932]);direction/=np.linalg.norm(direction);h=np.cross(direction,e2);det=np.einsum('ij,ij->i',e1,h);ok=abs(det)>1e-12;iv=np.zeros(len(det));iv[ok]=1/det[ok];hits=[]
 for index in cand:
  p=points[index];s=p-a;u=np.einsum('ij,ij->i',s,h)*iv;q=np.cross(s,e1);v=q@direction*iv;tt=np.einsum('ij,ij->i',e2,q)*iv;mask=ok&(u>1e-8)&(v>1e-8)&(u+v<1-1e-8)&(tt>1e-7);ts=np.sort(tt[mask]);count=int(np.r_[True,np.diff(ts)>1e-6].sum()) if len(ts) else 0
  if count%2:hits.append(index)
 return hits
for e in frames:
 if e['trial'] not in [2,5,8,12]:continue
 t=e['trace'];s=t['contact_status'];rg=Rotation.from_quat(s['gripper_world_orientation_xyzw']);rc=Rotation.from_quat(s['object_orientation_xyzw']);pts=rg.inv().apply(rc.apply(cup[::50])+np.array(s['object_position_m'])-s['gripper_world_position_m']);mv=Rotation.from_rotvec([0,t['joint_state'][-1],0]).apply(mover)+[.049,0,0]
 ins=inside_points(pts,stator,data['stator_faces']);inm=inside_points(pts,mv,data['mover_faces']);row=dict(trial=e['trial'],time=e['time'],sample_vertices=len(pts),inside_stator=len(ins),inside_mover=len(inm),inside_cup_indices=[int(i*50) for i in sorted(set(ins+inm))],jaw_position=t['joint_state'][-1],reported_penetration=s.get('target_finger_max_penetration_m'));out.append(row)
print(json.dumps(out,indent=2));(R/'visual_intersections.json').write_text(json.dumps(out,indent=2))
