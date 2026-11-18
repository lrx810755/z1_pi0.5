"""Read-only comparison of a failed rollout and three nearby successful demonstrations."""
from pathlib import Path
import json, csv, sys, xml.etree.ElementTree as ET
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
RUN=OUT.parent
EP=next((RUN/'episodes').iterdir())
DATA=Path('/home/lp531/smolvla_data/z1_tabletop_bottle_recovery_boundary_only_prepared_v1')
trace=[json.loads(s) for s in (EP/'trace.jsonl').read_text().splitlines()]
q=np.array([r['joint_state'] for r in trace]); actions=np.array([r['command'] for r in trace])
js=ET.parse('/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/z1.urdf').getroot()
chain=[js.find(f"joint[@name='joint{i}']") for i in range(1,7)]
def fk(q):
 t=np.eye(4)
 for a,j in zip(q,chain):
  o=j.find('origin'); m=np.eye(4);m[:3,3]=np.fromstring(o.get('xyz'),sep=' ')
  m[:3,:3]=Rotation.from_euler('xyz',np.fromstring(o.get('rpy'),sep=' ')).as_matrix()@Rotation.from_rotvec(a*np.fromstring(j.find('axis').get('xyz'),sep=' ')).as_matrix(); t=t@m
 m=np.eye(4);m[0,3]=.051
 return t@m
refs=[]; records=[]
for n in [89,235,53]:
 d=DATA/f'episode_{n:06d}'; r=[json.loads(s) for s in (d/'frames.jsonl').read_text().splitlines()]; dq=np.array([v['observation.state'] for v in r]); da=np.array([v['action'] for v in r]); c=int(np.flatnonzero(da[:,-1]>=-.75)[0]);
 # Match ascending shoulder progress before the reference first closure; no equal-time assumption.
 matches=np.abs(q[:,1,None]-dq[None,:c+1,1]).argmin(axis=1)
 for i,k in enumerate(matches):
  delta=q[i,:6]-dq[k,:6];angle=Rotation.from_matrix(fk(q[i])[:3,:3]@fk(dq[k])[:3,:3].T).magnitude()*180/np.pi
  records.append(dict(reference=n,trace_index=i,wall_s=trace[i]['elapsed_s'],demo_frame=int(k),shoulder_mismatch_rad=float(q[i,1]-dq[k,1]),orientation_difference_deg=angle,**{f'joint{a+1}_difference_deg':v for a,v in enumerate(np.rad2deg(delta))}))
 refs.append(dict(id=n,dir=d,rows=r,q=dq,action=da,close=c,matches=matches))
with (OUT/'aligned_joints.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=records[0].keys());w.writeheader();w.writerows(records)
fig,axes=plt.subplots(2,3,figsize=(13,7))
for j,ax in enumerate(axes.flat):
 ax.plot(q[:,1],np.rad2deg(q[:,j]),'k',lw=2,label='Failed rollout')
 for ref in refs:
  c=ref['close']; ax.plot(ref['q'][:c+1,1],np.rad2deg(ref['q'][:c+1,j]),alpha=.7,label=f"Demo {ref['id']}")
 ax.set(xlabel='Shoulder joint2 (rad), approach progress',ylabel=f'joint{j+1} (deg)');ax.grid(alpha=.3)
axes[0,0].legend();fig.tight_layout();fig.savefig(OUT/'joint_comparison.png',dpi=150);plt.close(fig)
def read_video(name,i):
 cap=cv2.VideoCapture(str(EP/'videos'/f'{name}.mp4'))
 assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==len(trace)+(len(trace)+4)//5
 # Writer emits a planning observation then each command observation per five-action chunk.
 cap.set(cv2.CAP_PROP_POS_FRAMES,i+i//5+1);ok,img=cap.read();cap.release();assert ok;return img
def tile(img,label):
 assert img is not None
 img=cv2.resize(img,(400,225));img=cv2.copyMakeBorder(img,28,0,0,0,cv2.BORDER_CONSTANT);cv2.putText(img,label,(5,19),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1);return img
for cam in ['global','wrist']:
 rows=[]
 for i in [15,30,45,60,75,92]:
  tiles=[tile(read_video('global_third_person' if cam=='global' else cam,i),f'FAIL i={i}, wall={trace[i]["elapsed_s"]:.2f}s')]
  for ref in refs:
   k=int(ref['matches'][i]);row=ref['rows'][k];tiles.append(tile(cv2.imread(str(ref['dir']/row[f'observation.images.{cam}'])),f'Demo {ref["id"]}, frame={k}, joint2 matched'))
  rows.append(np.hstack(tiles))
 cv2.imwrite(str(OUT/f'aligned_{cam}.jpg'),np.vstack(rows))
summary=[]
for ref in refs:
 rr=[r for r in records if r['reference']==ref['id']]; sustained=next((i for i in range(len(rr)-4) if all(abs(v['joint4_difference_deg'])>10 for v in rr[i:i+5])),None)
 close=ref['close'];finaldelta=np.rad2deg(q[-1,:6]-ref['q'][close,:6])
 v=dict(reference=ref['id'],close_frame=close,first_sustained_joint4_10deg_trace_index=sustained,first_sustained_joint4_10deg_wall_s=trace[sustained]['elapsed_s'] if sustained is not None else None,final_vs_closure_joint_difference_deg=finaldelta.tolist(),final_vs_closure_orientation_difference_deg=Rotation.from_matrix(fk(q[-1])[:3,:3]@fk(ref['q'][close])[:3,:3].T).magnitude()*180/np.pi)
 summary.append(v)
(OUT/'comparison_metrics.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
if '--probe' in sys.argv:
 import torch
 sys.path.insert(0,'/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge')
 from z1_isaac_bridge.pi05_policy_bridge import PI05ProcessPolicy
 policy=PI05ProcessPolicy(Path('/home/lp531/smolvla_training/pi05_bottle_lora_20261001/best'))
 results=[]
 try:
  samples=[]
  for ref in refs:
   for k in [int(ref['matches'][45]),ref['close']-5,ref['close']]:
    row=ref['rows'][k];imgs={c:cv2.imread(str(ref['dir']/row[f'observation.images.{c}'])) for c in ['global','wrist']}
    samples.append((f'demo_{ref["id"]}',k,ref['q'][k],imgs,ref['action'][k:k+5],ref['close']-k))
  for i in [45,75,90]:
   imgs={c:read_video('global_third_person' if c=='global' else c,i) for c in ['global','wrist']};samples.append(('failed_video_replay',i,q[i],imgs,None,None))
  for name,k,state,imgs,truth,close_offset in samples:
   batch={'observation.state':torch.tensor(state,dtype=torch.float32)[None], 'task':['Physically grasp the bottle, lift it clear, and place it in the box.']}
   for c,img in imgs.items():batch[f'observation.images.{c}']=torch.from_numpy(cv2.cvtColor(img,cv2.COLOR_BGR2RGB).copy()).permute(2,0,1)[None].float()/255
   for rep in range(5):
    policy.reset();pred=policy.predict_action_chunk(batch)[0].numpy();ix=np.flatnonzero(pred[:,-1]>=-.75)
    result=dict(source=name,frame=k,repeat=rep,true_close_offset=close_offset,predicted_close_offset=int(ix[0]) if len(ix) else None,first5_gripper=pred[:5,-1].tolist(),first_action_delta_deg=np.rad2deg(pred[0,:6]-state[:6]).tolist())
    if truth is not None:result['first5_joint_mae_deg']=np.rad2deg(np.abs(pred[:5,:6]-truth[:,:6]).mean(axis=0)).tolist()
    results.append(result)
   print('Probed',name,k,flush=True)
 finally:policy.close()
 (OUT/'repeated_inference.json').write_text(json.dumps(results,indent=2))
