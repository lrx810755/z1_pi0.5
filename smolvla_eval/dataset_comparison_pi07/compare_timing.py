"""Compare pre-close pose stability, without changing any input datasets."""
from pathlib import Path
import json,xml.etree.ElementTree as ET
from collections import defaultdict
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R
P=Path(__file__).resolve().parent
DATA=Path('/home/lp531/smolvla_data/z1_tabletop_bottle_recovery_boundary_only_prepared_v1')
urdf=ET.parse('/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/z1.urdf').getroot()
chain=[]
for n in range(1,7):
 j=urdf.find(f"joint[@name='joint{n}']");o=j.find('origin');m=np.eye(4);m[:3,3]=np.fromstring(o.get('xyz'),sep=' ');m[:3,:3]=R.from_euler('xyz',np.fromstring(o.get('rpy'),sep=' ')).as_matrix();chain.append((m,np.fromstring(j.find('axis').get('xyz'),sep=' ')))
def pose(q):
 t=np.eye(4)
 for angle,(origin,axis) in zip(q,chain):
  m=np.eye(4);m[:3,:3]=R.from_rotvec(angle*axis).as_matrix();t=t@origin@m
 t[:3,3]+=t[:3,:3]@np.array([.051,0,0]);return t[:3,3],t[:3,:3]
def stationary(c,getpose,pos_tol,rot_tol):
 p0,r0=getpose(c);k=c
 while k>0:
  p,r=getpose(k-1)
  if np.linalg.norm(p-p0)>pos_tol or R.from_matrix(r@r0.T).magnitude()>rot_tol:break
  k-=1
 return c-k
rows=[]
for d in sorted(DATA.glob('episode_*')):
 rr=[json.loads(s) for s in (d/'frames.jsonl').read_text().splitlines()];q=np.array([v['observation.state'] for v in rr]);a=np.array([v['action'] for v in rr]);c=int(np.flatnonzero(a[:,-1]>=-.75)[0]);cache={}
 def getpose(k):
  if k not in cache:cache[k]=pose(q[k])
  return cache[k]
 v={'dataset':'Z1','episode':d.name,'first_close':c,'fps_metadata':10}
 for mm,deg in [(1,.5),(2,1),(5,2)]:v[f'stationary_steps_{mm}mm_{deg}deg']=stationary(c,getpose,mm/1000,np.deg2rad(deg))
 rows.append(v)
meta={r['episode_index']:r['length'] for r in pq.read_table(P/'libero_episodes.parquet').to_pylist()};eps=defaultdict(list)
for f in sorted(P.glob('libero_sample_*.parquet')):
 for row in pq.read_table(f).to_pylist():eps[row['episode_index']].append(row)
for ep,rr in sorted(eps.items()):
 rr.sort(key=lambda x:x['frame_index']);assert [r['frame_index'] for r in rr]==list(range(meta[ep])),ep
 q=np.array([v['observation.state'] for v in rr]);a=np.array([v['action'] for v in rr]);closes=np.flatnonzero(a[:,-1]>0)
 if not len(closes):continue
 c=int(closes[0]);assert c>0 and a[c-1,-1]<0
 def getpose(k):return q[k,:3],R.from_rotvec(q[k,3:6]).as_matrix()
 v={'dataset':'LIBERO_sample','episode':ep,'task_index':rr[0]['task_index'],'first_close':c,'fps_metadata':10}
 for mm,deg in [(1,.5),(2,1),(5,2)]:v[f'stationary_steps_{mm}mm_{deg}deg']=stationary(c,getpose,mm/1000,np.deg2rad(deg))
 rows.append(v)
summary={}
for name in ['Z1','LIBERO_sample']:
 r=[v for v in rows if v['dataset']==name];summary[name]={'episodes':len(r)}
 for key in ['stationary_steps_1mm_0.5deg','stationary_steps_2mm_1deg','stationary_steps_5mm_2deg']:
  values=[v[key] for v in r];summary[name][key]={'min':min(values),'median':float(np.median(values)),'max':max(values)}
result={'definition':'Consecutive end-effector poses before first close command, relative to its pose. Z1 FK gripperStator; LIBERO stored EE pose. Counts in recorded steps, not asserted equal physical durations. Different tasks and controllers; not a quality ranking.','summary':summary,'episodes':rows}
(P/'pose_timing_comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps(summary,indent=2));print('LIBERO',json.dumps([r for r in rows if r['dataset']=='LIBERO_sample']))
