import json,sys,csv
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,'/home/lp531/ros2_ws/src/z1_ros2/scripts/cup_failure_recovery_20261019')
from geometry import Geometry
OUT=Path(__file__).resolve().parent
RUN=Path('/home/lp531/pi05_lora_workspace/act_cup_recovery_cached_20261020')
records=json.loads((RUN/'split_manifest.json').read_text())['source_records']
a=np.load(RUN/'sampling_arrays.npz');g=Geometry();result=[];segments=[];masks={}
def runs(mask):
 d=np.diff(np.r_[False,mask,False].astype(int));return list(zip(np.flatnonzero(d==1),np.flatnonzero(d==-1)))
for rec in records:
 path=Path(rec['source_path']);rows=[json.loads(l) for l in (path/'frames.jsonl').read_text().splitlines()]
 n=len(rows);c=rec['close_frame'];s=np.array([r['observation.state'] for r in rows]);cmd=np.array([r['action'] for r in rows]);t=np.array([r['timestamp'] for r in rows]);poses=[g.fk(q[:6]) for q in s];p=np.array([x[0] for x in poses]);rot=np.array([x[1] for x in poses]);summary=json.loads((path/'episode_summary.json').read_text());cup=np.array(summary['scene_randomization']['object_positions_m']['cup']);cup[2]+=.0305
 meta={}
 if (path/'correction.json').exists():
  meta=json.loads((path/'correction.json').read_text());cup=np.array(meta.get('actual_cup_position_m',summary['scene_randomization']['object_positions_m']['cup']));cup[2]+=.0305
 rel=np.einsum('nij,nj->ni',rot.transpose(0,2,1),cup-p)
 # Stationary requires a full half-second window with little command AND actual movement.
 stop=np.zeros(n,bool)
 for i in range(max(0,c-5)):
  j=i+6
  spatial=np.linalg.norm(np.ptp(p[i:j],axis=0))
  angular=Rotation.from_matrix(rot[i:j]@rot[i].T).magnitude().max()
  if spatial<=.001 and angular<=np.deg2rad(.5) and np.max(np.ptp(cmd[i:j,:6],axis=0))<=.003:
   stop[i:j]=True
 pre=np.arange(n)<c
 shallow=pre&(rel[:,0]>.13883045)&(rel[:,0]<.30)&(abs(rel[:,1])<.10)&(abs(rel[:,2])<.10)
 near=pre&(np.linalg.norm(rel,axis=1)<.30)
 terminal=pre&(np.arange(n)>=max(0,c-20))
 b=rec['dataset_from_index'];e=rec['dataset_to_index'];key=a['key'][b:e];w=a['weights'][b:e]
 mm={'preclose':pre,'stop_preclose':stop&pre,'shallow':shallow,'shallow_stop':shallow&stop,'near_stop':near&stop&pre,'last2s_stop':terminal&stop}
 masks[rec['episode_index']]=mm
 data={k:round(float(v.sum()*.1),4) for k,v in mm.items()};data.update({k+'_mass':float(w[v].sum()) for k,v in mm.items()})
 data.update(source=rec['source'],episode=path.name,episode_index=rec['episode_index'],split=rec['split'],close_frame=c,close_time=round(c*.1,3),firstclose_local_xyz=rel[c].tolist(),shallow_stop_key_s=round(float((shallow&stop&key).sum()*.1),3),max_shallow_stop_s=0,task_id=rec.get('task_id'))
 for start,end in runs(stop&pre):
  seg=dict(source=rec['source'],episode=path.name,episode_index=rec['episode_index'],split=rec['split'],start_s=float(t[start]),end_s=float(t[end-1]+.1),duration_s=round((end-start)*.1,3),depth_x_median=float(np.median(rel[start:end,0])),local_xyz_median=np.median(rel[start:end],axis=0).tolist(),shallow_s=float(shallow[start:end].sum()*.1),key_frames=int(key[start:end].sum()),phase=rows[start]['phase'],source_path=str(path))
  segments.append(seg)
 for start,end in runs(stop&shallow):data['max_shallow_stop_s']=max(data['max_shallow_stop_s'],round((end-start)*.1,3))
 data['sensitivity']={}
 for gap in [.005,.010,.018,.030]:
  mask=pre&stop&(rel[:,0]>.12083045+gap)&(rel[:,0]<.30)&(abs(rel[:,1])<.1)&(abs(rel[:,2])<.1)
  static16=np.zeros(n,bool)
  for z in np.flatnonzero(mask):
   if z+15<c and np.max(np.ptp(cmd[z:z+16,:6],axis=0))<=.003:static16[z]=True
  data['sensitivity'][str(gap)]=dict(seconds=round(float(mask.sum()*.1),3),mass=float(w[mask].sum()),static16_mass=float(w[static16].sum()),max_s=max([round((v-u)*.1,3) for u,v in runs(mask)]+[0]))
 data['preclose_stop_mass_key']=float(w[stop&pre&key].sum())
 data['initial_stop_s']=round(sum(z['duration_s'] for z in segments if z['episode_index']==rec['episode_index'] and z['start_s']<.1),3)
 data['future_static_open_mass']=0
 result.append(data)
allm={k:np.zeros(len(a['weights']),bool) for k in masks[0]}
for rec in records:
 for k,v in masks[rec['episode_index']].items():allm[k][rec['dataset_from_index']:rec['dataset_to_index']]=v
summary={}
for src in ['original','old_correction','new_correction']:
 rr=[r for r in result if r['source']==src];ss=[s for s in segments if s['source']==src]
 summary[src]=dict(episodes=len(rr),**{k+'_seconds':round(sum(r[k] for r in rr),3) for k in masks[0]},episodes_shallow_stop_ge_05=sum(r['max_shallow_stop_s']>=.5 for r in rr),episodes_shallow_stop_ge_1=sum(r['max_shallow_stop_s']>=1 for r in rr),episodes_shallow_stop_ge_2=sum(r['max_shallow_stop_s']>=2 for r in rr),max_shallow_stop_s=max(r['max_shallow_stop_s'] for r in rr),median_preclose_stop_s=float(np.median([r['stop_preclose'] for r in rr])),sampling_mass={k:sum(r[k+'_mass'] for r in rr) for k in masks[0]},key_shallow_stop_seconds=sum(r['shallow_stop_key_s'] for r in rr))
# Fraction of future actions stationary/open over the whole 16-step target, not merely current input.
low=np.load('/home/lp531/smolvla_data/act_cup_recovery_rgb224_cache_20261020/lowdim.npz')['actions']
staticchunk=np.zeros(len(low),bool)
for i in np.flatnonzero(a['eligible_train']):
 chunk=low[i:i+16]
 staticchunk[i]=(chunk[:,-1]==0).all() and np.max(np.ptp(chunk[:,:6],axis=0))<=.003
summary['overall_sampling']={k:float(a['weights'][v].sum()) for k,v in allm.items()}
summary['overall_sampling']['shallow_stop_full16_static_open']=float(a['weights'][allm['shallow_stop']&staticchunk].sum())
summary['overall_sampling']['full16_static_open']=float(a['weights'][staticchunk].sum())
np.savez_compressed(OUT/'masks.npz',**allm,staticchunk=staticchunk)
(OUT/'results.json').write_text(json.dumps(dict(summary=summary,episodes=result,segments=segments),ensure_ascii=False,indent=2,default=lambda x:x.item()))
with (OUT/'episodes.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(result[0]));writer.writeheader();writer.writerows(result)
print(json.dumps(summary,ensure_ascii=False,indent=2,default=lambda x:x.item()))
print('TOP',json.dumps(sorted(segments,key=lambda s:s['shallow_s'],reverse=True)[:12],ensure_ascii=False,indent=2,default=lambda x:x.item()))
