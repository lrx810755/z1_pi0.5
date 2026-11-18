from pathlib import Path
import json,collections,itertools,re
import numpy as np,cv2,pyarrow.parquet as pq
R=Path('/home/lp531/smolvla_data/z1_tabletop_left_wrist_prepared_all_v1'); L=Path('/home/lp531/smolvla_data/z1_tabletop_lerobot_left_wrist_2cam_all_v1');O=Path(__file__).parent
by=collections.defaultdict(list)
for p in sorted(R.glob('episode_*')):
 s=json.loads((p/'episode_summary.json').read_text());t=s['target_object'];xy=s['scene_randomization']['object_positions_m'][t][:2];by[t].append((p,s,np.array(xy)))
mapping={int(a):int(b) for a,b in re.findall(r'Converted raw episode_(\d+) -> LeRobot episode (\d+)',Path('/home/lp531/z1_queue_logs/smolvla_left_wrist_v1/train_pipeline.log').read_text())}
for n,a in re.findall(r'appended (\d+)/83: episode_(\d+)',Path('/home/lp531/z1_queue_logs/smolvla_left_wrist_v1/train_all_pipeline.log').read_text()):mapping[int(a)]=517+int(n)
results=[]
for t,eps in by.items():
 xy=np.array([e[2] for e in eps]);chosen=[int(np.argmin(np.linalg.norm(xy-xy.mean(0),axis=1)))]
 while len(chosen)<4:
  dist=np.min(np.linalg.norm(xy[:,None,:]-xy[chosen][None,:,:],axis=2),axis=1);chosen.append(int(dist.argmax()))
 tiles=[]
 for n in chosen:
  p,s,pos=eps[n];frames=[json.loads(z) for z in (p/'frames.jsonl').read_text().splitlines()];idx=int(p.name.split('_')[-1]);idx=mapping[idx];file=L/'data/chunk-000'/f'episode_{idx:06d}.parquet';df=pq.read_table(file).to_pandas();q=np.stack(df['observation.state']);a=np.stack(df['action']);assert len(q)==len(frames)
  rawq=np.array([f['observation.state'] for f in frames]);rawa=np.array([f['action'] for f in frames]);
  close=np.flatnonzero(a[:,-1]>-.75);ci=int(close[0]) if len(close) else len(a)//2
  inds=list(dict.fromkeys([0,max(0,ci-10),ci,min(len(q)-1,ci+20)]))
  gr=np.flatnonzero(np.diff((a[:,-1]>-.75).astype(int))==1)
  stationary=np.max(np.abs(np.diff(q,axis=0)),axis=1)<.002
  sync={k:max(abs(f.get(k, f['simulation_time'])-f['simulation_time']) for f in frames) for k in ['wrist_image_stamp','global_image_stamp']}
  stats=[];row=[]
  for fi in inds:
   pair=[]
   for cam in ['global','wrist']:
    im=cv2.imread(str(p/frames[fi][f'observation.images.{cam}']));assert im is not None
    stats.append(float(im.std()));im=cv2.resize(im,(320,240));cv2.putText(im,f'{cam} f{fi}',(6,20),0,.5,(0,0,255),1);pair.append(im)
   row.append(np.vstack(pair))
  tile=np.hstack(row);cv2.putText(tile,f'{p.name} {t} xy={pos.round(3)} close={ci}',(5,465),0,.5,(0,0,255),1);tiles.append(tile)
  result=dict(episode=p.name,training_episode=idx,target=t,xy=pos.tolist(),frames=len(q),first_close_frame=ci,close_events=len(gr),stationary_fraction=float(stationary.mean()),phase_counts=dict(collections.Counter(f.get('phase') for f in frames)),image_std_min=min(stats),sync_max_s=sync,action_age_max_s=max(f.get('action_age_s',0) for f in frames),source_state_max_error=float(abs(q-rawq).max()),source_action_max_error=float(abs(a-rawa).max()),initial_q=q[0].tolist(),preclose_q=q[max(0,ci-1)].tolist(),first30_action_end=a[min(29,len(a)-1)].tolist())
  results.append(result);print(json.dumps(result),flush=True)
 cv2.imwrite(str(O/f'{t}_contact_sheet.jpg'),np.vstack(tiles))
json.dump({'metadata_counts':{k:len(v) for k,v in by.items()},'samples':results},open(O/'results.json','w'),indent=2)
