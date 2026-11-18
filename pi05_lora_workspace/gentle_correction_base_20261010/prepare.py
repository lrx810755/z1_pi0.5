"""Immutable source snapshot, binary correction conversion, and merged dataset."""
import copy,hashlib,json,sys,random
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
RUN=Path(__file__).resolve().parent
WORK=RUN.parent
RAW=Path('/home/lp531/smolvla_data/z1_bottle_corrections_20261009')
OLD=Path('/home/lp531/smolvla_data/z1_bottle_three_camera_pi05_binary_20261007')
CORR=Path('/home/lp531/smolvla_data/z1_corrections_binary_20261010')
MERGED=Path('/home/lp531/smolvla_data/z1_bottle_gentle_correction_20261010')
REPO='lp531/z1_bottle_gentle_correction_20261010'
sys.path.insert(0,str(WORK/'binary_gripper_20261007'))
from convert import label_episode
sys.path.insert(0,'/home/lp531/ros2_ws/src/z1_ros2/scripts/correction_recording')
from core import strict_final

def write(name,value):
 (RUN/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str)+'\n')
def digest(p):
 return hashlib.sha256(p.read_bytes()).hexdigest()
def rows_at(p):return [json.loads(x) for x in (p/'frames.jsonl').read_text().splitlines()]
def snapshot():
 if (RUN/'snapshot.json').exists():return json.loads((RUN/'snapshot.json').read_text())
 entries=[]
 for p in sorted(RAW.glob('episode_*')):
  m=json.loads((p/'correction.json').read_text());s=json.loads((p/'episode_summary.json').read_text())
  assert m['accepted'] and m['recovery_complete'] and not m['table_contact'] and strict_final(m['final_status']),p
  assert s['task_success'] and s['dropped_frames']==0,p
  rows=rows_at(p);labels,event=label_episode(rows)
  assert len(rows)==s['frame_count']
  entries.append(dict(source=str(p),frames=len(rows),kind=m['task']['kind'],task_id=m['task_id'],
                      frames_sha256=digest(p/'frames.jsonl'),correction_sha256=digest(p/'correction.json'),event=event))
 assert len(entries)>=24
 assert len({x['task_id'] for x in entries})==len(entries)
 # True holdout only among new corrections; the original 160 were seen by source adapter.
 rng=random.Random(20261010);held=[]
 for kind in ['tilt','low','low_tilt','near']:
  eligible=[i for i,e in enumerate(entries) if e['kind']==kind];assert len(eligible)>=2
  held.append(rng.choice(eligible))
 for i,e in enumerate(entries):e.update(correction_index=i,episode_index=160+i,split='validation' if i in held else 'train')
 write('snapshot.json',entries);return entries

def prepare():
 entries=snapshot()
 from lerobot.datasets.lerobot_dataset import LeRobotDataset
 from lerobot.datasets.aggregate import aggregate_datasets
 from PIL import Image
 features=copy.deepcopy(json.loads((OLD/'meta/info.json').read_text())['features'])
 features={k:v for k,v in features.items() if k in ['action','observation.state'] or k.startswith('observation.images.')}
 cameras=[k for k in features if k.startswith('observation.images.')]
 for k in cameras:features[k].pop('info',None)
 if not (RUN/'correction_conversion_complete.json').exists():
  if CORR.exists():raise RuntimeError('Incomplete correction conversion exists; inspect before retry')
  ds=LeRobotDataset.create(repo_id='lp531/z1_corrections_binary_20261010',root=CORR,fps=10,robot_type='unitree_z1',features=features,use_videos=True,image_writer_threads=4)
  expected={'action':[],'observation.state':[]}
  for e in entries:
   p=Path(e['source']);assert digest(p/'frames.jsonl')==e['frames_sha256']
   rows=rows_at(p);labels,event=label_episode(rows)
   for i,row in enumerate(rows):
    action=np.asarray(row['action'],dtype=np.float32);action[-1]=labels[i]
    state=np.asarray(row['observation.state'],dtype=np.float32)
    assert np.isfinite(action).all() and np.isfinite(state).all()
    frame={'action':action,'observation.state':state,'task':row['task']}
    for cam in cameras:
     with Image.open(p/row[cam]) as im:frame[cam]=np.array(im.convert('RGB'))
    ds.add_frame(frame);expected['action'].append(action);expected['observation.state'].append(state)
   ds.save_episode();print('CONVERTED',e['correction_index'],p.name,len(rows),flush=True)
  ds.finalize()
  t=pq.read_table(CORR/'data').sort_by([('index','ascending')])
  for k in expected:assert np.array_equal(np.asarray(t[k].to_pylist(),dtype=np.float32),np.asarray(expected[k])),k
  write('correction_conversion_complete.json',{'episodes':len(entries),'frames':len(t),'all_labels_exact':True,'all_source_images_decoded':True})
 if not MERGED.exists():
  aggregate_datasets(['lp531/z1_bottle_three_camera_pi05_binary_20261007','lp531/z1_corrections_binary_20261010'],REPO,roots=[OLD,CORR],aggr_root=MERGED,concatenate_videos=False,concatenate_data=False)
 t=pq.read_table(MERGED/'data').sort_by([('index','ascending')])
 original=pq.read_table(OLD/'data').sort_by([('index','ascending')])
 correction=pq.read_table(CORR/'data').sort_by([('index','ascending')])
 for key in ['action','observation.state']:
  assert t[key].to_pylist()==original[key].to_pylist()+correction[key].to_pylist(),key
 # Preserve the 80000 checkpoint's normalization coordinate system.
 import shutil
 shutil.copy2(MERGED/'meta/stats.json',RUN/'merged_empirical_stats.json')
 shutil.copy2(OLD/'meta/stats.json',MERGED/'meta/stats.json')
 old_events=json.loads((WORK/'binary_gripper_20261007/episode_labels.json').read_text())
 records=[]
 for e in old_events:
  records.append(dict(episode_index=e['episode_index'],source='original',split='train',frames=e['frames'],
                      key_frames=list(range(max(0,e['close_frame']-20),e['close_frame'])),close_frame=e['close_frame']))
 for e in entries:
  p=Path(e['source']);rows=rows_at(p);close=e['event']['close_frame'];m=json.loads((p/'correction.json').read_text())
  key=set(range(max(0,close-20),close))
  for step in m['stages']:
   key.update(i for i,r in enumerate(rows[:close]) if step['start_sim_time']<=r['simulation_time']<=step['end_sim_time'])
  records.append(dict(episode_index=e['episode_index'],source='correction',kind=e['kind'],split=e['split'],frames=e['frames'],key_frames=sorted(key),close_frame=close))
 assert len(t)==sum(e['frames'] for e in records)
 episode=t['episode_index'].to_numpy();frame=t['frame_index'].to_numpy()
 assert set(episode)==set(range(len(records)))
 key=np.zeros(len(t),dtype=bool)
 for e in records:
  mask=episode==e['episode_index'];assert np.array_equal(frame[mask],np.arange(e['frames']))
  key[mask]=np.isin(frame[mask],e['key_frames'])
 actions=np.asarray(t['action'].to_pylist());assert set(actions[:,-1])=={0.,1.};assert np.all(actions[key,-1]==0)
 np.savez(RUN/'sampling_arrays.npz',episode=episode,key=key,correction=episode>=160)
 write('phase_manifest.json',records)
 write('preparation_audit.json',dict(passed=True,original_episodes=160,correction_episodes=len(entries),train_corrections=len(entries)-4,validation_corrections=4,total_frames=len(t),normalization='original checkpoint coordinate system; original dataset stats preserved',phase_definition='open-intent anchors in last 2s before close union logged recovery intervals',old_original_training_data_not_holdout=True,merged=str(MERGED)))
 print('PREPARATION PASSED',len(records),len(t),flush=True)
if __name__=='__main__':prepare()
