"""Convert verified partial episodes and append to the immutable 192-episode dataset."""
from pathlib import Path
import sys,json,hashlib,copy,shutil
import numpy as np
import pyarrow.parquet as pq
W=Path(__file__).resolve().parent;ROOT=Path('/home/lp531/smolvla_data/z1_bottle_correction_fragments_20261012');BASE=Path('/home/lp531/smolvla_data/z1_bottle_gentle_correction_20261010');OUT=Path('/home/lp531/smolvla_data/z1_bottle_fragments_binary_20261012');MERGED=Path('/home/lp531/smolvla_data/z1_bottle_with_fragments_20261012')
sys.path.insert(0,'/home/lp531/ros2_ws/src/z1_ros2/scripts/correction_fragments_20261012');from fragment_core import partial_labels,valid_hold
from core import atomic_json,completed_ids
from selection import check_requested_tasks

def main(accepted_snapshot=None):
 plan=json.loads((ROOT/'correction_plan.json').read_text())
 selected=None if accepted_snapshot is None else [r for r in json.loads(Path(accepted_snapshot).read_text())['records'] if r['source']=='new_fragment']
 requested=[t['id'] for t in plan['tasks']] if selected is None else [r['task_id'] for r in selected]
 count=check_requested_tasks({t['id'] for t in plan['tasks']},completed_ids(ROOT,plan['hash']),requested)
 entries=[];seeds=[]
 for root in sorted(ROOT.glob('episode_[0-9]*')):
  m=json.loads((root/'correction.json').read_text());s=json.loads((root/'episode_summary.json').read_text());assert m['accepted'] and m['recovery_complete'] and not m['table_contact'] and valid_hold(m['final_status']);assert s['fragment_success'] and s['dropped_frames']==0
  rows=[json.loads(x) for x in (root/'frames.jsonl').read_text().splitlines()];labels,close,valid=partial_labels(rows);assert len(rows)==s['frame_count'];assert len(rows)-close>=20
  assert all(all((root/r[k]).is_file() for k in ['observation.images.global','observation.images.global_right','observation.images.wrist']) for r in rows)
  seed=s['scene_randomization']['seed'];assert seed not in seeds;seeds.append(seed)
  entries.append(dict(source=str(root),task_id=m['task_id'],kind=m['task']['kind'],group=m['task_id'].split('_')[0],split=m['split'],frames=len(rows),valid_anchor_count=valid,close_frame=close,frames_sha256=hashlib.sha256((root/'frames.jsonl').read_bytes()).hexdigest(),seed=seed,bottle_position_m=s['scene_randomization']['object_positions_m']['beverage_bottle']))
 assert len(entries)==count and sum(e['split']=='validation' for e in entries)==4
 if selected is not None:
  assert [(e['task_id'],e['source'],e['frames_sha256'],e['split']) for e in entries]==[(r['task_id'],r['source_path'],r['source_frames_sha256'],r['split']) for r in selected]
 atomic_json(W/'snapshot.json',entries)
 if not (W/'conversion_complete.json').exists():
  if OUT.exists():raise RuntimeError('Incomplete binary output exists; inspect before resuming')
  from lerobot.datasets.lerobot_dataset import LeRobotDataset
  from PIL import Image
  features=copy.deepcopy(json.loads((BASE/'meta/info.json').read_text())['features']);features={k:v for k,v in features.items() if k in ['action','observation.state'] or k.startswith('observation.images.')};cameras=[k for k in features if k.startswith('observation.images.')]
  for k in cameras:features[k].pop('info',None)
  ds=LeRobotDataset.create(repo_id='lp531/z1_bottle_fragments_binary_20261012',root=OUT,fps=10,robot_type='unitree_z1',features=features,use_videos=True,image_writer_threads=4)
  expected=[]
  for e in entries:
   root=Path(e['source']);rows=[json.loads(x) for x in (root/'frames.jsonl').read_text().splitlines()];labels,close,valid=partial_labels(rows)
   for i,r in enumerate(rows):
    action=np.array(r['action'],dtype=np.float32);action[-1]=labels[i];frame={'action':action,'observation.state':np.array(r['observation.state'],dtype=np.float32),'task':r['task']};expected.append(action.tolist())
    for cam in cameras:
     with Image.open(root/r[cam]) as im:frame[cam]=np.array(im.convert('RGB'))
    ds.add_frame(frame)
   ds.save_episode();print('CONVERTED',e['task_id'],e['frames'],flush=True)
  ds.finalize();table=pq.read_table(OUT/'data').sort_by([('index','ascending')]);assert table['action'].to_pylist()==expected
  atomic_json(W/'conversion_complete.json',dict(episodes=count,frames=len(table),labels_exact=True,all_images_decoded=True))
 assert json.loads((W/'conversion_complete.json').read_text())['episodes']==count
 from lerobot.datasets.aggregate import aggregate_datasets
 if not MERGED.exists():aggregate_datasets(['lp531/z1_bottle_gentle_correction_20261010','lp531/z1_bottle_fragments_binary_20261012'],'lp531/z1_bottle_with_fragments_20261012',roots=[BASE,OUT],aggr_root=MERGED,concatenate_videos=False,concatenate_data=False)
 table=pq.read_table(MERGED/'data').sort_by([('index','ascending')]);bt=pq.read_table(BASE/'data').sort_by([('index','ascending')]);nt=pq.read_table(OUT/'data').sort_by([('index','ascending')])
 for k in ['action','observation.state']:assert table[k].to_pylist()==bt[k].to_pylist()+nt[k].to_pylist()
 if not (W/'merged_empirical_stats.json').exists():shutil.copy2(MERGED/'meta/stats.json',W/'merged_empirical_stats.json')
 shutil.copy2(BASE/'meta/stats.json',MERGED/'meta/stats.json')
 records=json.loads((W.parent/'grip_stage_base_20261011/phase_manifest.json').read_text())
 for i,e in enumerate(entries):
  r=dict(e);r['source_path']=r.pop('source');r.update(episode_index=192+i,source='new_correction_fragment');records.append(r)
 ep=table['episode_index'].to_numpy();fi=table['frame_index'].to_numpy();held=[r['episode_index'] for r in records if r['split']=='validation'];eligible=~np.isin(ep,held)
 for r in records[192:]:eligible[(ep==r['episode_index'])&(fi>=r['valid_anchor_count'])]=False
 np.savez(W/'sampling_eligibility.npz',episode=ep,frame=fi,eligible_train_anchor=eligible)
 assert len(records)==192+count and len(held)==8
 atomic_json(W/'phase_manifest.json',records);atomic_json(W/'preparation_audit.json',dict(passed=True,total_episodes=192+count,train_episodes=192+count-len(held),validation_episodes=held,new_train_fragments=count-4,new_validation_fragments=4,total_frames=len(table),eligible_train_anchors=int(eligible.sum()),old_actions_states_unchanged=True,new_binary_labels_exact=True,normalization_preserved=True,tail14_anchors_excluded_for_new_fragments=True,merged_dataset=str(MERGED),training_started=False))
 print('PREPARATION VERIFIED',len(records),len(table),flush=True)
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--accepted-snapshot',type=Path,help='Explicit frozen window manifest after collection was stopped early')
 main(parser.parse_args().accepted_snapshot)
