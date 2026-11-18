import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,'/home/lp531/ros2_ws/src/z1_ros2/scripts/cup_failure_recovery_20261019')
from geometry import Geometry
P=Path(__file__).resolve().parent;R=Path('/home/lp531/pi05_lora_workspace/act_cup_recovery_cached_20261020')
records=json.loads((R/'split_manifest.json').read_text())['source_records'];g=Geometry();samples=[];indices=[]
for rec in records:
 if rec['source']=='original':continue
 path=Path(rec['source_path']);rows=[json.loads(x) for x in (path/'frames.jsonl').read_text().splitlines()];meta=json.loads((path/'correction.json').read_text());c=rec['close_frame'];cup=np.array(meta['actual_cup_position_m']);cup[2]+=.0305
 start=rows[0]['simulation_time'];recovery_start=min((x['start_sim_time'] for x in meta['stages']),default=start)-start
 for i in range(0,c,3):
  if i+15>=len(rows):continue
  p,rot=g.fk(rows[i]['observation.state'][:6]);rel=rot.T@(cup-p)
  if not(.12583045<rel[0]<.30 and abs(rel[1])<.1 and abs(rel[2])<.1):continue
  samples.append(dict(episode_index=rec['episode_index'],episode=path.name,source=rec['source'],split=rec['split'],frame=i,time_s=i/10,close_frame=c,source_path=str(path),actual_position=p.tolist(),actual_rotation=rot.tolist(),cup_center=cup.tolist(),local_xyz=rel.tolist(),initial_wait=i/10<recovery_start,recovery_start_s=recovery_start))
  indices.append(rec['dataset_from_index']+i)
assert len(samples)>30
(P/'probes.json').write_text(json.dumps(dict(samples=samples,absolute_indices=indices),ensure_ascii=False,indent=2))
print('samples',len(samples));print({str((src,split)):sum(x['source']==src and x['split']==split for x in samples) for src in ['old_correction','new_correction'] for split in ['train','validation']})
