import json,collections
from pathlib import Path
import numpy as np
T=Path('/home/lp531/pi05_lora_workspace/act_cup_cached_20261019'); R=Path(__file__).resolve().parent
records=json.loads((T/'split_manifest.json').read_text())['source_records'];a=np.load(T/'sampling_arrays.npz');w=a['weights'];mass=collections.defaultdict(float);dur=[];close_times=[];positions=[];phase=collections.defaultdict(float)
for rec in records:
 rows=[json.loads(x) for x in (Path(rec['source_path'])/'frames.jsonl').read_text().splitlines()]
 if rec['source']=='original':
  dur.append(rows[-1]['timestamp']);close_times.append(rows[rec['close_frame']]['timestamp']);s=json.loads((Path(rec['source_path'])/'episode_start.json').read_text());positions.append((rec,s['scene_randomization']['object_positions_m']['cup'][:2]))
 start=rec['dataset_from_index'];c=rec['close_frame'];o=rec['open_frame'];end=rec['frames']
 for i,row in enumerate(rows):
  ww=float(w[start+i]);stage='before_close' if i<c else ('closed_before_release' if o is None or i<o else 'release_and_after');mass[rec['source']+'/'+stage]+=ww;phase[rec['source']+'/'+row['phase']]+=ww
  if i<c and i+15>=c:mass[rec['source']+'/open_anchor_chunk_contains_close']+=ww
  if i==c:mass[rec['source']+'/first_close_anchor']+=ww
m=json.loads((R/'metrics.json').read_text());near=[]
for e in m:
 choices=[(float(np.linalg.norm(np.array(p)-np.array(e['initial_position'][:2]))),rec) for rec,p in positions if rec['split']=='train'];d,rec=min(choices,key=lambda x:x[0]);near.append({'episode':e['episode'],'success':e['success'],'category':e['category'],'nearest_train_initial_xy_mm':round(d*1000,2),'nearest_train_grid':rec['grid']})
result={'weighted_anchor_mass':dict(mass),'weighted_phase_mass':dict(phase),'base_duration_s_percentiles':np.percentile(dur,[0,25,50,75,100]).tolist(),'base_first_close_s_percentiles':np.percentile(close_times,[0,25,50,75,100]).tolist(),'eval_nearest_training_position':near}
(R/'training_optimization_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))
