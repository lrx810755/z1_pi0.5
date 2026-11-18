"""Choose four nearby physically replayed, bottom-placed full episodes."""
import itertools,json,math,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
labels={x['raw_episode']:x for x in json.loads((ROOT/'upright_candidates.json').read_text())}
results=[]
for screening in ['screen2','screen3']:
 for x in json.loads((ROOT/screening/'replay_results.json').read_text()):
  x=dict(x,screening=screening)
  p=x['final_status']['object_position_m']
  # Box interior x .25..39 / y .07..21, plus a conservative 30 mm
  # horizontal footprint margin; base at the .01 m inner floor.
  x['interior_floor_check']=bool(.280<=p[0]<=.360 and .100<=p[1]<=.180 and .007<=p[2]<=.014)
  if x['success'] and x['interior_floor_check']:results.append(x)
assert len(results)>=4,[(x['source_episode'],x['success'],x['interior_floor_check']) for x in results]
def diameter(group):
 return max(math.dist(labels[a['source_episode']]['position'],labels[b['source_episode']]['position']) for a,b in itertools.combinations(group,2))
chosen=list(min(itertools.combinations(results,4),key=diameter))
archive=ROOT/'screen1';archive.mkdir(exist_ok=True)
for name in ['selected.json','replay_results.json','replay_complete.json','dataset_verification.json','resolved_train_config.json']:
 p=ROOT/name
 if p.exists() and not (archive/name).exists():shutil.copy2(p,archive/name)
selected=[labels[x['source_episode']] for x in chosen]
(ROOT/'selected.json').write_text(json.dumps(selected,indent=2));(ROOT/'replay_results.json').write_text(json.dumps(chosen,indent=2))
manifest=dict(experiment='PI05 four-episode training-set fitting diagnostic',selected=[dict(raw_episode=x['raw_episode'],dataset_episode_index=x['episode_index'],frames=x['frames'],seed=x['seed'],grid=x['grid'],position=x['position'],raw_frames_sha256=hashlib.sha256((Path(x['raw_root'])/'frames.jsonl').read_bytes()).hexdigest()) for x in selected],frames=sum(x['frames'] for x in selected),diameter_m=diameter(chosen),source_dataset='/home/lp531/smolvla_data/z1_bottle_three_camera_pi05_binary_20261007',selection='native episodes filter; all selected frames retained',normalization='same full160 stats as previous model',validation_split=0,reference_replay_passed=True,strict_placement_proxy='object base on box floor and conservative horizontal interior margin; video review also required',seed=20261007)
(ROOT/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
