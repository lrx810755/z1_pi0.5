"""Read-only source audit, safe while corrections are still being completed."""
import collections,json,time
import numpy as np
from prepare import RUN,BASE,CORR,write,required_source_hashes,validate_physics
from rules import full_labels,correction_labels,split_for_grid,validate_reference

def main():
 started=time.monotonic();sources={};errors=[];records=[];references={}
 for name,root in [('basic',BASE),('correction',CORR)]:
  episodes=[];categories=collections.Counter();multi=0;excluded=0
  for path in sorted(root.glob('episode_[0-9]*')):
   try:
    required_source_hashes(path,name)
    summary=json.loads((path/'episode_summary.json').read_text());scene=summary['scene_randomization'];grid=scene['target_grid'];seed=scene['seed'];n=summary['frame_count'];split=split_for_grid(grid)
    if name=='basic':
     labels,_,_=full_labels([json.loads(x) for x in (path/'frames.jsonl').read_text().splitlines()]);expert=np.ones(n,bool);valid=np.arange(n)+15<n;references[str(path.resolve())]=(grid,seed)
    else:
     c=json.loads((path/'correction.json').read_text());validate_physics(path/'physics_status.jsonl',c);labels,expert,valid=correction_labels(json.loads((path/'gripper_labels.json').read_text()),json.loads((path/'training_mask.json').read_text()),n)
     if labels[0]!=0 or labels[-1]!=0:raise ValueError('Incomplete gripper sequence')
     ref=references[str(__import__('pathlib').Path(c['task']['reference_basic_episode']).resolve())];validate_reference(grid,ref[0],seed,ref[1]);categories[c['task']['kind']]+=1
     if len(np.flatnonzero(np.diff(labels)))>2:multi+=1
    excluded+=int((~expert).sum());episodes.append(dict(frames=n,split=split,eligible_windows=int(valid.sum())));records.append(dict(seed=seed,split=split))
   except Exception as exc:errors.append(dict(source=name,episode=path.name,error=str(exc)))
  sources[name]=dict(episodes=len(episodes),frames=sum(e['frames'] for e in episodes),train_episodes=sum(e['split']=='train' for e in episodes),validation_episodes=sum(e['split']=='validation' for e in episodes),
   training_windows=sum(e['eligible_windows'] for e in episodes if e['split']=='train'),validation_windows=sum(e['eligible_windows'] for e in episodes if e['split']=='validation'),categories=dict(categories),multiple_gripper_cycle_episodes=multi,excluded_intervention_frames=excluded)
 overlap={r['seed'] for r in records if r['split']=='train'}&{r['seed'] for r in records if r['split']=='validation'}
 total=sum(s['frames'] for s in sources.values());trainwindows=sum(s['training_windows'] for s in sources.values())
 report=dict(sources=sources,errors=errors,seed_overlap_train_validation=sorted(overlap),total_frames=total,crop_cache_gib=total*3*3*224*224*4/2**30,proposed_anchor_exposures=90000*32,proposed_equivalent_window_passes=90000*32/trainwindows,elapsed_s=time.monotonic()-started)
 write('read_only_source_inventory.json',report);print(json.dumps(report,indent=2))
 if errors or overlap:raise SystemExit(1)
if __name__=='__main__':main()
