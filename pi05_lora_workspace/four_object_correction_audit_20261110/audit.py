from pathlib import Path
from collections import Counter
import json,math
ROOT=Path('/home/lp531/smolvla_data');names=['apple_scale110_correction60_20261106','peach_scale120_correction60_20261106','cup_minus90_rerecord_20261022/corrections_full','z1_bottle_corrections_20261009','z1_bottle_correction_fragments_20261012'];report={}
for name in names:
 summary=Counter();problems=[];durations=[];totalframes=0
 for path in sorted((ROOT/name).glob('episode_*/episode_summary.json')):
  d=json.loads(path.read_text());summary['episodes']+=1;summary['full_success']+=d.get('task_success') is True;summary['fragment_success']+=d.get('fragment_success') is True;summary['released']+=d.get('release_sequence_complete') is True;summary['dropped_frames']+=d.get('dropped_frames',0);durations.append(d.get('duration_s',0))
  frames=path.parent/'frames.jsonl';n=0;previous=None;issues=Counter();last={}
  try:
   for line in frames.open():
    f=json.loads(line);n+=1;last=f;t=f.get('timestamp')
    if not isinstance(t,(int,float)) or not math.isfinite(t) or (previous is not None and t<=previous):issues['nonmonotonic_or_invalid_timestamp']+=1
    previous=t
    for k in ['action','observation.state']:
     v=f.get(k,[])
     if len(v)!=7 or not all(isinstance(x,(int,float)) and math.isfinite(x) for x in v):issues['invalid_'+k]+=1
    for cam in ['wrist','global','global_right']:
     relative=f.get('observation.images.'+cam)
     if not isinstance(relative,str) or not (path.parent/relative).is_file():issues['missing_'+cam]+=1
  except (OSError,ValueError) as e:issues['unreadable_frame_file']+=1
  if n!=d.get('frame_count'):issues['frame_count_mismatch']+=1
  totalframes+=n;summary['last_phase_'+str(last.get('phase'))]+=1
  if issues:problems.append({'episode':str(path.parent),'issues':dict(issues)})
 report[name]={'summary':dict(summary),'frames':totalframes,'duration_range_s':[min(durations),max(durations)],'problem_episodes':problems};print(name,json.dumps(report[name],ensure_ascii=False)[:2200],flush=True)
Path(__file__).with_name('report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
