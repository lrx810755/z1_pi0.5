"""Read-only audit of 50 stratified demonstrations and deployed binary labels."""
import csv,json,random,statistics
from pathlib import Path
import pyarrow.parquet as pq
W=Path(__file__).resolve().parent
M=Path('/home/lp531/pi05_lora_workspace/fine_grip_recovery_base_20261012/phase_manifest.json')
m=json.loads(M.read_text());rng=random.Random(20261014)
selected=[]
for group,n in [('original',35),('old_correction',10),('new_fragment',5)]:
 selected.extend(rng.sample([e for e in m if e['source']==group],n))
selected.sort(key=lambda e:e['episode_index'])
t=pq.read_table('/home/lp531/smolvla_data/z1_bottle_with_fragments_20261012/data',columns=['episode_index','frame_index','action']).to_pylist()
stored={}
for r in t:stored.setdefault(r['episode_index'],{})[r['frame_index']]=r['action'][-1]
def onset(a,c,threshold):
 # Local open baseline avoids interpreting initial opening or open-state noise as closing.
 baseline=statistics.median(a[max(0,c-10):max(1,c-5)])
 for i in range(max(0,c-8),min(len(a)-3,c+6)):
  if min(a[i:i+2])>baseline+threshold and max(a[i:i+4])>baseline+.02:
   return i,baseline
 return None,baseline
out=[];examples=[]
for e in selected:
 rows=[json.loads(x) for x in (Path(e['source_path'])/'frames.jsonl').read_text().splitlines()]
 c=e['close_frame'];g=next(i for i,r in enumerate(rows) if r['phase']=='GRASPING')
 a=[r['action'][-1] for r in rows];s=[r['observation.state'][-1] for r in rows]
 cmd,b=onset(a,c,.0005)
 threshold=next((i for i in range(max(0,c-8),min(len(a)-1,c+6)) if min(a[i:i+2])>-.895 and max(a[i:i+4])>-.875),None)
 actual={str(x):onset(s,c,x)[0] for x in [.002,.005,.01]}
 binary=next(i for i in sorted(stored[e['episode_index']]) if stored[e['episode_index']][i]>.5)
 rec=dict(episode_index=e['episode_index'],source=e['source'],split=e['split'],source_path=e['source_path'],grasping_frame=g,label_frame=c,stored_label_frame=binary,command_onset_frame=cmd,command_threshold_frame=threshold,actual_onset_frame_002=actual['0.002'],actual_onset_frame_005=actual['0.005'],actual_onset_frame_010=actual['0.01'],command_baseline=b,actual_baseline=onset(s,c,.005)[1],command_at_label=a[c],actual_at_label=s[c],actual_before_label=s[c-1],phase_before_label=rows[c-1]['phase'])
 for name,frame in [('command_onset',cmd),('command_threshold',threshold),('actual_002',actual['0.002']),('actual_005',actual['0.005']),('actual_010',actual['0.01'])]:
  rec['grasping_minus_'+name+'_s']=None if frame is None else round(rows[g]['timestamp']-rows[frame]['timestamp'],6)
  rec['label_minus_'+name+'_s']=None if frame is None else round(rows[c]['timestamp']-rows[frame]['timestamp'],6)
 rec['label_minus_grasping_s']=round(rows[c]['timestamp']-rows[g]['timestamp'],6)
 out.append(rec)
 examples.append(dict(episode_index=e['episode_index'],source=e['source'],events=rec,frames=[dict(frame=i,time=rows[i]['timestamp'],phase=rows[i]['phase'],command=a[i],actual=s[i],binary_label=stored[e['episode_index']][i]) for i in range(max(0,c-10),min(len(rows),c+8))]))
def describe(rs):
 result={'episodes':len(rs),'label_matches_saved_dataset':all(r['label_frame']==r['stored_label_frame'] for r in rs)}
 for key in ['grasping_minus_command_onset_s','grasping_minus_command_threshold_s','grasping_minus_actual_002_s','grasping_minus_actual_005_s','grasping_minus_actual_010_s','label_minus_command_onset_s','label_minus_actual_005_s','label_minus_grasping_s']:
  vals=[r[key] for r in rs if r[key] is not None]
  result[key]=dict(detected=len(vals),earlier=sum(v<0 for v in vals),same_frame=sum(v==0 for v in vals),later=sum(v>0 for v in vals),min=min(vals) if vals else None,max=max(vals) if vals else None,median=statistics.median(vals) if vals else None)
 return result
summary=dict(selection_seed=20261014,selection='35 original + 10 old correction + 5 new fragments; randomly selected within each source',fps=10,positive_delta_means='phase/label later than movement',actual_onset_rule='median local open baseline (label-10 through label-6); exceeds baseline by threshold for 2 samples and rises >0.02 rad within 4 samples',limitation='10Hz sampled records cannot resolve sub-frame events; measured angles include tracking noise. No videos decoded.',all=describe(out),groups={g:describe([r for r in out if r['source']==g]) for g in ['original','old_correction','new_fragment']})
for name,v in [('summary.json',summary),('episodes.json',out),('onset_windows.json',examples)]: (W/name).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
with (W/'episodes.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(out[0]));writer.writeheader();writer.writerows(out)
print(json.dumps(summary,ensure_ascii=False,indent=2))
print('EXCEPTIONS',json.dumps([r for r in out if r['grasping_minus_command_onset_s']!=0 or r['grasping_minus_actual_005_s']>0],ensure_ascii=False))
