"""Keep ACT's split and statistics; add historical gentle anchor weights."""
import json,shutil,hashlib
from pathlib import Path
import numpy as np
RUN=Path(__file__).resolve().parent;OLD=RUN.parent/'act_correction_20261013'
DATA=Path('/home/lp531/smolvla_data/z1_bottle_with_fragments_20261012')
def write(name,value):
 p=RUN/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str)+'\n');t.replace(p)
def main():
 from sampling import weighted_probabilities
 for name in ['split_manifest.json','train_stats.json']:shutil.copy2(OLD/name,RUN/name)
 split=json.loads((RUN/'split_manifest.json').read_text())
 with np.load(OLD/'sampling_arrays.npz') as a:arrays={k:a[k].copy() for k in a.files}
 key=np.zeros(len(arrays['episode']),bool);events=[]
 for r in split['source_records']:
  close=r['close_frame'];frames=set(range(max(0,close-20),close));raw=Path(r['source_path'])
  if r['source']!='original':
   rows=[json.loads(x) for x in (raw/'frames.jsonl').read_text().splitlines()];meta=json.loads((raw/'correction.json').read_text())
   for stage in meta.get('stages',[]):
    frames.update(i for i,row in enumerate(rows[:close]) if stage['start_sim_time']<=row['simulation_time']<=stage['end_sim_time'])
  mask=arrays['episode']==r['episode_index'];key[mask]=np.isin(arrays['frame'][mask],sorted(frames))
  events.append(dict(episode_index=r['episode_index'],source=r['source'],close_frame=close,key_frames=sorted(frames)))
 arrays['key']=key;arrays['weights']=weighted_probabilities(arrays['episode'],arrays['eligible_train'],key)
 assert not np.isin(arrays['episode'][arrays['weights']>0],split['validation_episodes']).any()
 np.savez_compressed(RUN/'sampling_arrays.npz',**arrays)
 p=arrays['weights'];c=arrays['episode']>=160;eligible=arrays['eligible_train']
 write('phase_manifest.json',events)
 write('sampling_audit.json',dict(original_mass=float(p[~c].sum()),correction_mass=float(p[c].sum()),key_mass=float(p[key].sum()),within_pool_key_weight=2,other_weight=1,loss_weight=1,rule='open anchors in 2s before first close UNION logged preclose recovery stages',training_windows=int(eligible.sum()),validation_windows=int(arrays['eligible_validation'].sum()),train_episodes=190,validation_episodes=21,chunk_size=16,complete_approved_windows=True,heldout_excluded=True,split_preserved=True,stats_preserved=True,split_sha256=hashlib.sha256((RUN/'split_manifest.json').read_bytes()).hexdigest()))
 print((RUN/'sampling_audit.json').read_text(),flush=True)
if __name__=='__main__':main()
