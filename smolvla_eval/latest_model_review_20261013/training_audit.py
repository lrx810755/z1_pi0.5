from pathlib import Path
import json,numpy as np,pyarrow.parquet as pq,re
work=Path('/home/lp531/pi05_lora_workspace');out=Path(__file__).resolve().parent;report={}
for name in ['gentle_correction_base_20261010','fine_grip_recovery_base_20261012']:
 d=work/name;cfg=json.loads((d/'resolved_train_config.json').read_text());a=np.load(d/'sampling_arrays.npz');tab=pq.read_table(Path(cfg['dataset']['root'])/'data',columns=['index','action','observation.state','episode_index']).sort_by([('index','ascending')]);actions=np.array(tab['action'].to_pylist());states=np.array(tab['observation.state'].to_pylist());ep=np.array(tab['episode_index']);eligible=np.isin(ep,cfg['dataset']['episodes']);eligible&=a['eligible_train_anchor'] if 'eligible_train_anchor' in a.files else True;ids=np.flatnonzero(eligible);assert len(ids)==json.loads((d/'sampling_audit.json').read_text())['training_windows'];w=np.array(a['phase_weight'][ids],float) if 'phase_weight' in a.files else np.where(a['key'][ids],2.,1.);corr=ep[ids]>=160
 for flag,mass in [(False,.7),(True,.3)]:w[corr==flag]*=mass/w[corr==flag].sum()
 close=actions[ids,-1]>=.5
 row={'eligible_anchors':len(ids),'closed_anchor_mass':float(w[close].sum()),'open_state_close_action_mass':float(w[(states[ids,-1]<-.88)&close].sum()),'midclosing_state_close_action_mass':float(w[(states[ids,-1]>=-.88)&(states[ids,-1]<-.75)&close].sum())}
 future=np.minimum(ids[:,None]+np.arange(15),len(ep)-1);valid=ep[future]==ep[ids,None];isclosed=actions[future,-1]>=.5
 transition=(~isclosed[:,0])&np.any(isclosed&valid,axis=1)
 row['open_to_close_chunk_mass']=float(w[transition].sum())
 row['closed_label_unique']=np.unique(actions[:,-1]).tolist()
 log=(d/'training.log').read_text();matches=re.findall(r'step (\d+): eval_loss=([\d.]+)',log);row['validation_loss']={step:float(loss) for step,loss in matches}
 if 'phase_id' in a.files:
  durations=json.loads((d/'phase_manifest.json').read_text());row['initial_lift_duration_s']={str(r['episode_index']):(r['stable_confirmed']-r['lift_start'])/10 for r in durations if r.get('stable_confirmed') is not None}
 report[name]=row
(out/'training_audit.json').write_text(json.dumps(report,indent=2))
print('saved training audit')
