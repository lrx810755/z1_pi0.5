import os,sys,json,time,pathlib,hashlib
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset,default_collate
A=pathlib.Path(__file__).resolve().parent;E=A.parent
TRAIN=pathlib.Path('/home/lp531/pi05_lora_workspace/act_cup_renderfix500_chunk30_batch64_20261028_003937')
sys.path.insert(0,str(TRAIN));sys.path.insert(0,str(E/'execution_audit'))
from cache_data import CupDataset
from check import fk
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies import make_pre_post_processors
from lerobot.processor import ImageCropResizeProcessorStep
REF=(.12083045,-.0028452,.0194797)
def dump(name,data):
 p=A/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(data,ensure_ascii=False,indent=2));t.replace(p)
def main():
 torch.set_num_threads(2);np.random.seed(20261028);torch.manual_seed(20261028)
 split=json.loads((TRAIN/'split_manifest.json').read_text());records=split['source_records']
 ds=CupDataset(list(range(500)))
 with np.load(TRAIN/'sampling_arrays.npz') as z:eligible=z['eligible_train']|z['eligible_validation']
 chosen=set();selection=[]
 for r in records:
  start,end=r['dataset_from_index'],r['dataset_to_index'];valid=np.flatnonzero(eligible[start:end]);anchors=set(valid[np.unique(np.linspace(0,len(valid)-1,12,dtype=int))].tolist())
  for trans in r['transitions']:
   for offset in [-29,-15,-5,-1,0,1,5,14,29]:
    f=trans+offset
    if 0<=f<end-start and eligible[start+f]:anchors.add(f)
  chosen.update(start+f for f in anchors)
  selection.append(dict(episode=r['episode_index'],source=r['source'],split=r['split'],frames=sorted(anchors)))
 chosen=sorted(chosen);dump('selection.json',dict(episodes=500,anchors=len(chosen),selection=selection,rule='12 uniformly spaced eligible anchors per episode plus exact eligible transition offsets[-29,-15,-5,-1,0,1,5,14,29]; no padding, intervention or cross-episode windows'))
 ck=json.loads((E/'selected_checkpoint.json').read_text())['checkpoint'];policy=ACTPolicy.from_pretrained(ck,local_files_only=True).eval();pre,post=make_pre_post_processors(policy_cfg=policy.config,pretrained_path=ck)
 fullsteps=list(pre.steps);cachedsteps=[x for x in fullsteps if not isinstance(x,ImageCropResizeProcessorStep)]
 # Compare raw deployment preprocessing and cache preprocessing across both sources/splits and transition states.
 raw=CupDataset(list(range(500)),raw_images=True);checks=[]
 for source in ['basic','correction']:
  for partition in ['train','validation']:
   rec=next(r for r in records if r['source']==source and r['split']==partition)
   ids=[i for i in chosen if rec['dataset_from_index']<=i<rec['dataset_to_index']];indices=[ids[0],min(ids,key=lambda i:abs(i-rec['dataset_from_index']-rec['transitions'][0]))]
   for i in indices:
    rb=default_collate([raw[i]]);cb=default_collate([ds[i]])
    for cam in raw.meta.camera_keys:rb[cam]=rb[cam].float()/255
    rb={k:v for k,v in rb.items() if k.startswith('observation.')};cb={k:v for k,v in cb.items() if k.startswith('observation.')}
    pre.steps=fullsteps;rp=pre(rb);pre.steps=cachedsteps;cp=pre(cb)
    feature_difference=max(float((rp[k]-cp[k]).abs().max()) for k in rb)
    for k in rb:torch.testing.assert_close(rp[k],cp[k],rtol=0,atol=5e-5)
    with torch.inference_mode():
     p1=post(policy.predict_action_chunk(rp));p2=post(policy.predict_action_chunk(cp))
    assert float((p1[:,:,:6]-p2[:,:,:6]).abs().max())<.002
    assert float((p1[:,:,6]-p2[:,:,6]).abs().max())<.01
    checks.append(dict(source=source,split=partition,index=i,max_feature_difference=feature_difference,max_prediction_difference=float((p1-p2).abs().max())))
 dump('preprocessing_verified.json',dict(passed=True,checks=checks,targets_excluded_from_policy_input=True,eval_mode=True))
 pre.steps=fullsteps
 dl=DataLoader(Subset(raw,chosen),batch_size=32,num_workers=2,multiprocessing_context='spawn',prefetch_factor=2,persistent_workers=True)
 predictions=[];truths=[];states=[];indices=[];started=time.monotonic()
 with torch.inference_mode():
  for batch in dl:
   truth=batch['action'].numpy().copy();assert truth.shape[1:]==(30,7) and not batch['action_is_pad'].any()
   obs={k:v for k,v in batch.items() if k.startswith('observation.')}
   for cam in raw.meta.camera_keys:obs[cam]=obs[cam].float()/255
   pred=post(policy.predict_action_chunk(pre(obs))).cpu().numpy();assert pred.shape==truth.shape and np.isfinite(pred).all()
   predictions.append(pred);truths.append(truth);states.append(batch['observation.state'].numpy());indices.extend(batch['index'].tolist())
   dump('progress.json',dict(processed=len(indices),total=len(chosen),elapsed_s=time.monotonic()-started))
 assert indices==chosen
 pred=np.concatenate(predictions);truth=np.concatenate(truths);state=np.concatenate(states)
 np.savez_compressed(A/'predictions.npz',indices=np.array(indices),predictions=pred,expert_actions=truth,states=state)
 rows=[]
 for n,i in enumerate(indices):
  ep=int(ds.arrays['episode'][i]);frame=int(ds.arrays['frame'][i]);r=records[ep];labels=ds.arrays['actions'][r['dataset_from_index']:r['dataset_to_index'],-1]
  close=next((t for t in r['transitions'] if labels[t]>.5),None);release=next((t for t in r['transitions'] if close is not None and t>close and labels[t]<.5),None)
  delta=frame-close if close is not None else None
  phase='approach' if delta is not None and delta< -15 else 'preclose' if delta is not None and delta<0 else 'closing' if delta is not None and delta<=14 else 'transport' if release is None or frame<release-15 else 'placement'
  pp,_=fk(pred[n,:,:6],REF);gt,_=fk(truth[n,:,:6],REF);baseline,_=fk(np.repeat(state[n:n+1,:6],30,axis=0),REF)
  errors=np.linalg.norm(pp-gt,axis=1)*1000;no_motion=np.linalg.norm(baseline-gt,axis=1)*1000
  row=dict(index=i,episode=ep,frame=frame,source=r['source'],split=r['split'],kind=r['kind'],phase=phase,delta_to_first_close=delta,arm_mae_deg=float(np.abs(pred[n,:,:6]-truth[n,:,:6]).mean()*180/np.pi),chunk_cartesian_mean_mm=float(errors.mean()),expert_close_now=bool(truth[n,0,6]>.5),predicted_close_now=bool(pred[n,0,6]>=.6),gripper_prediction_now=float(pred[n,0,6]))
  for h in [0,4,14,29]:row[f'error_h{h}_mm']=float(errors[h]);row[f'no_motion_h{h}_mm']=float(no_motion[h])
  gtclose=np.flatnonzero(truth[n,:,6]>.5);pc=np.flatnonzero(pred[n,:,6]>=.6)
  row.update(expert_first_close=int(gtclose[0]) if len(gtclose) else None,predicted_first_close=int(pc[0]) if len(pc) else None)
  rows.append(row)
 dump('sample_metrics.json',rows)
 def summarize(sub):
  if not sub:return None
  result=dict(samples=len(sub),episodes=len({x['episode'] for x in sub}))
  for key in ['arm_mae_deg','chunk_cartesian_mean_mm',*[f'error_h{h}_mm' for h in [0,4,14,29]],*[f'no_motion_h{h}_mm' for h in [0,4,14,29]]]:
   result[key]=dict(zip(['median','p95','mean'],map(float,[np.median([x[key] for x in sub]),np.percentile([x[key] for x in sub],95),np.mean([x[key] for x in sub])])))
  closed=[x for x in sub if x['expert_close_now']];result['closed_now_samples']=len(closed);result['closed_now_recall']=sum(x['predicted_close_now'] for x in closed)/len(closed) if closed else None
  open_now=[x for x in sub if not x['expert_close_now']];result['open_now_false_close_fraction']=sum(x['predicted_close_now'] for x in open_now)/len(open_now) if open_now else None
  future=[x for x in sub if x['expert_first_close'] is not None];both=[x for x in future if x['predicted_first_close'] is not None]
  result.update(expert_close_in_chunk_samples=len(future),missed_close_chunk_fraction=sum(x['predicted_first_close'] is None for x in future)/len(future) if future else None,first_close_step_delta_median=float(np.median([x['predicted_first_close']-x['expert_first_close'] for x in both])) if both else None)
  return result
 report=dict(checkpoint=ck,episodes=500,samples=len(rows),horizons_steps=[0,4,14,29],demonstration_rate_hz=10,reference_point_local_m=list(REF),summary={})
 for part in ['train','validation']:
  report['summary'][part]=dict(overall=summarize([x for x in rows if x['split']==part]),phases={p:summarize([x for x in rows if x['split']==part and x['phase']==p]) for p in ['approach','preclose','closing','transport','placement']},sources={s:summarize([x for x in rows if x['split']==part and x['source']==s]) for s in ['basic','correction']})
 dump('results.json',report);dump('completed.json',dict(completed=True,episodes=500,samples=len(rows),elapsed_s=time.monotonic()-started));print(json.dumps(dict(completed=True,samples=len(rows),elapsed_s=time.monotonic()-started)),flush=True)
if __name__=='__main__':main()
