from run import *
def main():
 torch.set_num_threads(2);torch.manual_seed(20261028)
 out=A/'preclose_1s';out.mkdir(exist_ok=True)
 split=json.loads((TRAIN/'split_manifest.json').read_text());recs=split['source_records'];raw=CupDataset(range(500),raw_images=True)
 with np.load(TRAIN/'sampling_arrays.npz') as z:eligible=z['eligible_train']|z['eligible_validation']
 chosen=[];meta={};excluded=[]
 for r in recs:
  start,end=r['dataset_from_index'],r['dataset_to_index'];labels=raw.arrays['actions'][start:end,6]
  close=next((t for t in r['transitions'] if labels[t]>.5),None)
  frame=None if close is None else close-10
  if frame is None or frame<0 or not eligible[start+frame]:
   excluded.append(dict(episode=r['episode_index'],source=r['source'],split=r['split'],close=close,frame=frame,reason='Exact minus10 frame has no complete30 expert window'));continue
  i=start+frame;rows=[json.loads(l) for l in (pathlib.Path(r['source_path'])/'frames.jsonl').read_text().splitlines()]
  dt=rows[close]['simulation_time']-rows[frame]['simulation_time'];assert abs(dt-1)<.051
  assert np.all(labels[frame:close]==0) and labels[close]==1
  meta[i]=dict(episode=r['episode_index'],source=r['source'],split=r['split'],frame=frame,close_frame=close,seconds_before_close=float(dt),input_gripper_rad=float(raw.arrays['states'][i,6]),source_path=r['source_path'])
  chosen.append(i)
 ck=json.loads((E/'selected_checkpoint.json').read_text())['checkpoint'];policy=ACTPolicy.from_pretrained(ck,local_files_only=True).eval();pre,post=make_pre_post_processors(policy_cfg=policy.config,pretrained_path=ck)
 dl=DataLoader(Subset(raw,chosen),batch_size=32,num_workers=2,multiprocessing_context='spawn',prefetch_factor=2,persistent_workers=True)
 result=[];allpred=[];allgt=[]
 with torch.inference_mode():
  for batch in dl:
   gt=batch['action'].numpy().copy();obs={k:v for k,v in batch.items() if k.startswith('observation.')}
   for cam in raw.meta.camera_keys:obs[cam]=obs[cam].float()/255
   pred=post(policy.predict_action_chunk(pre(obs))).cpu().numpy();assert np.isfinite(pred).all() and not batch['action_is_pad'].any()
   for n,i in enumerate(batch['index'].tolist()):
    row=meta[i].copy();p,_=fk(pred[n,:,:6],REF);g,_=fk(gt[n,:,:6],REF);err=np.linalg.norm(p-g,axis=1)*1000
    state=batch['observation.state'][n].numpy();base,_=fk(np.repeat(state[None,:6],30,axis=0),REF);b=np.linalg.norm(base-g,axis=1)*1000
    close=np.flatnonzero(pred[n,:,6]>=.6);gtc=np.flatnonzero(gt[n,:,6]>.5);assert gtc[0]==10
    first=int(close[0]) if len(close) else None
    row.update(expert_close_step=10,predicted_close_step=first,close_step_error=None if first is None else first-10,close_in_first15=first is not None and first<15,predicted_gripper=pred[n,:,6].tolist(),expert_gripper=gt[n,:,6].tolist(),cartesian_error_mm=err.tolist(),no_motion_error_mm=b.tolist(),joint_mae_deg=(np.mean(abs(pred[n,:,:6]-gt[n,:,:6]),axis=1)*180/np.pi).tolist())
    result.append(row)
   allpred.extend(pred);allgt.extend(gt)
 assert len(result)==len(chosen)
 def stats(x):return dict(zip(['median','p95','mean'],map(float,[np.median(x),np.percentile(x,95),np.mean(x)]))) if len(x) else None
 def summarize(rows):
  if not rows:return None
  predicted=[r for r in rows if r['predicted_close_step'] is not None]
  return dict(episodes=len(rows),input_gripper_rad=stats([r['input_gripper_rad'] for r in rows]),input_gripper_above_minus075=sum(r['input_gripper_rad']>-.75 for r in rows),close_in_first15=sum(r['close_in_first15'] for r in rows),no_close_in30=sum(r['predicted_close_step'] is None for r in rows),close_only_after_first15=sum(r['predicted_close_step'] is not None and r['predicted_close_step']>=15 for r in rows),close_step_error=stats([r['close_step_error'] for r in predicted]),close_within_plusminus2_steps=sum(abs(r['close_step_error'])<=2 for r in predicted),errors={str(h):stats([r['cartesian_error_mm'][h] for r in rows]) for h in [0,4,9,10,14,29]},no_motion_errors={str(h):stats([r['no_motion_error_mm'][h] for r in rows]) for h in [0,4,9,10,14,29]})
 report=dict(checkpoint=ck,exact_offset_frames=-10,nominal_seconds_before_close=1,input_path='native JPEG -> saved deployment preprocessor, no expert action in policy input',total_sources=500,evaluated=len(result),excluded=excluded,summary={p:summarize([r for r in result if r['split']==p]) for p in ['train','validation']},sources={p:summarize([r for r in result if r['source']==p]) for p in ['basic','correction']},episodes=result)
 (out/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));np.savez_compressed(out/'predictions.npz',indices=np.array(chosen),predictions=np.array(allpred),expert_actions=np.array(allgt))
 print(json.dumps({k:v for k,v in report.items() if k not in ['episodes','sources']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
