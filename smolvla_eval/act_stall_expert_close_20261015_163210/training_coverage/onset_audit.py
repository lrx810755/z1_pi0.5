"""Current checkpoint on teacher observations around close onset; no edits."""
from pathlib import Path
import json,numpy as np,torch
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import ImageCropResizeProcessorStep
OUT=Path(__file__).resolve().parent
RUN=Path('/home/lp531/pi05_lora_workspace/act_gentle_cached_20261015')
CACHE=Path('/home/lp531/smolvla_data/act_rgb224_float32_cache_20261015')
CP='/home/lp531/smolvla_training/act_gentle_cached_chunk16_20261015/checkpoints/050000/pretrained_model'
a=np.load(OUT/'geometry_arrays.npz');sampling=np.load(RUN/'sampling_arrays.npz')
split=json.loads((RUN/'split_manifest.json').read_text());records=split['source_records'];states=a['states'];actions=a['actions'];ep=a['episode'];frame=a['frame']
for r in records:r['split']='train' if r['episode_index'] in split['train_episodes'] else 'validation'
ids=[]
for r in records:
    for offset in [-5,-2,-1,0,1,2,5]:
        matches=np.flatnonzero((ep==r['episode_index'])&(frame==r['close_frame']+offset))
        if len(matches):
            i=int(matches[0]);eligible=sampling['eligible_train'][i] if r['split']=='train' else sampling['eligible_validation'][i]
            if eligible:ids.append(i)
torch.set_num_threads(4);policy=ACTPolicy.from_pretrained(CP,local_files_only=True,strict=True).eval()
pre,post=make_pre_post_processors(policy.config,pretrained_path=CP)
pre.steps=[s for s in pre.steps if not isinstance(s,ImageCropResizeProcessorStep)]
cams=['wrist','global','global_right'];maps={};rows=[]
for start in range(0,len(ids),16):
    chosen=ids[start:start+16];batch={'observation.state':torch.tensor(states[chosen],dtype=torch.float32)}
    for cam in cams:
        images=[]
        for i in chosen:
            e=int(ep[i]);pair=(e,cam)
            if pair not in maps:maps[pair]=np.load(CACHE/f'observation.images.{cam}'/f'episode_{e:06d}.npy',mmap_mode='r')
            images.append(np.array(maps[pair][int(frame[i])],copy=True))
        batch[f'observation.images.{cam}']=torch.tensor(np.array(images),dtype=torch.float32)
    with torch.inference_mode():pred=post(policy.predict_action_chunk(pre(batch))).cpu().numpy()
    for i,p in zip(chosen,pred):
        e=int(ep[i]);r=records[e];truth=actions[i:i+16,-1];pv=p[:,-1]
        rows.append(dict(index=i,episode=e,source=r['source'],split=r['split'],frame=int(frame[i]),offset=int(frame[i]-r['close_frame']),input_gripper=float(states[i,-1]),label=float(truth[0]),key=bool(sampling['key'][i]),weight=float(sampling['weights'][i]),first_prediction=float(pv[0]),first_closed_step=next((h+1 for h,v in enumerate(pv) if v>=.6),None),prediction=pv.tolist(),truth=truth.tolist()))
    if start%160==0:print('onset inference',start,len(ids),flush=True)
summary={}
for part in ['train','validation']:
    for name,cond in [('open_input_first_close',lambda r:r['offset']==0 and r['input_gripper']<=-.88),('all_first_close',lambda r:r['offset']==0),('before_1_frame',lambda r:r['offset']==-1),('after_1_frame',lambda r:r['offset']==1)]:
        rr=[r for r in rows if r['split']==part and cond(r)]
        summary[f'{part}_{name}']=dict(windows=len(rr),first_step_closed=sum(r['first_prediction']>=.6 for r in rr),any_16_step_close=sum(r['first_closed_step'] is not None for r in rr),first_15_step_close=sum(r['first_closed_step'] is not None and r['first_closed_step']<=15 for r in rr),prediction_range=[min(r['first_prediction'] for r in rr),max(r['first_prediction'] for r in rr)] if rr else None,key_windows=sum(r['key'] for r in rr))
(OUT/'onset_predictions.json').write_text(json.dumps(rows,indent=2));(OUT/'onset_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
