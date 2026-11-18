import os
os.environ['TOKENIZERS_PARALLELISM']='false'
from pathlib import Path
import json,collections
import numpy as np,cv2,torch,pyarrow.parquet as pq
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
O=Path(__file__).parent;R=Path('/home/lp531/smolvla_data/z1_tabletop_left_wrist_prepared_all_v1');L=Path('/home/lp531/smolvla_data/z1_tabletop_lerobot_left_wrist_2cam_all_v1')
audit=json.load(open('/home/lp531/smolvla_eval/dataset_sample_audit_20260928/results.json'))['samples'];counts=collections.Counter();examples=[];tiles=[]
for s in audit:
 t=s['target']
 if counts[t]>=2:continue
 counts[t]+=1
 frames=[json.loads(z) for z in (R/s['episode']/'frames.jsonl').read_text().splitlines()]
 df=pq.read_table(L/'data/chunk-000'/f"episode_{s['training_episode']:06d}.parquet").to_pandas();q=np.stack(df['observation.state']);a=np.stack(df['action']);ci=s['first_close_frame']
 candidates=[i for i in range(int(ci*.4),ci-12) if np.linalg.norm(a[i+9,:6]-q[i,:6])>.05]
 assert candidates
 i=min(candidates,key=lambda i:abs(i-ci*.65));ims={};row=[]
 for cam in ['global','wrist']:
  im=cv2.imread(str(R/s['episode']/frames[i][f'observation.images.{cam}']));ims[cam]=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
  thumb=cv2.resize(im,(320,240));cv2.putText(thumb,f"{t} {s['episode'][-3:]} f{i} {cam}",(4,18),0,.45,(0,0,255),1);row.append(thumb)
 tiles.append(np.hstack(row));examples.append((s,i,q[i],a[i:i+10],ims,frames[i]['task']))
cv2.imwrite(str(O/'inputs.jpg'),np.vstack(tiles))
ck='/home/lp531/smolvla_training/z1_left_wrist_no_phase_all_batch8_lr_restart_20260928_004609/checkpoints/037803/pretrained_model'
p=SmolVLAPolicy.from_pretrained(ck,local_files_only=True,strict=False);p.eval();device=torch.device(p.config.device)
conditions=['normal','wrist_gray','global_only_masked','both_gray'];results=[];predictions={}
for s,fi,state,truth,ims,task in examples:
 for seed in [123,456]:
  for cond in conditions:
   p.reset();torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
   batch={'task':[task],'observation.state':torch.tensor(state,device=device).unsqueeze(0)}
   for cam in ['wrist','global']:
    rgb=ims[cam]
    if cond=='both_gray' or (cam=='wrist' and cond in ['wrist_gray','global_only_masked']):rgb=np.broadcast_to(np.array([195,197,200],dtype=np.uint8),rgb.shape).copy()
    batch[f'observation.images.{cam}']=torch.from_numpy(np.ascontiguousarray(rgb)).to(device=device,dtype=torch.float32).permute(2,0,1).unsqueeze(0)/255
   if cond=='global_only_masked':batch['observation.images.wrist_padding_mask']=torch.zeros(1,dtype=torch.bool,device=device)
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):out=p.predict_action_chunk(batch).float().cpu().numpy()[0,:10]
   delta=out[-1,:6]-state[:6];ref=truth[-1,:6]-state[:6];cos=float(np.dot(delta,ref)/(np.linalg.norm(delta)*np.linalg.norm(ref)+1e-12))
   r=dict(target=s['target'],episode=s['episode'],frame=fi,xy=s['xy'],condition=cond,seed=seed,arm_rmse_rad=float(np.sqrt(np.mean((out[:,:6]-truth[:,:6])**2))),direction_cosine=cos,reference_motion_norm_rad=float(np.linalg.norm(ref)),predicted_motion_norm_rad=float(np.linalg.norm(delta)))
   results.append(r);predictions[f"{s['episode']}_{fi}_{seed}_{cond}"]=out
  print(s['episode'],fi,'seed',seed,[(r['condition'],round(r['arm_rmse_rad'],4),round(r['direction_cosine'],3)) for r in results[-4:]],flush=True)
 json.dump({'checkpoint':ck,'results':results},open(O/'results.json','w'),indent=2)
np.savez(O/'predictions.npz',**predictions)
for cond in conditions:
 rr=[r for r in results if r['condition']==cond];print('TOTAL',cond,'rmse',np.mean([r['arm_rmse_rad'] for r in rr]),'cos',np.mean([r['direction_cosine'] for r in rr]),'opposite',sum(r['direction_cosine']<0 for r in rr),flush=True)
