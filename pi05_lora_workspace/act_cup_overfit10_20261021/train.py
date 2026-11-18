"""Fine-tune ten known training demonstrations to diagnose closed-loop fit."""
import argparse,fcntl,hashlib,json,math,os,random,shutil,sys,time
from pathlib import Path
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset,WeightedRandomSampler,default_collate
BASE=Path('/home/lp531/pi05_lora_workspace/act_cup_dynamic_cached_20261020')
R=Path(__file__).resolve().parent
OUT=Path('/home/lp531/smolvla_training/act_cup_overfit10_20261021')
SCENES=Path('/home/lp531/smolvla_eval/act_cup_train_closedloop10_20261021/selected_scenes.json')
sys.path.insert(0,str(BASE))
from cache_data import CupDataset
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import ImageCropResizeProcessorStep

def save(name,value):
 p=R/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,ensure_ascii=False,indent=2));t.replace(p)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=10000);args=parser.parse_args()
 with (R/'.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (R/'completed.json').exists():print((R/'completed.json').read_text());return
  random.seed(20261013);np.random.seed(20261013);torch.manual_seed(20261013);torch.cuda.manual_seed_all(20261013)
  torch.set_num_threads(8);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=True
  assert torch.cuda.is_available(),'GPU required for this training diagnostic'
  plan=json.loads(SCENES.read_text());checkpoint=Path(plan['checkpoint']);episodes=[s['training_episode'] for s in plan['scenes']]
  original_hash=hashlib.sha256((checkpoint/'model.safetensors').read_bytes()).hexdigest();assert original_hash==plan['policy_sha256']
  manifest=json.loads((BASE/'split_manifest.json').read_text());records=manifest['source_records']
  assert len(set(episodes))==10 and all(records[e]['split']=='train' and records[e]['source']=='original' for e in episodes)
  for s in plan['scenes']:assert hashlib.sha256((Path(s['source_path'])/'frames.jsonl').read_bytes()).hexdigest()==s['frames_sha256']
  dataset=CupDataset(episodes)
  with np.load(BASE/'sampling_arrays.npz') as a:
   absolute=np.flatnonzero(a['eligible_train']&np.isin(a['episode'],episodes));relative=[dataset.absolute_to_relative_idx[int(i)] for i in absolute];weights=a['weights'][absolute].copy();weights/=weights.sum()
  assert relative and all(not dataset[i]['action_is_pad'].any() for i in [relative[0],relative[-1]])
  fixed=[];anchors=[]
  for ep in episodes:
   r=records[ep];eligible=[i for i in range(r['frames']-15)]
   indices=set(np.linspace(0,len(eligible)-1,32,dtype=int).tolist())
   for event in [r['close_frame'],r['open_frame']]:
    if event is not None:indices.update(event+off for off in [-10,-5,-1,0,1,5,10])
   for frame in sorted(indices):
    if 0<=frame<r['frames']-15:fixed.append(dataset.absolute_to_relative_idx[r['dataset_from_index']+frame]);anchors.append(dict(episode=ep,frame=frame))
  eval_loader=DataLoader(Subset(dataset,fixed),batch_size=16,num_workers=0)
  sampler=WeightedRandomSampler(torch.as_tensor(weights,dtype=torch.double),num_samples=args.steps*16,replacement=True,generator=torch.Generator().manual_seed(20261013))
  loader=DataLoader(Subset(dataset,relative),batch_size=16,sampler=sampler,num_workers=4,persistent_workers=True,prefetch_factor=2,pin_memory=True,multiprocessing_context='spawn')
  cfg=PreTrainedConfig.from_pretrained(str(checkpoint),local_files_only=True);cfg.device='cuda';cfg.pretrained_backbone_weights=None
  policy=ACTPolicy.from_pretrained(str(checkpoint),config=cfg,local_files_only=True,strict=True)
  pre,post=make_pre_post_processors(cfg,pretrained_path=str(checkpoint),preprocessor_overrides={'device_processor':{'device':'cuda'}})
  pre.steps=[s for s in pre.steps if not isinstance(s,ImageCropResizeProcessorStep)]
  optimizer=torch.optim.AdamW(policy.get_optim_params(),lr=1e-5,weight_decay=1e-4,betas=(.9,.999),eps=1e-8)
  recipe=dict(diagnostic_only=True,initial_checkpoint=str(checkpoint),original_sha256=original_hash,episodes=episodes,training_windows=len(relative),fit_check_windows=len(fixed),fit_check_anchors=anchors,fit_check_is_training_data=True,seed=20261013,max_updates=args.steps,minimum_updates_before_early_stop=5000,early_stop_rule='fixed-window normalized L1 <= .008 and max six-joint MAE <= .008rad',batch_size=16,lr=1e-5,backbone_lr=1e-5,weight_decay=1e-4,grad_clip_norm=1.,optimizer='fresh AdamW state; continue original 050000 weights',image_augmentation=False,normalization='original saved statistics unchanged',sampling='original key2/other1 window probabilities restricted and renormalized to ten episodes',architecture='unchanged ACT chunk16, dropout0.1, VAE and original KL weight',checkpoint_selection='lowest fixed-window inference L1; no simulator outcome used')
  save('recipe.json',recipe)
  history=[];best=float('inf');best_path=None;best_step=None;started=time.monotonic()

  def evaluate(step):
   policy.eval();joint_sum=np.zeros(6);grip_sum=0.;normalized_sum=0.;elements=0;gcount=0;correct=0;onsets=[]
   with torch.inference_mode():
    for batch in eval_loader:
     processed=pre(batch);normalized=policy.predict_action_chunk(processed);truth=processed['action'];n=truth.shape[0]*truth.shape[1]
     normalized_sum+=float((normalized-truth).abs().sum());elements+=truth.numel()
     pred=post(normalized).cpu().numpy();target=batch['action'].numpy();joint_sum+=np.abs(pred[:,:,:6]-target[:,:,:6]).sum(axis=(0,1));grip_sum+=float(np.abs(pred[:,:,-1]-target[:,:,-1]).sum());gcount+=n;correct+=int(((pred[:,:,-1]>=.6)==(target[:,:,-1]>=.6)).sum())
   # Explicit closure-onset check, using each episode's first closing-label observation.
   onset_batch=default_collate([dataset[dataset.absolute_to_relative_idx[records[e]['dataset_from_index']+records[e]['close_frame']]] for e in episodes]);processed=pre(onset_batch)
   with torch.inference_mode():onset_pred=post(policy.predict_action_chunk(processed)).cpu().numpy()
   for ep,values in zip(episodes,onset_pred):
    ids=np.flatnonzero(values[:,-1]>=.6);onsets.append(dict(episode=ep,first_close_step=int(ids[0]) if len(ids) else None,first_action_value=float(values[0,-1])))
   result=dict(step=step,normalized_l1=normalized_sum/elements,joint_mae_rad=(joint_sum/gcount).tolist(),gripper_mae=grip_sum/gcount,gripper_threshold_accuracy=correct/gcount,closure_onsets=onsets,elapsed_s=time.monotonic()-started)
   history.append(result);save('fit_history.json',history);print('FIT '+json.dumps({k:v for k,v in result.items() if k!='closure_onsets'}),flush=True);policy.train();return result

  def checkpoint_save(step):
   destination=OUT/'checkpoints'/f'{step:06d}'/'pretrained_model';destination.mkdir(parents=True,exist_ok=False);policy.save_pretrained(destination)
   for p in checkpoint.glob('policy_*'):shutil.copy2(p,destination/p.name)
   (destination/'diagnostic_training_recipe.json').write_text(json.dumps(recipe,indent=2));return destination

  baseline=evaluate(0);save('baseline_fit.json',baseline)
  batch_time=time.monotonic();losses=[]
  for step,batch in enumerate(loader,1):
   policy.train();processed=pre(batch);optimizer.zero_grad(set_to_none=True);loss,components=policy(processed)
   if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
   loss.backward();grad=torch.nn.utils.clip_grad_norm_(policy.parameters(),1.0);optimizer.step();losses.append(float(loss.detach()))
   if step==1:
    assert {int(s['step'].item()) for s in optimizer.state.values() if 'step' in s}=={1}
   if step==1 or step%100==0:
    status=dict(stage='training',optimizer_updates=step,target_updates=args.steps,loss_mean=sum(losses)/len(losses),lr=optimizer.param_groups[0]['lr'],grad_norm=float(grad),elapsed_s=time.monotonic()-started)
    save('status.json',status);print(json.dumps(status),flush=True);losses=[]
   if step%1000==0 or step==args.steps:
    result=evaluate(step)
    if result['normalized_l1']<best:
     best=result['normalized_l1'];best_path=checkpoint_save(step);best_step=step;save('best_checkpoint.json',dict(step=step,checkpoint=str(best_path),fit=result))
    if step>=5000 and result['normalized_l1']<=.008 and max(result['joint_mae_rad'])<=.008:break
  assert best_path is not None and hashlib.sha256((checkpoint/'model.safetensors').read_bytes()).hexdigest()==original_hash
  reloaded=ACTPolicy.from_pretrained(str(best_path),local_files_only=True,strict=True).eval()
  # Deployment processors must retain crop/resize and exact original normalization state.
  reload_pre,reload_post=make_pre_post_processors(reloaded.config,pretrained_path=str(best_path))
  raw_dataset=CupDataset([episodes[0]],raw_images=True);raw=default_collate([raw_dataset[0]])
  for key in raw_dataset.meta.camera_keys:raw[key]=raw[key].float()/255
  with torch.inference_mode():pred=reload_post(reloaded.predict_action_chunk(reload_pre(raw)))
  assert pred.shape==(1,16,7) and torch.isfinite(pred).all()
  result=dict(completed=True,diagnostic_only=True,optimizer_updates=step,best_step=best_step,best_checkpoint=str(best_path),best_model_sha256=hashlib.sha256((best_path/'model.safetensors').read_bytes()).hexdigest(),baseline_normalized_l1=baseline['normalized_l1'],best_normalized_l1=best,relative_fit_improvement=1-best/baseline['normalized_l1'],original_model_unchanged=True,deployment_reload_verified=True,elapsed_s=time.monotonic()-started)
  save('completed.json',result);save('status.json',dict(stage='completed',**result));print('COMPLETE '+json.dumps(result),flush=True)
if __name__=='__main__':main()
