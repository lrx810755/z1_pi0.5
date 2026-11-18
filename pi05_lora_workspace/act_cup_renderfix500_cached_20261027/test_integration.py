"""Bounded CPU integration tests; no real dataset cache or GPU training."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

class IntegrationTests(unittest.TestCase):
 def test_sampler_filters_masks_and_resumes_exactly(self):
  import sampling
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);eligible=np.array([1,0,1,1,0,1,0,0],bool);w=np.array([.25,0,.25,.25,0,.25,0,0])
   np.savez(root/'sampling_arrays.npz',eligible_train=eligible,weights=w)
   with patch.object(sampling,'RUN',root):
    mapping={i:j for j,i in enumerate(range(4,8))}
    s=sampling.BalancedSampler([0,4],[4,8],episode_indices_to_use=[1],absolute_to_relative_idx=mapping,seed=42,shuffle=True)
    self.assertEqual(s.absolute_indices.tolist(),[5]);self.assertEqual(s.indices,[1])
    order=list(iter(s));s.load_state_dict({'epoch':0,'start_index':0});self.assertEqual(order,list(iter(s)))
    s=sampling.BalancedSampler([0],[8],seed=42,shuffle=True)
    order=list(iter(s));s.load_state_dict({'epoch':0,'start_index':2});self.assertEqual(order[2:],list(iter(s)))
    self.assertTrue(set(order)<={0,2,3,5})
 def test_real_native_jpeg_decode_is_pixel_exact_and_pins_source_bytes(self):
  import torch
  from PIL import Image
  from torchvision.transforms import functional as F
  import prepare,build_cache
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);source=root/'source';source.mkdir();run=root/'run';run.mkdir();data=run/'cache';data.mkdir();pin=run/'source_pins'/'episode_000000';pin.mkdir(parents=True)
   rows=[]
   for i in range(2):
    row={}
    for cam in prepare.CAMERAS:
     h=360 if cam.endswith('wrist') else 480
     # Structured color at all edges catches crop axis, padding and RGB/BGR mistakes.
     y,x=np.indices((h,640));pixels=np.stack([(x+i)%256,(y+3*i)%256,(x+y)%256],axis=-1).astype(np.uint8)
     p=source/f'{cam}_{i}.jpg';Image.fromarray(pixels).save(p,quality=93);row[cam]=p.name
    rows.append(row)
   content='\n'.join(json.dumps(r) for r in rows)+'\n';(source/'frames.jsonl').write_text(content);(pin/'frames.jsonl').write_text(content)
   record=dict(episode_index=0,source_path=str(source),frames=2,source_hashes={'frames.jsonl':hashlib.sha256(content.encode()).hexdigest()})
   with patch.object(prepare,'RUN',run),patch.object(build_cache,'RUN',run),patch.object(build_cache,'DATA',data):
    self.assertEqual(build_cache.job(record),(0,2))
    for cam in prepare.CAMERAS:
     a=np.load(data/cam/'episode_000000.npy',mmap_mode='r')
     with Image.open(source/rows[0][cam]) as image:raw=torch.from_numpy(np.array(image.convert('RGB'))).permute(2,0,1).float()/255
     expected=F.resize(F.crop(raw,*prepare.CROPS[cam]),[224,224]).clamp(0,1).numpy()
     np.testing.assert_array_equal(a[0],expected)
    # A cached camera must reject a changed JPEG rather than silently reusing pixels.
    (source/rows[0][prepare.CAMERAS[0]]).write_bytes(b'changed')
    with self.assertRaisesRegex(ValueError,'Source image changed'):build_cache.job(record)
 def test_full500_prepare_enforces_spatial_split_stats_and_corrected_intent(self):
  import prepare
  from rules import PLAN_HASH,EXPECTED_CATEGORIES,full_labels,valid_anchors,split_for_grid
  from types import SimpleNamespace
  with tempfile.TemporaryDirectory() as tmp:
   run=Path(tmp)/'run';run.mkdir();base=Path(tmp)/'basic';corr=Path(tmp)/'correction';base.mkdir();corr.mkdir();planpath=Path(tmp)/'plan.json'
   kinds=[k for k,n in EXPECTED_CATEGORIES.items() for _ in range(n)];tasks=[];rows0=[]
   command=[-.9]*4+[-.85,-.7]+[0.]*18+[-.1,-.4,-.7,-.9]+[-.9]*4
   for i,value in enumerate(command):rows0.append(dict(frame_index=i,action=[0.]*6+[value],**{'observation.state':[0.]*6+[value]},target_object='cup',task='pick cup',**{cam:'image.jpg' for cam in prepare.CAMERAS}))
   intent,_,_=full_labels(rows0)
   for source,count in [(base,400),(corr,100)]:
    for idx in range(count):
     p=source/f'episode_{idx:06d}';p.mkdir();(p/'image.jpg').write_bytes(b'JPEG placeholder: preparation checks existence; decode tested separately');cell=idx//(20 if source==base else 5);grid=[cell%5,cell//5];seed=cell*20+(idx%20 if source==base else 0)
     rows=[dict(r,simulation_time=float(idx+(0 if source==base else 1000))) for r in rows0];(p/'frames.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
     summary=dict(target_object='cup',frame_count=32,dropped_frames=0,scene_randomization=dict(target_grid=grid,seed=seed),physical_grasp_ok=True,task_success=True,release_sequence_complete=True,placement_stable=True)
     (p/'episode_summary.json').write_text(json.dumps(summary));(p/'episode_start.json').write_text('{}')
     if source==corr:
      tid=f'correction_{idx}';task=dict(id=tid,kind=kinds[idx],grid=grid,split=split_for_grid(grid),reference_basic_episode=str(base/f'episode_{cell*20:06d}'));tasks.append(task)
      final={k:True for k in ['task_success','physical_grasp_ok','released','placement_stable','release_sequence_complete','target_collisions_enabled']};final.update(frozen=False,holding_observed=False)
      meta=dict(task=task,task_id=tid,plan_hash=PLAN_HASH,accepted=True,recovery_complete=True,table_contact=False,dynamic_violations=[],final_status=final,measured_preclose_roll_deg=-90,split=task['split'],dynamic_samples=2)
      physics=[dict(simulation_time=float(idx)+j/30,status=dict(frozen=False,target_collisions_enabled=True,left_finger_touching_table=False,right_finger_touching_table=False)) for j in range(2)]
      (p/'physics_status.jsonl').write_text('\n'.join(json.dumps(x) for x in physics)+'\n')
      expert=np.ones(32,bool);expert[12]=False
      for name,value in [('correction.json',meta),('gripper_labels.json',dict(labels=intent.tolist())),('training_mask.json',dict(expert_frame_mask=expert.tolist(),valid_action_horizon16_anchors=valid_anchors(expert).tolist()))]: (p/name).write_text(json.dumps(value))
   planpath.write_text(json.dumps(dict(hash=PLAN_HASH,tasks=tasks)))
   with patch.multiple(prepare,RUN=run,DATA=run/'cache',BASE=base,CORR=corr,PLAN=planpath),patch('sys.argv',['prepare.py']),patch('builtins.print'):
    prepare.main()
   split=json.loads((run/'split_manifest.json').read_text());self.assertEqual((len(split['train_episodes']),len(split['validation_episodes'])),(450,50))
   with np.load(run/'sampling_arrays.npz') as a:
    self.assertFalse(np.isin(a['episode'][a['weights']>0],split['validation_episodes']).any())
    self.assertAlmostEqual(a['weights'][a['source_pool']==1].sum(),.25)
   stats=json.loads((run/'train_stats.json').read_text());self.assertEqual(stats['action']['count'],[360*32+90*31])
   self.assertTrue((run/'source_pins'/'collection_plan.json').is_file())
   # Exercise damaged evidence through prepare itself, after the complete fixture passed.
   damaged=[(base/'episode_000000'/'episode_start.json',None,'required source'),(corr/'episode_000000'/'physics_status.jsonl',None,'required source'),(corr/'episode_000000'/'physics_status.jsonl',b'{broken}\n','physics')]
   for evidence,replacement,message in damaged:
    original=evidence.read_bytes()
    try:
     if replacement is None:evidence.unlink()
     else:evidence.write_bytes(replacement)
     with patch.multiple(prepare,RUN=run,DATA=run/'cache',BASE=base,CORR=corr,PLAN=planpath),patch('sys.argv',['prepare.py','--audit-only']),patch('builtins.print'):
      with self.subTest(evidence=evidence.name,replacement=replacement),self.assertRaisesRegex(ValueError,message):prepare.main()
    finally:evidence.write_bytes(original)
 def test_validation_evidence_survives_trainer_logging_setup(self):
  import logging
  from types import SimpleNamespace
  import train,prepare
  import lerobot.scripts.lerobot_train as trainer
  names=['make_train_eval_datasets','make_dataloaders','make_pre_post_processors','_preprocess_dataset_batch','update_policy','save_checkpoint','init_logging']
  originals={name:getattr(trainer,name) for name in names};handlers=list(logging.getLogger().handlers)
  try:
   with tempfile.TemporaryDirectory() as tmp:
    run=Path(tmp)
    with patch.object(train,'RUN',run),patch.object(prepare,'RUN',run):
     train.install_hooks(SimpleNamespace(resume=False),False);trainer.init_logging()
     logging.info('step 5000: eval_loss=1.2345')
     self.assertTrue((run/'validation_latest.json').is_file())
     self.assertEqual(json.loads((run/'validation_latest.json').read_text()),dict(optimizer_updates=5000,heldout_loss=1.2345))
  finally:
   for name,value in originals.items():setattr(trainer,name,value)
   logging.getLogger().handlers=handlers
 def test_config_is_fresh_act_and_scaled_without_starting_gpu(self):
  import train
  with tempfile.TemporaryDirectory() as tmp:
   run=Path(tmp);(run/'split_manifest.json').write_text(json.dumps(dict(train_episodes=list(range(450)),validation_episodes=list(range(450,500)),source_records=[{}]*500)))
   with patch.object(train,'RUN',run):cfg=train.make_config(False)
   self.assertEqual((cfg.steps,cfg.batch_size,cfg.policy.chunk_size,cfg.policy.n_action_steps),(90000,32,16,5))
   self.assertFalse(cfg.resume);self.assertIsNone(cfg.policy.pretrained_path)
   self.assertEqual(cfg.policy.pretrained_backbone_weights,'ResNet18_Weights.IMAGENET1K_V1')
   self.assertEqual(cfg.num_workers,2);self.assertEqual(cfg.accelerator.gradient_accumulation.steps,1)

if __name__=='__main__':unittest.main()
