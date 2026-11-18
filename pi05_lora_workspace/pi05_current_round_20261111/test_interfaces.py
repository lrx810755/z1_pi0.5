import unittest,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from rules import valid_anchors
from sampling import draw_block,BalancedSampler
from pipeline import resolve
from config import make_config,ACCUMULATION

class Contracts(unittest.TestCase):
 def test_boundary_and_intervention(self):
  expert=np.ones(95,bool);expert[40:45]=False
  valid=valid_anchors(expert,30)
  self.assertEqual(np.flatnonzero(valid).tolist(),list(range(11))+list(range(45,66)))
  self.assertFalse(valid_anchors(np.ones(29,bool),30).any())
 def test_every_update_and_pair(self):
  pools=[np.arange(i*100,(i+1)*100) for i in range(8)]
  probs=[np.ones(100)/100 for _ in pools];rng=np.random.default_rng(1)
  for _ in range(200):
   sample=draw_block(rng,pools,probs);counts=np.bincount(sample//100,minlength=8)
   self.assertEqual(counts.tolist(),[3,1]*4)
   pairs=(sample//200).reshape(8,2);self.assertTrue(np.all(pairs[:,0]!=pairs[:,1]))
   self.assertEqual(len(set(sample)),16)
 def test_sampler_resume_and_epochs(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);pool=np.repeat(np.arange(8),40);n=len(pool)
   np.savez(root/'sampling_arrays.npz',eligible_train=np.ones(n,bool),source_pool=pool,weights=np.ones(n))
   with patch('sampling.RUN',root):
    s=BalancedSampler([0],[n],seed=25);whole=list(s._iter_epoch(3,0))
    self.assertEqual(list(s._iter_epoch(3,48)),whole[48:])
    self.assertNotEqual(list(s._iter_epoch(4,0)),whole)
    self.assertEqual(len(whole)%16,0)
 def test_replacement_chain_and_cycle(self):
  self.assertEqual(resolve('/a',{'/a':'/b','/b':'/c'}),'/c')
  with self.assertRaises(ValueError):resolve('/a',{'/a':'/b','/b':'/a'})
 def test_optimizer_units(self):
  cfg=make_config(validate=False)
  self.assertEqual((cfg.batch_size,ACCUMULATION,cfg.steps//ACCUMULATION,cfg.save_freq//ACCUMULATION),(2,8,100000,5000))
  self.assertEqual(cfg.policy.scheduler_warmup_steps//ACCUMULATION,2500)
  self.assertEqual(cfg.eval_steps,0);self.assertEqual(cfg.dataset.eval_split,0)
  self.assertEqual(cfg.policy.chunk_size,30)


class ConversionIntegration(unittest.TestCase):
 def test_raw_to_arrays_and_rgb_loader(self):
  import json
  from PIL import Image
  import dataset_prepare as prep
  import dataset as loader
  import pipeline
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);sources=[]
   for obj in prep.OBJECTS:
    for correction in [False,True]:
     ep=root/(obj+str(correction));ep.mkdir();n=40
     for cam in ['wrist','global','global_right']:Image.new('RGB',(640,360 if cam=='wrist' else 480),(50,80,100)).save(ep/(cam+'.jpg'))
     rawobj='beverage_bottle' if obj=='bottle' else obj
     task='Physically grasp '+obj+' and place it in the box.'
     rows=[]
     for i in range(n):
      row=dict(frame_index=i,simulation_time=i/10,action_stamp=i/10,target_object=rawobj,task=task,phase='READY',action=[0]*6+[(-.9 if i<19 or i>=35 else -.88 if i==19 else -.80 if i==20 else -.5)])
      row['observation.state']=[0]*6+[-.89]
      row.update(wrist_image_stamp=i/10,global_image_stamp=i/10,global_right_image_stamp=i/10,joint_state_source_stamps=[i/10,i/10],sync_tolerance_s=.045)
      row.update({c:c.rsplit('.',1)[1]+'.jpg' for c in prep.CAMERAS});rows.append(row)
     (ep/'frames.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
     summary=dict(fps=10,frame_count=n,dropped_frames=0,target_object=rawobj,task=task,task_success=True,scene_randomization=dict(target_grid=[0,0],seed=1))
     (ep/'episode_summary.json').write_text(json.dumps(summary));(ep/'episode_start.json').write_text('{}')
     if correction:
      (ep/'correction.json').write_text(json.dumps(dict(accepted=True,task=dict(kind='approach'),events=[dict(event='gripper_intent',time=1.9,state='close'),dict(event='gripper_intent',time=3.5,state='open')])))
      (ep/'gripper_labels.json').write_text(json.dumps(dict(labels=[0]*19+[1]*16+[0]*5)))
     sources.append(dict(path=str(ep),frames_sha256=prep.sha(ep/'frames.jsonl')))
   frozen=dict(sources=sources,input_pins={},explicit_exclusions=[])
   (root/'sources_frozen.json').write_text(json.dumps(frozen))
   with patch.object(prep,'RUN',root),patch.object(pipeline,'RUN',root),patch.object(loader,'RUN',root):
    prep.main()
    split=json.loads((root/'split_manifest.json').read_text());self.assertEqual(len(split['source_records']),8)
    self.assertEqual(split['validation_episodes'],[])
    ds=loader.CupDataset(list(range(8)));item=ds[0]
    self.assertEqual(tuple(item['action'].shape),(30,7));self.assertEqual(tuple(item[prep.CAMERAS[0]].shape),(3,224,224))
    self.assertIn('place it in the box',item['task']);self.assertAlmostEqual(float(item['observation.state'][-1]),-.89,places=5)
    self.assertEqual(set(item['action'][:,-1].tolist()),{0.,1.})
    with np.load(root/'sampling_arrays.npz') as a:
     self.assertEqual(int(a['eligible_train'].sum()),8*11)
     for ep in range(8):self.assertFalse(a['eligible_train'][ep*40+11:(ep+1)*40].any())
    with self.assertRaises(ValueError):ds[11]
    (Path(sources[0]['path'])/'episode_summary.json').write_text('{}')
    with self.assertRaises(ValueError):prep.assert_pins(split['source_records'])

if __name__=='__main__':unittest.main()
