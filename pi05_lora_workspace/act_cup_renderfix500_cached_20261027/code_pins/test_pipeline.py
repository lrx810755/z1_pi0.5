"""Behavior tests: interventions, repeated gripper intent, spatial leakage and sampling."""
import unittest
import numpy as np

class PipelineTests(unittest.TestCase):
 def test_horizon_rejects_any_intervention_and_short_tail(self):
  from rules import valid_anchors
  expert=np.ones(40,bool);expert[20:22]=False
  expected=np.array([i+16<=40 and expert[i:i+16].all() for i in range(40)])
  np.testing.assert_array_equal(valid_anchors(expert),expected)
  self.assertTrue(expected[4]);self.assertFalse(expected[5]);self.assertTrue(expected[22]);self.assertFalse(expected[25])
 def test_declared_anchor_mask_must_match_expert_mask(self):
  from rules import correction_labels
  with self.assertRaisesRegex(ValueError,'horizon'):
   correction_labels({'labels':[0]*32},{'expert_frame_mask':[True]*32,'valid_action_horizon16_anchors':[True]*32},32)
 def test_multiple_close_open_regrasp_intents_preserved(self):
  from rules import correction_labels,valid_anchors
  labels=[0]*5+[1]*5+[0]*5+[1]*5+[0]*20
  mask=np.ones(40,bool);mask[9]=False
  y,e,v=correction_labels({'labels':labels},{'expert_frame_mask':mask.tolist(),'valid_action_horizon16_anchors':valid_anchors(mask).tolist()},40)
  np.testing.assert_array_equal(y,labels);np.testing.assert_array_equal(e,mask)
  self.assertEqual(np.flatnonzero(np.diff(y)).tolist(),[4,9,14,19])
  with self.assertRaisesRegex(ValueError,'binary'):
   correction_labels({'labels':[.5]*40},{'expert_frame_mask':mask.tolist(),'valid_action_horizon16_anchors':v.tolist()},40)
 def test_spatial_holdout_and_samegrid_reference(self):
  from rules import split_for_grid,validate_reference
  self.assertEqual(split_for_grid([1,1]),'validation');self.assertEqual(split_for_grid([2,3]),'validation')
  self.assertEqual(split_for_grid([0,0]),'train')
  with self.assertRaisesRegex(ValueError,'reference'):
   validate_reference([1,1],[0,0],1,1)
  validate_reference([1,1],[1,1],17,17)
 def test_source_masses_and_modest_transition_emphasis(self):
  from weights import source_balanced_weights
  eligible=np.array([1,1,0,1,1,0],bool);key=np.array([1,0,1,1,0,0],bool);source=np.array([0,0,0,1,1,1])
  w=source_balanced_weights(eligible,key,source)
  self.assertAlmostEqual(w[source==0].sum(),.75);self.assertAlmostEqual(w[source==1].sum(),.25)
  self.assertAlmostEqual(w[0]/w[1],1.5);self.assertEqual(w[2],0);self.assertEqual(w[5],0)
 def test_gate_rejects_incomplete_or_wrong_categories_hash(self):
  from rules import validate_collection,EXPECTED_CATEGORIES,PLAN_HASH
  records=[{'kind':k,'plan_hash':PLAN_HASH,'task_id':f'{k}_{i}'} for k,n in EXPECTED_CATEGORIES.items() for i in range(n)]
  validate_collection(400,records)
  with self.assertRaisesRegex(ValueError,'100'):validate_collection(400,records[:-1])
  records[0]['plan_hash']='bad'
  with self.assertRaisesRegex(ValueError,'hash'):validate_collection(400,records)
 def test_required_source_evidence_missing_or_empty_is_rejected(self):
  import tempfile
  from pathlib import Path
  from prepare import required_source_hashes
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)
   names=['frames.jsonl','episode_summary.json','episode_start.json','correction.json','gripper_labels.json','training_mask.json','physics_status.jsonl']
   for name in names:(root/name).write_text('{}\n')
   self.assertEqual(set(required_source_hashes(root,'correction')),set(names))
   for name in names:
    original=(root/name).read_bytes();(root/name).unlink()
    with self.subTest(name=name,damage='missing'),self.assertRaisesRegex(ValueError,'Missing or empty required source'):
     required_source_hashes(root,'correction')
    (root/name).touch()
    with self.subTest(name=name,damage='empty'),self.assertRaisesRegex(ValueError,'Missing or empty required source'):
     required_source_hashes(root,'correction')
    (root/name).write_bytes(original)
 def test_physics_stream_rejects_corruption_mismatched_count_and_unsafe_sample(self):
  import json,tempfile
  from pathlib import Path
  from prepare import validate_physics
  good=dict(simulation_time=1.25,status=dict(frozen=False,target_collisions_enabled=True,left_finger_touching_table=False,right_finger_touching_table=False))
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'physics_status.jsonl';path.write_text(json.dumps(good)+'\n');validate_physics(path,dict(dynamic_samples=1))
   for field,value in [('frozen',True),('target_collisions_enabled',False),('left_finger_touching_table',True),('right_finger_touching_table',True)]:
    bad=dict(good,status=dict(good['status']));bad['status'][field]=value;path.write_text(json.dumps(bad)+'\n')
    with self.subTest(field=field),self.assertRaisesRegex(ValueError,'physics'):validate_physics(path,dict(dynamic_samples=1))
   for content in ['', '{invalid}\n',json.dumps(dict(good,status=[]))+'\n',json.dumps(dict(simulation_time=1.25,status={'frozen':False}))+'\n',json.dumps(dict(good,simulation_time=float('nan')))+'\n']:
    path.write_text(content)
    with self.subTest(content=content),self.assertRaisesRegex(ValueError,'physics'):validate_physics(path,dict(dynamic_samples=1))
   path.write_text(json.dumps(good)+'\n')
   for count in [0,2,True]:
    with self.subTest(count=count),self.assertRaisesRegex(ValueError,'physics'):validate_physics(path,dict(dynamic_samples=count))
   path.write_text(json.dumps(good)+'\n'+json.dumps(dict(good,simulation_time=1.0))+'\n')
   with self.assertRaisesRegex(ValueError,'physics'):validate_physics(path,dict(dynamic_samples=2))
 def test_train_stats_exclude_val_and_interventions(self):
  from rules import feature_stats
  x=np.array([[1,2],[3,4],[999,999],[1000,1000]],np.float32)
  s=feature_stats(x,np.array([1,1,0,0],bool))
  np.testing.assert_array_equal(s['mean'],[2,3]);self.assertEqual(s['count'],[2])

if __name__=='__main__':unittest.main()
