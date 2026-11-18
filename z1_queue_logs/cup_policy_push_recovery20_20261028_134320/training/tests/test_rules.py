import copy, unittest
import numpy as np
try:
 import rules
except ModuleNotFoundError:
 rules=None

class TrainingRules(unittest.TestCase):
 def test_horizon30_excludes_policy_actions_and_episode_tail(self):
  self.assertIsNotNone(rules, 'training rules must be implemented')
  expert=np.ones(65,bool);expert[31]=False
  self.assertEqual(np.flatnonzero(rules.valid_anchors(expert)).tolist(),[0,1,32,33,34,35])
  self.assertFalse(rules.valid_anchors(np.ones(29,bool)).any())
 def test_weights_take_max_and_normalize_source_pools(self):
  self.assertIsNotNone(rules, 'training rules must be implemented')
  w=rules.source_balanced_weights(np.array([1,1,1,1,0],bool),np.array([1,0,1,0,0],bool),np.array([0,0,1,1,1]),np.array([0,0,1,0,1],bool))
  np.testing.assert_allclose(w,[.45,.30,.1875,.0625,0],atol=1e-15)
 def test_verified_count_gate_rejects_success_without_acceptance_and_duplicates(self):
  self.assertIsNotNone(rules, 'training rules must be implemented')
  records=[dict(accepted=True,recovery_complete=True,source_type='actual_policy_push_recovery',source_identity=f'model:{i}',grid=[0,0],seed=i) for i in range(20)]
  rules.validate_new_collection(records,20)
  rules.validate_new_collection(records[:16],16)
  for count in [15,19,21]:
   with self.assertRaises(ValueError):rules.validate_new_collection((records+records)[:count],20)
  bad=copy.deepcopy(records);bad[0]['accepted']=False;bad[0]['task_success']=True
  with self.assertRaises(ValueError):rules.validate_new_collection(bad,20)
  bad=copy.deepcopy(records);bad[0]['source_type']='synthetic_failure_informed'
  with self.assertRaises(ValueError):rules.validate_new_collection(bad,20)
  bad=copy.deepcopy(records);bad[1]['source_identity']=bad[0]['source_identity']
  with self.assertRaises(ValueError):rules.validate_new_collection(bad,20)
 def test_old400_100_gate_still_applies(self):
  self.assertIsNotNone(rules, 'training rules must be implemented')
  records=[dict(source='basic') for _ in range(400)]+[dict(source='correction') for _ in range(100)]
  rules.validate_old_records(records)
  with self.assertRaises(ValueError):rules.validate_old_records(records[:-1])
 def test_spatial_and_source_layout_grouping_and_seed_leak(self):
  self.assertIsNotNone(rules, 'training rules must be implemented')
  records=[dict(grid=[1,1],seed=11,source_frames_sha256='a',split='validation'),dict(grid=[0,0],seed=12,source_frames_sha256='b',split='train',source_grid=[0,0])]
  rules.validate_split(records)
  bad=copy.deepcopy(records);bad[1]['seed']=11
  with self.assertRaises(ValueError):rules.validate_split(bad)
  bad=copy.deepcopy(records);bad[1]['source_grid']=[1,1]
  with self.assertRaises(ValueError):rules.validate_split(bad)
  same_split=[dict(grid=[0,0],seed=42,source_frames_sha256='c',split='train'),dict(grid=[0,1],seed=42,source_frames_sha256='d',split='train')]
  with self.assertRaises(ValueError):rules.validate_split(same_split)
 def test_new_intervals_use_simulation_time_and_only_expert_frames(self):
  self.assertIsNotNone(rules, 'training rules must be implemented')
  rows=[dict(simulation_time=100+i*.1,physical_grasp_ok=False) for i in range(60)]
  meta=dict(source_type='actual_policy_push_recovery',recovery_intervals_s=[dict(start=101.,end=102.)])
  expert=np.ones(60,bool);expert[15]=False
  recovery,intervals=rules.recovery_mask(rows,meta,expert)
  self.assertEqual(np.flatnonzero(recovery).tolist(),[10,11,12,13,14,16,17,18,19,20])
  self.assertEqual(intervals[0]['origin'],'explicit_expert_recovery')
 def test_old_regrasp_and_loaded_recovery_are_preserved(self):
  self.assertIsNotNone(rules, 'training rules must be implemented')
  rows=[dict(simulation_time=100+i*.1,physical_grasp_ok=i>=30) for i in range(80)]
  meta=dict(task=dict(kind='regrasp'),recovery_stages=[dict(stage='recover_approach',start=101,end=101.4)],events=[dict(event='retry_opened',time=100.5),dict(event='gripper_intent',time=102,state='close'),dict(event='loaded_recovery_start',time=105),dict(event='loaded_recovery_complete',time=106)])
  recovery,intervals=rules.recovery_mask(rows,meta,np.ones(80,bool))
  self.assertTrue(recovery[5]);self.assertTrue(recovery[33]);self.assertFalse(recovery[34]);self.assertTrue(recovery[50]);self.assertTrue(recovery[60]);self.assertFalse(recovery[61])
  self.assertEqual({x['origin'] for x in intervals},{'recovery_stages','retry_opened','loaded_recovery'})

if __name__=='__main__':unittest.main()
