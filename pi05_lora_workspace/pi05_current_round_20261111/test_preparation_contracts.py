import json,tempfile,unittest
from pathlib import Path
from quality_decisions import accepted_reverse
from label_rules import full_labels
from data_contracts import validate_timeline

class LabelAndRetentionContracts(unittest.TestCase):
 def test_intent_starts_with_command_ramp_and_opens_before_mechanical_open(self):
  values=[-.9]*5+[-.88,-.82,-.76,-.70]+[-.67]*8+[-.76,-.80,-.85,-.9]+[-.9]*5
  labels,close,release=full_labels([dict(action=[0]*6+[x]) for x in values])
  self.assertEqual(close,5);self.assertEqual(release,17);self.assertEqual(labels[5],1);self.assertEqual(labels[17],0)
 def test_user_retention_requires_exact_episode_hash(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);(root/'quality_waivers.json').write_text(json.dumps(dict(episodes=[dict(path='/example/episode',frames_sha256='exact',decision='include_without_rerecord')])))
   self.assertTrue(accepted_reverse(root,'/example/episode','exact'))
   self.assertFalse(accepted_reverse(root,'/example/episode','changed'));self.assertFalse(accepted_reverse(root,'/other/episode','exact'))
 def test_sensor_timeline_rejects_reused_rgb_and_future_command(self):
  def make():return [dict(simulation_time=i/10,action_stamp=i/10,joint_state_source_stamps=[i/10,i/10],sync_tolerance_s=.045,wrist_image_stamp=i/10,global_image_stamp=i/10,global_right_image_stamp=i/10) for i in range(4)]
  rows=make();validate_timeline(rows);rows[2]['global_right_image_stamp']=rows[1]['global_right_image_stamp']
  with self.assertRaisesRegex(ValueError,'Repeated'):validate_timeline(rows)
  rows=make();rows[2]['action_stamp']=.3
  with self.assertRaisesRegex(ValueError,'noncausal'):validate_timeline(rows)

if __name__=='__main__':unittest.main()
