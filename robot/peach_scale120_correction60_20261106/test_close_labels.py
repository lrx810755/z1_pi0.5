import json,math,unittest
from pathlib import Path
from fragment_core import partial_labels


def frames(actions):
 return [dict(action=[0.]*6+[a],simulation_time=i*.1,action_stamp=i*.1) for i,a in enumerate(actions)]

class Labels(unittest.TestCase):
 def test_opening_settle_before_explicit_close_is_not_closing(self):
  a=[-.9]*3+[-.884,-.887,-.891,-.897,-.9,-.9]+[-.899,-.894,-.89,-.885,-.879,-.85,-.5]
  labels,onset,_=partial_labels(frames(a),close_command_time=.9)
  self.assertEqual(onset,10);self.assertEqual(labels[:10],[0]*10);self.assertTrue(all(labels[10:]))
 def test_slow_close_does_not_count_as_reopening(self):
  a=[-.9,-.9,-.898,-.894,-.892,-.889,-.885,-.879,-.85]
  _,onset,_=partial_labels(frames(a),close_command_time=.2)
  self.assertEqual(onset,3)
 def test_explicit_partial_reopen_is_rejected(self):
  with self.assertRaisesRegex(ValueError,'unexpected reopening'):
   partial_labels(frames([-.9,-.85,-.5,-.6]),close_command_time=.1,reopen_command_times=[.3])
 def test_actual_return_to_open_after_closing_is_rejected(self):
  with self.assertRaisesRegex(ValueError,'unexpected reopening'):
   partial_labels(frames([-.9,-.894,-.88,-.6,-.9]),close_command_time=.1)
 def test_old_causal_command_sample_is_not_close_onset(self):
  r=frames([-.9,-.85,-.7]);r[1]['simulation_time']=2.;r[1]['action_stamp']=.05;r[2]['simulation_time']=2.1;r[2]['action_stamp']=.2
  _,onset,_=partial_labels(r,close_command_time=.1)
  self.assertEqual(onset,2)
 def test_nonfinite_action_rejected(self):
  with self.assertRaises(ValueError):partial_labels(frames([-.9,float('nan'),-.5]),close_command_time=.1)
 def test_real_failed_sequences_ignore_initial_opening(self):
  files=sorted(Path('/home/lp531/smolvla_data/apple_scale110_correction60_20261106/collection_logs').glob('*/*.label_failure.json'))[-12:]
  self.assertTrue(files)
  for p in files:
   d=json.loads(p.read_text());a=d['actions'];ts=d['simulation_times']
   # These traces finish a real open-to-close grasp after the earlier opening ramp.
   last_open=max(i for i,x in enumerate(a) if x<=-.8999)
   if last_open==len(a)-1:continue
   expected=next(i for i in range(last_open+1,len(a)) if a[i]>-.895)
   r=[dict(action=[0]*6+[x],simulation_time=t) for x,t in zip(a,ts)]
   _,onset,_=partial_labels(r,close_command_time=ts[last_open])
   self.assertEqual(onset,expected,str(p))
if __name__=='__main__':unittest.main()
