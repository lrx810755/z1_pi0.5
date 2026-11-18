import copy,json,unittest
from pathlib import Path
from layout_fallback import selected_task,layout_is_allowed
class LayoutTests(unittest.TestCase):
 def setUp(self):
  plan=json.loads(Path(__file__).with_name('plan.json').read_text());self.stage=next(s for s in plan['stages'] if s['id']=='bottle_basic');self.task=next(t for t in self.stage['tasks'] if t['id'] in self.stage['layout_fallbacks'])
 def test_preserves_source_identity_and_grid(self):
  chosen=selected_task(self.task,self.stage,1)
  self.assertEqual(chosen['source'],self.task['source']);self.assertEqual(chosen['source_frames_sha256'],self.task['source_frames_sha256']);self.assertEqual(chosen['grid'],self.task['grid']);self.assertEqual(chosen['original_seed'],self.task['seed']);self.assertNotEqual(chosen['seed'],self.task['seed'])
 def test_rejects_unregistered_seed_or_cell(self):
  t=selected_task(self.task,self.stage,1);s=dict(scene_randomization=t['scene']);p={k:t.get(k) for k in ['seed','original_seed','layout_reference','layout_reference_frames_sha256']}
  self.assertTrue(layout_is_allowed(self.task,self.stage,s,p));p['seed']=-1;self.assertFalse(layout_is_allowed(self.task,self.stage,s,p))
  p['seed']=t['seed'];s=copy.deepcopy(s);s['scene_randomization']['target_grid']=[4,3];self.assertFalse(layout_is_allowed(self.task,self.stage,s,p))
 def test_keeps_all_15_previous_saves_valid(self):
  tasks={t['id']:t for t in self.stage['tasks']};count=0
  for path in Path(self.stage['data']).glob('episode_*'):
   proof=json.loads((path/'rerecord_provenance.json').read_text());summary=json.loads((path/'episode_summary.json').read_text())
   self.assertTrue(layout_is_allowed(tasks[proof['task_id']],self.stage,summary,proof));count+=1
  self.assertGreaterEqual(count,15)
if __name__=='__main__':unittest.main()
