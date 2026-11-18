import copy,json,tempfile,unittest
from pathlib import Path
from core import make_tasks,completed_ids
from preflight import routes

class Revision(unittest.TestCase):
 def test_preparation_reaches_clearance_before_entering_grasp(self):
  task=make_tasks()[0]
  r=next(routes([.5,0,.15],[0,0,0,1],[.5,0,0],task))
  self.assertFalse(r['preparation'][0]['linear'])
  self.assertGreater(r['preparation'][0]['position'][2],r['bad_position'][2])
  self.assertEqual(len({tuple(s['position']) for s in r['preparation']}),2)
 def test_lineage_preserves_valid_saved_clips_without_rewriting_them(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);ep=root/'episode_000000';ep.mkdir()
   (root/'correction_plan.json').write_text(json.dumps({'hash':'new','accepted_plan_hashes':['old']}))
   (ep/'correction.json').write_text(json.dumps({'accepted':True,'plan_hash':'old','task_id':'a'}))
   (ep/'episode_summary.json').write_text(json.dumps({'fragment_success':True,'physical_grasp_ok':True,'target_object':'peach'}))
   (ep/'frames.jsonl').write_text('{}\n')
   self.assertEqual(completed_ids(root,'new'),{'a'})
   self.assertEqual(completed_ids(root,'unrelated'),set())
 def test_revision_changes_only_missing_tasks_and_preserves_split(self):
  from revise_plan import revised_plan
  p={'hash':'old','tasks':make_tasks(),'target_object':'peach'}
  saved={p['tasks'][0]['id']};before=copy.deepcopy(p)
  new=revised_plan(p,saved)
  self.assertEqual(new['tasks'][0],before['tasks'][0])
  for t in new['tasks'][1:]:
   self.assertFalse(t['grid'][1]==4 and t['grid'][0]<3)
   self.assertEqual(t['split']=='validation',tuple(t['grid']) in {(1,1),(2,3)})
   self.assertLessEqual(max(map(abs,t['rotation_tool_deg'])),2.5)
   self.assertLessEqual(abs(t['offset_approach_m'][1]),.005)
  self.assertEqual(p,before)
if __name__=='__main__':unittest.main()
