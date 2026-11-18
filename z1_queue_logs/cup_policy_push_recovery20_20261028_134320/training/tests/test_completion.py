import json,tempfile,unittest
from pathlib import Path
from prepare import completion_gate

class AcceptanceGate(unittest.TestCase):
 def test_missing_or_unverified_completion_cannot_admit_sources(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp)
   with self.assertRaises(FileNotFoundError):completion_gate(root)
   (root/'completion_verified.json').write_text(json.dumps(dict(passed=False,source_type='actual_policy_push_recovery',accepted_count=20,episodes=[])))
   with self.assertRaises(ValueError):completion_gate(root)
 def test_verified_list_must_match_all_episode_dirs(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);episodes=[]
   for i in range(19):
    path=root/f'episode_{i:06d}';path.mkdir();episodes.append(str(path))
    (path/'correction.json').write_text(json.dumps(dict(accepted=True,recovery_complete=True,source_type='actual_policy_push_recovery',source_identity=f'model:episode:{i}')))
   evidence=dict(passed=True,source_type='actual_policy_push_recovery',accepted_count=19,episodes=episodes)
   (root/'completion_verified.json').write_text(json.dumps(evidence));paths,_=completion_gate(root);self.assertEqual(len(paths),19)
   (root/'episode_000020').mkdir()
   with self.assertRaises(ValueError):completion_gate(root)

class FrozenPreparation(unittest.TestCase):
 def test_prepared_data_rejects_changed_acceptance_or_source(self):
  import prepare
  self.assertTrue(hasattr(prepare,'assert_prepared'),'Preparation needs immutable reuse check')
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);new=root/'new';new.mkdir();source=new/'episode_000000';source.mkdir();(source/'frames.jsonl').write_text('original')
   (new/'completion_verified.json').write_text('{}')
   manifest=dict(completion_evidence_sha256=prepare.sha(new/'completion_verified.json'),source_records=[dict(source_path=str(source),source_hashes={'frames.jsonl':prepare.sha(source/'frames.jsonl')})])
   (root/'split_manifest.json').write_text(json.dumps(manifest));(root/'preparation_complete.json').write_text(json.dumps(dict(manifest_sha256=prepare.sha(root/'split_manifest.json'))))
   original=prepare.RUN,prepare.NEW;prepare.RUN=root;prepare.NEW=new
   try:
    prepare.assert_prepared();(source/'frames.jsonl').write_text('changed')
    with self.assertRaises(ValueError):prepare.assert_prepared()
    (source/'frames.jsonl').write_text('original');(new/'completion_verified.json').write_text('{"passed":false}')
    with self.assertRaises(ValueError):prepare.assert_prepared()
   finally:prepare.RUN,prepare.NEW=original
