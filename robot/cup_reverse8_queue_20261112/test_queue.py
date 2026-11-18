import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import run_extension as executor

ROOT=Path(__file__).resolve().parent

class QueueContracts(unittest.TestCase):
 def test_missing_prior_audit_never_launches(self):
  with tempfile.TemporaryDirectory() as temp:
   previous=Path(temp)/'previous';original=Path(temp)/'original';previous.mkdir();original.mkdir();calls=[]
   with self.assertRaisesRegex(RuntimeError,'not passed'):
    executor.after_previous({},previous,original,lambda cmd:calls.append(cmd))
   self.assertEqual(calls,[])
 def test_finish_is_idempotent(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);previous=root/'previous';original=root/'original';logs=root/'logs'
   for p in [previous,original,logs]:p.mkdir()
   for p in [previous,original]:(p/'completion_verified.json').write_text('{"verified":true}')
   plan=dict(logs=str(logs),source_pins={},command=['fake']);calls=[]
   def run(command):calls.append(command);(logs/'completion_verified.json').write_text('{"verified":true}')
   with patch.object(executor.subprocess,'run') as proc,patch.object(executor.subprocess,'check_output',return_value='24000'):
    proc.return_value.returncode=1
    executor.after_previous(plan,previous,original,run);executor.after_previous(plan,previous,original,run)
   self.assertEqual(len(calls),1)
 def test_all_seven_bad_sources_rejected_without_changing_raw(self):
  spec=importlib.util.spec_from_file_location('cup7gate',ROOT/'collector/quality_gate.py');gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
  plan=json.loads((ROOT/'plan.json').read_text())
  for t in plan['tasks']:
   source=Path(t['rerecord_source']);before={p.name:executor.digest(p) for p in source.glob('*.json')};frames=executor.digest(source/'frames.jsonl')
   with tempfile.TemporaryDirectory() as temp:
    p=Path(temp)/'episode';p.mkdir()
    for name in ['frames.jsonl','episode_start.json','episode_summary.json','correction.json','gripper_labels.json','training_mask.json']:
     if (source/name).exists():(p/name).symlink_to(source/name)
    with self.assertRaisesRegex(ValueError,'Retake rejected'):gate.screen(p)
    result=json.loads((p/'retake_quality.json').read_text());self.assertGreaterEqual(result['max_reverse_excursion_m'],.08)
   self.assertEqual(frames,executor.digest(source/'frames.jsonl'))
   self.assertEqual(before,{p.name:executor.digest(p) for p in source.glob('*.json')})
 def test_tasks_keep_original_correction_types(self):
  plan=json.loads((ROOT/'plan.json').read_text());self.assertEqual(len(plan['tasks']),7)
  for t in plan['tasks']:
   original=json.loads((Path(t['rerecord_source'])/'correction.json').read_text())['task']
   for key in ['kind','grid','offset_approach_m','loaded_offset_world_m','requested_shift_world_m']:
    self.assertEqual(t[key],original[key])

if __name__=='__main__':unittest.main()
