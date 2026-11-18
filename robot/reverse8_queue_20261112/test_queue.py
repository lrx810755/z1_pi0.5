import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from build_queue import deduplicate
import run_extension as executor
from quality_gate import screen


class QueueTests(unittest.TestCase):
    def test_dedup_both_alias_paths_and_copied_frames(self):
        existing=[dict(source='/tmp/source_a',source_frames_sha256='aaa')]
        candidates=[dict(physical_path='/tmp/source_a',source_frames_sha256='new_hash'),
            dict(physical_path='/tmp/copied_source',source_frames_sha256='aaa'),
            dict(physical_path='/tmp/new_source',source_frames_sha256='bbb'),
            dict(physical_path='/tmp/new_source_alias',source_frames_sha256='bbb')]
        new,duplicate=deduplicate(candidates,existing)
        self.assertEqual(len(new),1);self.assertEqual(len(duplicate),3)

    def test_parent_failure_does_not_start_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);called=[]
            with self.assertRaisesRegex(RuntimeError,'without verified completion'):
                executor.run_plan(dict(stages=[{}]),root,lambda stage:called.append(stage))
            self.assertEqual(called,[])

    def test_completed_parent_runs_once_and_resumes_without_duplicates(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);logs=root/'stage';logs.mkdir()
            (root/'completion_verified.json').write_text(json.dumps(dict(verified=True)))
            stage=dict(id='test',tasks=[],source_pins={},logs=str(logs),command=[])
            calls=[]
            def child(s):
                calls.append(s['id']);(logs/'completion_verified.json').write_text(json.dumps(dict(verified=True)))
            with patch.object(executor,'LOGS',root),patch.object(executor.subprocess,'run') as proc,patch.object(executor.subprocess,'check_output',return_value='24000'):
                proc.return_value.returncode=1
                executor.run_plan(dict(stages=[stage],target=1),root,child)
                executor.run_plan(dict(stages=[stage],target=1),root,child)
            self.assertEqual(calls,['test'])

    def test_source_pin_change_blocks_collection(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'completion_verified.json').write_text('{"verified":true}')
            source=root/'runtime.py';source.write_text('changed')
            stage=dict(logs=str(root/'stage'),source_pins={str(source):'wrong'},tasks=[])
            called=[]
            with self.assertRaisesRegex(RuntimeError,'runtime changed'):
                executor.run_plan(dict(stages=[stage]),root,lambda stage:called.append(stage))
            self.assertEqual(called,[])

    def gate_fixture(self,source):
        temp=tempfile.TemporaryDirectory();root=Path(temp.name)/'dataset';episode=root/'episode_000000';episode.mkdir(parents=True)
        for name in ['frames.jsonl','episode_start.json','episode_summary.json','correction.json','gripper_labels.json','training_mask.json']:
            path=source/name
            if path.exists():(episode/name).symlink_to(path)
        (root/'dataset_info.json').symlink_to(source.parent/'dataset_info.json')
        return temp,episode

    def test_known_large_reverse_source_is_rejected_without_source_write(self):
        source=Path('/home/lp531/smolvla_data/cup_cylinder_horizontal200_20261108/episode_000174')
        before=set(source.iterdir());temp,episode=self.gate_fixture(source)
        try:
            with self.assertRaisesRegex(ValueError,'Retake rejected'):screen(episode)
            report=json.loads((episode/'retake_quality.json').read_text())
            self.assertGreater(report['max_reverse_excursion_m'],.49)
            self.assertEqual(set(source.iterdir()),before)
        finally:temp.cleanup()

    def test_known_short_good_cup_passes(self):
        source=Path('/home/lp531/smolvla_data/cup_quality200_from_pi05_513_20261111/episode_000002').resolve()
        temp,episode=self.gate_fixture(source)
        try:self.assertTrue(screen(episode)['passed'])
        finally:temp.cleanup()

    def test_subtle_apple_8cm_source_is_rejected(self):
        source=Path('/home/lp531/smolvla_data/apple_scale110_demo400_20261105/episode_000229')
        temp,episode=self.gate_fixture(source)
        try:
            with self.assertRaisesRegex(ValueError,'Retake rejected'):screen(episode)
            report=json.loads((episode/'retake_quality.json').read_text())
            self.assertGreaterEqual(report['max_reverse_excursion_m'],.08)
        finally:temp.cleanup()


if __name__=='__main__':unittest.main()
