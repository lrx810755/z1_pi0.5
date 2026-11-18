import json,tempfile,unittest
from pathlib import Path
from failure_recorder import archive_failed_episode
class Tests(unittest.TestCase):
 def test_preserves_all_views_and_frames_outside_training_root(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);source=root/'training/.recording_000001';source.mkdir(parents=True)
   (source/'frames.jsonl').write_text('{"frame_index":0}\n')
   for camera in ['wrist','global','global_right']:
    p=source/'images'/camera;p.mkdir(parents=True);(p/'frame_000000.jpg').write_bytes(b'original image bytes')
   dest=archive_failed_episode(source,root/'failures',{'frame_count':1,'task_success':False})
   self.assertFalse(source.exists());self.assertTrue((dest/'frames.jsonl').exists())
   self.assertEqual(len(list(dest.glob('images/*/*.jpg'))),3)
   self.assertFalse(json.loads((dest/'failure_summary.json').read_text())['task_success'])
 def test_repeated_episode_index_does_not_overwrite_previous_failure(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);destinations=[]
   for _ in range(2):
    source=root/'.recording_000001';source.mkdir();(source/'frames.jsonl').write_text('data')
    destinations.append(archive_failed_episode(source,root/'failures',{}))
   self.assertNotEqual(*destinations);self.assertTrue(all(p.exists() for p in destinations))
if __name__=='__main__':unittest.main()
