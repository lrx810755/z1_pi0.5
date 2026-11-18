import ast
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace


class RecorderStartTests(unittest.TestCase):
    def test_status_callback_cannot_arrive_before_physics_stream(self):
        tree = ast.parse(Path(__file__).with_name('nodes.py').read_text())
        recorder = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == 'CorrectionRecorder')
        method = next(n for n in recorder.body if isinstance(n, ast.FunctionDef) and n.name == '_start_callback')
        started = threading.Event()
        class Base:
            def _start_callback(self, req, res):
                with self._lock:
                    self._recording = True
                def status():
                    started.set()
                    with self._lock:
                        self.dynamic_samples += 1
                        if self.contact_stream is not None:
                            self.contact_stream.write('{}\n')
                self.worker = threading.Thread(target=status)
                self.worker.start()
                assert started.wait(2)
                # Give the concurrent callback an opportunity while Base returns.
                self.worker.join(.05)
                res.success = True
                return res
        cls = ast.ClassDef(name='Recorder', bases=[ast.Name(id='Base', ctx=ast.Load())], keywords=[], body=[method], decorator_list=[])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'evidence.json').write_text('{}')
            env = dict(Base=Base, context=lambda: {}, evidence_path=lambda ctx: root/'evidence.json', json=json)
            exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), 'recorder_start', 'exec'), env)
            obj = env['Recorder'](); obj._lock = threading.RLock(); obj._episode_dir = root
            obj.contact_stream = None; obj.dynamic_samples = 0
            obj._start_callback(None, SimpleNamespace(success=False))
            obj.worker.join(2); self.assertFalse(obj.worker.is_alive())
            obj.contact_stream.flush(); obj.contact_stream.close()
            self.assertEqual(obj.dynamic_samples, len((root/'physics_status.jsonl').read_text().splitlines()))
            self.assertEqual(obj.dynamic_samples, 1)


if __name__ == '__main__': unittest.main()
