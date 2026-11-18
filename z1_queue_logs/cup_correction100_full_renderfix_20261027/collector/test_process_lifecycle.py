import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import unittest
import json
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import stop_scene, previous_round


class ProcessLifecycleTests(unittest.TestCase):
    def test_restart_continues_layout_rotation_instead_of_repeating_round1(self):
        with tempfile.TemporaryDirectory() as temp:
            runs = Path(temp)
            self.assertEqual(previous_round(runs, runs/'new'), 0)
            for name, value in [('20261027_010000', 3), ('20261027_020000', 5)]:
                folder = runs/name; folder.mkdir()
                (folder/'current.json').write_text(json.dumps({'round': value}))
            current = runs/'20261027_030000'; current.mkdir()
            (current/'current.json').write_text('{}')
            self.assertEqual(previous_round(runs, current), 5)

    def test_cleanup_kills_remaining_child_after_launch_parent_exits(self):
        code = "import subprocess,sys; p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],stdout=subprocess.DEVNULL); print(p.pid,flush=True)"
        parent = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE,
                                  text=True, start_new_session=True)
        try:
            child = int(parent.stdout.readline())
            parent.wait(timeout=5)
            stop_scene(parent)
            deadline = time.monotonic() + 3
            live = True
            while time.monotonic() < deadline:
                try:
                    state = Path(f'/proc/{child}/stat').read_text().split()[2]
                    live = state != 'Z'
                except FileNotFoundError:
                    live = False
                if not live:
                    break
                time.sleep(.05)
            self.assertFalse(live, 'Simulator child survived after launcher parent exited')
        finally:
            try:
                os.killpg(parent.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            parent.stdout.close()


if __name__ == '__main__':
    unittest.main()
