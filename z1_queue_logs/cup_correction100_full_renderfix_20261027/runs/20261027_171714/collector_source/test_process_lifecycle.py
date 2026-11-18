import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import stop_scene


class ProcessLifecycleTests(unittest.TestCase):
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
