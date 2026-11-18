"""Run the pre-change snapshot once and restore the authorized final files in finally.
No simulator/MoveIt processes are running during this offline test.
"""
from pathlib import Path
import subprocess,json
L=Path(__file__).resolve().parent;B=Path('/home/lp531/ros2_ws/src/z1_ros2');saved={}
try:
 for backup in (L/'backup').rglob('*'):
  if not backup.is_file():continue
  dst=B/backup.relative_to(L/'backup');saved[dst]=dst.read_bytes();dst.write_bytes(backup.read_bytes())
 command='source /opt/ros/humble/setup.bash; source /home/lp531/ros2_ws/install/setup.bash; cd /home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge; PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /usr/bin/python3 -m pytest -q'
 with (L/'pytest_baseline.log').open('w') as out:
  result=subprocess.run(['bash','-c',command],stdout=out,stderr=subprocess.STDOUT,timeout=120)
finally:
 for dst,content in saved.items():dst.write_bytes(content)
print((L/'pytest_baseline.log').read_text().splitlines()[-1])
def failures(path):return {line for line in path.read_text().splitlines() if line.startswith(('FAILED ','ERROR '))}
before=failures(L/'pytest_baseline.log');after=failures(L/'pytest.log');comparison=dict(new_failures=sorted(after-before),resolved_failures=sorted(before-after),unchanged_failures=sorted(before&after))
(L/'baseline_comparison.json').write_text(json.dumps(comparison,indent=2))
print(json.dumps({k:len(v) if k=='unchanged_failures' else v for k,v in comparison.items()},indent=2))
