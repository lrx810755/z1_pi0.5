"""Reload failure retention between attempts, preserving successful episodes."""
import json,os,signal,subprocess,time,urllib.request
from pathlib import Path
UNIT='cup-horizontal-demo400-20261108.service'
LOG=Path('/home/lp531/z1_queue_logs/cup_horizontal_demo_20261108/failure_archive_reload.json')
def log(stage):
 LOG.write_text(json.dumps(dict(stage=stage,time=time.strftime('%Y-%m-%d %H:%M:%S')),ensure_ascii=False,indent=2)+'\n')
 print(stage,flush=True)
def state():
 return subprocess.run(['systemctl','--user','show',UNIT,'-p','ActiveState','--value'],capture_output=True,text=True).stdout.strip()

def main():
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   args=(p/'cmdline').read_bytes().split(b'\0')
   if len(args)>1 and args[0].endswith(b'python3') and any(a.endswith(b'/continuous_training.py') for a in args) and b'8837' in args:os.kill(int(p.name),signal.SIGSTOP)
  except OSError:pass
 log('WAITING_FOR_CURRENT_RECORDING')
 while state() in ('active','activating'):
  with urllib.request.urlopen('http://127.0.0.1:8837/api/status',timeout=5) as response:s=json.load(response)
  if not s.get('recording',{}).get('recording'):break
  time.sleep(1)
 log('RELOADING')
 subprocess.run(['systemctl','--user','stop','--no-block',UNIT],check=True)
 for _ in range(15):
  if state() not in ('active','activating','deactivating'):break
  time.sleep(1)
 if state() in ('active','activating','deactivating'):
  subprocess.run(['systemctl','--user','kill','--signal=SIGKILL','--kill-who=all',UNIT],check=True)
  time.sleep(2)
 subprocess.run(['systemctl','--user','restart',UNIT],check=True)
 log('RECORDER_RESTARTED_WITH_FAILURE_RETENTION')
if __name__=='__main__':main()
