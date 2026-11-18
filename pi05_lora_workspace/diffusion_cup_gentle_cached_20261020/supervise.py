import json,os,subprocess,sys,time
from datetime import datetime
from pathlib import Path
from prepare import RUN,write
if __name__=='__main__':
 assert (RUN/'checkpoint_reload_verified.json').exists() and (RUN/'data_verified.json').exists()
 with (RUN/'training.log').open('a',buffering=1) as log:
  child=subprocess.Popen([sys.executable,str(RUN/'train.py')],cwd=RUN,stdout=log,stderr=subprocess.STDOUT)
  write('status.json',dict(state='training',pid=child.pid,supervisor_pid=os.getpid(),started_at=datetime.now().astimezone().isoformat(),target_optimizer_updates=50000))
  code=child.wait()
 write('status.json',dict(state='complete' if code==0 else 'failed',exit_code=code,pid=child.pid,ended_at=datetime.now().astimezone().isoformat(),completion_record=(RUN/'completed.json').exists()))
 sys.exit(code)
