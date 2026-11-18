import json,datetime
from pathlib import Path
def latch_fault(path,reason,episode):
 if not reason:return
 path=Path(path)
 # Exclusive creation preserves the first failure across discard/reset/restart.
 try:
  with path.open('x') as f:json.dump(dict(reason=str(reason),episode=episode,time=datetime.datetime.now().astimezone().isoformat()),f)
 except FileExistsError:pass
def check_fault(path):
 path=Path(path)
 if path.exists():raise RuntimeError('Latched recording synchronization fault; inspect '+str(path))
