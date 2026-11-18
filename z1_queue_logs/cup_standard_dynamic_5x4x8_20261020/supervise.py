import subprocess, json, datetime, shutil
from pathlib import Path
r=Path(__file__).resolve().parent
entry='/home/lp531/start_cup_dynamic_5x4x8_collection.sh'
with (r/'launcher.log').open('a') as log:
 p=subprocess.Popen(['bash',entry],stdout=log,stderr=log)
 d=json.loads((r/'collection_plan.json').read_text());d['pid']=p.pid;d['entry']=entry
 (r/'launch.json').write_text(json.dumps(d,ensure_ascii=False,indent=2))
 w=subprocess.Popen(['/usr/bin/python3',str(r/'watch_collection.py')],stdout=log,stderr=log)
 code=p.wait()
 w.terminate()
 data=Path(d['raw_root']);count=0
 for path in data.glob('episode_*/episode_summary.json'):
  try:
   e=json.loads(path.read_text());count+=bool(e.get('task_success') and e.get('target_object')=='cup')
  except Exception:pass
 (r/'completed.json').write_text(json.dumps(dict(exit_code=code,saved=count,goal=160,complete=code==0 and count==160,finished=datetime.datetime.now().astimezone().isoformat()),indent=2))
