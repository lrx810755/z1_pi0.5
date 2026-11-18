from pathlib import Path
import subprocess,json,datetime
root=Path(__file__).resolve().parent
meta=json.loads((root/'launch.json').read_text())
with (root/'evaluation.log').open('a',buffering=1) as log:
 p=subprocess.Popen(meta['command'],stdout=log,stderr=subprocess.STDOUT)
 meta['evaluation_pid']=p.pid
 (root/'launch.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
 code=p.wait()
reports=list(root.glob('act_sim_eval_*/report.json'))
results=list(root.glob('act_sim_eval_*/episodes/*/result.json'))
complete={'finished_at':datetime.datetime.now().isoformat(),'returncode':code,'episodes':len(results),'target_episodes':30,'report':[str(p) for p in reports],'running':False,'completed':code==0 and len(results)==30 and bool(reports)}
(root/'completed.json').write_text(json.dumps(complete,ensure_ascii=False,indent=2))
