"""GPU work starts only after both authorized evaluations fully complete."""
from pathlib import Path
import json,subprocess,sys,time,datetime,os
R=Path(__file__).resolve().parent
QUEUE=Path('/home/lp531/smolvla_eval/pi05_cup_update24k_execute30_40_queue_20261105')
EVALS=[Path(f'/home/lp531/smolvla_eval/pi05_cup20hz_chunk50_update24k_execute{s}_paired30_20261105') for s in [30,40]]
def write(stage,**details):
 p=R/'queue_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(stage=stage,updated=datetime.datetime.now().astimezone().isoformat(),**details),indent=2));t.replace(p)
try:
    write('waiting_for_cpu_preparation')
    while not (R/'preparation_complete.json').exists():time.sleep(10)
    write('cpu_verification')
    with (R/'cpu_verification.log').open('w') as log:subprocess.run([sys.executable,'verify.py'],cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
    write('waiting_for_evaluations',dependencies=[str(p) for p in EVALS])
    while True:
        states=[json.loads((p/'supervisor_status.json').read_text()) if (p/'supervisor_status.json').exists() else {} for p in EVALS]
        if any(s.get('stage') in ['failed','stopped'] for s in states):raise RuntimeError('Evaluation failed/stopped; refusing to start training before successful dependency completion')
        queue=json.loads((QUEUE/'status.json').read_text()) if (QUEUE/'status.json').exists() else {}
        if queue.get('stage')=='completed' and all(s.get('stage')=='completed' for s in states):
            for p in EVALS:
                d=json.loads((p/'completed.json').read_text());assert d['completed'] and d['total']==30
            active=subprocess.run(['systemctl','--user','is-active','--quiet','pi05-cup20hz-update24k-execute30-40-20261105.service']).returncode==0
            if not active:break
        time.sleep(30)
    memory=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    if memory<18000:raise RuntimeError(f'GPU free memory insufficient after evaluation: {memory}MiB')
    write('starting_training',free_gpu_mib=memory)
    with (R/'runner.log').open('a') as log:subprocess.run([sys.executable,'-u','runner.py'],cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
    write('completed',optimizer_updates=10000)
except BaseException as exc:
    write('failed',error=str(exc));raise
