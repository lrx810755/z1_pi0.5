"""One authorized preparation/smoke/formal lifecycle, without auto-restarts."""
import fcntl
import os
import signal
import subprocess
import sys
import time
import traceback
from common import RUN, OUTPUT, read, write


def main():
    lock=(RUN/'.workflow.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=os.environ.copy()
    for key in ['PYTHONPATH','PYTHONHOME']:env.pop(key,None)
    env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',WANDB_DISABLED='true',PYTHONUNBUFFERED='1')
    child=None
    def stop(signum,frame):
        if child is not None and child.poll() is None:os.killpg(child.pid,signum)
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    stages=[('contracts',['-m','unittest','test_contracts','-q']),
            ('cpu_verification',['verify.py','--cpu']),
            ('smoke_training',['train.py','--smoke']),
            ('smoke_reload',['verify.py','--reload-smoke']),
            ('formal_training',['train.py','--train']),
            ('final_reload',['verify.py','--reload-final'])]
    try:
        if not read(RUN/'cache/completed.json').get('complete'):raise ValueError('Decode first')
        if OUTPUT.exists() or (RUN/'smoke').exists():raise ValueError('Refusing to overwrite existing training')
        for stage,args in stages:
            log=RUN/(stage+'.log');command=[sys.executable,*args]
            with log.open('x') as stream:
                child=subprocess.Popen(command,cwd=RUN,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                write('workflow_status.json',dict(stage=stage,status='running',workflow_pid=os.getpid(),child_pid=child.pid,command=command,log=str(log),started_unix=time.time()))
                code=child.wait()
            if code:raise RuntimeError(f'{stage} exited {code}; see {log}')
            print(stage+' passed',flush=True)
        write('workflow_status.json',dict(stage='complete',status='completed',workflow_pid=os.getpid(),finished_unix=time.time()))
    except Exception as error:
        prior=read(RUN/'workflow_status.json') if (RUN/'workflow_status.json').exists() else {}
        write('workflow_status.json',dict(prior,status='failed',error=str(error),finished_unix=time.time()))
        traceback.print_exc();raise


if __name__=='__main__':main()
