"""Timed recovery of recollection and automatic training handoff."""
import json
import subprocess
from datetime import datetime
from pathlib import Path
import workflow

UNIT='pi05-three-camera-recollect-train-20261006.service'
TIMER='pi05-three-camera-check-20261006.timer'

def decide(count,phase,active):
    if phase=='complete':return 'finished'
    if active and phase=='training':return 'training_started'
    if active:return 'continue_running'
    if count<160:return 'resume_collection'
    return 'start_audit_and_training'

def main():
    count,counts=workflow.saved_counts()
    debt=workflow.deficits(counts)
    if not 125<=count<=160 or count-125+sum(debt.values())!=35:
        raise RuntimeError(f'Unexpected quota: {count}, {debt}')
    status_path=workflow.RUN/'status.json'
    status=json.loads(status_path.read_text()) if status_path.exists() else {}
    active=subprocess.run(['systemctl','--user','is-active','--quiet',UNIT]).returncode==0
    action=decide(count,status.get('phase'),active)
    entry=dict(time=datetime.now().astimezone().isoformat(),saved_episodes=count,
               replacement_successes=count-125,remaining=sum(debt.values()),
               phase=status.get('phase'),active=active,action=action)
    with (workflow.RUN/'scheduled_checks.jsonl').open('a') as log:
        log.write(json.dumps(entry,ensure_ascii=False)+'\n')
    print(json.dumps(entry,ensure_ascii=False),flush=True)
    if action in ('finished','training_started'):
        subprocess.run(['systemctl','--user','stop',TIMER],check=True)
    elif action in ('resume_collection','start_audit_and_training'):
        subprocess.run(['systemctl','--user','reset-failed',UNIT],check=False)
        subprocess.run(['systemctl','--user','start',UNIT],check=True)

if __name__=='__main__':main()
