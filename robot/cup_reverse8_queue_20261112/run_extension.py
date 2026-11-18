#!/usr/bin/python3
"""Serialize cup7 after the verified original172 and reverse41 executors."""
from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
LOGS=Path('/home/lp531/z1_queue_logs/cup_reverse8_queue_20261112')
PREVIOUS=Path('/home/lp531/z1_queue_logs/reverse8_queue_20261112')
ORIGINAL=Path('/home/lp531/z1_queue_logs/bottle_rerecord172_queue_20261111')
sys.path.insert(0,'/home/lp531/robot/dataset_trajectory_review')
from review import atomic_json

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def after_previous(plan,previous=PREVIOUS,original=ORIGINAL,runner=None):
    runner=runner or (lambda command:subprocess.run(command,check=True))
    # The reverse41 executor holds this lock throughout its wait and recording.
    # Taking it first prevents this job from racing reverse41 for the parent lock.
    with (previous/'.executor.lock').open('a') as prior:
        fcntl.flock(prior,fcntl.LOCK_EX)
        for folder in [original,previous]:
            audit=folder/'completion_verified.json'
            if not audit.exists() or not json.loads(audit.read_text()).get('verified'):
                raise RuntimeError('Preceding queue has not passed its completion audit: '+str(folder))
        with (original/'.queue.lock').open('a') as parent:
            fcntl.flock(parent,fcntl.LOCK_EX)
            for p,h in plan['source_pins'].items():
                if digest(p)!=h:raise RuntimeError('Queued runtime/source changed: '+p)
            complete=Path(plan['logs'])/'completion_verified.json'
            if complete.exists() and json.loads(complete.read_text()).get('verified'):return
            if subprocess.run(['pgrep','-f','[/]run_z1_ros2_scene.py'],stdout=subprocess.DEVNULL).returncode==0:
                raise RuntimeError('A simulator is active; refusing concurrent collection')
            free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
            if int(free.strip().splitlines()[0])<18000:raise RuntimeError('Insufficient free GPU memory for collection')
            atomic_json(Path(plan['logs'])/'status.json',dict(stage='starting_cup7',target=7,updated_at=datetime.now().astimezone().isoformat()))
            runner(plan['command'])
            if not complete.exists() or not json.loads(complete.read_text()).get('verified'):
                raise RuntimeError('Cup7 returned without verified completion')

def main():
    plan=json.loads((ROOT/'plan.json').read_text())
    with (LOGS/'.executor.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic_json(LOGS/'status.json',dict(stage='waiting_for_reverse41',target=7,completed=0,
            previous_queue=str(PREVIOUS),new_simulator_started=False,updated_at=datetime.now().astimezone().isoformat()))
        try:after_previous(plan)
        except BaseException as exc:
            atomic_json(LOGS/'status.json',dict(stage='error',target=7,error=str(exc),updated_at=datetime.now().astimezone().isoformat()))
            raise

if __name__=='__main__':main()
