#!/usr/bin/python3
"""Hold behind the existing queue lock, then execute the attached fixed plan."""
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,'/home/lp531/robot/dataset_trajectory_review')
from review import atomic_json
from screen_fruit_motion import digest

LOGS=Path('/home/lp531/z1_queue_logs/reverse8_queue_20261112')
PARENT=Path('/home/lp531/z1_queue_logs/bottle_rerecord172_queue_20261111')


def run_plan(plan,parent_logs=PARENT,run_stage=None):
    """Injectable coordinator: never start children until parent audit AND lock."""
    run_stage=run_stage or (lambda stage:subprocess.run(stage['command'],check=True,
        env=dict(os.environ,ROS_DOMAIN_ID=str(stage['domain']))))
    with (parent_logs/'.queue.lock').open('a') as parent_lock:
        fcntl.flock(parent_lock,fcntl.LOCK_EX)
        completion=parent_logs/'completion_verified.json'
        if not completion.exists() or not json.loads(completion.read_text()).get('verified'):
            raise RuntimeError('Original queue released its lock without verified completion; extension will not start a simulator')
        for stage in plan['stages']:
            complete=Path(stage['logs'])/'completion_verified.json'
            if complete.exists() and json.loads(complete.read_text()).get('verified'):continue
            for path,expected in stage['source_pins'].items():
                if digest(path)!=expected:raise RuntimeError('Queued runtime changed: '+path)
            for task in stage['tasks']:
                if digest(Path(task['source'])/'frames.jsonl')!=task['source_frames_sha256']:
                    raise RuntimeError('Pinned original source changed: '+task['source'])
            # This only inspects simulator ownership; no existing job is killed.
            if subprocess.run(['pgrep','-f','[/]run_z1_ros2_scene.py'],stdout=subprocess.DEVNULL).returncode==0:
                raise RuntimeError('Another simulator is active; refusing overlapping GPU collection')
            free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip()
            if int(free.splitlines()[0])<18000:raise RuntimeError('Less than18GB GPU memory free; saved data retained')
            atomic_json(LOGS/'status.json',dict(stage=stage['id'],target=plan['target'],
                current_stage_target=len(stage['tasks']),updated_at=datetime.now().astimezone().isoformat()))
            run_stage(stage)
            if not complete.exists() or not json.loads(complete.read_text()).get('verified'):
                raise RuntimeError('Stage returned without independent completion audit: '+stage['id'])


def main():
    LOGS.mkdir(parents=True,exist_ok=True)
    plan=json.loads((HERE/'plan.json').read_text())
    with (LOGS/'.executor.lock').open('w') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic_json(LOGS/'status.json',dict(stage='waiting_for_original_queue',
            target=plan['target'],original_target=172,original_queue=str(PARENT),
            service='reverse8-rerecord41-20261112.service',
            updated_at=datetime.now().astimezone().isoformat(),new_simulator_started=False))
        try:
            run_plan(plan)
            replacement=[]
            for stage in plan['stages']:
                replacements=json.loads((Path(stage['logs'])/'replacement_map.json').read_text())
                lookup={str(Path(t['source']).resolve()):t for t in stage['tasks']}
                for row in replacements:
                    task=lookup[str(Path(row['old']).resolve())]
                    replacement.append(dict(row,task_id=task['id'],mark_id=task['mark_id']))
            assert len(replacement)==plan['target'] and len({r['mark_id'] for r in replacement})==plan['target']
            atomic_json(LOGS/'replacement_map.json',replacement)
            atomic_json(LOGS/'completion_verified.json',dict(verified=True,target=plan['target'],
                completed_at=datetime.now().astimezone().isoformat(),replacement_map=str(LOGS/'replacement_map.json'),
                training_started=False))
            atomic_json(LOGS/'status.json',dict(stage='complete',saved=len(replacement),target=plan['target'],verified=True))
        except BaseException as exc:
            atomic_json(LOGS/'status.json',dict(stage='error',target=plan['target'],error=str(exc),
                updated_at=datetime.now().astimezone().isoformat()))
            raise


if __name__=='__main__':main()
