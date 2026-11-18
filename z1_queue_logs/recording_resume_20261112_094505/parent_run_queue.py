"""Run the authorized160+12 rerecording queue, then verify both stages."""
from datetime import datetime
import fcntl
import json
from pathlib import Path
import subprocess

ROOT=Path('/home/lp531/z1_queue_logs/bottle_rerecord172_queue_20261111')
STAGES=[
    dict(name='base160',target=160,data='/home/lp531/smolvla_data/bottle_rerecord160_renderfix_20261111',
         logs='/home/lp531/z1_queue_logs/bottle_rerecord160_renderfix_20261111',
         command=['bash','/home/lp531/SH/start_bottle_rerecord160_renderfix_20261111.sh']),
    dict(name='wait8_retake12',target=12,data='/home/lp531/smolvla_data/bottle_wait8_retake12_20261111',
         logs='/home/lp531/z1_queue_logs/bottle_wait8_retake12_20261111',
         command=['bash','/home/lp531/SH/start_bottle_wait8_retake12_20261111.sh']),
]


def write_status(stage,**extra):
    counts={s['name']:len(list(Path(s['data']).glob('episode_*/episode_summary.json'))) for s in STAGES}
    value=dict(stage=stage,updated_at=datetime.now().astimezone().isoformat(),target=172,
               saved=sum(counts.values()),by_stage=counts,**extra)
    temp=ROOT/'status.json.tmp';temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));temp.replace(ROOT/'status.json')
    print(json.dumps(value,ensure_ascii=False),flush=True)


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'.queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (ROOT/'queue_manifest.json').write_text(json.dumps(dict(target=172,base_demonstrations=170,
            correction_demonstrations=2,stages=STAGES,column0='original approach',other_columns='horizontal approach',
            waiting_gate_s=8,old_sources_preserved=True,training_started=False),ensure_ascii=False,indent=2))
        try:
            for s in STAGES:
                complete=Path(s['logs'])/'completion_verified.json'
                if complete.exists() and json.loads(complete.read_text()).get('verified'):
                    continue
                write_status(s['name'])
                subprocess.run(s['command'],check=True)
                if not complete.exists() or not json.loads(complete.read_text()).get('verified'):
                    raise RuntimeError('Stage exited without passing final audit: '+s['name'])
            write_status('complete',verified=True)
            (ROOT/'completion_verified.json').write_text(json.dumps(dict(verified=True,target=172,
                completed_at=datetime.now().astimezone().isoformat(),training_started=False),ensure_ascii=False,indent=2))
        except BaseException as exc:
            write_status('error',error=str(exc));raise


if __name__=='__main__':main()
