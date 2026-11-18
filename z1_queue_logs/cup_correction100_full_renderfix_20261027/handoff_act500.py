"""Wait for complete collection, independently audit, then run decoded fresh ACT."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

RUN = Path(__file__).resolve().parent
TRAIN = Path('/home/lp531/pi05_lora_workspace/act_cup_renderfix500_cached_20261027')
DATA = Path('/home/lp531/smolvla_data/z1_cup_correction100_full_renderfix_20261027')


def status(stage, **values):
    result = dict(stage=stage, updated_at=datetime.datetime.now().astimezone().isoformat(), **values)
    temp = RUN/'act_handoff_status.tmp'
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    temp.replace(RUN/'act_handoff_status.json')
    print(json.dumps(result, ensure_ascii=False), flush=True)


def main():
    with (RUN/'.act_handoff.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            while True:
                episodes = list(DATA.glob('episode_[0-9]*'))
                if len(episodes) == 100:
                    break
                if len(episodes) > 100:
                    raise RuntimeError('Unexpected extra correction episodes')
                status('waiting_for_complete_collection', accepted=len(episodes), target=100)
                time.sleep(20)
            # The collector owns simulator shutdown; wait for its normal exit.
            collector = json.loads((RUN/'launch.json').read_text())['pid']
            deadline = time.monotonic() + 120
            while True:
                try:
                    os.kill(collector, 0)
                except ProcessLookupError:
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError('Collector has not released simulator after100 episodes')
                time.sleep(2)
            status('auditing100_corrections')
            with (RUN/'handoff_integrity.log').open('a') as log:
                subprocess.run(['/usr/bin/python3', str(RUN/'collector/audit_saved.py')], stdout=log, stderr=subprocess.STDOUT, check=True)
            audit = json.loads((RUN/'saved_integrity.json').read_text())
            if not (audit['verified'] and audit['complete'] and audit['accepted'] == 100):
                raise RuntimeError('Independent complete100 audit failed')
            status('starting_decode_and_training_workflow', training_directory=str(TRAIN))
            with (TRAIN/'workflow.log').open('a') as log:
                subprocess.run(['/home/lp531/SH/start_act_cup_renderfix500_20261027.sh'], stdout=log, stderr=subprocess.STDOUT, check=True)
            status('training_completed', training_directory=str(TRAIN))
        except BaseException as exc:
            status('failed', error=str(exc), traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    main()
