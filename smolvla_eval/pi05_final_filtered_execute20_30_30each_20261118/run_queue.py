"""Run both requested conditions sequentially and write the matched comparison."""
from datetime import datetime
import json
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REQUEST = json.loads((ROOT / 'request.json').read_text())


def write_json(path, data):
    staged = path.with_suffix(path.suffix + '.tmp')
    staged.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    staged.replace(path)


def status(phase, groups):
    item = dict(phase=phase, groups=groups, updated_at=datetime.now().astimezone().isoformat())
    write_json(ROOT / 'status.json', item)
    print(json.dumps(item, ensure_ascii=False), flush=True)


def combine(groups):
    baseline = json.loads(Path(REQUEST['baseline']).read_text())
    comparisons = {steps: json.loads(Path(job['paired_comparison']).read_text())
                   for steps, job in groups.items()}
    by_condition = {steps: {(row['target'], row['seed']): row for row in comp['trials']}
                    for steps, comp in comparisons.items()}
    if len(by_condition[20]) != 120 or by_condition[20].keys() != by_condition[30].keys():
        raise ValueError('The two conditions must have 120 identical object/seed pairs')
    trials = []
    for key, row20 in by_condition[20].items():
        row30 = by_condition[30][key]
        if row20['baseline_success'] != row30['baseline_success']:
            raise ValueError(f'Conflicting baseline for {key}')
        outcomes = {
            '15': dict(success=row20['baseline_success'], failure_category=row20['baseline_failure']),
            '20': dict(success=row20['filtered_success'], failure_category=row20['filtered_failure'],
                       penetration_estimate_mm=row20['max_penetration_estimate_mm']),
            '30': dict(success=row30['filtered_success'], failure_category=row30['filtered_failure'],
                       penetration_estimate_mm=row30['max_penetration_estimate_mm']),
        }
        trials.append(dict(target=key[0], seed=key[1], episode=row20['episode'], results=outcomes))
    write_json(ROOT / 'comparison_15_20_30.json', dict(
        checkpoint=REQUEST['model'], baseline_report=REQUEST['baseline'],
        per_object={'15': baseline['per_object'], **{str(k): v['per_object'] for k,v in groups.items()}},
        overall={'15': baseline['overall'], **{str(k): v['overall'] for k,v in groups.items()}},
        trials=trials))


def interrupt(signum, _frame):
    raise KeyboardInterrupt(f'Signal {signum}')


def main():
    groups = {steps: dict(phase='queued') for steps in REQUEST['execute_steps']}
    child = None
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupt)
    try:
        for steps in groups:
            job = ROOT / f'execute_{steps}' / 'job'
            with (job / 'runner.log').open('x') as log:
                child = subprocess.Popen([sys.executable, '-u', str(job / 'run_job.py')],
                                         cwd=job, stdin=subprocess.DEVNULL, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                groups[steps] = dict(phase='running', runner_pid=child.pid, status=str(job/'status.json'))
                status('running', groups)
                code = child.wait()
                child = None
            item = json.loads((job/'status.json').read_text())
            groups[steps] = dict(item, exit_code=code)
        if all(item['phase']=='completed' and item['exit_code']==0 for item in groups.values()):
            combine(groups)
            status('completed', groups)
        else:
            status('incomplete', groups)
            raise SystemExit(1)
    except KeyboardInterrupt:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait(timeout=55)
        status('interrupted', groups)
        raise


if __name__ == '__main__':
    main()
