"""One evaluation job lifecycle, no monitoring service or automatic restarts."""
from pathlib import Path
from datetime import datetime
import json
import os
import signal
import subprocess
import time
import urllib.request

JOB = Path(__file__).resolve().parent
REQUEST = json.loads((JOB / 'request.json').read_text())
OUTPUT = Path(REQUEST['output'])
scene = None
evaluation = None


def status(phase, **extra):
    item = dict(phase=phase, updated_at=datetime.now().astimezone().isoformat(), **extra)
    tmp = JOB / 'status.json.tmp'
    tmp.write_text(json.dumps(item, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(JOB / 'status.json')
    print(json.dumps(item, ensure_ascii=False), flush=True)


def stop_child(child):
    if child is None or child.poll() is not None:
        return
    for sig, wait_s in [(signal.SIGINT, 20), (signal.SIGTERM, 10), (signal.SIGKILL, 5)]:
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            return
        try:
            child.wait(timeout=wait_s)
            return
        except subprocess.TimeoutExpired:
            continue


def interrupted(signum, frame):
    raise KeyboardInterrupt('job interrupted')


for sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(sig, interrupted)


def main():
    global scene, evaluation
    # Verification is completed separately before this process starts.
    import sys
    sys.path.insert(0, '/home/lp531/robot/pi05_four_objects_eval_20261117')
    from contracts import check_final, check_pins, check_bridge_verification
    check_pins()
    check_bridge_verification(check_final(require_reload=True))
    with (JOB / 'scene.log').open('w') as scene_log, (JOB / 'evaluation.log').open('w') as eval_log:
        scene = subprocess.Popen(['/home/lp531/SH/pi05_eval_scene.sh', '--headless', '--output', str(OUTPUT)],
                                 stdin=subprocess.DEVNULL, stdout=scene_log, stderr=subprocess.STDOUT, start_new_session=True)
        status('starting_scene', scene_pid=scene.pid)
        deadline = time.monotonic() + 420
        last = 'not yet available'
        while time.monotonic() < deadline:
            if scene.poll() is not None:
                raise RuntimeError('Scene exited during startup: ' + str(scene.returncode))
            try:
                with urllib.request.urlopen('http://127.0.0.1:8868/api/status', timeout=3) as response:
                    state = json.load(response)
                components = ['recording', 'episode_reset', 'collision_scene', 'object_collisions']
                if state.get('connected') and all(state.get(k, {}).get('ready') for k in components):
                    (JOB / 'scene_ready.json').write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
                    break
                last = json.dumps({k: state.get(k) for k in ['connected', *components]})
            except Exception as exc:
                last = str(exc)
            time.sleep(2)
        else:
            raise RuntimeError('Scene startup timeout: ' + last)
        command = ['/home/lp531/SH/pi05_eval.sh', '--run', '--objects', 'all',
                   '--episodes-per-object', '10', '--execute-steps', str(REQUEST['execute_steps']), '--output', str(OUTPUT)]
        (JOB / 'evaluation_command.json').write_text(json.dumps(command, indent=2) + '\n')
        evaluation = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=eval_log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
        status('evaluating', scene_pid=scene.pid, evaluation_pid=evaluation.pid,
               objects=REQUEST['objects'], episodes_per_object=10, total_episodes=40, execute_steps=REQUEST['execute_steps'], predicted_chunk_size=30,
               failure_reasons_recorded=True, output=str(OUTPUT))
        result = evaluation.wait()
        if result != 0:
            raise RuntimeError('Evaluation exited with status ' + str(result))
        reports = sorted(OUTPUT.glob('pi05_sim_eval_*/report.json'))
        if len(reports) != 1:
            raise RuntimeError('Expected one evaluation report; got ' + str(len(reports)))
        report = json.loads(reports[0].read_text())
        counts = {obj: report['per_object'].get(obj, {}).get('episodes', 0) for obj in REQUEST['objects']}
        if any(count != 10 for count in counts.values()):
            raise RuntimeError('Incomplete episode counts: ' + str(counts))
        status('completed', per_object=report['per_object'], overall=report['overall'],
               failure_categories=report['failure_categories'], report=str(reports[0]), total_episodes=40)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        status('failed' if not isinstance(exc, KeyboardInterrupt) else 'interrupted', error=str(exc))
        raise
    finally:
        stop_child(evaluation)
        stop_child(scene)
