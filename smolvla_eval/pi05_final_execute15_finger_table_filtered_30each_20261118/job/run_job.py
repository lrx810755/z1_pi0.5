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
        scene = subprocess.Popen(json.loads((JOB/'scene_command.json').read_text()),
                                 stdin=subprocess.DEVNULL, stdout=scene_log, stderr=subprocess.STDOUT, start_new_session=True)
        status('verifying_scene_physics', scene_pid=scene.pid)
        deadline = time.monotonic() + 420
        last = 'not yet available'
        while time.monotonic() < deadline:
            if scene.poll() is not None:
                raise RuntimeError('Scene exited during startup: ' + str(scene.returncode))
            try:
                with urllib.request.urlopen('http://127.0.0.1:8868/api/status', timeout=3) as response:
                    state = json.load(response)
                components = ['recording', 'episode_reset', 'collision_scene', 'object_collisions']
                if state.get('connected') and all(state.get(k, {}).get('ready') for k in components) and (JOB/'physics_verified.json').is_file():
                    (JOB / 'scene_ready.json').write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
                    break
                last = json.dumps({k: state.get(k) for k in ['connected', *components]})
            except Exception as exc:
                last = str(exc)
            time.sleep(2)
        else:
            raise RuntimeError('Scene startup timeout: ' + last)
        physics = json.loads((JOB/'physics_verified.json').read_text())
        if not physics.get('passed') or set(physics['support']) != set(REQUEST['objects']):
            raise RuntimeError('Scene physics verification did not pass')
        command = json.loads((JOB/'evaluation_command.json').read_text())
        evaluation = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=eval_log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
        status('evaluating', scene_pid=scene.pid, evaluation_pid=evaluation.pid,
               objects=REQUEST['objects'], episodes_per_object=30, total_episodes=120, execute_steps=REQUEST['execute_steps'], predicted_chunk_size=30,
               failure_reasons_recorded=True, output=str(OUTPUT))
        result = evaluation.wait()
        if result != 0:
            raise RuntimeError('Evaluation exited with status ' + str(result))
        reports = sorted(OUTPUT.glob('pi05_sim_eval_*/report.json'))
        if len(reports) != 1:
            raise RuntimeError('Expected one evaluation report; got ' + str(len(reports)))
        report = json.loads(reports[0].read_text())
        counts = {obj: report['per_object'].get(obj, {}).get('episodes', 0) for obj in REQUEST['objects']}
        if any(count != 30 for count in counts.values()):
            raise RuntimeError('Incomplete episode counts: ' + str(counts))
        from collections import Counter
        from math import dist
        baseline = json.loads((JOB/'baseline.json').read_text())['objects']
        records = [json.loads(path.read_text()) for path in sorted(reports[0].parent.glob('episodes/*/result.json'))]
        comparison = {}
        pairs = []
        for target in REQUEST['objects']:
            old = {item['seed']: item for item in baseline[target]}
            new = [item for item in records if item['target_object'] == target]
            if len(new) != 30 or {item['seed'] for item in new} != set(old):
                raise RuntimeError(f'Incomplete paired trials for {target}')
            for item in new:
                if item.get('execute_steps') != 15 or not item.get('penetration_samples'):
                    raise RuntimeError(f'Missing execution or penetration evidence for {target} seed {item["seed"]}')
                if item['failure_category'] == 'finger_table_contact':
                    raise RuntimeError(f'Actual finger/table contact detected with filter active for {target} seed {item["seed"]}')
                prior = old[item['seed']]
                pairs.append(dict(target=target, seed=item['seed'], episode=item['episode_number'],
                    baseline_success=prior['success'], filtered_success=item['success'],
                    baseline_failure=prior['failure_category'], filtered_failure=item['failure_category'],
                    baseline_grasp=prior['physical_predicates'].get('physical_grasp_ok'),
                    filtered_grasp=item['physical_predicates'].get('physical_grasp_ok'),
                    max_penetration_estimate_mm=round(item['max_finger_table_penetration_estimate_m']*1000,3),
                    settled_target_displacement_mm=round(dist(item['target_position_m'],prior['target_position_m'])*1000,3)))
            matched = [row for row in pairs if row['target'] == target]
            comparison[target] = dict(
                baseline_successes=sum(row['baseline_success'] for row in matched),
                filtered_successes=sum(row['filtered_success'] for row in matched),
                baseline_stable_grasps=sum(row['baseline_grasp'] is True for row in matched),
                filtered_stable_grasps=sum(row['filtered_grasp'] is True for row in matched),
                estimated_penetration_over_2mm=sum(row['max_penetration_estimate_mm']>2 for row in matched),
                successes_with_estimated_penetration_over_2mm=sum(row['filtered_success'] and row['max_penetration_estimate_mm']>2 for row in matched),
                baseline_failures=dict(Counter(row['baseline_failure'] for row in matched if not row['baseline_success'])),
                filtered_failures=dict(Counter(row['filtered_failure'] for row in matched if not row['filtered_success'])))
        paired_path = OUTPUT / 'paired_comparison.json'
        paired_path.write_text(json.dumps(dict(per_object=comparison, trials=pairs), ensure_ascii=False, indent=2)+'\n')
        status('completed', per_object=report['per_object'], overall=report['overall'],
               failure_categories=report['failure_categories'], report=str(reports[0]), paired_comparison=str(paired_path), total_episodes=120)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        status('failed' if not isinstance(exc, KeyboardInterrupt) else 'interrupted', error=str(exc))
        raise
    finally:
        stop_child(evaluation)
        stop_child(scene)
