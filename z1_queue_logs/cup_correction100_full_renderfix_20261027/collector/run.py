#!/usr/bin/python3
"""Resumable fixed100-task collection with physical acceptance and source snapshots."""
import argparse
from collections import Counter
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import uuid

from protocol import DATA, QUOTAS, atomic_json, build_tasks

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
PORT = 8807


def api(path, body=None, timeout=8):
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}{path}',
        data=None if body is None else json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            value = json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(exc.read().decode(errors='replace')) from exc
    if value.get('success') is False:
        raise RuntimeError(value.get('message', str(value)))
    return value


def completed(plan):
    known = {t['id']: t for t in plan['tasks']}
    result = {}
    for path in sorted(DATA.glob('episode_[0-9]*')):
        try:
            meta = json.loads((path / 'correction.json').read_text())
            summary = json.loads((path / 'episode_summary.json').read_text())
            task_id = meta['task_id']
            if meta.get('accepted') and meta.get('plan_hash') == plan['hash'] and task_id in known:
                if not all(summary.get(k) is True for k in
                           ('task_success', 'physical_grasp_ok', 'release_sequence_complete', 'placement_stable')):
                    raise RuntimeError('Accepted episode has failed physical summary: ' + str(path))
                if task_id in result:
                    raise RuntimeError('Duplicate accepted task ID: ' + task_id)
                result[task_id] = str(path)
        except (OSError, ValueError, KeyError):
            continue
    return result


def status(plan, stage, **extra):
    done = completed(plan)
    kinds = Counter(t['kind'] for t in plan['tasks'] if t['id'] in done)
    value = dict(stage=stage, updated_at=datetime.datetime.now().astimezone().isoformat(),
                 completed=len(done), quota=100, by_kind=dict(kinds), category_quotas=QUOTAS, **extra)
    atomic_json(RUN / 'status.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def prepare_plan():
    tasks = build_tasks()
    digest = hashlib.sha256(json.dumps(tasks, sort_keys=True).encode()).hexdigest()
    path = RUN / 'collection_plan.json'
    if path.exists():
        plan = json.loads(path.read_text())
        if plan['hash'] != digest:
            raise RuntimeError('Existing plan differs; do not silently change quota/tasks')
        return plan
    plan = dict(hash=digest, tasks=tasks, data_root=str(DATA), quotas=QUOTAS,
        specification='/home/lp531/docs/杯子纠偏数据集要求_20261026.md',
        created_at=datetime.datetime.now().astimezone().isoformat(),
        source_type='synthetic_failure_informed', renderer_world_eps=.0001,
        instruction='Keep100 unique successful task IDs; never train on masked intervention actions.')
    atomic_json(path, plan)
    return plan


def stop_scene(process):
    if process is None:
        return
    for sig, delay in ((signal.SIGINT, 12), (signal.SIGTERM, 5), (signal.SIGKILL, 2)):
        try:
            os.killpg(process.pid, sig)
            process.wait(timeout=delay)
            # The launch parent may have exited while a native crash handler
            # still owns GPU memory. Continue signaling the owned process group.
            time.sleep(.2)
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            pass


def previous_round(runs, current):
    history = sorted(p for p in runs.glob('*/current.json') if p.parent != current)
    return int(json.loads(history[-1].read_text())['round']) if history else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--max-attempts', type=int, default=0)
    args = parser.parse_args()
    plan = prepare_plan()
    if args.status or args.dry_run:
        done = completed(plan)
        print(json.dumps(dict(plan=str(RUN / 'collection_plan.json'), total=100,
            completed=len(done), quotas=QUOTAS, data_root=str(DATA)), ensure_ascii=False, indent=2))
        return 0
    RUN.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    with (RUN / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if len(completed(plan)) == 100:
            status(plan, 'complete')
            return 0
        try:
            api('/api/status', timeout=1)
        except (OSError, ValueError):
            pass
        else:
            raise RuntimeError('Collection GUI port8807 is already occupied; refusing duplicate scene')
        runtime = RUN / 'runs' / datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        runtime.mkdir(parents=True)
        snapshot = runtime / 'collector_source'
        snapshot.mkdir()
        hashes = {}
        for path in HERE.glob('*.py'):
            shutil.copy2(path, snapshot / path.name)
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        atomic_json(runtime / 'source_manifest.json', dict(files=hashes, plan_hash=plan['hash']))
        current = runtime / 'current.json'
        os.environ['Z1_CORRECTION100_CURRENT'] = str(current)
        for pending in DATA.glob('.recording_*'):
            interrupted = runtime / 'interrupted'
            interrupted.mkdir(exist_ok=True)
            pending.rename(interrupted / pending.name)
        command = ['ros2', 'launch', str(snapshot / 'collection.launch.py'),
            'external_camera_layout:=dual', 'headless:=true', 'rviz:=false',
            'target_object:=cup', 'task_gui:=true', f'task_gui_port:={PORT}',
            'run_smolvla_policy:=false', 'freeze_target_keep_collisions:=true',
            f'dataset_root:={DATA}', 'wrist_rgb_topic:=/left_wrist_camera/color/image_raw',
            'vla_wrist_rgb_topic:=none', 'recording_fps:=10.0',
            'recording_sync_tolerance_s:=0.045', 'recording_sync_delay_s:=0.5', 'recording_action_age_s:=0.40']
        atomic_json(runtime / 'command.json', command)
        process = None
        attempts = 0
        try:
            with (runtime / 'scene.log').open('w') as output:
                process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            status(plan, 'starting_scene', scene_pid=process.pid, runtime=str(runtime))
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError('Simulator exited; inspect ' + str(runtime / 'scene.log'))
                try:
                    live = api('/api/status', timeout=2)
                    if live.get('connected') and all(live.get(k, {}).get('ready') for k in
                            ('recording', 'episode_reset', 'collision_scene', 'object_collisions')):
                        if Path(live['recording']['dataset_root']).resolve() != DATA:
                            raise RuntimeError('Recorder points at another dataset')
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(2)
            else:
                raise RuntimeError('Scene readiness timed out')
            # Exercise one task of each category first, then retain the fixed full quota.
            first = [min((t for t in plan['tasks'] if t['kind'] == kind),
                         key=lambda t: abs(t['grid'][0] - 3) + abs(t['grid'][1] - 1))
                     for kind in QUOTAS]
            first_ids = {t['id'] for t in first}
            ordered = first + [t for t in plan['tasks'] if t['id'] not in first_ids]
            stalled = 0
            round_index = previous_round(RUN / 'runs', runtime)
            while True:
                round_index += 1
                before = len(completed(plan))
                for task in ordered:
                    if task['id'] in completed(plan):
                        continue
                    if process.poll() is not None:
                        raise RuntimeError('Scene process exited during collection')
                    if args.max_attempts and attempts >= args.max_attempts:
                        status(plan, 'attempt_limit', attempts=attempts, runtime=str(runtime))
                        return 0
                    ctx = dict(task=task, plan_hash=plan['hash'], attempt_id=uuid.uuid4().hex,
                               round=round_index, source_manifest=str(runtime / 'source_manifest.json'))
                    atomic_json(current, ctx)
                    attempts += 1
                    status(plan, 'collecting', task_id=task['id'], kind=task['kind'], grid=task['grid'],
                           attempts=attempts, round=round_index, runtime=str(runtime), scene_pid=process.pid)
                    row = dict(task_id=task['id'], attempt_id=ctx['attempt_id'], kind=task['kind'],
                               grid=task['grid'], round=round_index,
                               started_at=datetime.datetime.now().astimezone().isoformat())
                    try:
                        row['result'] = api('/api/demonstration/auto',
                            {'target_object': 'cup', 'curriculum_level': 'full', 'target_grid': task['grid']}, timeout=900)
                        if task['id'] not in completed(plan):
                            raise RuntimeError('API success without accepted complete episode')
                        row['saved'] = True
                        print('SAVED ' + task['id'], flush=True)
                    except (OSError, ValueError, RuntimeError) as exc:
                        row.update(saved=False, error=str(exc))
                        print('FAILED ' + task['id'] + ': ' + str(exc), flush=True)
                        for path in ('/api/force_stop', '/api/recording/discard'):
                            try:
                                api(path, {}, timeout=15)
                            except (OSError, ValueError, RuntimeError):
                                pass
                        deadline = time.monotonic() + 45
                        while api('/api/status').get('auto_demonstration', {}).get('running'):
                            if time.monotonic() > deadline:
                                raise RuntimeError('Failed attempt did not stop; refusing overlapping context')
                            time.sleep(.5)
                    with (runtime / 'attempts.jsonl').open('a') as output:
                        output.write(json.dumps(row, ensure_ascii=False) + '\n')
                        output.flush()
                after = len(completed(plan))
                if after == 100:
                    status(plan, 'complete', attempts=attempts, runtime=str(runtime))
                    return 0
                stalled = stalled + 1 if after == before else 0
                if stalled >= 3:
                    status(plan, 'needs_investigation', reason='Three full rounds without a new accepted episode',
                           runtime=str(runtime))
                    return 1
                status(plan, 'retrying_missing', attempts=attempts, round=round_index, runtime=str(runtime))
        except BaseException as exc:
            status(plan, 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'error', error=str(exc))
            raise
        finally:
            if process is not None:
                try:
                    api('/api/force_stop', {}, timeout=5)
                except Exception:
                    pass
                stop_scene(process)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    raise SystemExit(main())
