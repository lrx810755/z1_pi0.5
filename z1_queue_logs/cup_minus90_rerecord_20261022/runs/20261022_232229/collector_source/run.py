"""Resumable ordered collection: 36 basic replacements, then 89 full corrections."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from protocol import atomic_json, round_indices, ProgressGuard

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
PLAN = json.loads((RUN / 'collection_plan.json').read_text())
DATA = Path(PLAN['data_root'])
PORT = 8799


def status(stage, **extra):
    value = dict(stage=stage, updated_at=datetime.datetime.now().astimezone().isoformat(), **extra)
    atomic_json(RUN / 'status.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def api(path, body=None, timeout=10):
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}{path}',
        data=None if body is None else json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            value = json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(exc.read().decode(errors='replace')) from exc
    if value.get('success') is False:
        raise RuntimeError(value.get('message', str(value)))
    return value


def completed(root):
    result = {}
    for p in root.glob('episode_[0-9]*'):
        try:
            meta = json.loads((p/'rerecord.json').read_text())
            summary = json.loads((p/'episode_summary.json').read_text())
            if meta.get('accepted') and meta.get('plan_hash') == PLAN['hash'] and summary.get('task_success') and summary.get('physical_grasp_ok'):
                result[meta['task_id']] = str(p)
        except (OSError, ValueError, KeyError):
            pass
    return result


def stop_scene(process):
    if process is None:
        return
    for sig, delay in [(signal.SIGINT, 8), (signal.SIGTERM, 4), (signal.SIGKILL, 1)]:
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=delay)
        except subprocess.TimeoutExpired:
            pass


def assemble_basic():
    done = completed(DATA/'basic_replacements')
    originals = PLAN['basic_retained']
    replacements = [t for t in PLAN['tasks'] if t['phase'] == 'basic']
    if len(done) != 36:
        raise ValueError('Cannot publish basic160 until all 36 replacements pass')
    mapping = {r['episode_index']:r['source_path'] for r in originals}
    mapping.update({t['original_episode_index']:done[t['id']] for t in replacements})
    assert set(mapping) == set(range(160))
    root = DATA/'basic160'; root.mkdir(exist_ok=True)
    rows = []; manifest = []
    for i, source in sorted(mapping.items()):
        link = root/f'episode_{i:06d}'
        if not link.exists():
            link.symlink_to(source, target_is_directory=True)
        elif link.resolve() != Path(source).resolve():
            raise ValueError('Existing basic160 link differs from verified mapping')
        summary = json.loads((Path(source)/'episode_summary.json').read_text())
        summary['episode_index'] = i
        rows.append(summary)
        manifest.append(dict(episode_index=i, path=str(link), source_path=source,
            replaced=any(t['original_episode_index']==i for t in replacements)))
    for record in originals:
        actual = hashlib.sha256((Path(record['source_path'])/'frames.jsonl').read_bytes()).hexdigest()
        if actual != record['source_frames_sha256']:
            raise ValueError('Retained original demonstration changed')
    original_root = Path(originals[0]['source_path']).parent
    info = json.loads((original_root/'dataset_info.json').read_text())
    atomic_json(root/'dataset_info.json', info)
    (root/'episodes.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    atomic_json(root/'assembly_manifest.json', dict(complete=True, episodes=160, retained=124, replacements=36,
        plan_hash=PLAN['hash'], authoritative_episode_indices='assembly manifest and folder names; linked raw metadata retains recording-local indices', episodes_mapping=manifest))


def collect_phase(phase, limit, until_complete):
    root = DATA / ('basic_replacements' if phase == 'basic' else 'corrections_full')
    root.mkdir(exist_ok=True)
    tasks = [t for t in PLAN['tasks'] if t['phase']==phase]
    done = completed(root)
    if len(done) == len(tasks):
        if phase == 'basic': assemble_basic()
        return True
    runtime = RUN/'runs'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    runtime.mkdir(parents=True)
    snapshot = runtime/'collector_source'; snapshot.mkdir()
    hashes = {}
    for source in HERE.glob('*.py'):
        raw = source.read_bytes(); (snapshot/source.name).write_bytes(raw)
        hashes[source.name] = hashlib.sha256(raw).hexdigest()
    atomic_json(runtime/'collector_manifest.json', dict(files=hashes, plan_hash=PLAN['hash'],
        config_sha256=hashlib.sha256((RUN/'grasp_physics_minus90.json').read_bytes()).hexdigest()))
    current = runtime/'current.json'; os.environ['Z1_RERECORD_CURRENT'] = str(current)
    for pending in root.glob('.recording_*'):
        quarantine = runtime/'interrupted'; quarantine.mkdir(exist_ok=True)
        pending.rename(quarantine/pending.name)
    command = ['ros2', 'launch', str(HERE/'collection.launch.py'), 'external_camera_layout:=dual',
        'target_object:=cup', 'headless:=true', 'rviz:=false', 'task_gui:=true', 'run_smolvla_policy:=false',
        'freeze_target_keep_collisions:=true', 'wrist_rgb_topic:=/left_wrist_camera/color/image_raw',
        'vla_wrist_rgb_topic:=none', f'task_gui_port:={PORT}', f'dataset_root:={root}',
        'recording_fps:=10.0', 'recording_sync_tolerance_s:=0.045', 'recording_sync_delay_s:=0.5',
        'recording_action_age_s:=0.40']
    atomic_json(runtime/'command.json', command)
    process = None; attempts = 0
    try:
        api('/api/status', timeout=1)
    except Exception:
        pass
    else:
        raise RuntimeError('GUI port already occupied; refuse unknown scene')
    with (runtime/'scene.log').open('w') as log:
        try:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            status('starting_scene', phase=phase, scene_pid=process.pid, runtime=str(runtime), completed=len(done), quota=len(tasks))
            deadline = time.monotonic()+300
            while time.monotonic()<deadline:
                if process.poll() is not None:
                    raise RuntimeError('Simulator exited: '+str(runtime/'scene.log'))
                try:
                    s = api('/api/status', timeout=2)
                    if s.get('connected') and all(s.get(k,{}).get('ready') for k in ['recording','episode_reset','collision_scene','object_collisions']) and Path(s['recording']['dataset_root']).resolve()==root:
                        break
                except Exception:
                    pass
                time.sleep(2)
            else:
                raise RuntimeError('Scene readiness timed out')
            progress_guard = ProgressGuard()
            for round_index in round_indices(until_complete):
                before_round = len(completed(root))
                for task in tasks:
                    done = completed(root)
                    if task['id'] in done:
                        continue
                    selected = os.environ.get('Z1_RERECORD_TASKS','').split(',')
                    if selected != [''] and task['id'] not in selected:
                        continue
                    if limit and attempts>=limit:
                        status('attempt_limit_reached', phase=phase, completed=len(done), quota=len(tasks))
                        return False
                    frames = Path(task['original_path'])/'frames.jsonl'
                    if hashlib.sha256(frames.read_bytes()).hexdigest()!=task['original_frames_sha256']:
                        raise RuntimeError('Original demonstration changed: '+str(frames))
                    ctx = dict(task, plan_hash=PLAN['hash'], attempt_id=uuid.uuid4().hex, reference_trial=round_index-1,
                        collector_manifest=str(runtime/'collector_manifest.json'))
                    atomic_json(current, ctx)
                    attempts+=1
                    status('collecting', phase=phase, task_id=task['id'], original_seed=task['seed'],
                        original_grid=task['grid'], round=round_index, attempts=attempts,
                        completed=len(done), quota=len(tasks), runtime=str(runtime), scene_pid=process.pid)
                    row = dict(task_id=task['id'], attempt_id=ctx['attempt_id'], round=round_index)
                    try:
                        row['result'] = api('/api/demonstration/auto', {'target_object':'cup','curriculum_level':'full','target_grid':task['grid']}, timeout=900)
                        if task['id'] not in completed(root):
                            raise RuntimeError('API success without accepted saved demonstration')
                        row['saved'] = True
                        print('SAVED '+task['id'], flush=True)
                    except Exception as exc:
                        row.update(saved=False, error=str(exc))
                        print('FAILED '+task['id']+': '+str(exc), flush=True)
                        for path in ['/api/force_stop','/api/recording/discard']:
                            try: api(path, {}, timeout=20)
                            except Exception: pass
                        deadline = time.monotonic()+45
                        while api('/api/status').get('auto_demonstration',{}).get('running'):
                            if time.monotonic()>deadline:
                                raise RuntimeError('Previous attempt did not stop')
                            time.sleep(.5)
                    with (runtime/'attempts.jsonl').open('a') as f:
                        f.write(json.dumps(row,ensure_ascii=False)+'\n')
                done = completed(root)
                if len(done)==len(tasks):
                    if phase=='basic': assemble_basic()
                    status('phase_complete', phase=phase, completed=len(done), quota=len(tasks), runtime=str(runtime))
                    return True
                if not progress_guard.observe(before_round,len(done)):
                    status('needs_investigation', phase=phase, completed=len(done), quota=len(tasks),
                        reason='Three complete rounds saved no new demonstrations; stopped futile retries',
                        missing=[t['id'] for t in tasks if t['id'] not in done], runtime=str(runtime))
                    return False
                status('retrying_missing', phase=phase, completed=len(done), quota=len(tasks),
                    round=round_index, missing=[t['id'] for t in tasks if t['id'] not in done],
                    runtime=str(runtime), until_complete=until_complete)
            status('needs_investigation', phase=phase, completed=len(done), quota=len(tasks),
                missing=[t['id'] for t in tasks if t['id'] not in done], runtime=str(runtime))
            return False
        finally:
            if process:
                try: api('/api/force_stop', {}, timeout=8)
                except Exception: pass
                stop_scene(process)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['all','basic','correction'], default='all')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--until-complete', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    def terminate(*_): raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    with (RUN/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if args.phase in ['basic','all'] and not collect_phase('basic', args.limit, args.until_complete):
                return 2
            if args.phase in ['correction','all']:
                if not (DATA/'basic160/assembly_manifest.json').exists():
                    raise RuntimeError('Basic160 must complete before correction collection')
                if not collect_phase('correction', args.limit, args.until_complete):
                    return 2
            from verify_saved import verify
            integrity = verify()
            if args.phase == 'all' and not integrity['complete']:
                raise RuntimeError('Final data integrity audit is incomplete')
            status('completed', basic_replacements=len(completed(DATA/'basic_replacements')),
                full_corrections=len(completed(DATA/'corrections_full')))
            return 0
        except BaseException as exc:
            status('interrupted' if isinstance(exc,KeyboardInterrupt) else 'error', error=str(exc))
            raise


if __name__ == '__main__':
    raise SystemExit(main())
