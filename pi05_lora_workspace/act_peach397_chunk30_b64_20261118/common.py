"""Paths and immutable-artifact helpers for the independent peach ACT run."""
from pathlib import Path
import hashlib
import json

RUN = Path(__file__).resolve().parent
PARENT = RUN.parent / 'pi05_current_round_20261111'
OUTPUT = Path('/home/lp531/smolvla_training/act_peach397_chunk30_b64_20261118')
CACHE = RUN / 'cache'
CAMERAS = ['observation.images.' + x for x in ['wrist', 'global', 'global_right']]
CROPS = {k: (-140 if k.endswith('wrist') else -80, 0, 640, 640) for k in CAMERAS}
SEED = 20261118
HORIZON = 30


def read(path):
    return json.loads(Path(path).read_text())


def write(name, data):
    dest = RUN / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + '\n')
    tmp.replace(dest)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_files(files):
    for path, expected in files.items():
        if sha(path) != expected:
            raise ValueError('Pinned artifact changed: ' + str(path))


def assert_sources(records):
    for record in records:
        root = Path(record['source_path'])
        for name, digest in record['source_hashes'].items():
            if sha(root / name) != digest:
                raise ValueError('Frozen source changed: ' + str(root / name))


def verify_prepared():
    marker = read(RUN / 'prepared.json')
    verify_files(marker['files'])
    assert_sources(read(RUN / 'split_manifest.json')['source_records'])
    return marker


def checkpoint_fingerprints(checkpoint):
    checkpoint=Path(checkpoint)
    return {str(p.relative_to(checkpoint)):sha(p) for p in sorted(checkpoint.rglob('*')) if p.is_file()}


def require_training_evidence(run=RUN,formal=False):
    run=Path(run);prep=sha(run/'prepared.json');cache_path=run/'cache/completed.json';cache=read(cache_path)
    if not cache.get('complete') or (cache.get('episodes'),cache.get('frames'),cache.get('images'))!=(397,117537,352611) or cache.get('prepared_sha256')!=prep:
        raise ValueError('Incomplete or changed cache preparation')
    snapshot=run/'code_snapshot.json';verify_files(read(snapshot)['files']);code=sha(snapshot)
    cpu=read(run/'cpu_verified.json')
    if not cpu.get('passed') or cpu.get('prepared_sha256')!=prep or cpu.get('cache_manifest_sha256')!=sha(cache_path) or cpu.get('code_snapshot_sha256')!=code:
        raise ValueError('Current CPU/cache/code verification required')
    if not formal:return
    completed=read(run/'smoke_completed.json');first=read(run/'smoke_first_update_verified.json');live=read(run/'smoke_live_training.json')
    if completed.get('optimizer_updates')!=4 or Path(completed.get('output','')).resolve()!=(run/'smoke').resolve() or live.get('optimizer_updates')!=4 or live.get('actual_optimizer_steps')!=[4] or first.get('actual_optimizer_steps')!=[1] or not first.get('action_head_changed'):
        raise ValueError('Four actual smoke optimizer updates are required')
    latest=read(run/'smoke_latest_checkpoint.json');checkpoint=Path(latest['checkpoint'])
    if latest.get('optimizer_updates')!=4 or not checkpoint.resolve().is_relative_to((run/'smoke').resolve()):
        raise ValueError('Unexpected smoke checkpoint')
    reload=read(run/'smoke_reload_verified.json')
    if not reload.get('passed') or reload.get('prepared_sha256')!=prep or reload.get('cache_manifest_sha256')!=sha(cache_path) or reload.get('code_snapshot_sha256')!=code or reload.get('checkpoint')!=str(checkpoint) or reload.get('checkpoint_files')!=checkpoint_fingerprints(checkpoint):
        raise ValueError('Smoke checkpoint reload evidence is stale or checkpoint changed')
