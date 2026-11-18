"""Create a binary-command dataset only; this script never starts training."""
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

RUN = Path(__file__).resolve().parent
SOURCE = Path('/home/lp531/smolvla_data/z1_bottle_three_camera_pi05_20261006')
OUTPUT = Path('/home/lp531/smolvla_data/z1_bottle_three_camera_pi05_binary_20261007')
MANIFEST = Path('/home/lp531/pi05_lora_workspace/three_camera_20261006/split_manifest.json')


def label_episode(rows):
    """Infer intent from causal commanded actions and demonstrator phases, not feedback."""
    angles = np.asarray([r['action'][-1] for r in rows], dtype=np.float64)
    if not np.isfinite(angles).all():
        raise ValueError('Nonfinite gripper command')
    close = next((i for i, r in enumerate(rows)
                  if r['phase'] == 'GRASPING' and angles[i] > -.895), None)
    if close is None:
        raise ValueError('No recorded close onset')
    transport = [i for i in range(close + 1, len(rows)) if rows[i]['phase'] == 'TRANSPORT']
    full_open = next((i for i in transport if angles[i] <= -.899), None)
    if full_open is None:
        raise ValueError('No recorded release in TRANSPORT')
    before = full_open
    while before > transport[0] and angles[before] < angles[before - 1] - 1e-4:
        before -= 1
    opening = before + 1
    if not (transport[0] < opening <= full_open and angles[opening - 1] > -.75):
        raise ValueError('Ambiguous release boundary')
    if not all(rows[i]['phase'] == 'TRANSPORT' for i in range(opening, full_open + 1)):
        raise ValueError('Release ramp crosses incompatible phases')
    if np.any(np.diff(angles[opening:full_open + 1]) > 1e-4) or np.max(angles[full_open:]) >= -.88:
        raise ValueError('Release is not a final opening ramp')
    if close == 0 or any(r['phase'] in ('READY', 'PLACING', 'SUCCESS') for r in rows[close:opening]):
        raise ValueError('Unexpected episode phase order')
    labels = np.zeros(len(rows), dtype=np.float32)
    labels[close:opening] = 1
    return labels, dict(close_frame=close, open_frame=opening, fully_open_command_frame=full_open,
                        close_source_command=float(angles[close]),
                        open_source_command=float(angles[opening]),
                        hold_source_command=float(angles[opening - 1]))


def stats(array):
    a = np.asarray(array, dtype=np.float64)
    result = {name: fn(a, axis=0).tolist() for name, fn in
              [('min', np.min), ('max', np.max), ('mean', np.mean), ('std', np.std)]}
    result.update({f'q{q:02d}': np.quantile(a, q / 100, axis=0).tolist() for q in [1, 10, 50, 90, 99]})
    result['count'] = [len(a)]
    return result


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    staging = OUTPUT.with_name(OUTPUT.name + '.converting')
    if OUTPUT.exists() or staging.exists():
        raise FileExistsError('Refusing to overwrite existing output or incomplete conversion')
    manifest = json.loads(MANIFEST.read_text())
    entries = manifest['episodes']
    table = pq.read_table(SOURCE / 'data').sort_by([('index', 'ascending')])
    original_actions = np.asarray(table['action'].to_pylist(), dtype=np.float32)
    original_states = np.asarray(table['observation.state'].to_pylist(), dtype=np.float32)
    episode_ids = np.asarray(table['episode_index'].to_pylist())
    frame_ids = np.asarray(table['frame_index'].to_pylist())
    binary_actions = original_actions.copy()
    events = []
    for e in entries:
        raw = Path(e['source'])
        rows = [json.loads(line) for line in (raw / 'frames.jsonl').read_text().splitlines()]
        mask = episode_ids == e['episode_index']
        expected_actions = np.asarray([r['action'] for r in rows], dtype=np.float32)
        expected_states = np.asarray([r['observation.state'] for r in rows], dtype=np.float32)
        assert np.array_equal(original_actions[mask], expected_actions), raw
        assert np.array_equal(original_states[mask], expected_states), raw
        assert np.array_equal(frame_ids[mask], np.arange(len(rows))), raw
        labels, event = label_episode(rows)
        binary_actions[mask, -1] = labels
        threshold_labels = (expected_actions[:, -1] >= -.75).astype(np.float32)
        event.update(episode_index=e['episode_index'], raw_episode=raw.name, raw_root=str(raw),
                     frames=len(rows), close_labels=int(labels.sum()), open_labels=int((labels == 0).sum()),
                     differs_from_angle_threshold_frames=int(np.sum(labels != threshold_labels)),
                     source_frames_sha256=digest(raw / 'frames.jsonl'), split='train')
        events.append(event)
    assert len(events) == 160 and len(table) == 37557
    assert set(np.unique(binary_actions[:, -1])) == {0.0, 1.0}
    source_hashes = {str(p.relative_to(SOURCE)): digest(p) for parent in ['data', 'meta']
                     for p in (SOURCE / parent).rglob('*') if p.is_file()}
    shutil.copytree(SOURCE, staging, ignore=shutil.ignore_patterns('videos'))
    # Videos are immutable here: reuse their bytes without transcoding or changing timestamps.
    shutil.copytree(SOURCE / 'videos', staging / 'videos', copy_function=os.link)
    all_stats = json.loads((staging / 'meta/stats.json').read_text())
    all_stats['action'] = stats(binary_actions)
    all_stats['observation.state'] = stats(original_states)
    write_json(staging / 'meta/stats.json', all_stats)
    info = json.loads((staging / 'meta/info.json').read_text())
    info['features']['action']['names'][-1] = 'gripper_close_binary'
    info['splits'] = {'train': '0:160'}
    write_json(staging / 'meta/info.json', info)
    for p in (staging / 'data').rglob('*.parquet'):
        t = pq.read_table(p)
        indices = np.asarray(t['index'].to_pylist())
        updated = pa.array(binary_actions[indices].tolist(), type=t.schema.field('action').type)
        pq.write_table(t.set_column(t.schema.get_field_index('action'), 'action', updated), p)
    for p in (staging / 'meta/episodes').rglob('*.parquet'):
        t = pq.read_table(p)
        records = t.to_pylist()
        for row in records:
            mask = episode_ids == row['episode_index']
            for feature, a in [('action', binary_actions[mask]), ('observation.state', original_states[mask])]:
                for key, value in stats(a).items():
                    column = f'stats/{feature}/{key}'
                    if column in row:
                        row[column] = value
        pq.write_table(pa.Table.from_pylist(records, schema=t.schema), p)
    converted = pq.read_table(staging / 'data').sort_by([('index', 'ascending')])
    for name in table.column_names:
        if name != 'action':
            assert converted[name].to_pylist() == table[name].to_pylist(), name
    actual = np.asarray(converted['action'].to_pylist(), dtype=np.float32)
    assert np.array_equal(actual, binary_actions)
    assert np.array_equal(actual[:, :6], original_actions[:, :6])
    videos = []
    for p in (SOURCE / 'videos').rglob('*.mp4'):
        new = staging / p.relative_to(SOURCE)
        assert os.path.samefile(p, new)
        videos.append(dict(path=str(p.relative_to(SOURCE)), sha256=digest(p)))
    for path, sha in source_hashes.items():
        assert digest(SOURCE / path) == sha, f'Source modified: {path}'
    contract = dict(action_gripper_encoding={'open': 0, 'close': 1},
                    action_arm_encoding='six absolute joint target angles, unchanged',
                    observation_gripper_encoding='actual jointGripper angle in radians, unchanged',
                    deployment_mapping={'threshold': 0.5, 'open_target_rad': -.9, 'close_target_rad': 0.0},
                    label_source='recorded causal gripper commands plus phase and final monotonic release ramp',
                    inferred_intent=True, explicit_intent_field_in_raw=False,
                    source_dataset=str(SOURCE), raw_manifest=str(MANIFEST),
                    episodes=160, frames=len(table), train_episodes=160, validation_episodes=0,
                    frames_removed=0, fps=10, training_started=False,
                    training_batch_or_chunk_15='awaiting user clarification',
                    requires_binary_model_output_adapter=True)
    write_json(staging / 'meta/gripper_contract.json', contract)
    staging.rename(OUTPUT)
    write_json(RUN / 'episode_labels.json', events)
    audit = dict(passed=True, dataset=str(OUTPUT), **contract,
                 open_frames=int(np.sum(actual[:, -1] == 0)), closed_frames=int(np.sum(actual[:, -1] == 1)),
                 differences_from_naive_threshold=sum(e['differs_from_angle_threshold_frames'] for e in events),
                 all_non_action_columns_identical=True, arm_action_columns_identical=True,
                 original_source_hashes=source_hashes, videos=videos,
                 episodes_with_exactly_one_close_and_one_open=160)
    write_json(RUN / 'conversion_audit.json', audit)
    print(json.dumps({k: audit[k] for k in ['passed','dataset','episodes','frames','open_frames','closed_frames','differences_from_naive_threshold','training_started']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
