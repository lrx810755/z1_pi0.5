"""Reuse the immutable 192-episode dataset and verify new phase annotations."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from phases import build_phase_weights, probabilities

RUN = Path(__file__).resolve().parent
WORK = RUN.parent
PREVIOUS = WORK / 'gentle_correction_base_20261010'
MERGED = Path('/home/lp531/smolvla_data/z1_bottle_gentle_correction_20261010')
REPO = 'lp531/z1_bottle_gentle_correction_20261010'
sys.path.insert(0, str(WORK / 'binary_gripper_20261007'))
from convert import label_episode


def write(name, value):
    (RUN / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    old_records = json.loads((PREVIOUS / 'phase_manifest.json').read_text())
    originals = json.loads((WORK / 'binary_gripper_20261007/episode_labels.json').read_text())
    corrections = json.loads((PREVIOUS / 'snapshot.json').read_text())
    assert len(originals) == 160 and len(corrections) == 32
    table = pq.read_table(MERGED / 'data').sort_by([('index', 'ascending')])
    episode = table['episode_index'].to_numpy()
    frames = table['frame_index'].to_numpy()
    actions = np.asarray(table['action'].to_pylist(), dtype=np.float32)
    states = np.asarray(table['observation.state'].to_pylist(), dtype=np.float32)
    phase_weights = np.ones(len(table), dtype=np.float64)
    records = []
    for old in old_records:
        ep = old['episode_index']
        source = originals[ep] if ep < 160 else corrections[ep - 160]
        raw = Path(source['raw_root'] if ep < 160 else source['source'])
        source_hash = source['source_frames_sha256'] if ep < 160 else source['frames_sha256']
        assert digest(raw / 'frames.jsonl') == source_hash, raw
        rows = [json.loads(line) for line in (raw / 'frames.jsonl').read_text().splitlines()]
        labels, event = label_episode(rows)
        mask = episode == ep
        assert len(rows) == old['frames'] == int(mask.sum())
        assert np.array_equal(frames[mask], np.arange(len(rows)))
        assert np.array_equal(actions[mask, -1], labels), (ep, 'binary labels')
        raw_actions = np.asarray([r['action'] for r in rows], dtype=np.float32)
        raw_states = np.asarray([r['observation.state'] for r in rows], dtype=np.float32)
        assert np.array_equal(actions[mask, :6], raw_actions[:, :6]), (ep, 'arm actions')
        assert np.array_equal(states[mask], raw_states), (ep, 'states')
        weights, stages = build_phase_weights(rows, event['close_frame'], event['open_frame'])
        assert np.all(labels[weights == 1.5] == 0)
        assert np.all(labels[weights == 2] == 1)
        phase_weights[mask] = weights
        record = {k: v for k, v in old.items() if k != 'key_frames'}
        record.update(stages, source_path=str(raw), source_frames_sha256=source_hash,
                      phase_frame_counts={str(w): int((weights == w).sum()) for w in [1., 1.5, 2.]})
        records.append(record)
    held = [r['episode_index'] for r in records if r['split'] == 'validation']
    assert len(held) == 4 and held == [r['episode_index'] for r in old_records if r['split'] == 'validation']
    train = ~np.isin(episode, held)
    p = probabilities((episode >= 160)[train], phase_weights[train])
    source_stats = Path('/home/lp531/smolvla_data/z1_bottle_three_camera_pi05_binary_20261007/meta/stats.json')
    assert json.loads((MERGED / 'meta/stats.json').read_text()) == json.loads(source_stats.read_text())
    np.savez(RUN / 'sampling_arrays.npz', episode=episode, correction=episode >= 160,
             phase_weight=phase_weights, key=phase_weights != 1)
    write('phase_manifest.json', records)
    write('snapshot.json', corrections)
    write('preparation_audit.json', dict(passed=True, original_episodes=160, correction_episodes=32,
          train_episodes=188, train_frames=int(train.sum()), validation_episodes=held,
          total_frames=len(table), all_binary_labels_exact=True, all_arm_actions_exact=True,
          all_state_values_exact=True, all_source_frame_hashes_match=True, normalization_preserved=True,
          dataset=str(MERGED), dataset_stats_sha256=digest(MERGED / 'meta/stats.json'),
          train_phase_sampling_mass={str(w): float(p[phase_weights[train] == w].sum()) for w in [1., 1.5, 2.]},
          phase_definition='last 1s before close:1.5; close through 5 consecutive physical_grasp_ok AND off-table frames:2; otherwise:1',
          loss_weight=1, old_recovery_interval_extra_weight=False))
    print('PHASE PREPARATION VERIFIED', len(records), len(table), held, flush=True)


if __name__ == '__main__':
    prepare()
