"""Subset the approved peach records, labels and expert masks without relabeling."""
import copy
from collections import Counter
from pathlib import Path
import shutil
import numpy as np
from common import RUN, PARENT, CAMERAS, HORIZON, read, write, sha, assert_sources, verify_files


def valid_anchors(expert):
    expert = np.asarray(expert, dtype=bool)
    result = np.zeros(len(expert), dtype=bool)
    if len(expert) >= HORIZON:
        sums = np.r_[0, np.cumsum(expert.astype(np.int32))]
        result[:len(expert)-HORIZON+1] = sums[HORIZON:] - sums[:-HORIZON] == HORIZON
    return result


def statistics(values, effective):
    array = values[effective].astype(np.float64)
    return dict(mean=array.mean(0).tolist(), std=array.std(0).tolist(), min=array.min(0).tolist(),
                max=array.max(0).tolist(), count=[len(array)])


def main():
    if (RUN / 'prepared.json').exists():
        from common import verify_prepared
        verify_prepared(); print('Existing peach preparation verified'); return
    parent_marker = read(PARENT / 'preparation_complete.json')
    verify_files(parent_marker['files'])
    source_manifest = read(PARENT / 'split_manifest.json')
    originals = [r for r in source_manifest['source_records'] if r['object'] == 'peach']
    assert len(originals) == 397
    assert Counter(r['source'] for r in originals) == {'peach_basic': 340, 'peach_correction': 57}
    assert all(r['split'] == 'train' and r['task'] == r['original_task'] for r in originals)
    assert_sources(originals)
    with np.load(PARENT / 'lowdim.npz') as archive:
        low = {k: archive[k] for k in archive.files}
    with np.load(PARENT / 'sampling_arrays.npz') as archive:
        old_sampling = {k: archive[k] for k in archive.files}
    indices = np.concatenate([np.arange(r['dataset_from_index'], r['dataset_to_index']) for r in originals])
    records = []
    cursor = 0
    for index, old in enumerate(originals):
        record = copy.deepcopy(old)
        record.update(original_episode_index=old['episode_index'], episode_index=index,
                      dataset_from_index=cursor, dataset_to_index=cursor+old['frames'])
        record['source'] = 'correction' if old['source'].endswith('_correction') else 'basic'
        cursor += old['frames']; records.append(record)
        dest = RUN / 'source_pins' / f'episode_{index:06d}'
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(old['source_path']) / 'frames.jsonl', dest / 'frames.jsonl')
        assert sha(dest/'frames.jsonl') == old['source_hashes']['frames.jsonl']
    arrays = {k: v[indices].copy() for k, v in low.items()}
    arrays['episode'] = np.repeat(np.arange(397), [r['frames'] for r in records])
    sampling = {k: v[indices].copy() for k, v in old_sampling.items() if k not in ['weights', 'source_pool']}
    sampling['episode'] = arrays['episode'].copy()
    sampling['source_pool'] = np.repeat([int(r['source']=='correction') for r in records], [r['frames'] for r in records])
    sampling['weights'] = np.maximum(np.where(sampling['key'], 1.5, 1.), np.where(sampling['recovery'], 3., 1.))
    sampling['weights'][~sampling['eligible_train']] = 0
    for record in records:
        sl = slice(record['dataset_from_index'], record['dataset_to_index'])
        np.testing.assert_array_equal(sampling['eligible_train'][sl], valid_anchors(sampling['expert'][sl]))
        assert not sampling['eligible_validation'][sl].any()
        assert record['frames'] >= HORIZON
    assert len(indices) == 117537 and int(sampling['eligible_train'].sum()) == 106024
    assert np.bincount(sampling['source_pool'][sampling['eligible_train']], minlength=2).tolist() == [100580, 5444]
    assert np.isin(arrays['actions'][:, -1], [0., 1.]).all()
    np.savez(RUN / 'lowdim.npz', **arrays)
    np.savez(RUN / 'sampling_arrays.npz', **sampling)
    with np.load(PARENT / 'image_pins.npz') as archive:
        np.savez(RUN / 'image_pins.npz', **{k: archive[k][indices] for k in archive.files})
    split = dict(source_records=records, train_episodes=list(range(397)), validation_episodes=[],
                 parent_manifest_sha256=sha(PARENT/'split_manifest.json'), source_global_indices=indices.tolist())
    write('split_manifest.json', split)
    stats = {k: statistics(arrays[v], sampling['effective']) for k, v in [('observation.state','states'), ('action','actions')]}
    for cam in CAMERAS:
        stats[cam] = dict(mean=[[v] for v in [.485,.456,.406]], std=[[v] for v in [.229,.224,.225]])
        stats[cam] = {k: np.asarray(v).reshape(3,1,1).tolist() for k,v in stats[cam].items()}
    write('train_stats.json', stats)
    files = {str(RUN/n): sha(RUN/n) for n in ['split_manifest.json','lowdim.npz','sampling_arrays.npz','image_pins.npz','train_stats.json']}
    files.update({str(PARENT/n): sha(PARENT/n) for n in ['split_manifest.json','preparation_complete.json']})
    write('prepared.json', dict(passed=True, episodes=397, basic=340, correction=57, frames=117537,
          eligible_windows=106024, basic_windows=100580, correction_windows=5444,
          normalization_frames=int(sampling['effective'].sum()), files=files, all_training=True,
          copied_labels_exact=True, copied_expert_mask_exact=True, image_format='native JPEG', original_files_modified=False))
    print((RUN/'prepared.json').read_text(), flush=True)


if __name__ == '__main__': main()
