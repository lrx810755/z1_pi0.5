"""Permute complete episodes for stratified holdout; preserve every recorded frame."""
import copy
import json
import os
from pathlib import Path
import random
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

WORK = Path(__file__).resolve().parent
SOURCE = Path('/home/lp531/smolvla_data/z1_tabletop_lerobot_bottle_recovery_boundary_only_v1')
DEST = Path('/home/lp531/smolvla_data/z1_tabletop_bottle_pi05_lora_v1')
MANIFEST = Path('/home/lp531/smolvla_data/z1_tabletop_bottle_recovery_boundary_only_prepared_v1/selection_manifest.json')


def split_episodes(entries):
    rng = random.Random(20261001)
    train, val = [], []
    for group, count in [('baseline', 19), ('left', 2), ('right', 2), ('high', 2)]:
        ids = [i for i, entry in enumerate(entries) if entry['group'] == group]
        rng.shuffle(ids)
        val.extend(ids[:count])
        train.extend(ids[count:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n')


def prepare():
    if (WORK / 'split_manifest.json').exists() and (DEST / 'meta/info.json').exists():
        print('Dataset copy already prepared', flush=True)
        return
    if DEST.exists():
        raise FileExistsError(f'Refusing to overwrite incomplete dataset: {DEST}')
    entries = json.loads(MANIFEST.read_text())['episodes']
    train, val = split_episodes(entries)
    assert len(entries) == 246 and len(set(train + val)) == 246
    seeds = [json.loads((Path(e['source_episode']) / 'episode_start.json').read_text())
             ['scene_randomization']['seed'] for e in entries]
    assert len(set(seeds)) == 246, 'Scene seeds must not cross train/validation boundary'
    info = json.loads((SOURCE / 'meta/info.json').read_text())
    episodes = [json.loads(line) for line in (SOURCE / 'meta/episodes.jsonl').read_text().splitlines()]
    stats = [json.loads(line) for line in (SOURCE / 'meta/episodes_stats.jsonl').read_text().splitlines()]
    (DEST / 'meta').mkdir(parents=True)
    shutil.copy2(SOURCE / 'meta/tasks.jsonl', DEST / 'meta/tasks.jsonl')
    reordered_episodes, reordered_stats, mapping = [], [], []
    offset = 0
    train_arrays = {'observation.state': [], 'action': []}
    for new_id, old_id in enumerate(train + val):
        old_rel = info['data_path'].format(episode_chunk=0, episode_index=old_id)
        new_rel = info['data_path'].format(episode_chunk=0, episode_index=new_id)
        table = pq.read_table(SOURCE / old_rel)
        count = len(table)
        ep_stat = copy.deepcopy(stats[old_id])
        for key in train_arrays:
            array = np.asarray(table[key].to_pylist(), dtype=np.float32)
            ep_stat['stats'][key]['q01'] = np.quantile(array, .01, axis=0).tolist()
            ep_stat['stats'][key]['q99'] = np.quantile(array, .99, axis=0).tolist()
            if new_id < len(train):
                train_arrays[key].append(array)
        for key, values in [('episode_index', np.full(count, new_id, dtype=np.int64)),
                            ('index', np.arange(offset, offset + count, dtype=np.int64))]:
            table = table.set_column(table.schema.get_field_index(key), table.schema.field(key), pa.array(values))
            ep_stat['stats'][key] = {'min': [int(values.min())], 'max': [int(values.max())],
                                    'mean': [float(values.mean())], 'std': [float(values.std())], 'count': [count]}
        target = DEST / new_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, target)
        reread = pq.read_table(target)
        original = pq.read_table(SOURCE / old_rel)
        for key in original.column_names:
            if key not in ('episode_index', 'index'):
                assert original[key].equals(reread[key]), (old_id, key)
        assert count == entries[old_id]['retained_frames']
        for camera in ('observation.images.global', 'observation.images.wrist'):
            old_video = SOURCE / info['video_path'].format(episode_chunk=0, episode_index=old_id, video_key=camera)
            new_video = DEST / info['video_path'].format(episode_chunk=0, episode_index=new_id, video_key=camera)
            new_video.parent.mkdir(parents=True, exist_ok=True)
            os.link(old_video, new_video)
        episode = copy.deepcopy(episodes[old_id])
        episode['episode_index'] = new_id
        ep_stat['episode_index'] = new_id
        reordered_episodes.append(episode)
        reordered_stats.append(ep_stat)
        mapping.append({'episode_index': new_id, 'original_episode_index': old_id,
                        'split': 'train' if new_id < len(train) else 'validation',
                        'group': entries[old_id]['group'], 'frames': count, 'scene_seed': seeds[old_id]})
        offset += count
    assert offset == info['total_frames'] == 65773
    for filename, rows in [('episodes.jsonl', reordered_episodes), ('episodes_stats.jsonl', reordered_stats)]:
        (DEST / 'meta' / filename).write_text(''.join(json.dumps(row) + '\n' for row in rows))
    write_json(DEST / 'meta/info.json', info)
    training_stats = {}
    for key, pieces in train_arrays.items():
        array = np.concatenate(pieces)
        training_stats[key] = {name: func(array, axis=0).tolist() for name, func in
                               [('min', np.min), ('max', np.max), ('mean', np.mean), ('std', np.std)]}
        training_stats[key].update(q01=np.quantile(array, .01, axis=0).tolist(),
                                   q99=np.quantile(array, .99, axis=0).tolist(), count=[len(array)])
    write_json(WORK / 'training_stats.json', training_stats)
    write_json(WORK / 'split_manifest.json', {'seed': 20261001, 'train_episodes': 221,
                                             'validation_episodes': 25, 'total_frames': offset,
                                             'all_non_index_columns_exact': True,
                                             'videos_hardlinked_without_reencoding': True,
                                             'episodes': mapping})
    print(f'Prepared {len(entries)} complete episodes, {offset} frames at {DEST}', flush=True)


if __name__ == '__main__':
    prepare()
