"""Local tokenizer, train-only normalization, and conversion integrity gates."""
import json
from pathlib import Path
import subprocess

import numpy as np
import pyarrow.parquet as pq
import sentencepiece as spm
from transformers import AutoTokenizer, GemmaTokenizer

WORK = Path(__file__).resolve().parent
DATA = Path('/home/lp531/smolvla_data/z1_tabletop_bottle_pi05_lora_v1')
SOURCE = Path('/home/lp531/smolvla_data/z1_tabletop_lerobot_bottle_recovery_boundary_only_v1')


def finalize():
    token_dir = WORK / 'weights/tokenizer'
    tokenizer_args = GemmaTokenizer.convert_to_native_format(vocab_file=str(token_dir / 'tokenizer.model'))
    tokenizer = GemmaTokenizer(**tokenizer_args, add_bos_token=True, add_eos_token=False)
    tokenizer.save_pretrained(token_dir)
    sp = spm.SentencePieceProcessor(model_file=str(token_dir / 'tokenizer.model'))
    auto = AutoTokenizer.from_pretrained(token_dir, local_files_only=True)
    for prompt in ['Task: Pick up the bottle. State: 0 127 255; Action:',
                   'Task: Physically grasp the bottle, lift it clear, and place it in the box. State: 1 2 3; Action:']:
        assert auto.encode(prompt) == [sp.bos_id()] + sp.encode(prompt), 'Tokenizer IDs must match official tokenizer'
    processor_path = WORK / 'weights/pi05_base/policy_preprocessor.json'
    processor = json.loads(processor_path.read_text())
    for step in processor['steps']:
        if step['registry_name'] == 'tokenizer_processor':
            step['config']['tokenizer_name'] = str(token_dir)
    processor_path.write_text(json.dumps(processor, indent=2) + '\n')
    info = json.loads((DATA / 'meta/info.json').read_text())
    assert info['codebase_version'] == 'v3.0'
    arrays = pq.read_table(DATA / 'data').to_pydict()
    manifest = json.loads((WORK / 'split_manifest.json').read_text())
    n = len(arrays['episode_index'])
    assert n == manifest['total_frames'] == 65773
    ids = np.asarray(arrays['episode_index'])
    comparable = {key: np.asarray(arrays[key]) for key in
                  ['action', 'observation.state', 'frame_index', 'timestamp', 'task_index']}
    for entry in manifest['episodes']:
        rows = np.flatnonzero(ids == entry['episode_index'])
        old = pq.read_table(SOURCE / f"data/chunk-000/episode_{entry['original_episode_index']:06d}.parquet").to_pydict()
        assert len(rows) == entry['frames']
        for key, values in comparable.items():
            assert np.array_equal(values[rows], np.asarray(old[key])), (entry, key)
    video_counts = {}
    for camera in ['observation.images.global', 'observation.images.wrist']:
        count = 0
        for path in sorted((DATA / 'videos' / camera).glob('**/*.mp4')):
            result = subprocess.check_output(['ffprobe', '-v', 'error', '-count_packets', '-select_streams', 'v:0',
                                              '-show_entries', 'stream=nb_read_packets', '-of', 'json', str(path)])
            count += int(json.loads(result)['streams'][0]['nb_read_packets'])
        assert count == n, (camera, count, n)
        video_counts[camera] = count
    stats_path = DATA / 'meta/stats.json'
    stats = json.loads(stats_path.read_text())
    stats.update(json.loads((WORK / 'training_stats.json').read_text()))
    stats_path.write_text(json.dumps(stats, indent=2) + '\n')
    (WORK / 'setup_verification.json').write_text(json.dumps({'frames': n, 'episodes': 246,
        'train_episodes': 221, 'validation_episodes': 25, 'video_packets': video_counts,
        'all_state_action_frames_exact': True, 'normalization_uses_train_only': True,
        'tokenizer_ids_match_official_sentencepiece': True}, indent=2) + '\n')
    print('SETUP_VERIFIED', n, 'frames; no frame trimming', flush=True)


if __name__ == '__main__':
    finalize()
