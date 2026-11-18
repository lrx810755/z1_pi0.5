"""Verify the local Z1 archive without loading Isaac or a policy."""
import csv
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_HOME = Path('/home/lp531')
RUN = ROOT / 'pi05_lora_workspace/pi05_current_round_20261111'
EVAL = ROOT / 'robot/pi05_four_objects_eval_20261117'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def file_sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(data)
    return result.hexdigest()


def check_inventory():
    with (ROOT / 'manifests/file_inventory.tsv').open(encoding='utf-8', newline='') as stream:
        entries = list(csv.DictReader(stream, delimiter='\t'))
    hashes = {}
    total_bytes = 0
    for entry in entries:
        relative = entry['archive_path']
        path = ROOT / relative
        if relative in hashes or not path.resolve().is_relative_to(ROOT):
            raise ValueError(f'Duplicate or out-of-tree archive path: {relative}')
        if entry['type'] == 'symlink':
            if not path.is_symlink() or not path.exists():
                raise ValueError(f'Invalid archived symlink: {relative}')
            actual_hash = hashlib.sha256(os.fsencode(os.readlink(path))).hexdigest()
        else:
            size = int(entry['bytes'])
            if not path.is_file() or path.is_symlink() or path.stat().st_size != size:
                raise ValueError(f'Missing file or wrong size; hydrate Git LFS first: {relative}')
            actual_hash = file_sha256(path)
            total_bytes += size
        if actual_hash != entry['sha256']:
            raise ValueError(f'Payload changed: {relative}')
        hashes[relative] = actual_hash
    print(f'Archive inventory verified: {len(hashes):,} files and links, {total_bytes / 2**30:.2f} GiB')
    return hashes


def check_pins(hashes):
    passed = []
    for filename, key in [(EVAL / 'source_snapshot.json', 'files'),
                          (RUN / 'preparation_complete.json', 'files'),
                          (RUN / 'sources_frozen.json', 'input_pins')]:
        pins = read_json(filename)[key]
        for original, expected in pins.items():
            p = Path(original)
            if p.is_absolute():
                relative = p.relative_to(ORIGINAL_HOME).as_posix()
            else:
                relative = (filename.parent.relative_to(ROOT) / p).as_posix()
            if hashes.get(relative) != expected:
                raise ValueError(f'Reproduction pin mismatch: {relative}')
        passed.append(len(pins))
    print('Evaluation source and training preparation pins verified:', passed)


def check_results():
    rows = read_json(RUN / 'split_manifest.json')['source_records']
    counts = Counter(r['object'] for r in rows)
    if dict(counts) != {'apple': 382, 'peach': 397, 'cup': 500, 'bottle': 483}:
        raise ValueError(f'Unexpected frozen object counts: {counts}')
    meta = read_json(ROOT / 'manifests/external_artifacts.json')['frozen_data']
    if meta['episodes'] != len(rows) or meta['jpeg_count'] != 1703757 or meta['physical_source_directories'] != 18:
        raise ValueError('External data manifest disagrees with frozen training')
    fixtures = [
        ('pi05_final_execute15_finger_table_filtered_30each_20261118/paired_comparison_from_first_inference.json',120,7,12),
        ('pi05_final_filtered_execute30_15each_20261118/paired_comparison.json',60,8,2),
    ]
    for name, total, baseline, filtered in fixtures:
        trials = read_json(ROOT/'smolvla_eval'/name)['trials']
        actual = (len({(t['target'],t['seed']) for t in trials}),sum(t['baseline_success'] for t in trials),
                  sum(t['filtered_success'] for t in trials))
        if actual != (total,baseline,filtered):
            raise ValueError(f'Archived pairing changed: {name}: {actual}')
    if read_json(RUN/'completed.json')['optimizer_updates'] != 100000:
        raise ValueError('Unexpected reported training completion')
    links = read_json(ROOT / 'manifests/cup_selection_symlinks.json')['links']
    if len({item['name'] for item in links}) != 200:
        raise ValueError('Historical cup selection link count differs')
    print('Frozen data, 200 historical cup links, optimization count, and both completed paired results verified')


def check_reference_links():
    files = [ROOT/'README.md', *(ROOT/'docs'/part/'README.md' for part in ('history','portfolio','reproduction'))]
    for f in files:
        for p in re.findall(r'\]\(([^)]+)\)',f.read_text(encoding='utf-8')):
            if '://' not in p and not p.startswith('#') and not (f.parent / p.split('#')[0]).exists():
                raise ValueError(f'Broken archive index link: {f.relative_to(ROOT)} -> {p}')
    print('Project index, showpiece, history, and reproduction links verified')


def main():
    hashes = check_inventory()
    check_pins(hashes)
    check_results()
    check_reference_links()


if __name__ == '__main__':
    main()
