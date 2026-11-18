"""Publish counts and a per-episode checklist from verified conversion artifacts."""
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from config import RUN
from dataset_prepare import write
from verify import verify_prepared


def main():
    verify_prepared()
    read = lambda name: json.loads((RUN/name).read_text())
    cpu = read('cpu_verification.json')
    images = read('image_audit_complete.json')
    labels = read('final_label_timeline_audit.json')
    records = read('split_manifest.json')['source_records']
    reverse = {r['path']: r for r in read('final_reverse_screen.json')}
    if not all(x['passed'] and x['episodes'] == len(records) for x in (cpu, images, labels)):
        raise ValueError('Final audits do not cover this conversion')
    by_object = {}
    with np.load(RUN/'sampling_arrays.npz') as arrays:
        valid = arrays['eligible_train']
        for obj in ('apple', 'peach', 'cup', 'bottle'):
            selected = [r for r in records if r['object'] == obj]
            basic = sum(r['source'] == obj+'_basic' for r in selected)
            by_object[obj] = dict(basic=basic, correction=len(selected)-basic,
                total=len(selected), frames=sum(r['frames'] for r in selected),
                partial_episodes=sum(r['fragment'] for r in selected))
        fields = ['episode_index', 'object', 'source', 'source_path', 'frames',
            'valid_30frame_windows', 'fragment', 'label_source', 'label_repair',
            'basic_label_changed_frames', 'causal_intervention_extra_excluded_frames',
            'max_reverse_excursion_cm', 'retained_by_user', 'original_task']
        path = RUN/'训练集逐条清单.csv'
        with path.with_suffix('.csv.tmp').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for r in records:
                row = {k: r.get(k) for k in fields}
                quality = reverse[r['source_path']]
                row.update(valid_30frame_windows=int(valid[r['dataset_from_index']:r['dataset_to_index']].sum()),
                    max_reverse_excursion_cm=round(quality['max_reverse_excursion_m']*100, 3),
                    retained_by_user=bool(quality['pending'] and quality['accepted_by_user']))
                writer.writerow(row)
        path.with_suffix('.csv.tmp').replace(path)
        effective_frames = int(arrays['effective'].sum())
    sources = Counter(str(Path(r['source_path']).parent) for r in records)
    counts = dict(checked_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        count_basis='Verified final split_manifest, recursive replacement maps and deduplication',
        total_episodes=len(records), basic_episodes=sum(x['basic'] for x in by_object.values()),
        correction_episodes=sum(x['correction'] for x in by_object.values()),
        by_object=by_object, by_source=dict(sources),
        total_frames=sum(r['frames'] for r in records), fps=10,
        registered_replacement_edges=len(read('sources_frozen.json')['replacement_edges']),
        collection_queues_audited=True, source_blockers=[], read_errors=[],
        final_conversion_done=True, training_started=False)
    if images['images'] != counts['total_frames']*3:
        raise ValueError('JPEG count differs from converted frame count')
    write('dataset_count_after_replacements.json', counts)
    report = dict(counts, ready_for_manual_training=True,
        training_windows=read('sampling_audit.json')['training_windows'],
        normalization_frames=effective_frames, decoded_images=images['images'],
        image_decode_errors=len(images['errors']),
        labels_and_timelines_passed=True, task_text_preserved=True,
        normalization_roundtrip_passed=cpu['normalization_roundtrip'],
        basic_label_changed_episodes=labels['basic_label_changed_episodes'],
        basic_label_changed_frames=labels['basic_label_changed_frames'],
        causal_boundary_extra_masked_frames=labels['causal_boundary_extra_masked_frames'],
        label_repairs=read('sampling_audit.json')['label_repairs'],
        reverse_episodes_retained_by_user=[r for r in reverse.values() if r['pending'] and r['accepted_by_user']],
        bottle_addition=dict(accepted=25, original_plan=30, cancelled=5, closed_by_user=True),
        validations=dict(image='image_audit_complete.json', labels='final_label_timeline_audit.json',
            conversion='preparation_complete.json', cpu='cpu_verification.json',
            sampling='sampling_audit.json', tests='preparation_tests.log'),
        manual_training_command='bash /home/lp531/SH/pi05_train.sh',
        gpu_smoke_started=False, raw_sources_modified=False)
    write('training_preparation_report.json', report)
    print(json.dumps({k: report[k] for k in ('total_episodes', 'total_frames', 'decoded_images',
        'training_windows', 'normalization_frames', 'ready_for_manual_training')}, indent=2))


if __name__ == '__main__':
    main()
