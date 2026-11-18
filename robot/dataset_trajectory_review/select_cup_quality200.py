#!/usr/bin/env python3
"""Select 200 original 10 Hz cup basics, preserving 10 episodes in each cell.

Never changes source episodes or starts training/recording. Full frames and JPEG
checks precede publication of an independent symlink subset and source manifest.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
from scipy.ndimage import median_filter
from scipy.stats import rankdata

import review

WORK = Path('/home/lp531')
OLD_RUN = WORK/'pi05_lora_workspace/pi05_cup_recovery513_lora_20261029'
SOURCE = WORK/'smolvla_data/z1_tabletop_raw_cup_basic_4x5x20_renderfix_20261026'
BASE_SCOPE = Path(__file__).with_name('current_round_sources.json')
RETAKES = WORK/'robot/bottle_wait8_retake12_20261111/retake_tasks.json'
QUEUE = WORK/'z1_queue_logs/bottle_rerecord172_queue_20261111/queue_manifest.json'
WEIGHTS = dict(stationary_s=.25, internal_max_wait_s=.20,
               backward_m=.25, backward_fraction=.15,
               detour_m=.10, duration_s=.05)
REJECT_TYPES = {'长停顿候选', '末尾停留偏长', '反向移动较多',
                '绕行或反复调整候选', '盒附近往返抖动候选',
                '局部往返抖动候选', '缓慢漂移或低进展',
                '采样时间间断', '摘要帧数不符'}
CAMERAS = ['wrist', 'global', 'global_right']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def motion_metrics(positions, phases, labels, fps):
    """Reverse projection inside a phase AND one jaw command intent only.

    Do not compare home->object->box as a single straight line. <1.5 mm movements
    are suppressed after a three-frame median filter and 0.3-second sampling.
    Segments with net displacement <3 cm have no reliable goal axis: score their
    path excess as detour instead of inventing a backward direction.
    """
    smooth = median_filter(positions, size=(3, 1), mode='nearest')
    edges = np.flatnonzero(np.r_[True,
        (phases[1:] != phases[:-1]) | (labels[1:] != labels[:-1]), True])
    backward = path = detour = 0.
    details = []
    for a, b in zip(edges[:-1], edges[1:]):
        if (b-a)/fps < 2:
            continue
        sampled = smooth[np.unique(np.r_[np.arange(a, b, max(1, round(.3*fps))), b-1])]
        kept = [sampled[0]]
        for point in sampled[1:]:
            if np.linalg.norm(point-kept[-1]) >= .0015:
                kept.append(point)
        if len(kept) < 2:
            continue
        points = np.asarray(kept)
        vectors = np.diff(points, axis=0)
        length = float(np.linalg.norm(vectors, axis=1).sum())
        net = float(np.linalg.norm(points[-1]-points[0]))
        reverse = 0.
        if net >= .03:
            projection = vectors @ ((points[-1]-points[0])/net)
            reverse = float(-projection[projection < 0].sum())
        backward += reverse
        path += length
        detour += max(0., length-net)
        details.append(dict(start_frame=int(a), end_frame_exclusive=int(b),
            phase=str(phases[a]), gripper_label=int(labels[a]), path_m=length,
            net_m=net, backward_m=reverse))
    return dict(backward_m=backward, backward_fraction=backward/max(path, 1e-12),
                motion_path_m=path, detour_m=detour, motion_segments=details)


def evaluate(record, binary_actions, chain):
    root = Path(record['source_path'])
    for filename, expected in record['source_hashes'].items():
        if sha(root/filename) != expected:
            raise ValueError('Historical source pin changed: '+str(root/filename))
    summary = review.read_json(root/'episode_summary.json')
    for key in ['task_success', 'physical_grasp_ok', 'bilateral_contact_ever',
                'object_lifted_clear_of_table_ever', 'release_sequence_complete',
                'placement_stable']:
        if summary.get(key) is not True:
            raise ValueError('Missing physical success: '+key)
    if summary['dropped_frames'] != 0 or summary['fps'] != 10:
        raise ValueError('Dropped frames or wrong fps')
    result = review.analyze(root/'frames.jsonl', chain, review.Settings())
    rows = [json.loads(line) for line in (root/'frames.jsonl').read_text().splitlines()]
    n = len(rows)
    q = np.asarray([r['observation.state'] for r in rows], float)
    action = np.asarray([r['action'] for r in rows], float)
    times = np.asarray([r['simulation_time'] for r in rows])
    if q.shape != (n, 7) or action.shape != (n, 7):
        raise ValueError('State/action dimension mismatch')
    if len(binary_actions) != n or not np.isin(binary_actions[:, -1], [0, 1]).all():
        raise ValueError('Invalid historical jaw labels')
    if not np.allclose(binary_actions[:, :6], action[:, :6], rtol=0, atol=1e-6):
        raise ValueError('Historical action alignment mismatch')
    if not np.allclose(np.diff(times), .1, atol=1e-5, rtol=0):
        raise ValueError('Not a regular 10 Hz sequence')
    for row in rows:
        if row.get('target_object') != 'cup':
            raise ValueError('Wrong object')
        if row.get('action_stamp', row['simulation_time']) > row['simulation_time']+1e-5:
            raise ValueError('Noncausal action')
    positions, rotations = review.forward(q[:, :6], chain)
    still = review.hold_mask(q, action, times, positions, rotations, 10., review.Settings())
    spans = review.runs(still)
    internal = [(a,b) for a,b in spans if a > 0 and b < n]
    result.update(motion_metrics(positions,
        np.asarray([r.get('phase', 'UNKNOWN') for r in rows]), binary_actions[:, -1], 10.))
    result.update(grid=record['grid'], seed=record['seed'],
        old_episode_index=record['episode_index'], old_split=record['split'],
        source_hashes=record['source_hashes'], task=record['task'],
        internal_max_wait_s=max([(b-a)/10. for a,b in internal], default=0.),
        max_stationary_run_s=max([(b-a)/10. for a,b in spans], default=0.),
        rejected_reasons=sorted(REJECT_TYPES.intersection(f['type'] for f in result['findings'])))
    return result


def select_balanced(items, per_cell=10):
    groups = defaultdict(list)
    for item in items:
        if not item.get('rejected_reasons'):
            groups[tuple(item['grid'])].append(item)
    expected = {(x,y) for x in range(5) for y in range(4)}
    if set(groups) != expected or any(len(g) < per_cell for g in groups.values()):
        raise ValueError('Insufficient clean episodes in a spatial cell: '+str(
            {str(k):len(v) for k,v in groups.items()}))
    chosen = []
    for grid in sorted(groups):
        group = groups[grid]
        for item in group:
            item['quality_score'] = 0.
            item['quality_percentiles'] = {}
        for key, weight in WEIGHTS.items():
            ranks = (rankdata([e[key] for e in group], method='average')-.5)/len(group)
            for item, rank in zip(group, ranks):
                item['quality_percentiles'][key] = float(rank)
                item['quality_score'] += weight*float(rank)
        ordered = sorted(group, key=lambda e:(e['quality_score'],e['old_episode_index']))
        for i,item in enumerate(ordered):
            item['rank_within_cell'] = i+1
        chosen.extend(ordered[:per_cell])
    return sorted(chosen, key=lambda e:e['old_episode_index'])


def verify_images(item):
    root = Path(item['path'])
    rows = [json.loads(l) for l in (root/'frames.jsonl').read_text().splitlines()]
    digest = hashlib.sha256()
    shapes = review.read_json(root.parent/'dataset_info.json')['image_shapes']
    count = 0
    for row in rows:
        for camera in CAMERAS:
            value = row['observation.images.'+camera]
            path = root/value
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError('Image escapes episode: '+str(path))
            blob = path.read_bytes()
            with Image.open(path) as image:
                image.load()
                if image.size != (shapes[camera][1],shapes[camera][0]):
                    raise ValueError('Image dimensions mismatch: '+str(path))
            digest.update(value.encode());digest.update(b'\0')
            digest.update(hashlib.sha256(blob).digest())
            count += 1
    return dict(episode=item['episode'], decoded_images=count,
                ordered_image_sha256=digest.hexdigest())


def stats(items):
    keys = ['duration_s','stationary_s','internal_max_wait_s',
            'max_stationary_run_s','backward_m','backward_fraction','detour_m']
    return {key:dict(mean=float(np.mean([e[key] for e in items])),
        median=float(np.median([e[key] for e in items])),
        p95=float(np.quantile([e[key] for e in items], .95)),
        max=float(max(e[key] for e in items))) for key in keys}


def write_selection_csv(path, items, chosen):
    selected = {e['path'] for e in chosen}
    review.write_csv(path, [dict(入选=e['path'] in selected, Episode=e['episode'],
        网格=e['grid'], 原训练索引=e['old_episode_index'], 原划分=e['old_split'],
        时长秒=e['duration_s'], 总近静止秒=e['stationary_s'],
        阶段中最长近静止秒=e['internal_max_wait_s'],
        全程最长近静止秒=e['max_stationary_run_s'],
        累计反向毫米=round(e['backward_m']*1000,3),
        反向路径比例=round(e['backward_fraction'],6),
        阶段路径超出直线毫米=round(e['detour_m']*1000,3),
        质量分=e.get('quality_score',''), 网格内排名=e.get('rank_within_cell',''),
        排除依据='；'.join(e['rejected_reasons']), 原始路径=e['path']) for e in items])


def register_current_training(subset, manifest_path):
    """Register addition separately from the historical-screening whitelist.

    The 172 replacement episodes must pass their existing queue audits before
    final training assembly. Never put old long-wait episodes and retakes in
    the same training list, and never pretend growing directories are frozen.
    """
    scope = review.read_json(BASE_SCOPE)
    tasks = review.read_json(RETAKES)
    replacements = [t['rerecord_source'] for t in tasks]
    queue = review.read_json(QUEUE)
    base = []
    for root in scope['roots']:
        episodes = [p for p in sorted(Path(root).glob('episode_*'))
                    if (p/'frames.jsonl').is_file() and (p/'episode_summary.json').is_file()]
        base.append(dict(root=root, saved_episodes_at_registration=len(episodes),
            exclude_episodes=[p for p in replacements if str(Path(p).parent)==root]))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(schema='pi05_raw_training_sources_v1',
        updated_at=dt.datetime.now().astimezone().isoformat(),
        status='sources_registered_pending_replacement_audits', training_started=False,
        intended_split='all_train_no_validation', fps=10,
        base_scope_file=str(BASE_SCOPE), base_sources=base,
        additions=[dict(root=str(subset), episodes=200, object='cup', pool='basic',
            manifest=str(subset/'selection_manifest.json'), status='verified')],
        replacement_queue=dict(manifest=str(QUEUE), target=queue['target'],
            sources_to_replace=replacements, stages=queue['stages'],
            require_completed_stage_audits=True),
        rules=['Resolve actual saved episodes and replacement maps after queue audits.',
               'Exclude original 12 long-wait sources when applying their replacement map.',
               'Deduplicate by resolved frames path and frames SHA256 before training.',
               'This registers data sources only; no sampling weights or optimizer configuration changed.'])
    review.atomic_json(manifest_path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--subset', type=Path, required=True)
    parser.add_argument('--training-manifest', type=Path, required=True)
    args = parser.parse_args()
    if args.subset.exists():
        raise FileExistsError('Refusing to overwrite subset: '+str(args.subset))
    args.out.mkdir(parents=True, exist_ok=True)
    records = review.read_json(OLD_RUN/'split_manifest.json')['source_records']
    records = [r for r in records if r['source']=='basic']
    assert len(records)==400 and all(Path(r['source_path']).parent==SOURCE for r in records)
    assert Counter(tuple(r['grid']) for r in records)==Counter({(x,y):20 for x in range(5) for y in range(4)})
    chain = review.chain_from_urdf(review.DEFAULT_URDF)
    with np.load(OLD_RUN/'lowdim.npz') as lowdim:
        binary_actions = lowdim['actions']
    items = []
    for i,record in enumerate(records):
        a,b = record['dataset_from_index'],record['dataset_to_index']
        items.append(evaluate(record, binary_actions[a:b], chain))
        if (i+1)%50==0:
            print('Analyzed',i+1,'/400',flush=True)
    chosen = select_balanced(items)
    assert len(chosen)==200
    review.atomic_json(args.out/'all_metrics.json',items)
    write_selection_csv(args.out/'全部400条评分清单.csv',items,chosen)
    review.atomic_json(args.out/'status.json',dict(stage='decoding_selected_images',selected=200))
    checks = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i,check in enumerate(pool.map(verify_images, chosen),1):
            checks.append(check)
            if i%25==0:
                print('Verified images',i,'/200',flush=True)
    # Recheck historical pins before publishing references after lengthy IO.
    for item in chosen:
        for filename,expected in item['source_hashes'].items():
            assert sha(Path(item['path'])/filename)==expected
    assert len({e['frames_sha256'] for e in chosen})==200
    assert len({e['seed'] for e in chosen})==200
    current_paths = [p for root in review.read_json(BASE_SCOPE)['roots']
                     for p in Path(root).glob('episode_*/frames.jsonl')]
    current_hashes = {sha(p) for p in current_paths}
    assert not current_hashes.intersection(e['frames_sha256'] for e in chosen)
    args.subset.mkdir(parents=True)
    shutil.copy2(SOURCE/'dataset_info.json',args.subset/'dataset_info.json')
    for item in chosen:
        (args.subset/item['episode']).symlink_to(Path(item['path']),target_is_directory=True)
    evidence_log = WORK/'z1_queue_logs/cup_basic_4x5x20_renderfix_20261026/scene.log'
    evidence = [dict(path=str(evidence_log),line=i,text=line.strip())
                for i,line in enumerate(evidence_log.read_text().splitlines(),1)
                if 'Close-range glass occlusion:' in line and '0.0001' in line]
    assert evidence
    selected_records = [dict(episode_index=i,source='cup_basic_quality200',
        source_path=e['path'],subset_path=str(args.subset/e['episode']),
        old_episode_index=e['old_episode_index'],old_split=e['old_split'],
        frames=e['frames'],grid=e['grid'],seed=e['seed'],task=e['task'],
        frames_sha256=e['frames_sha256'],source_hashes=e['source_hashes'],
        quality_score=e['quality_score'],motion_segments=e['motion_segments'])
        for i,e in enumerate(chosen)]
    report = dict(passed=True,source=str(SOURCE),source_training_run=str(OLD_RUN),
        candidates=400,selected=200,frames=sum(e['frames'] for e in chosen),fps=10,
        source_pool='basic_only_no_corrections',spatial_quota='20 cells, 10 per cell',
        selection_weights=WEIGHTS,hard_exclusions=sorted(REJECT_TYPES),
        rejected_by_rules=sum(bool(e['rejected_reasons']) for e in items),
        source_stats=stats(items),selected_stats=stats(chosen),
        grid_counts={str(k):v for k,v in sorted(Counter(tuple(e['grid']) for e in chosen).items())},
        image_checks=checks,images_decoded=sum(x['decoded_images'] for x in checks),
        render_fix_runtime_evidence=evidence,render_image_exhaustive_visual_review=False,
        no_frames_sha_overlap_with_current_round=True,
        original_episode_files_unchanged=True,training_started=False,
        records=selected_records)
    review.atomic_json(args.out/'selection_report.json',report)
    review.atomic_json(args.subset/'selection_manifest.json',report)
    write_selection_csv(args.out/'入选200条清单.csv',chosen,chosen)
    shutil.copy2(args.out/'入选200条清单.csv',args.subset/'入选200条清单.csv')
    training = register_current_training(args.subset,args.training_manifest)
    review.atomic_json(args.out/'status.json',dict(stage='complete',passed=True,
        selected=200,subset=str(args.subset),training_manifest=str(args.training_manifest),
        pending_replacement_target=training['replacement_queue']['target']))
    print(json.dumps(dict(passed=True,selected=200,images=report['images_decoded'],
        subset=str(args.subset),stats=report['selected_stats']),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
