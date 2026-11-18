#!/usr/bin/env python3
"""Freeze current training sources and mark >=8 cm reverse excursions for retake.

Does not edit raw episodes, launch recorders, or change the running bottle queue.
Training eligibility remains to be resolved during final dataset assembly.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import review
import screen_fruit_motion as motion
from screen_cup_training import cup_command_labels

WORK=Path('/home/lp531')
TRAINING=WORK/'pi05_lora_workspace/pi05_current_round_20261111/data_sources.json'
CACHED_REPORTS=[WORK/'z1_queue_logs/fruit_motion_height_review_20261111/report.json',
                WORK/'z1_queue_logs/cup_training_motion_review_20261111/report.json']
RETAKES=WORK/'robot/bottle_wait8_retake12_20261111/retake_tasks.json'
MASTER=TRAINING.with_name('pending_rerecord_reverse8.json')


def qualifying(segments,threshold=.08):
    # Compare the unrounded metric, not display-rounded centimetres.
    return [s for s in segments if float(s['max_reverse_excursion_m'])>=threshold]


def add_bottle_metrics(path,chain,fixed):
    item,data,view=motion.read_episode(path,chain,fixed,motion.Rules())
    root=path.parent
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    labels=review.read_json(root/'gripper_labels.json',{}).get('labels')
    if labels is not None and len(labels)==len(rows) and np.isin(labels,[0,1]).all():
        labels=np.asarray(labels,int);label_source='saved_gripper_labels'
    else:
        labels=cup_command_labels(rows);label_source='cylinder_command_hysteresis_close_-0.82_open_-0.875'
    times=np.asarray([r['simulation_time'] for r in rows])
    phases=np.asarray(view['phases'],dtype=object)
    meta=review.read_json(root/'correction.json',{})
    # Bottle corrections document intentional clearance/approach legs under a
    # different schema than the fruit collector. Preserve those boundaries.
    for stage in meta.get('recovery_stages',[]):
        if not all(k in stage for k in ['start','end','stage']):continue
        chosen=(times>=stage['start'])&(times<stage['end'])
        phases[chosen]=[str(p)+'#'+stage['stage'] for p in phases[chosen]]
    position,rotation=review.forward(np.asarray([r['observation.state'][:6] for r in rows]),chain)
    gripper=position+np.einsum('nij,j->ni',rotation,fixed)
    segments=motion.reverse_segments(gripper,phases,labels,data['expert'],times,
        item['fps'],item['fragment'],motion.Rules())
    item['motion_segments']=segments;item['gripper_label_source']=label_source
    item['reverse_analysis_note']='Deliberate intervention/nonexpert frames excluded; documented recovery legs split.'
    return item


def freeze_sources(manifest):
    sources=[]
    for source in manifest['base_sources']:
        sources.append(dict(root=source['root'],scope='registered_base',
            already_excluded=source.get('exclude_episodes',[])))
    for source in manifest['additions']:
        sources.append(dict(root=source['root'],scope='registered_addition',
            already_excluded=source.get('exclude_episodes',[])))
    for stage in manifest['replacement_queue']['stages']:
        sources.append(dict(root=stage['data'],scope='saved_replacement_snapshot',already_excluded=[]))
    result=[];seen=set();coverage=[]
    for source in sources:
        root=Path(source['root']);count=0;not_final=[]
        for summary in sorted(root.glob('episode_*/episode_summary.json')):
            path=summary.parent/'frames.jsonl'
            if not path.exists():continue
            if source['scope']=='saved_replacement_snapshot':
                # Summaries may appear before the running recorder has finished
                # provenance/time-gate sidecars; leave those to the next scan.
                gate=review.read_json(path.parent/'time_screen.json',{})
                if not gate.get('passed') or not (path.parent/'recording_provenance.json').exists():
                    not_final.append(str(path.parent));continue
                if gate['frames_sha256']!=motion.digest(path):
                    not_final.append(str(path.parent));continue
            physical=str(path.parent.resolve())
            if physical in seen:continue
            seen.add(physical);count+=1
            result.append(dict(path=str(path),physical_path=physical,root=str(root),
                scope=source['scope'],already_excluded=str(path.parent) in source['already_excluded']))
        coverage.append(dict(root=str(root),scope=source['scope'],frozen_saved_episodes=count,
            skipped_unfinished_episodes=not_final))
    return result,coverage


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--mark',action='store_true',help='Publish marks into training-source registration')
    args=parser.parse_args();out=args.out
    if out.exists() and any(out.iterdir()):raise FileExistsError('Refuse overwrite')
    out.mkdir(parents=True,exist_ok=True)
    original=TRAINING.read_bytes();manifest=json.loads(original)
    frozen_at=datetime.now().astimezone().isoformat()
    snapshot,coverage=freeze_sources(manifest)
    review.atomic_json(out/'input_snapshot.json',dict(frozen_at=frozen_at,coverage=coverage,episodes=snapshot))
    cached={};report_pins={}
    for path in CACHED_REPORTS:
        report=review.read_json(path);assert report['passed']
        report_pins[str(path)]=motion.digest(path)
        for item in report['results']:cached[str(Path(item['path']).resolve())]=item
    chain=review.chain_from_urdf(review.DEFAULT_URDF)
    robot=ET.parse(review.DEFAULT_URDF).getroot()
    fixed=np.fromstring(robot.find("joint[@name='gripperStatorLink']/origin").get('xyz'),sep=' ')
    tasks=review.read_json(RETAKES)
    existing_retake={str(Path(t['rerecord_source']).resolve()):t['id'] for t in tasks}
    assessed=[];marked=[];errors=[]
    for i,entry in enumerate(snapshot,1):
        path=Path(entry['path']);physical=entry['physical_path']
        try:
            item=cached.get(physical)
            if item is not None:
                for name,expected in item['source_hashes'].items():
                    if motion.digest(path.parent/name)!=expected:raise ValueError('Cached source changed: '+name)
                item=copy.deepcopy(item)
            else:
                if review.read_json(path.parent/'episode_summary.json')['target_object']!='beverage_bottle':
                    raise ValueError('No frozen report for non-bottle source')
                item=add_bottle_metrics(path,chain,fixed)
            hits=qualifying(item['motion_segments'])
            maximum=max([s['max_reverse_excursion_m'] for s in item['motion_segments']],default=0.)
            result=dict(logical_path=str(path.parent),physical_path=physical,scope=entry['scope'],
                object=item['object'],grid=item['grid'],kind=item['kind'],fps=item['fps'],
                source_hashes=item['source_hashes'],max_reverse_excursion_m=maximum,
                matched_segments=hits,gripper_label_source=item['gripper_label_source'],
                already_excluded=entry['already_excluded'],
                existing_retake_task_id=existing_retake.get(physical),
                source_analysis='frozen_report' if physical in cached else 'fresh_bottle_frames_analysis')
            assessed.append(result)
            if hits:
                summary=review.read_json(path.parent/'episode_summary.json')
                result=copy.deepcopy(result)
                result.update(status='pending_rerecord',status_zh='待重新录制',
                    reason='single_reverse_excursion_ge_8cm',threshold_m=.08,
                    seed=summary.get('scene_randomization',{}).get('seed'),
                    source_frames_sha256=item['source_hashes']['frames.jsonl'],
                    task_execution_status='already_queued_for_wait8_retake' if physical in existing_retake else 'not_scheduled',
                    mark_id=hashlib.sha256((physical+'\n'+item['source_hashes']['frames.jsonl']).encode()).hexdigest())
                marked.append(result)
        except Exception as exc:errors.append(dict(path=str(path),error=str(exc)))
        if i%200==0:print('Assessed',i,'/',len(snapshot),flush=True)
    review.write_csv(out/'读取问题.csv',errors)
    if errors:
        review.atomic_json(out/'failed_analysis.json',dict(errors=errors,assessed=len(assessed),expected=len(snapshot)))
        raise RuntimeError('Incomplete assessment; marks were NOT published')
    for item in assessed:
        for name,expected in item['source_hashes'].items():
            assert motion.digest(Path(item['physical_path'])/name)==expected
    assert TRAINING.read_bytes()==original,'Training source manifest changed concurrently'
    assert len({e['mark_id'] for e in marked})==len(marked)
    assert all(e['max_reverse_excursion_m']>=.08 and e['matched_segments'] for e in marked)
    marked.sort(key=lambda e:(e['object'],-e['max_reverse_excursion_m'],e['physical_path']))
    registry=dict(schema='pending_rerecord_reverse8_v1',status='marked_pending_rerecord',
        frozen_at=frozen_at,threshold_m=.08,comparison='>= applied before rounding',
        criterion='Per episode maximum reverse excursion in a phase with stable jaw intent; loaded XY, open XYZ.',
        training_source_manifest=str(TRAINING),training_source_before_sha256=hashlib.sha256(original).hexdigest(),
        assessed_episodes=len(assessed),marked_episodes=len(marked),
        assessed_by_object=dict(Counter(e['object'] for e in assessed)),
        marked_by_object=dict(Counter(e['object'] for e in marked)),
        marked_by_scope=dict(Counter(e['scope'] for e in marked)),
        already_queued_wait8_overlap=sum(bool(e['existing_retake_task_id']) for e in marked),
        pending_replacement_snapshot=True,require_rescan_after_replacement_completion=True,
        source_files_changed=False,active_recording_queue_changed=False,
        execution_started=False,marks_published=bool(args.mark),coverage=coverage,
        reused_report_pins=report_pins,analysis_code_sha256=motion.digest(__file__),episodes=marked)
    rows=[]
    for e in marked:
        rows.append(dict(状态=e['status_zh'],物体=e['object'],来源范围=e['scope'],
            数据集=Path(e['logical_path']).parent.name,Episode=Path(e['logical_path']).name,
            网格=e['grid'],种子=e['seed'],单次最大回退厘米=round(e['max_reverse_excursion_m']*100,2),
            命中区间=[dict(阶段=s['phase'],坐标=s['metric'],开始秒=round(s['reverse_start_frame']/e['fps'],2),
                结束秒=round(s['reverse_end_frame_exclusive']/e['fps'],2),回退厘米=round(s['max_reverse_excursion_m']*100,2)) for s in e['matched_segments']],
            原已排除训练=e['already_excluded'],原有重录任务=e['existing_retake_task_id'] or '',
            执行状态=e['task_execution_status'],原始路径=e['logical_path'],物理路径=e['physical_path'],
            Frames_SHA256=e['source_frames_sha256'],替换后的路径='',备注=''))
    review.write_csv(out/'全部物体反向8厘米以上待重录.csv',rows)
    review.atomic_json(out/'pending_rerecord_reverse8.json',registry)
    review.atomic_json(out/'assessment_all_episodes.json',assessed)
    (out/'data_sources.before_mark.json').write_bytes(original)
    if args.mark:
        review.atomic_json(MASTER,registry)
        marked_by_root={}
        for e in marked:
            marked_by_root.setdefault(str(Path(e['logical_path']).parent),[]).append(e['logical_path'])
        for source in manifest['base_sources']+manifest['additions']:
            source['pending_rerecord_episodes']=marked_by_root.get(source['root'],[])
        manifest.setdefault('quality_control',{})['reverse_excursion_ge_8cm']=dict(
            disposition='pending_rerecord',registry=str(MASTER),review_csv=str(out/'全部物体反向8厘米以上待重录.csv'),
            marked_episodes=len(marked),threshold_m=.08,frozen_at=frozen_at,
            handling='Resolve pending marks before final training assembly; raw episodes preserved.',
            saved_replacement_episode_marks=[e['logical_path'] for e in marked if e['scope']=='saved_replacement_snapshot'],
            require_rescan_after_replacement_completion=True,
            rescan_script=str(Path(__file__).resolve()),active_recording_queue_unchanged=True)
        manifest['updated_at']=datetime.now().astimezone().isoformat()
        review.atomic_json(TRAINING,manifest)
        assert review.read_json(MASTER)['marked_episodes']==len(marked)
        assert review.read_json(TRAINING)['quality_control']['reverse_excursion_ge_8cm']['marked_episodes']==len(marked)
    review.atomic_json(out/'summary.json',{k:v for k,v in registry.items() if k not in ['episodes','reused_report_pins','coverage']})
    print(json.dumps({k:v for k,v in registry.items() if k not in ['episodes','reused_report_pins','coverage']},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
