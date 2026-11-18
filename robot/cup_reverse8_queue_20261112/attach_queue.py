#!/usr/bin/python3
"""Idempotently register seven unique cup correction replacements after reverse41."""
from collections import Counter
from datetime import datetime
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
LOGS=Path('/home/lp531/z1_queue_logs/cup_reverse8_queue_20261112')
TRAIN=Path('/home/lp531/pi05_lora_workspace/pi05_current_round_20261111')
PARENT=Path('/home/lp531/z1_queue_logs/bottle_rerecord172_queue_20261111')
REVERSE=Path('/home/lp531/z1_queue_logs/reverse8_queue_20261112')
SERVICE='cup-reverse8-rerecord7-20261112.service'
sys.path.insert(0,'/home/lp531/robot/dataset_trajectory_review')
from review import atomic_json

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())

def main():
    with (PARENT/'.attachment.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        plan=read(ROOT/'plan.json');training=read(TRAIN/'data_sources.json');parent=read(PARENT/'queue_manifest.json');marks=read(TRAIN/'pending_rerecord_reverse8.json')
        existing=read('/home/lp531/robot/reverse8_queue_20261112/plan.json')['tasks']
        old_paths={str(Path(t['source']).resolve()) for t in existing};old_hashes={t['source_frames_sha256'] for t in existing}
        with (PARENT/'录制清单.csv').open(encoding='utf-8-sig') as stream:
            for r in csv.DictReader(stream):
                p=Path(r['原始记录']).resolve();old_paths.add(str(p))
                if (p/'frames.jsonl').exists():old_hashes.add(digest(p/'frames.jsonl'))
        paths=set();hashes=set()
        for task in plan['tasks']:
            p=str(Path(task['rerecord_source']).resolve());h=task['rerecord_source_frames_sha256']
            if p in old_paths or h in old_hashes or p in paths or h in hashes:raise RuntimeError('Duplicate recording source: '+p)
            if digest(Path(p)/'frames.jsonl')!=h:raise RuntimeError('Original source changed: '+p)
            paths.add(p);hashes.add(h)
        assert len(paths)==7
        extension=dict(id=plan['id'],plan=str(ROOT/'plan.json'),plan_hash=plan['plan_hash'],target=7,
            service=SERVICE,executor=str(ROOT/'run_extension.py'),
            completion_audit=str(LOGS/'completion_verified.json'),replacement_map=str(LOGS/'replacement_map.json'),
            starts_after_extension='reverse8_20261112',stage_data_roots=[plan['data']])
        for d in [parent,training['replacement_queue']]:
            extensions={e['id']:e for e in d.get('extensions',[])}
            if plan['id'] in extensions and extensions[plan['id']]['plan_hash']!=plan['plan_hash']:raise RuntimeError('Existing7-task identity differs')
            extensions[plan['id']]=extension;d['extensions']=list(extensions.values())
        parent['combined_recording_operations']=172+sum(e['target'] for e in parent['extensions'])
        assert parent['combined_recording_operations']==220
        atomic_json(PARENT/'queue_manifest.json',parent)
        for source in training['additions']:
            if source['root']=='/home/lp531/smolvla_data/z1_cup_correction100_full_renderfix_20261027':
                source.update(status='registered_pending_queued_replacements',quality_note='Seven reverse>=8cm corrections queued after reverse41; original files preserved.',queue_service=SERVICE,queue_plan=str(ROOT/'plan.json'))
        qc=training['quality_control']['reverse_excursion_ge_8cm'];qc.update(marked_episodes=48,queued_episodes=48)
        qc['queue_services']=['reverse8-rerecord41-20261112.service',SERVICE]
        training['updated_at']=datetime.now().astimezone().isoformat();atomic_json(TRAIN/'data_sources.json',training)
        bypath={str(Path(r['physical_path']).resolve()):r for r in marks['episodes']}
        for t in plan['tasks']:
            p=Path(t['rerecord_source']);physical=str(p.resolve());h=t['rerecord_source_frames_sha256']
            mark=bypath.get(physical,{})
            mark.update(logical_path=physical,physical_path=physical,scope='registered_addition',object='cup',
                kind=t['kind'],grid=t['grid'],seed=t['seed'],fps=10.,source_frames_sha256=h,
                source_hashes={name:digest(p/name) for name in ['frames.jsonl','episode_start.json','episode_summary.json']},
                max_reverse_excursion_m=t['original_max_reverse_m'],status='pending_rerecord',status_zh='待重新录制',
                reason='single_reverse_excursion_ge_8cm',threshold_m=.08,source_analysis='cup100_reverse_review',
                mark_id=hashlib.sha256((physical+'\n'+h).encode()).hexdigest(),task_execution_status='queued_after_reverse41',
                queue_task_id=t['id'],queue_plan=str(ROOT/'plan.json'),queue_service=SERVICE,destination_dataset=plan['data'])
            bypath[physical]=mark
        marks['episodes']=list(bypath.values());marks['marked_episodes']=len(marks['episodes']);assert marks['marked_episodes']==48
        marks['marked_by_object']=dict(Counter(r['object'] for r in marks['episodes']));marks['marked_by_scope']=dict(Counter(r['scope'] for r in marks['episodes']))
        marks.update(queued_episodes=48,queue_services=qc['queue_services']);marks['additional_screenings']=[dict(source=str(TRAIN/'cup100_reverse_review.json'),episodes=100,new_marks=7)]
        atomic_json(TRAIN/'pending_rerecord_reverse8.json',marks)
        report=read(TRAIN/'cup100_reverse_review.json');report.update(queued=True,queue_service=SERVICE,queue_plan=str(ROOT/'plan.json'));atomic_json(TRAIN/'cup100_reverse_review.json',report)
        with (REVERSE/'统一队列213条.csv').open(encoding='utf-8-sig') as stream:rows=list(csv.DictReader(stream))
        assert len(rows)==213
        rows += [dict(序号=214+i,队列阶段='杯子纠偏反向≥8cm补录',任务ID=t['id'],物体='cup',原始记录=t['rerecord_source'],示范池='correction',网格=t['grid'],新数据目录=plan['data']) for i,t in enumerate(plan['tasks'])]
        assert len(rows)==220 and len({str(Path(r['原始记录']).resolve()) for r in rows})==220
        temporary=LOGS/'统一队列220条.csv.tmp'
        with temporary.open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        temporary.replace(LOGS/'统一队列220条.csv')
        atomic_json(LOGS/'attachment_verified.json',dict(verified=True,added=7,duplicate_sources=0,total_recording_operations=220,
            service=SERVICE,after_extension='reverse8_20261112',attached_at=datetime.now().astimezone().isoformat()))
        print('7 unique cup corrections attached after reverse41; total220; repeated attachment keeps one extension')

if __name__=='__main__':main()
