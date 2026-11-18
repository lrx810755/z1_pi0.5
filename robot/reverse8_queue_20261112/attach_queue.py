#!/usr/bin/python3
"""Idempotently register this fixed continuation without touching active stages."""
from datetime import datetime
import copy
import csv
import fcntl
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,'/home/lp531/robot/dataset_trajectory_review')
from review import atomic_json,write_csv
from screen_fruit_motion import digest

WORK=Path('/home/lp531')
PARENT=WORK/'z1_queue_logs/bottle_rerecord172_queue_20261111'
LOGS=WORK/'z1_queue_logs/reverse8_queue_20261112'
TRAIN=WORK/'pi05_lora_workspace/pi05_current_round_20261111'
SERVICE='reverse8-rerecord41-20261112.service'


def main():
    plan=json.loads((HERE/'plan.json').read_text());LOGS.mkdir(parents=True,exist_ok=True)
    with (LOGS/'.attachment.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        parent=json.loads((PARENT/'queue_manifest.json').read_text())
        training=json.loads((TRAIN/'data_sources.json').read_text())
        pending=json.loads((TRAIN/'pending_rerecord_reverse8.json').read_text())
        expected={t['mark_id']:t for t in plan['tasks']}
        assert len(expected)==plan['target']==41
        marks={e['mark_id']:e for e in pending['episodes']}
        assert expected.keys()<=marks.keys()
        for task in plan['tasks']:
            assert digest(Path(task['source'])/'frames.jsonl')==task['source_frames_sha256']
        for source,backup in [(PARENT/'queue_manifest.json','parent_queue.before_attachment.json'),
                              (TRAIN/'data_sources.json','training_sources.before_attachment.json'),
                              (TRAIN/'pending_rerecord_reverse8.json','pending_marks.before_attachment.json')]:
            if not (LOGS/backup).exists():(LOGS/backup).write_bytes(source.read_bytes())
        extension=dict(id='reverse8_20261112',plan=str(HERE/'plan.json'),plan_hash=plan['plan_hash'],
            target=41,service=SERVICE,executor=str(HERE/'run_extension.py'),
            completion_audit=str(LOGS/'completion_verified.json'),replacement_map=str(LOGS/'replacement_map.json'),
            starts_after_original172=True,stage_data_roots=[s['data'] for s in plan['stages']])
        # A repeat attachment updates the same extension ID instead of appending.
        extensions={e['id']:e for e in parent.get('extensions',[])}
        if extension['id'] in extensions and extensions[extension['id']]['plan_hash']!=plan['plan_hash']:
            raise RuntimeError('Existing queue extension has a different immutable plan')
        extensions[extension['id']]=extension
        parent['extensions']=list(extensions.values());parent['combined_recording_operations']=172+sum(e['target'] for e in extensions.values())
        atomic_json(PARENT/'queue_manifest.json',parent)
        registered={e['id']:e for e in training['replacement_queue'].get('extensions',[])}
        registered[extension['id']]=extension;training['replacement_queue']['extensions']=list(registered.values())
        training['quality_control']['reverse_excursion_ge_8cm'].update(
            queue_manifest=str(LOGS/'queue_manifest.json'),queue_service=SERVICE,queued_episodes=41)
        training['updated_at']=datetime.now().astimezone().isoformat()
        atomic_json(TRAIN/'data_sources.json',training)
        stages={t['id']:s for s in plan['stages'] for t in s['tasks']}
        for mark_id,task in expected.items():
            mark=marks[mark_id];stage=stages[task['id']]
            mark.update(task_execution_status='queued_after_original172',queue_task_id=task['id'],
                queue_plan=str(HERE/'plan.json'),queue_service=SERVICE,destination_dataset=stage['data'])
        pending['queued_episodes']=41;pending['queue_service']=SERVICE
        atomic_json(TRAIN/'pending_rerecord_reverse8.json',pending)
        with (PARENT/'录制清单.csv').open(encoding='utf-8-sig') as stream:old=list(csv.DictReader(stream))
        rows=[dict(序号=i+1,队列阶段='原队列',任务ID=r['任务ID'],物体='beverage_bottle',
            原始记录=r['原始记录'],示范池=r['示范池'],网格=r['网格'],新数据目录=r['新数据目录']) for i,r in enumerate(old)]
        rows += [dict(序号=len(old)+i+1,队列阶段='反向≥8cm补录',任务ID=t['id'],物体=t['object'],
            原始记录=t['source'],示范池=t['source_pool'],网格=t['grid'],新数据目录=stages[t['id']]['data']) for i,t in enumerate(plan['tasks'])]
        assert len(rows)==213 and len({str(Path(r['原始记录']).resolve()) for r in rows})==213
        write_csv(LOGS/'统一队列213条.csv',rows)
        atomic_json(LOGS/'attachment_verified.json',dict(verified=True,added=41,duplicates_skipped=0,
            original_tasks=172,total_recording_operations=213,service=SERVICE,
            attached_at=datetime.now().astimezone().isoformat(),native_stage_runtime_tests_required=True,
            active_parent_stages_unchanged=parent['stages']==json.loads((LOGS/'parent_queue.before_attachment.json').read_text())['stages']))
        print('Attached41 distinct tasks after original172; combined recording operations213; repeat attachment is idempotent')


if __name__=='__main__':main()
