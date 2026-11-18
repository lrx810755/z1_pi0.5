#!/usr/bin/python3
"""Build immutable retake stages and deduplicate against all original172 tasks."""
import argparse
from collections import Counter
import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
WORK=Path('/home/lp531')
import sys
sys.path.insert(0,str(WORK/'robot/dataset_trajectory_review'))
from review import atomic_json,write_csv
from screen_fruit_motion import digest


def deduplicate(candidates,existing):
    paths={str(Path(e['source']).resolve()) for e in existing}
    hashes={e['source_frames_sha256'] for e in existing};new=[];duplicates=[]
    for item in candidates:
        path=str(Path(item['physical_path']).resolve());sha=item['source_frames_sha256']
        if path in paths or sha in hashes:duplicates.append(item);continue
        paths.add(path);hashes.add(sha);new.append(item)
    return new,duplicates


def clone_runtime(source,destination):
    if destination.exists():raise FileExistsError('Refuse replacing native runtime '+str(destination))
    destination.mkdir(parents=True)
    for path in source.iterdir():
        if path.is_file() and path.suffix in ['.py','.json','.sh','.md']:
            target=destination/path.name
            target.write_text(path.read_text().replace(str(source),str(destination)))
    return destination


def main():
    if (HERE/'plan.json').exists():
        plan=json.loads((HERE/'plan.json').read_text())
        print(json.dumps(dict(already_built=True,target=plan['target'],plan_hash=plan['plan_hash'])));return
    pending_path=WORK/'pi05_lora_workspace/pi05_current_round_20261111/pending_rerecord_reverse8.json'
    pending=json.loads(pending_path.read_text())
    parent=WORK/'z1_queue_logs/bottle_rerecord172_queue_20261111'
    with (parent/'录制清单.csv').open(encoding='utf-8-sig') as stream:existing_rows=list(csv.DictReader(stream))
    existing=[dict(source=row['原始记录'],source_frames_sha256=digest(Path(row['原始记录'])/'frames.jsonl')) for row in existing_rows]
    new,duplicates=deduplicate(pending['episodes'],existing)
    assert len(new)==41 and not duplicates
    tasks=[]
    for item in new:
        source=Path(item['physical_path']);summary=json.loads((source/'episode_summary.json').read_text())
        if digest(source/'frames.jsonl')!=item['source_frames_sha256']:raise RuntimeError('Marked source changed')
        tasks.append(dict(id='reverse8_'+item['object']+'_'+item['mark_id'][:12],mark_id=item['mark_id'],
            object=item['object'],kind=item['kind'],source=str(source),
            source_frames_sha256=item['source_frames_sha256'],grid=item['grid'],seed=item['seed'],
            source_pool='correction' if (source/'correction.json').exists() else 'basic',
            scene=summary['scene_randomization'],original_max_reverse_m=item['max_reverse_excursion_m']))
    stages=[]
    specs=[('apple_basic','apple',WORK/'robot/apple_scale110_demo400_20261105',144,8851),
           ('cup_basic','cup',WORK/'robot/cup_cylinder_horizontal200_20261108',145,8852),
           ('bottle_basic','beverage_bottle',WORK/'robot/bottle_rerecord160_renderfix_20261111',146,8853)]
    for name,obj,source,domain,port in specs:
        selected=[t for t in tasks if t['object']==obj and t['source_pool']=='basic']
        if not selected:continue
        native=clone_runtime(source,HERE/('native_'+name))
        if obj=='apple':
            shutil.copy2(WORK/'ros2_ws/src/z1_ros2/z1_isaac_bridge/z1_isaac_bridge/grasp_assistant.py',native/'base_assistant_snapshot.py')
            p=native/'assistant.py';p.write_text(p.read_text().replace('from z1_isaac_bridge import grasp_assistant as base','import base_assistant_snapshot as base'))
        else:
            file=native/('assistant.py' if obj=='cup' else 'horizontal50.py')
            text=file.read_text();needle="    def _collection_column(self):\n"
            assert text.count(needle)==1
            insert="        import os\n        with open(os.environ['Z1_REVERSE8_CURRENT']) as stream: return int(json.load(stream)['task']['grid'][0])\n"
            file.write_text(text.replace(needle,needle+insert))
        data=WORK/'smolvla_data'/f'reverse8_{name}_{len(selected)}_20261112'
        logs=WORK/'z1_queue_logs'/f'reverse8_{name}_{len(selected)}_20261112'
        for file in native.iterdir():
            if file.is_file() and file.suffix in ['.py','.json','.sh','.md']:
                file.write_text(file.read_text().replace(str(WORK/'z1_queue_logs'/source.name),str(logs)))
        stage=dict(id=name,type='basic',object=obj,native=str(native),data=str(data),logs=str(logs),
            domain=domain,port=port,tasks=selected,
            command=['bash',str(HERE/'run_stage.sh'),name])
        stage['plan_hash']=hashlib.sha256(json.dumps(selected,sort_keys=True).encode()).hexdigest()
        stages.append(stage)
    corr_tasks=[t for t in tasks if t['source_pool']=='correction']
    assert len(corr_tasks)==4 and all(t['object']=='beverage_bottle' for t in corr_tasks)
    old=WORK/'robot/bottle_wait8_retake12_20261111';native=clone_runtime(old,HERE/'bottle_corrections')
    data=WORK/'smolvla_data/bottle_reverse8_corrections4_20261112'
    logs=WORK/'z1_queue_logs/bottle_reverse8_corrections4_20261112'
    for path in native.iterdir():
        if not path.is_file() or path.suffix not in ['.py','.json','.sh','.md']:continue
        text=path.read_text().replace('/home/lp531/smolvla_data/bottle_wait8_retake12_20261111',str(data))
        text=text.replace('/home/lp531/z1_queue_logs/bottle_wait8_retake12_20261111',str(logs))
        text=text.replace('Z1_BOTTLE_WAIT8_RETAKE_CURRENT','Z1_REVERSE8_CORRECTION_CURRENT').replace('8845','8854')
        path.write_text(text)
    p=native/'protocol.py';text=p.read_text();needle="QUOTAS = {'basic_retake':10,'transport':1,'close_transition':1}"
    assert text.count(needle)==1;text=text.replace(needle,"QUOTAS = {'transport':1,'close_transition':2,'regrasp':1}");p.write_text(text)
    retakes=[]
    for task in corr_tasks:
        meta=json.loads((Path(task['source'])/'correction.json').read_text());item=copy.deepcopy(meta['task'])
        item.update(id=task['id'],mark_id=task['mark_id'],split='train',source_type='rerecorded_reverse8_episode',
            source_pool='correction',rerecord_source=task['source'],
            rerecord_source_frames_sha256=task['source_frames_sha256'],original_reverse_m=task['original_max_reverse_m'])
        retakes.append(item)
    atomic_json(native/'retake_tasks.json',retakes)
    (native/'time_gate.py').write_text("import sys\nsys.path.insert(0,"+repr(str(HERE))+')\nfrom quality_gate import screen\n')
    p=native/'audit_saved.py';text=p.read_text();needle="        timing = json.loads((path / 'time_screen.json').read_text())"
    assert text.count(needle)==1;text=text.replace(needle,
        "        import sys\n        sys.path.insert(0,"+repr(str(HERE))+')\n        from quality_gate import screen\n        screen(path)\n'+needle);p.write_text(text)
    launcher=HERE/'run_corrections.sh'
    launcher.write_text('''#!/usr/bin/env bash
set -eo pipefail
source /home/lp531/ros2_ws/src/z1_ros2/scripts/configure_environment.sh
set +u
source /opt/ros/humble/setup.bash
source /home/lp531/ros2_ws/install/setup.bash
set -u
export ROS_DOMAIN_ID=147 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export Z1_COLLECTION_GRID_ROWS=4 Z1_COLLECTION_GRID_COLUMNS=5
export Z1_SERVO_PREFIX=/home/lp531/.local/opt/ros-humble-moveit-servo/opt/ros/humble
export FASTDDS_DEFAULT_PROFILES_FILE="$FASTRTPS_DEFAULT_PROFILES_FILE"
export Z1_ISAAC_GRASP_CONFIG='''+str(native/'grasp_physics.json')+'\nexec /usr/bin/python3 -u '+str(native/'run.py')+' "$@"\n')
    stages.append(dict(id='bottle_corrections',type='correction',object='beverage_bottle',native=str(native),
        data=str(data),logs=str(logs),domain=147,port=8854,tasks=corr_tasks,command=['bash',str(launcher)]))
    # Pin every source used by a queued child, including the common gate/driver.
    common=[p for p in HERE.iterdir() if p.is_file() and p.suffix in ['.py','.sh']]
    for stage in stages:
        files=common+[p for p in Path(stage['native']).iterdir() if p.is_file() and p.suffix in ['.py','.json','.sh']]
        stage['source_pins']={str(p):digest(p) for p in files}
    plan=dict(schema='reverse8_queue_extension_v1',target=len(tasks),tasks=tasks,stages=stages,
        prerequisite_queue=str(parent),prerequisite_target=172,combined_recording_operations=172+len(tasks),
        pending_registry=str(pending_path),pending_registry_sha256=digest(pending_path),
        deduplicated_against=len(existing),overlap_skipped=len(duplicates),
        by_object=dict(Counter(t['object'] for t in tasks)),
        quality=dict(reverse_excursion_m_must_be_less_than=.08,stationary_wait_s_must_be_less_than=8),
        starts_after_original_queue_lock_and_completion_audit=True)
    plan['plan_hash']=hashlib.sha256(json.dumps(tasks,sort_keys=True).encode()).hexdigest()
    atomic_json(HERE/'plan.json',plan)
    output=WORK/'z1_queue_logs/reverse8_queue_20261112';output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/'queue_manifest.json',plan)
    write_csv(output/'新增重录任务41条.csv',[dict(任务ID=t['id'],物体=t['object'],类型=t['source_pool'],
        纠偏类别=t['kind'],网格=t['grid'],种子=t['seed'],原始路径=t['source'],原Frames_SHA256=t['source_frames_sha256'],
        原最大回退厘米=round(t['original_max_reverse_m']*100,2),状态='已挂载，等待原队列完成') for t in tasks])
    print(json.dumps(dict(target=len(tasks),by_object=plan['by_object'],duplicates=len(duplicates),
        stages=[(s['id'],len(s['tasks'])) for s in stages]),ensure_ascii=False))


if __name__=='__main__':main()
