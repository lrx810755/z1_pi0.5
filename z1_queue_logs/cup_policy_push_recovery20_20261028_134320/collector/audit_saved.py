#!/usr/bin/python3
"""Independent, fail-closed audit of 19 actual policy-push expert recoveries.

Freeze source pins BEFORE collection:
  python3 collector/audit_saved.py --freeze-source-pins
Audit saved data after collection:
  python3 collector/audit_saved.py
Only a complete, exact identity set produces completion_verified.json. No ROS,
scene, model inference, or GPU is used. Geometry is imported from its original
location so its URDF/mesh ROOT resolution remains correct.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import sys

ROOT=Path(__file__).resolve().parents[1]
DATA=Path('/home/lp531/smolvla_data/z1_cup_policy_push_recovery20_20261028')
MANIFEST=ROOT/'candidates/candidates_manifest.json'
PINS=ROOT/'source_pins.json'
SOURCE_TYPE='actual_policy_push_recovery'
EXPECTED_COUNT=19
OLD=Path('/home/lp531/z1_queue_logs/cup_correction100_full_renderfix_20261027/collector')
MINUS=Path('/home/lp531/z1_queue_logs/cup_minus90_rerecord_20261022/collector')
GEOMETRY=Path('/home/lp531/ros2_ws/src/z1_ros2/scripts/cup_correction_fragments_20261019/geometry.py')
CONFIG=Path('/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge/config/grasp_physics.json')
REQUIRED=['episode_start.json','episode_summary.json','frames.jsonl','correction.json','gripper_labels.json','training_mask.json','physics_status.jsonl']
CAMERAS={'observation.images.wrist':(640,360),'observation.images.global':(640,480),'observation.images.global_right':(640,480)}


def load(path):return json.loads(Path(path).read_text())
def lines(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)
def require(condition,message):
    if not condition:raise ValueError(message)
def identity(candidate):return candidate['candidate_id']+f"_takeover_{candidate['takeover_simulation_elapsed_s']:.6f}"
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def validate_identity_set(actual,expected):
    require(len(actual)==len(set(actual)),'duplicate source identity')
    require(set(actual)==set(expected),'missing or extra source identities: '+str(dict(missing=sorted(set(expected)-set(actual)),extra=sorted(set(actual)-set(expected)))))


def verify_pins(pins):
    for path,digest in pins.items():
        require(Path(path).is_file(),'missing pinned source: '+path)
        require(sha(path)==digest,'pinned source changed: '+path)


def expected_sources(manifest=MANIFEST):
    payload=load(manifest);candidates=payload['candidates']
    require(len(candidates)==EXPECTED_COUNT,'source manifest must contain exactly19 candidates')
    expected={identity(c):c for c in candidates}
    require(len(expected)==EXPECTED_COUNT,'duplicate source identity in candidate manifest')
    require(len({c['source_episode_path'] for c in candidates})==EXPECTED_COUNT,'duplicate source rollout episode')
    return expected


def freeze_pins(manifest=MANIFEST,pins_path=PINS):
    expected=expected_sources(manifest)
    if Path(pins_path).exists():
        pinned=load(pins_path);verify_pins(pinned['files'])
        validate_identity_set(pinned['source_identities'],expected)
        return pinned
    paths={Path(manifest).resolve(),GEOMETRY,CONFIG,OLD/'protocol.py',MINUS/'protocol.py'}
    snapshots={}
    for candidate in expected.values():
        for key,value in candidate.items():
            if key.endswith('_path') and isinstance(value,str) and Path(value).is_file():paths.add(Path(value).resolve())
        for name in ['source_snapshot.json','final_box_results.json']:
            path=Path(candidate['source_eval_path'])/name;paths.add(path.resolve())
            if name=='source_snapshot.json':snapshots.update(load(path))
        require(candidate.get('source_final_box_success') is False,'successful source candidate forbidden')
    # Pin the original evaluator/scene/config/checkpoint files already recorded
    # by each source evaluation, and first confirm the evaluation's own hashes.
    for name,digest in snapshots.items():
        path=Path(name).resolve()
        require(path.is_file() and sha(path)==digest,'source evaluation snapshot mismatch: '+str(path))
        paths.add(path)
    value=dict(source_type=SOURCE_TYPE,expected_count=EXPECTED_COUNT,manifest_path=str(Path(manifest).resolve()),
        source_identities=sorted(expected),files={str(p):sha(p) for p in sorted(paths)},
        frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write(pins_path,value);return value


def validate_takeover(status,initial):
    require(status.get('cylinder_tilt_valid') is True and 0<=float(status.get('cylinder_tilt_deg',180))<=10,'takeover cup tilted or unknown tilt')
    position=status.get('object_position_m',[])
    require(len(position)==3 and all(math.isfinite(float(x)) for x in position),'invalid takeover position')
    require(abs(position[2])<=.008 and status.get('object_touching_table') is True,'takeover not supported on table')
    require(float(status.get('object_speed_m_s',math.inf))<=.02,'takeover cup moving')
    require(status.get('frozen') is False and status.get('target_collisions_enabled') is True,'takeover physics changed')
    require(status.get('gripper_position_rad',0)<=-.75 and not status.get('gripper_commanded_closed') and not status.get('physical_grasp_ok') and not status.get('was_grasped'),'takeover is not open and ungrasped')
    require(math.dist(position[:2],initial[:2])>=.009,'takeover has no measurable policy push')
    require(.32<=position[0]<=.72 and -.30<=position[1]<=.12,'takeover outside screened tabletop workspace')
    return position


def dynamic_ok(status):
    return all(status.get(k) is value for k,value in [('frozen',False),('target_collisions_enabled',True),('left_finger_touching_table',False),('right_finger_touching_table',False)])


def inside_box(position):
    # Exact evaluator defaults pinned above; base.launch.py does not override.
    return (isinstance(position,(list,tuple)) and len(position)==3
        and all(math.isfinite(float(x)) for x in position)
        and abs(position[0]-.32)<=.055+1e-9 and abs(position[1]-.14)<=.055+1e-9 and 0<=position[2]<=.12)


def stable_release(status):
    return (dynamic_ok(status) and status.get('inside_success_volume') is True and inside_box(status.get('object_position_m'))
        and status.get('released') is True and status.get('holding_observed') is False
        and status.get('gripper_commanded_closed') is False and float(status.get('gripper_position_rad',math.inf))<=-.88
        and 0<=float(status.get('object_speed_m_s',math.inf))<=.02)


def validate_tail(samples):
    require(len(samples)>=2,'missing stable physics tail')
    latest=float(samples[-1]['simulation_time'])
    anchors=[i for i,s in enumerate(samples) if float(s['simulation_time'])<=latest-.5+1e-7]
    require(bool(anchors),'physics tail shorter than0.5 seconds')
    tail=samples[anchors[-1]:];duration=latest-float(tail[0]['simulation_time'])
    require(duration>=.5-1e-7 and all(stable_release(s['status']) for s in tail),'unstable/outside-box release tail')
    return duration


def validate_masks(mask,n):
    require(mask.get('expert_frame_mask')==[True]*n,'expert mask must align and contain expert actions only')
    require(mask.get('valid_action_horizon16_anchors')==[i+16<=n for i in range(n)],'horizon16 mask mismatch')
    require(mask.get('valid_action_horizon30_anchors')==[i+30<=n for i in range(n)],'horizon30 mask mismatch')


def jpeg_dimensions(path):
    """Read actual JPEG SOF dimensions and require complete SOI/EOI framing."""
    path=Path(path)
    require(path.is_file() and path.stat().st_size>4,'missing/empty JPEG: '+str(path))
    with path.open('rb') as stream:
        require(stream.read(2)==b'\xff\xd8','invalid JPEG SOI: '+str(path))
        stream.seek(-2,2);require(stream.read(2)==b'\xff\xd9','truncated JPEG EOI: '+str(path));stream.seek(2)
        while True:
            byte=stream.read(1)
            require(bool(byte),'missing JPEG SOF: '+str(path))
            if byte!=b'\xff':continue
            marker=stream.read(1)
            while marker==b'\xff':marker=stream.read(1)
            require(bool(marker),'invalid JPEG marker')
            code=marker[0]
            if code in (0xD8,0x01) or 0xD0<=code<=0xD7:continue
            require(code not in (0xD9,0xDA),'JPEG has no image dimensions')
            raw=stream.read(2);require(len(raw)==2,'truncated JPEG segment')
            size=struct.unpack('>H',raw)[0];require(size>=2,'invalid JPEG segment size')
            if code in (0xC0,0xC1,0xC2,0xC3,0xC5,0xC6,0xC7,0xC9,0xCA,0xCB,0xCD,0xCE,0xCF):
                raw=stream.read(5);require(len(raw)==5,'invalid JPEG SOF')
                height,width=struct.unpack('>HH',raw[1:5]);require(width>0 and height>0,'zero-size JPEG');return width,height
            stream.seek(size-2,1)


def source_grid(candidate):
    cfg=load(CONFIG)['scene_randomization'];x,y=candidate['cup_initial_xyz_m'][:2]
    limits=[cfg['workspace_x_m'],cfg['workspace_y_m']];shape=cfg['levels']['full']['target_grid_shape']
    require(all(limits[i][0]<=v<=limits[i][1] for i,v in enumerate((x,y))),'source initial layout outside grid workspace')
    return [min(shape[i]-1,int((v-limits[i][0])/(limits[i][1]-limits[i][0])*shape[i])) for i,v in enumerate((x,y))]


def audit_source(candidate):
    require(candidate.get('source_final_box_success') is False,'successful source marked as failure')
    final=load(Path(candidate['source_eval_path'])/'final_box_results.json')
    records=[r for r in final['episodes'] if r['seed']==candidate['seed']]
    require(len(records)==1 and records[0]['success'] is False,'source final-box success/missing seed')
    require(Path(records[0]['source_result']).resolve()==Path(candidate['source_result_path']).resolve(),'source result identity mismatch')
    original=load(candidate['source_result_path'])
    require(original['seed']==candidate['seed'] and original['initial_scene_randomization']==candidate['raw_initial_scene_randomization'],'source original layout changed')
    count=candidate['takeover_trace_index_zero_based']+1
    previous=-math.inf;last=None
    with Path(candidate['source_trace_path']).open() as stream:
        for index,line in enumerate(stream):
            if index>=count:break
            row=json.loads(line);stamp=float(row['simulation_elapsed_s']);require(stamp>=previous,'nonmonotonic original trace');previous=stamp
            s=row['contact_status'];require(s.get('cylinder_tilt_valid') is True and float(s.get('cylinder_tilt_deg',180))<=10,'source already tilted before takeover')
            require(abs(s['object_position_m'][2]-candidate['cup_initial_xyz_m'][2])<=.008,'source already left tabletop before takeover')
            require(not s.get('was_grasped') and not s.get('physical_grasp_ok') and not row.get('lifted'),'source grasped/lifted before takeover')
            last=row
    require(last is not None and index>=count-1,'source trace prefix truncated')
    require(abs(last['simulation_elapsed_s']-candidate['takeover_simulation_elapsed_s'])<1e-6,'source takeover time mismatch')
    require(last['contact_status']['object_position_m']==candidate['cup_xyz_m'],'source takeover pose changed')
    return True


def audit_episode(folder,expected,geometry,old_protocol,minus_protocol):
    folder=Path(folder).resolve()
    for name in REQUIRED:require((folder/name).is_file() and (folder/name).stat().st_size>0,'missing episode evidence '+name)
    correction=load(folder/'correction.json');summary=load(folder/'episode_summary.json');start=load(folder/'episode_start.json')
    source_id=correction['source_identity'];require(isinstance(source_id,str) and source_id in expected,'unexpected source identity')
    candidate=expected[source_id]
    require(correction.get('source_type')==SOURCE_TYPE and correction.get('accepted') is True and correction.get('recovery_complete') is True,'unaccepted or wrong-type correction')
    require(correction.get('source_candidate')==candidate,'saved source candidate differs from pinned manifest')
    audit_source(candidate)
    require(correction.get('task_id')==source_id and correction.get('task',{}).get('kind')=='push_recovery','task identity mismatch')
    grid=source_grid(candidate);split='validation' if tuple(grid) in [(1,1),(2,3)] else 'train'
    require(correction.get('source_grid')==grid and correction.get('split')==split and correction.get('source_seed')==candidate['seed'],'source grid/seed/split mismatch')
    for scene in [summary['scene_randomization'],start['scene_randomization']]:
        require(scene.get('seed')==candidate['seed'] and scene.get('target_grid')==grid and scene.get('source_grid')==grid,'source scene grouping mismatch')
        for key in ['object_positions_m','object_orientations_xyzw']:
            require(scene.get(key)==candidate['raw_initial_scene_randomization'][key],'original five-object layout overwritten: '+key)
    actual=correction['actual_takeover_status'];position=validate_takeover(actual,candidate['cup_initial_xyz_m'])
    require(position==correction.get('actual_takeover_cup_position_m'),'actual takeover cup pose mismatch')
    require(math.dist(position,candidate['cup_xyz_m'])<=.025+1e-8,'replay differs25mm from source cup')
    replay=Path(correction['replay_trace']);require(replay.is_file() and replay.stat().st_size>0,'missing actual replay trace')
    replay_rows=lines(replay);require(bool(replay_rows),'empty replay trace')
    for sample in replay_rows:
        s=sample['status'];require(s.get('cylinder_tilt_valid') is True and float(s.get('cylinder_tilt_deg',180))<=10,'actual replay tilted before takeover')
        require(abs(s.get('object_position_m',[0,0,999])[2])<=.008,'actual replay flew before takeover')
        require(s.get('frozen') is False and s.get('target_collisions_enabled') is True,'actual replay physics changed')
    rows=lines(folder/'frames.jsonl');n=len(rows)
    require(n>=30 and summary.get('frame_count')==n and summary.get('dropped_frames')==0,'frame count/drops invalid')
    require(float(summary.get('fps',0))==10 and summary.get('target_object')=='cup','unexpected fps/target')
    for key in ['task_success','physical_grasp_ok','release_sequence_complete','placement_stable','bilateral_contact_ever','object_lifted_clear_of_table_ever']:
        require(summary.get(key) is True,'missing summary physical evidence '+key)
    previous=-math.inf;image_paths=set();images=0
    for index,row in enumerate(rows):
        stamp=float(row['simulation_time']);require(math.isfinite(stamp) and stamp>previous,'nonmonotonic frame time');previous=stamp
        require(index==row.get('frame_index') and row.get('target_object')=='cup','frame alignment/target mismatch')
        require(abs(stamp-float(rows[0]['simulation_time'])-index*.1)<1e-5,'frames not continuous10Hz')
        for key in ['action','observation.state']:
            require(len(row[key])==7 and all(math.isfinite(float(v)) for v in row[key]),'invalid joint/action vector')
        for key,dimensions in CAMERAS.items():
            path=(folder/row[key]).resolve();require(path.is_relative_to(folder),'camera path escapes episode')
            require(path not in image_paths,'camera frames reused across frame/camera');image_paths.add(path)
            require(jpeg_dimensions(path)==dimensions,'actual camera dimensions mismatch: '+str(path));images+=1
    masks=load(folder/'training_mask.json');validate_masks(masks,n)
    labels,close,release=old_protocol.command_labels(rows,correction['events']);stored=load(folder/'gripper_labels.json')
    require(stored.get('labels')==labels and stored.get('close_command_frame')==close and stored.get('open_command_frame')==release,'gripper intent labels misaligned')
    require(len(rows)-release>=16,'missing16 release-tail frames')
    actual_close=next(i for i in range(close,release) if float(rows[i]['action'][-1])>-.75)
    require(actual_close>0,'no preclose open frame');_,rotation=geometry.link_fk(rows[actual_close-1]['observation.state'][:6]);roll=float(minus_protocol.tool_roll(rotation))
    require(math.isfinite(roll) and abs(roll+90)<=3,'measured preclose FK is not minus90 +/-3 degrees')
    require(abs(roll-float(correction['measured_preclose_roll_deg']))<1e-5 and correction.get('preclose_fk_frame')==actual_close-1,'recorded preclose FK evidence inconsistent')
    require(correction.get('recording_contains_failed_policy_actions') is False and start.get('failed_policy_prefix_recorded') is False,'failed policy actions present in expert recording')
    require(not any(str(e.get('event','')).startswith('intervention_') for e in correction['events']),'intervention actions in expert recording')
    intervals=correction.get('recovery_intervals_s',[]);require(len(intervals)==1,'missing complete recovery interval')
    a,b=intervals[0]['start'],intervals[0]['end']
    require(rows[0]['simulation_time']-.25<=a<=correction['recovery_end_sim_s']<=b==correction['stable_grasp_end_sim_s']<rows[release]['simulation_time'],'invalid recovery/stable-grasp interval')
    require(any(a<=r['simulation_time']<=b and i+30<=n for i,r in enumerate(rows)),'no complete30-action recovery anchor')
    withdrawal=[e for e in correction['events'] if e.get('event')=='withdrawal_measured'];require(bool(withdrawal),'missing measured withdrawal event')
    rise=float(withdrawal[-1]['after'][2])-float(withdrawal[-1]['before'][2])
    require(rise>=.03 and abs(rise-float(correction.get('withdrawal_rise_m',0)))<1e-6,'insufficient/inconsistent withdrawal')
    samples=lines(folder/'physics_status.jsonl')
    require(len(samples)==correction.get('dynamic_samples') and len(samples)>0,'logged dynamic sample count mismatch')
    require(correction.get('dynamic_frame_samples')==n and not correction.get('dynamic_frame_violations') and not correction.get('dynamic_violations') and correction.get('table_contact') is False,'invalid dynamic frame counts/violations')
    previous=-math.inf
    for sample in samples:
        stamp=float(sample['simulation_time']);require(math.isfinite(stamp) and stamp>=previous,'nonmonotonic physics clock');previous=stamp
        require(dynamic_ok(sample['status']),'non-dynamic/collision-disabled/finger-table physics sample')
    require(samples[0]['simulation_time']<=rows[0]['simulation_time']+.1 and samples[-1]['simulation_time']>=rows[-1]['simulation_time']-.05,'physics stream does not cover recorded frames')
    for key in ['physical_grasp_ok','lifted','bilateral_contact','release_sequence_complete','task_success']:
        require(any(s['status'].get(key) is True for s in samples),'missing independently logged physical evidence '+key)
    final=correction['final_status']
    for key in ['task_success','physical_grasp_ok','placement_stable','release_sequence_complete']:
        require(final.get(key) is True,'final physical acceptance missing '+key)
    require(stable_release(final),'current final cup is not quietly released inside box')
    tail=validate_tail(samples)
    return dict(path=str(folder),source_identity=source_id,candidate_id=candidate['candidate_id'],seed=candidate['seed'],
        source_grid=grid,split=split,frames=n,images_checked=images,measured_preclose_roll_deg=roll,
        stable_release_tail_duration_s=tail,dynamic_samples=len(samples),replay_trace_sha256=sha(replay),
        evidence_hashes={name:sha(folder/name) for name in REQUIRED},passed=True)


def audit(data=DATA,manifest=MANIFEST,pins_path=PINS):
    data=Path(data).resolve();report=dict(passed=False,accepted_count=0,expected_count=EXPECTED_COUNT,source_type=SOURCE_TYPE,
        episodes=[],episode_details=[],errors=[],audited_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    try:
        expected=expected_sources(manifest);pinned=load(pins_path)
        verify_pins(pinned['files']);validate_identity_set(pinned['source_identities'],expected)
        require(str(Path(manifest).resolve())==pinned.get('manifest_path'),'manifest differs from source pin baseline')
        geometry=load_module('audit_original_geometry',GEOMETRY).Geometry()
        old_protocol=load_module('audit_old_protocol',OLD/'protocol.py');minus_protocol=load_module('audit_minus_protocol',MINUS/'protocol.py')
        folders=sorted(data.glob('episode_[0-9]*'));actual=[]
        for folder in folders:
            try:
                detail=audit_episode(folder,expected,geometry,old_protocol,minus_protocol)
                report['episode_details'].append(detail);report['episodes'].append(str(folder.resolve()));actual.append(detail['source_identity'])
            except Exception as exc:report['errors'].append(dict(episode=str(folder),error=str(exc)))
        report['accepted_count']=len(report['episodes'])
        try:validate_identity_set(actual,expected)
        except ValueError as exc:report['errors'].append(dict(error=str(exc)))
        if len(folders)!=EXPECTED_COUNT:report['errors'].append(dict(error=f'expected exactly19 saved episode directories, found{len(folders)}'))
        if list(data.glob('.recording_*')):report['errors'].append(dict(error='unfinished recording directories remain'))
        if not report['errors'] and len(actual)==EXPECTED_COUNT:
            # Recheck source pins after image/data scanning to detect mutation
            # during a long audit; publish completion last.
            verify_pins(pinned['files'])
            report.update(passed=True,source_pins_path=str(Path(pins_path).resolve()),source_pins_sha256=sha(pins_path),manifest_sha256=sha(manifest),image_validation='all3-camera JPEG SOF dimensions plus nonempty complete SOI/EOI framing')
    except Exception as exc:report['errors'].append(dict(error=str(exc)))
    if report['passed']:
        write(data/'completion_verified.json',report);write(ROOT/'completion_verified.json',report)
    else:
        # A failed fresh audit must never leave a stale success marker usable by
        # training. Preserve its contents in the partial audit before removal.
        previous={}
        for marker in [data/'completion_verified.json',ROOT/'completion_verified.json']:
            if marker.exists():previous[str(marker)]=load(marker);marker.unlink()
        if previous:report['invalidated_previous_completion']=previous
        write(ROOT/'saved_audit_partial.json',report)
        if data.exists():write(data/'saved_audit_partial.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze-source-pins',action='store_true')
    parser.add_argument('--data',type=Path,default=Path(os.environ.get('Z1_PUSH_RECOVERY_DATA',DATA)))
    args=parser.parse_args()
    if args.freeze_source_pins:
        pinned=freeze_pins();print(json.dumps(dict(source_pins=str(PINS),files=len(pinned['files']),source_identities=len(pinned['source_identities']))));return 0
    report=audit(args.data)
    print(json.dumps({k:report[k] for k in ['passed','accepted_count','expected_count','source_type','errors']},ensure_ascii=False,indent=2))
    return 0 if report['passed'] else 1
if __name__=='__main__':sys.exit(main())
