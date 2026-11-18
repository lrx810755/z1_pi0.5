#!/usr/bin/python3
"""Resumable native-expert basic retakes for one object, serialized by parent."""
import argparse
from collections import Counter
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request
import urllib.error
import uuid

HERE=Path(__file__).resolve().parent
from quality_gate import screen
from basic_nodes import collection_ready
from layout_fallback import selected_task,layout_is_allowed
from review import atomic_json


def api(port,path,body=None,timeout=8):
    request=urllib.request.Request(f'http://127.0.0.1:{port}{path}',
        data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(request,timeout=timeout) as response:value=json.load(response)
    except urllib.error.HTTPError as exc:raise RuntimeError(exc.read().decode(errors='replace')) from exc
    if value.get('success') is False:raise RuntimeError(value.get('message',str(value)))
    return value


def completed(stage):
    expected={t['id'] for t in stage['tasks']};result={}
    for p in Path(stage['data']).glob('episode_*/rerecord_provenance.json'):
        if not (p.parent/'episode_summary.json').exists():continue
        proof=json.loads(p.read_text());task=proof.get('task_id')
        if proof.get('plan_hash')!=stage['plan_hash'] or task not in expected:
            raise RuntimeError('Dataset contains a different task plan: '+str(p))
        if task in result:raise RuntimeError('Duplicate saved task ID: '+task)
        gate=json.loads((p.parent/'retake_quality.json').read_text())
        if not proof.get('accepted') or not gate['passed']:raise RuntimeError('Unaccepted saved episode')
        result[task]=str(p.parent)
    return result


def status(stage,state,**extra):
    atomic_json(Path(stage['logs'])/'status.json',dict(stage=state,
        updated_at=datetime.now().astimezone().isoformat(),completed=len(completed(stage)),
        target=len(stage['tasks']),object=stage['object'],**extra))
    print(state,extra,flush=True)


def audit(stage):
    import numpy as np
    from PIL import Image
    done=completed(stage)
    if len(done)!=len(stage['tasks']):raise RuntimeError('Retake quota incomplete')
    replacements=[]
    for task in stage['tasks']:
        root=Path(done[task['id']]);summary=json.loads((root/'episode_summary.json').read_text())
        for key in ['task_success','physical_grasp_ok','bilateral_contact_ever',
                    'object_lifted_clear_of_table_ever','release_sequence_complete','placement_stable']:
            if summary.get(key) is not True:raise RuntimeError('Physical audit failed: '+key)
        assert summary['target_object']==stage['object'] and summary['dropped_frames']==0 and summary['fps']==10
        proof=json.loads((root/'rerecord_provenance.json').read_text())
        assert layout_is_allowed(task,stage,summary,proof),'Unregistered recording layout'
        rows=[json.loads(line) for line in (root/'frames.jsonl').read_text().splitlines()]
        assert summary['frame_count']==len(rows)
        for key in ['observation.state','action']:
            values=np.asarray([r[key] for r in rows]);assert values.shape==(len(rows),7) and np.isfinite(values).all()
        assert all(r['action_stamp']<=r['simulation_time']+1e-7 for r in rows)
        quality=screen(root)
        for row in rows:
            for camera,shape in [('wrist',(640,360)),('global',(640,480)),('global_right',(640,480))]:
                with Image.open(root/row['observation.images.'+camera]) as image:
                    image.load();assert image.size==shape
        replacements.append(dict(task_id=task['id'],mark_id=task['mark_id'],old=task['source'],
            new=str(root),pool=task['source_pool'],frames=len(rows),frames_sha256=quality['frames_sha256']))
    logs=Path(stage['logs'])
    atomic_json(logs/'replacement_map.json',replacements)
    atomic_json(logs/'completion_verified.json',dict(verified=True,accepted=len(done),
        completed_at=datetime.now().astimezone().isoformat(),plan_hash=stage['plan_hash'],
        renderer_world_eps=.0001,replacement_map=str(logs/'replacement_map.json'),training_started=False))


def stop_scene(process):
    if process is None:return
    for sig,delay in [(signal.SIGINT,12),(signal.SIGTERM,5),(signal.SIGKILL,2)]:
        try:
            os.killpg(process.pid,sig)
            try:process.wait(timeout=delay)
            except subprocess.TimeoutExpired:pass
            time.sleep(.2)
        except ProcessLookupError:return


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage_id');parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args();plan=json.loads((HERE/'plan.json').read_text())
    stage=next(s for s in plan['stages'] if s['id']==args.stage_id)
    logs=Path(stage['logs']);logs.mkdir(parents=True,exist_ok=True)
    if args.dry_run:
        print(json.dumps(dict(stage=stage['id'],tasks=len(stage['tasks']),completed=len(completed(stage)),
            native_expert=stage['native'],domain=stage['domain'],port=stage['port']),ensure_ascii=False));return
    with (logs/'.collection.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if len(completed(stage))==len(stage['tasks']):audit(stage);return
        env=os.environ.copy();env.update(Z1_REVERSE8_NATIVE=stage['native'],
            Z1_REVERSE8_CURRENT=str(logs/'current.json'),Z1_ISAAC_GRASP_CONFIG=str(Path(stage['native'])/'grasp_physics.json'))
        command=['ros2','launch',str(HERE/'basic_wrapper.launch.py'),
            'external_camera_layout:=dual','headless:=true','rviz:=false',
            'task_gui:=true',f"task_gui_port:={stage['port']}",f"target_object:={stage['object']}",
            'run_smolvla_policy:=false','freeze_target_keep_collisions:=true',f"dataset_root:={stage['data']}",
            'wrist_rgb_topic:=/left_wrist_camera/color/image_raw','vla_wrist_rgb_topic:=none',
            'recording_fps:=10.0','recording_sync_tolerance_s:=0.045',f"recording_sync_delay_s:={stage.get('recording_sync_delay_s',.5)}",'recording_action_age_s:=0.40']
        atomic_json(logs/'command.json',command)
        process=None
        try:
            scene_log=logs/('scene_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.log')
            with scene_log.open('w') as output:
                process=subprocess.Popen(command,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
            status(stage,'starting_scene',scene_pid=process.pid,scene_log=str(scene_log))
            deadline=time.monotonic()+300
            while time.monotonic()<deadline:
                if process.poll() is not None:raise RuntimeError('Owned simulator exited')
                startup_text=scene_log.read_text(errors='replace')
                if 'ImportError:' in startup_text or 'ModuleNotFoundError:' in startup_text:
                    raise RuntimeError('A required node failed to import; inspect '+str(scene_log))
                try:
                    live=api(stage['port'],'/api/status',timeout=2)
                    if collection_ready(live,stage['object']):
                        assert Path(live['recording']['dataset_root']).resolve()==Path(stage['data']).resolve();break
                except (OSError,ValueError):pass
                time.sleep(2)
            else:raise RuntimeError('Scene readiness timed out')
            import ast
            lines=[l for l in scene_log.read_text().splitlines() if 'Close-range glass occlusion:' in l]
            if not lines or ast.literal_eval(lines[-1].split('Close-range glass occlusion:',1)[1].strip()).get('/rtx/translucency/worldEps')!=.0001:
                raise RuntimeError('Runtime rendering fix not verified')
            atomic_json(logs/'render_verified.json',dict(verified=True,world_eps=.0001))
            stalled=0;round_index=0
            while len(completed(stage))<len(stage['tasks']):
                before=len(completed(stage));round_index+=1
                for task in stage['tasks']:
                    if task['id'] in completed(stage):continue
                    if process.poll() is not None:raise RuntimeError('Owned simulator exited during recording')
                    if __import__('shutil').disk_usage(stage['data'] if Path(stage['data']).exists() else Path(stage['data']).parent).free<10*1024**3:
                        raise RuntimeError('Less than 10GB free; collection stopped with saved data preserved')
                    ctx=dict(task=selected_task(task,stage,round_index),attempt_id=uuid.uuid4().hex,plan_hash=stage['plan_hash'],
                        round=round_index,logs=stage['logs'])
                    atomic_json(logs/'current.json',ctx);status(stage,'collecting',task_id=task['id'],round=round_index)
                    attempt=dict(task_id=task['id'],attempt_id=ctx['attempt_id'],round=round_index)
                    try:
                        attempt['result']=api(stage['port'],'/api/demonstration/auto',
                            dict(target_object=stage['object'],curriculum_level='full',target_grid=task['grid']),timeout=21600)
                        if task['id'] not in completed(stage):raise RuntimeError('API success without verified task save')
                        attempt['saved']=True
                    except (OSError,ValueError,RuntimeError) as exc:
                        attempt.update(saved=False,error=str(exc))
                        for endpoint in ['/api/force_stop','/api/recording/discard']:
                            try:api(stage['port'],endpoint,{},timeout=15)
                            except (OSError,ValueError,RuntimeError):pass
                        deadline=time.monotonic()+45
                        while api(stage['port'],'/api/status').get('auto_demonstration',{}).get('running'):
                            if time.monotonic()>deadline:raise RuntimeError('Failed task did not stop; context will not overlap')
                            time.sleep(.5)
                    with (logs/'attempts.jsonl').open('a') as stream:stream.write(json.dumps(attempt,ensure_ascii=False)+'\n')
                after=len(completed(stage));stalled=stalled+1 if after==before else 0
                if stalled>=max([3]+[len(x) for x in stage.get('layout_fallbacks',{}).values()]):
                    raise RuntimeError('All bounded task/layout rounds exhausted; inspect owned scene logs')
            status(stage,'auditing');audit(stage);status(stage,'complete')
        finally:
            if process is not None:
                try:api(stage['port'],'/api/force_stop',{},timeout=5)
                except Exception:pass
                stop_scene(process)


if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    main()
