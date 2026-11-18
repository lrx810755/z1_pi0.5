"""Refill only the 35 removed successes, audit, then fine-tune without polling an LLM."""
import collections,json,os,signal,subprocess,sys,time
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError
import run
RUN=Path(__file__).resolve().parent
AUDIT=Path('/home/lp531/smolvla_eval/three_camera_dataset_strict_audit_20261006')
PY=RUN.parent/'.venv/bin/python'
CONT=Path('/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge/z1_isaac_bridge/continuous_training.py')

def deficits(counts):
    return {(x,y):8-counts.get((x,y),0) for y in range(4) for x in range(5) if counts.get((x,y),0)<8}

def saved_counts():
    entries=[json.loads(l) for l in (run.RAW/'episodes.jsonl').read_text().splitlines() if l.strip()]
    counts=collections.Counter(tuple(e['scene_randomization']['target_grid']) for e in entries if e.get('task_success') and e.get('target_object')=='beverage_bottle')
    return len(entries),counts

def collect_missing(before):
    round_number=0
    while True:
        current,counts=saved_counts()
        debt=deficits(counts)
        if not debt:return
        if current-before+sum(debt.values())!=35:
            raise RuntimeError(f'Unexpected remaining quota: {current}, {debt}')
        round_number+=1
        for x,y in debt:
            current,counts=saved_counts();needed=max(0,8-counts.get((x,y),0))
            if not needed:continue
            run.status('collecting',saved_episodes=current,new_successes=current-before,
                       grid=[x,y],remaining_in_cell=needed,round=round_number,retry_round_limit=None)
            args=['/usr/bin/python3',str(CONT),'--target','beverage_bottle','--curriculum','full','--episodes',str(needed),'--target-grid',f'{x},{y}','--max-failures','50']
            with (RUN/'collection.log').open('a') as log:
                result=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode in (130,143,-signal.SIGINT,-signal.SIGTERM):
                raise InterruptedError('Collection stopped by signal')
            if result.returncode not in (0,3,4):
                run.status('retrying_collection',grid=[x,y],collector_exit_code=result.returncode,round=round_number)
                time.sleep(5)
        # No overall failure or round limit; revisit only cells still missing successes.
        time.sleep(2)

def main():
    scene=None
    try:
        before=125
        current,counts=saved_counts();debt=deficits(counts)
        if not before<=current<=160 or current-before+sum(debt.values())!=35:
            raise RuntimeError(f'Unexpected resume quota: {current}, {debt}')
        if not (RUN/'recollection_plan.json').exists():
            if current!=before:raise RuntimeError('Missing original recollection plan')
            run.write('recollection_plan.json',dict(initial_episodes=before,new_successes_required=35,cells=[dict(column=x,row=y,episodes=n) for (x,y),n in debt.items()],skip_row=4,roll_deg=-90))
        run.status('starting_collection',saved_episodes=current,required_new=35,new_successes=current-before,retry_round_limit=None)
        if debt:
            cmd=['ros2','launch','z1_isaac_bridge','z1_isaac.launch.py','external_camera_layout:=dual','rviz:=false','task_gui:=true','run_smolvla_policy:=false','freeze_target_keep_collisions:=true',f'dataset_root:={run.RAW}','target_object:=beverage_bottle','wrist_rgb_topic:=/left_wrist_camera/color/image_raw','vla_wrist_rgb_topic:=none','recording_fps:=10.0','recording_sync_tolerance_s:=0.045','recording_sync_delay_s:=0.5','recording_action_age_s:=0.40']
            scene_log=(RUN/'scene.log').open('a')
            scene=subprocess.Popen(cmd,stdout=scene_log,stderr=subprocess.STDOUT,start_new_session=True)
            deadline=time.monotonic()+300
            while True:
                if scene.poll() is not None:raise RuntimeError('Collection scene exited during startup')
                try:
                    with urlopen('http://127.0.0.1:8767/api/status',timeout=3) as response:payload=json.load(response)
                    # HTTP readiness alone is sufficient for the collector, which
                    # performs the ROS/state readiness checks before recording.
                    break
                except (URLError,TimeoutError,OSError):
                    if time.monotonic()>deadline:raise RuntimeError('Collection API startup timeout')
                    time.sleep(2)
            collect_missing(before)
        count,counts=saved_counts()
        if count-before!=35 or deficits(counts):raise RuntimeError(f'Incomplete recollection: {count-before}/35')
        run.write('recollection_complete.json',dict(initial_episodes=before,final_episodes=count,new_successful_episodes=count-before,roll_deg=-90))
        run.status('auditing',saved_episodes=count)
        env=os.environ.copy();env.pop('PYTHONPATH',None);env.pop('LD_LIBRARY_PATH',None)
        with (RUN/'audit.log').open('w') as log:
            subprocess.run([str(PY),str(AUDIT/'audit.py')],stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
        summary=json.loads((AUDIT/'summary.json').read_text())
        failures={k:v for k,v in summary['violations'].items() if k!='action_joint_limit'}
        if summary['audit_errors'] or failures or summary['roll_groups']['positive90']['count'] or summary['duplicate_seeds']:
            raise RuntimeError(f'Dataset audit gate failed: {failures}; see audit.log')
        # Previous tiny positive joint3 targets were measured Home holds.
        # Reject larger violations rather than silently clipping labels.
        import xml.etree.ElementTree as ET
        robot=ET.parse(AUDIT/'robot.urdf').getroot()
        limits=[]
        for name in ['joint1','joint2','joint3','joint4','joint5','joint6','jointGripper']:
            limit=robot.find(f"joint[@name='{name}']/limit")
            limits.append((float(limit.get('lower')),float(limit.get('upper'))))
        maxovershoot=0.0
        for p in run.RAW.glob('episode_[0-9][0-9][0-9][0-9][0-9][0-9]'):
            for line in (p/'frames.jsonl').read_text().splitlines():
                r=json.loads(line)
                for value,(lower,upper) in zip(r['action'],limits):maxovershoot=max(maxovershoot,lower-value,value-upper)
        if maxovershoot>.001:raise RuntimeError(f'Action label out of range by {maxovershoot} rad')
        run.write('audit_gate.json',dict(passed=True,episodes=summary['episodes'],frames=summary['frames'],negative90_episodes=summary['roll_groups']['negative90']['count'],maximum_action_boundary_overshoot_rad=maxovershoot,minor_home_boundary_residual_retained=True))
        if scene is not None:
            os.killpg(scene.pid,signal.SIGTERM)
            try:scene.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(scene.pid,signal.SIGKILL);scene.wait()
            scene=None;scene_log.close()
        with (RUN/'pipeline.log').open('a') as log:
            subprocess.run([str(PY),str(RUN/'run.py')],stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
    except BaseException as exc:
        run.status('failed',error=str(exc));raise
    finally:
        if scene is not None and scene.poll() is None:
            os.killpg(scene.pid,signal.SIGTERM)
            try:scene.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(scene.pid,signal.SIGKILL);scene.wait()

if __name__=='__main__':main()
