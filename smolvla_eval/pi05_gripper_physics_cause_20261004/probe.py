"""Bounded simulation-only contact controls using recorded arm commands."""
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

sys.path.insert(0, '/home/lp531/smolvla_eval/replay_debug_20261003')
from debug_replay import DebugReplay
from z1_isaac_bridge.smolvla_sim_rollout import parse_args, JOINT_NAMES
from z1_isaac_bridge.smolvla_rollout_core import GraspPhaseTracker

ROOT = Path(__file__).resolve().parent
STAGED = ROOT.parent / 'pi05_staged_takeover_20261004'
SOURCE = STAGED/'midway/trial_03/episodes/beverage_bottle_003_seed_342470491/trace.jsonl'
TEACHER = ROOT.parent/'replay_debug_20261003/diagnostic_sparse_commands/episode_000000'
records = [json.loads(x) for x in SOURCE.read_text().splitlines()]
reference = np.array([json.loads(x)['action'] for x in (TEACHER/'frames.jsonl').read_text().splitlines()])
source_times = np.array([r['simulation_elapsed_s'] for r in records])
source_commands = np.array([r['command'] for r in records])

def snapshot(node, start, command, phase, scheduled):
    _,actual,status,_ = node._snapshot()
    with node._lock:
        stamp = node._joint_sim_stamp
        contact = dict(getattr(node, 'contact', {}))
    return {'phase':phase, 'scheduled_s':scheduled, 'simulation_elapsed_s':stamp-start,
            'joint_state':actual.tolist(), 'command':command.tolist(), 'status':status, 'contact':contact}

def trial(node, label, mode):
    node.recorded_scene = json.loads((TEACHER/'episode_start.json').read_text())['scene_randomization']
    node._publish_target('beverage_bottle')
    node._randomize_scene('beverage_bottle',342470491)
    node._set_active(True)
    node.freeze(True)
    tracker = GraspPhaseTracker()
    with node._lock:start = node._joint_sim_stamp
    rows=[];reason='bounded_complete'
    # Identical teacher approach up to the original mid-way handoff.
    for tick in range(301):
        t=tick/60;node.wait_until(start+t)
        _,actual,status,_=node._snapshot()
        lo=min(int(t*10),50);hi=min(lo+1,50);ratio=t*10-lo
        command=reference[lo]*(1-ratio)+reference[hi]*ratio;command[-1]=-.9
        node._publish_grasp_phase('beverage_bottle',tracker.update(-.9,actual[-1],status))
        node._publish_command(command)
        rows.append(snapshot(node,start,command,'prefix',t))
        if np.max(np.abs(actual[:6]-command[:6]))>.20:
            raise RuntimeError('guided_prefix_tracking_error')
    if mode=='collision_off':node.freeze(False)
    elif mode=='dynamic':node._set_target_frozen('beverage_bottle',False)
    # Hold the same mid-way arm pose before the fixed gripper/no-model test.
    for tick in range(31):
        with node._lock:stamp=node._joint_sim_stamp
        command=reference[50].copy();command[-1]=-.9;node._publish_command(command)
        node.wait_until(stamp+1/60)
    _,actual,status,_=node._snapshot()
    handoff={'joint_state':actual.tolist(),'arm_error_from_reference':float(np.max(np.abs(actual[:6]-reference[50,:6]))),'frozen':status.get('frozen'),'target_collisions_enabled':status.get('target_collisions_enabled')}
    assert handoff['arm_error_from_reference']<.02, handoff
    with node._lock:test_start=node._joint_sim_stamp
    # No model and no gripper limiter: the gripper target is always -0.9.
    for tick in range(361):
        t=tick/60;node.wait_until(test_start+t)
        command=np.array([np.interp(t,source_times,source_commands[:,j]) for j in range(7)])
        command[-1]=-.9
        row=snapshot(node,test_start,command,'test',t);actual=np.array(row['joint_state']);status=row['status']
        rows.append(row)
        if actual[-1]<-1.65 or actual[-1]>.10:reason='gripper_feedback_abort';break
        if np.max(np.abs(actual[:6]-command[:6]))>.20:reason='arm_tracking_abort';break
        if status.get('left_finger_touching_table') or status.get('right_finger_touching_table'):reason='finger_table_contact';break
        node._publish_grasp_phase('beverage_bottle',tracker.update(-.9,actual[-1],status))
        node._publish_command(command)
    test=[r for r in rows if r['phase']=='test']
    result={'label':label,'mode':mode,'reason':reason,'handoff':handoff,'test_rows':len(test),
            'fixed_gripper_target':-.9,'test_feedback_range':[min(r['joint_state'][-1] for r in test),max(r['joint_state'][-1] for r in test)],
            'max_gripper_error_rad':max(abs(r['joint_state'][-1]+.9) for r in test),
            'max_arm_tracking_error_rad':max(float(np.max(np.abs(np.array(r['joint_state'])[:6]-np.array(r['command'])[:6]))) for r in test),
            'max_target_penetration_m':max((r['status'].get('target_finger_max_penetration_m') or 0) for r in test),
            'final_object_position':test[-1]['status'].get('object_position_m')}
    (ROOT/(label+'_trace.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (ROOT/(label+'_result.json')).write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    # End each diagnostic in home, with target physical again.
    node._set_target_frozen('beverage_bottle',False)
    node._call_trigger(node._reset_episode,120)
    node._set_active(False)
    return result

def main():
    if (ROOT/'started.json').exists():raise RuntimeError('Refuse to overwrite run')
    (ROOT/'started.json').write_text(json.dumps({'source':str(SOURCE),'teacher':str(TEACHER),'test_duration_s':6,'fixed_gripper':-.9,'model_loaded':False},indent=2))
    args=parse_args(['--checkpoint',str(TEACHER),'--objects','beverage_bottle','--wrist-rgb-topic','/left_wrist_camera/color/image_raw'])
    args.training_manifest=None
    rclpy.init();node=DebugReplay(args);executor=MultiThreadedExecutor(num_threads=4);executor.add_node(node)
    thread=threading.Thread(target=executor.spin,daemon=True);thread.start();results=[]
    try:
        node._disable_other_control()
        for label,mode in [('fixed_collision_on','fixed'),('fixed_collision_off','collision_off'),('dynamic_collision_on','dynamic')]:
            print('START',label,flush=True);results.append(trial(node,label,mode));(ROOT/'results.json').write_text(json.dumps(results,indent=2))
        (ROOT/'completed.json').write_text(json.dumps({'trials':len(results)},indent=2))
    finally:
        try:
            node._set_target_frozen('beverage_bottle',False)
            node._call_trigger(node._reset_episode,120)
        finally:
            node._set_active(False);executor.shutdown();thread.join(timeout=2);node.destroy_node();rclpy.shutdown()

if __name__=='__main__':main()
