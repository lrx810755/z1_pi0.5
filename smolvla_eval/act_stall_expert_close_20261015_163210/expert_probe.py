"""Physical close-only takeover at recorded ACT stall poses, without inference."""
import json, time, threading, traceback, xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import rclpy
from rclpy.executors import MultiThreadedExecutor
from z1_isaac_bridge.smolvla_sim_rollout import SmolVLASimRollout, parse_args, VideoSet, JOINT_NAMES

ROOT=Path(__file__).resolve().parent
SOURCE=Path('/home/lp531/smolvla_eval/act_gentle_15vs8_repaired_20261015_133937')
URDF=Path('/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/generated_dual_camera/z1_with_dual_d435_resolved.urdf')
J={j.attrib['name']:j for j in ET.parse(URDF).getroot().findall('joint')}
CHAIN=[J[f'joint{i}'] for i in range(1,7)]+[J['gripperStatorLink']]
LOW=np.array([float(j.find('limit').attrib['lower']) for j in CHAIN[:6]])
HIGH=np.array([float(j.find('limit').attrib['upper']) for j in CHAIN[:6]])
def fk(q):
    T=np.eye(4)
    for i,j in enumerate(CHAIN):
        o=j.find('origin');A=np.eye(4)
        A[:3,:3]=Rotation.from_euler('xyz',np.fromstring(o.attrib.get('rpy','0 0 0'),sep=' ')).as_matrix()
        A[:3,3]=np.fromstring(o.attrib.get('xyz','0 0 0'),sep=' ');T=T@A
        if i<6:
            A=np.eye(4);A[:3,:3]=Rotation.from_rotvec(np.fromstring(j.find('axis').attrib['xyz'],sep=' ')*q[i]).as_matrix();T=T@A
    return T

def lift_plan(q,world_quat):
    T=fk(q);Rworld=Rotation.from_quat(world_quat).as_matrix()@T[:3,:3].T
    direction=Rworld.T@np.array([0.,0.,.10]);plan=[];seed=np.asarray(q)
    for f in np.linspace(.05,1.,20):
        pos=T[:3,3]+direction*f
        def residual(x):
            A=fk(x)
            return np.r_[A[:3,3]-pos,.15*Rotation.from_matrix(T[:3,:3].T@A[:3,:3]).as_rotvec()]
        sol=least_squares(residual,seed,bounds=(LOW,HIGH),xtol=1e-10,ftol=1e-10,gtol=1e-10,max_nfev=200)
        e=residual(sol.x)
        if np.linalg.norm(e[:3])>.0005 or np.linalg.norm(e[3:])/.15>np.deg2rad(.2):
            raise RuntimeError('Cartesian lift IK failed')
        if np.max(abs(sol.x-seed))>.2:raise RuntimeError('Discontinuous lift IK')
        plan.append(sol.x.tolist());seed=sol.x
    return plan

def save(name,value):
    p=ROOT/name;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(p)

class Probe(SmolVLASimRollout):
    def _load_policy(self,_):
        self.camera_keys=['wrist','global','global_right'];self.state_dim=7
        return SimpleNamespace(reset=lambda:None)
    def snapshot(self):
        with self._lock:
            return dict(sim_stamp_s=self._joint_sim_stamp,joint_state=[self._joints[n] for n in JOINT_NAMES],status=dict(self._status))
    def phase(self,name,closed):
        self.stage=name
        self._publish_grasp_phase('beverage_bottle',dict(gripper_commanded_closed=closed,physical_test_lift_active=name in ['lift','hold_lift'],release_started=False,release_sequence_complete=False,physical_grasp_ok=False))
    def send(self,q):
        self._publish_command(q)
        self.log_step(q)
    def log_step(self,q):
        if not getattr(self,'trace',None):return
        data=self.snapshot();data.update(stage=self.stage,command=list(map(float,q)))
        self.trace.write(json.dumps(data)+'\n')
        if data['sim_stamp_s']-getattr(self,'last_video',-1e9)>=.099:
            with self._lock:images={k:v.copy() for k,v in self._images.items()}
            self.video.write(images);self.last_video=data['sim_stamp_s']
        if self.stage in ['close','hold_close','lift','hold_lift']:
            s=data['status']
            if s.get('left_finger_touching_table') or s.get('right_finger_touching_table'):
                raise RuntimeError('finger_table_contact_during_takeover')
    def wait_until(self,t):
        deadline=time.monotonic()+15
        while self._joint_sim_stamp<t:
            if time.monotonic()>deadline:raise RuntimeError('simulation_clock_stalled')
            time.sleep(.002)
    def move(self,q,duration):
        start=np.array(self.snapshot()['joint_state']);goal=np.array(q);t=self._joint_sim_stamp
        for i in range(1,int(duration*60)+1):
            f=i/(duration*60);f=f*f*(3-2*f)
            self.send(start+(goal-start)*f);self.wait_until(t+i/60)
    def hold(self,q,duration):
        t=self._joint_sim_stamp
        while self._joint_sim_stamp-t<duration:
            stamp=self._joint_sim_stamp;self.send(q);self.wait_until(stamp+1/60)
    def trial(self,ep,which):
        source=next((SOURCE/'steps15').glob(f'*/episodes/beverage_bottle_{ep:03d}_*/result.json'))
        original=json.loads(source.read_text());rows=[json.loads(l) for l in (source.parent/'trace.jsonl').read_text().splitlines()]
        analysis=json.loads((SOURCE/'review/analysis.json').read_text())
        rec=next(r for r in analysis['episodes'] if r['steps']==15 and r['episode']==ep)
        target=rows[-1] if which=='final' else min(rows,key=lambda r:abs(r['simulation_elapsed_s']-rec['closest_before_close']['time_s']))
        q=np.array(target['joint_state']);q[-1]=-.9
        name=f'ep{ep:02d}_{which}';folder=ROOT/name;folder.mkdir(exist_ok=False)
        self.video=VideoSet(folder/'videos',10);self.trace=(folder/'trace.jsonl').open('w',buffering=1);self.last_video=-1e9
        result=dict(episode=ep,snapshot=which,seed=original['seed'],source=str(source),source_sim_elapsed_s=target['simulation_elapsed_s'],commanded_preclose_joints=q.tolist(),status='running')
        save('status.json',dict(status='running',trial=name,stage='reset'))
        try:
            self.phase('reset',False);self._prepare_episode('beverage_bottle',original['seed']);self._set_target_frozen('beverage_bottle',False)
            with self._lock:layout=dict(self._randomization_status)
            initial=layout['object_positions_m']['beverage_bottle']
            if np.linalg.norm(np.array(initial)-original['target_position_m'])>.001:raise RuntimeError('Seed layout differs')
            # Replay the recorded approach up to 12 s; then settle to the saved
            # static pose. All three source episodes have no finger contact.
            self.phase('restore_approach',False)
            end=min(12.,target['simulation_elapsed_s']);t=self._joint_sim_stamp
            for dt in np.arange(0,end,.1):
                row=min(rows,key=lambda r:abs(r['simulation_elapsed_s']-dt));cmd=np.array(row['command']);cmd[-1]=-.9
                self.send(cmd);self.wait_until(t+dt+.1)
            self.move(q,4.);self.hold(q,1.)
            before=self.snapshot();s=before['status'];actual=np.array(before['joint_state'])
            joint_error=float(max(abs(actual[:6]-q[:6])))
            rel_error=float(np.linalg.norm(np.array(s['relative_position_gripper_m'])-target['contact_status']['relative_position_gripper_m']))
            bottle_error=float(np.linalg.norm(np.array(s['object_position_m'])-original['target_position_m']))
            if joint_error>.01 or rel_error>.003 or bottle_error>.001:
                raise RuntimeError(f'Restored pose mismatch: joint={joint_error},relative={rel_error},bottle={bottle_error}')
            result.update(restoration=dict(max_joint_error_rad=joint_error,relative_position_error_m=rel_error,bottle_position_error_m=bottle_error),before_close=before)
            self.phase('close',True)
            # Arm targets remain identical while the gripper closes.
            closed=q.copy();closed[-1]=0.;self.move(closed,1.5);self.phase('hold_close',True);self.hold(closed,1.5)
            after=self.snapshot();result['after_close']=after
            # Derive a pure world-Z lift from the actual tool pose; no XY or
            # orientation correction is performed before grasping.
            actual=np.array(after['joint_state']);plan=lift_plan(actual[:6],after['status']['gripper_world_orientation_xyzw'])
            (folder/'lift_plan.json').write_text(json.dumps(plan,indent=2))
            self.phase('lift',True)
            for waypoint in plan:self.move([*waypoint,0.],.3)
            self.phase('hold_lift',True);self.hold([*plan[-1],0.],2.)
            final=self.snapshot();result['after_lift']=final
            self.trace.flush();logged=[json.loads(l) for l in (folder/'trace.jsonl').read_text().splitlines()]
            tail=[r for r in logged if r['stage']=='hold_lift' and r['sim_stamp_s']>=final['sim_stamp_s']-1.]
            baseline=before['status']['object_position_m'][2]
            rises=np.array([r['status']['object_position_m'][2]-baseline for r in tail])
            contacts=[r['status'].get('left_finger_contact_object')=='beverage_bottle' and r['status'].get('right_finger_contact_object')=='beverage_bottle' for r in tail]
            rel=np.array([r['status']['relative_position_gripper_m'] for r in tail]);spread=float(np.max(np.linalg.norm(rel-rel[0],axis=1)))
            success=bool(min(rises)>=.05 and np.mean(contacts)>=.8 and spread<=.01)
            tool_rise=final['status']['gripper_world_position_m'][2]-after['status']['gripper_world_position_m'][2]
            if tool_rise<.085:raise RuntimeError(f'Lift execution incomplete: {tool_rise}')
            result.update(status='complete',physically_graspable=success,proof=dict(final_object_rise_m=float(rises[-1]),minimum_last_second_rise_m=float(min(rises)),last_second_bilateral_contact_fraction=float(np.mean(contacts)),last_second_relative_position_spread_m=spread,tool_rise_m=tool_rise),criterion='Object remains lifted >=50mm with >=80% bilateral contact in final second and <=10mm relative position drift; evaluator grasp-envelope/5deg gate ignored')
        except Exception as e:
            result.update(status='invalid',error=str(e),traceback=traceback.format_exc())
        finally:
            self.trace.close();self.trace=None;self.video.close()
            (folder/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
            self._set_active(False)
        return result

def main():
    args=parse_args(['--policy-type','act','--checkpoint',str(ROOT),'--objects','beverage_bottle','--target-xy-bounds','.35','.70','-.27','.16','--wrist-rgb-topic','/left_wrist_camera/color/image_raw','--no-freeze-target-until-close','--execution-rate','60','--rate','10','--output',str(ROOT)])
    args.training_manifest=None
    rclpy.init();node=Probe(args);executor=MultiThreadedExecutor(num_threads=4);executor.add_node(node)
    thread=threading.Thread(target=executor.spin,daemon=True);thread.start();results=[]
    try:
        deadline=time.monotonic()+30
        while (not node._joints or not node._status) and time.monotonic()<deadline:time.sleep(.1)
        if not node._joints or not node._status:raise RuntimeError('No live simulation streams')
        for which in ['final','closest']:
            for ep in [16,21,26]:
                r=node.trial(ep,which);results.append(r);save('results.json',results)
                print('TRIAL_RESULT',ep,which,r['status'],r.get('physically_graspable'),r.get('proof',r.get('error')),flush=True)
        save('status.json',dict(status='complete',trials=len(results),valid=sum(r['status']=='complete' for r in results),graspable=sum(bool(r.get('physically_graspable')) for r in results)))
    except Exception as e:
        save('status.json',dict(status='failed',error=str(e)));raise
    finally:
        node._set_active(False);executor.shutdown();thread.join(timeout=2);node.destroy_node();rclpy.shutdown()
if __name__=='__main__':main()
