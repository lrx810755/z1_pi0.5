import json, threading, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from std_msgs.msg import String
import rclpy
from rclpy.executors import MultiThreadedExecutor
from z1_isaac_bridge.smolvla_sim_rollout import SmolVLASimRollout, parse_args
ROOT=Path(__file__).resolve().parent
DATA=Path('/home/lp531/smolvla_data/z1_tabletop_bottle_recovery_boundary_only_prepared_v1')
class Replay(SmolVLASimRollout):
    def _load_policy(self, checkpoint):
        self.camera_keys=['global','wrist']; self.state_dim=7
        return SimpleNamespace(reset=lambda:None)
    def _policy_batch(self,*args): return None
    def _predict(self,batch):
        if self.index>=len(self.frames): raise RuntimeError('recorded_trajectory_complete')
        row=self.frames[self.index]; self.index+=1
        return np.asarray([row['action']],dtype=float)
    def _wait_sim_period(self,start_stamp):
        if self.playback_start is None: self.playback_start=start_stamp
        super()._wait_sim_period(self.playback_start+(self.index-1)/self.args.rate)
    def _set_target_frozen(self,target,frozen):
        if not frozen: return super()._set_target_frozen(target,False)
        with self._lock: self._scene_grasp_reply={}
        message=String();message.data=json.dumps({'command':'freeze','target':target,'keep_collisions':False})
        self._scene_grasp_pub.publish(message)
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            with self._lock:reply=dict(self._scene_grasp_reply)
            if reply.get('phase')=='FROZEN' and reply.get('target_collisions_enabled') is False:return
            time.sleep(.01)
        raise RuntimeError('Legacy diagnostic freeze acknowledgement missing')
    def _randomize_scene(self,target,fixed_seed=None):
        original=self._randomization_command_pub
        grid=self.recorded_scene.get('target_grid')
        class Publisher:
            def publish(self,message):
                obj=json.loads(message.data);obj['target_grid']=grid
                message.data=json.dumps(obj);original.publish(message)
        self._randomization_command_pub=Publisher()
        try: return super()._randomize_scene(target,fixed_seed)
        finally: self._randomization_command_pub=original

def main():
    args=parse_args(['--checkpoint',str(DATA),'--objects','beverage_bottle','--episodes-per-object','5','--execute-steps','1','--wrist-rgb-topic','/left_wrist_camera/color/image_raw','--freeze-target-until-close','--timeout','60','--output',str(ROOT/'replay_legacy_diagnostic')])
    args.training_manifest=None
    args.max_joint_step=3.2; args.tracking_error_duration=100.0
    rclpy.init(); node=Replay(args); executor=MultiThreadedExecutor(num_threads=4); executor.add_node(node)
    thread=threading.Thread(target=executor.spin,daemon=True); thread.start()
    output=ROOT/'replay_legacy_diagnostic';output.mkdir(exist_ok=True);results=[]
    try:
        for number,index in enumerate([89],1):
            episode=DATA/f'episode_{index:06d}'
            start=json.loads((episode/'episode_start.json').read_text())
            node.recorded_scene=start['scene_randomization']
            node.frames=[json.loads(line) for line in (episode/'frames.jsonl').read_text().splitlines()]
            node.index=0;node.playback_start=None
            seed=node.recorded_scene['seed']
            result=node.run_episode('beverage_bottle',number,output,fixed_seed=seed)
            result.update(source_episode=str(episode),source_frames=len(node.frames),sent_frames=node.index,replay_rate_hz=10,physics='legacy_collision_off_diagnostic_only',timing='absolute_simulation_deadlines',pure_joint_action=True, step_limit_disabled=True, joint_limits_retained=True, tracking_abort=False)
            before=node.recorded_scene.get('object_positions_m',{}); after=result.get('initial_scene_randomization',{}).get('object_positions_m',{})
            result['layout_errors_m']={name:float(np.linalg.norm(np.array(value)-after[name])) for name,value in before.items() if name in after}
            results.append(result);(output/'summary.json').write_text(json.dumps(results,indent=2)); print('REPLAY_RESULT',index,result['stop_reason'],result['physical_predicates'],flush=True)
    finally:
        node._set_active(False);executor.shutdown();thread.join(timeout=2);node.destroy_node();rclpy.shutdown()
if __name__=='__main__':main()
