"""No-policy, paired fixed-demonstration friction/drive-cap experiment."""
import argparse, json, random, threading, time, uuid
from pathlib import Path
from types import SimpleNamespace
from protocol import PROFILES, BASELINE, rank_profiles, strict_placement

ROOT=Path(__file__).resolve().parent
EPISODES=['episode_000157','episode_000154','episode_000160','episode_000155']

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dry-run',action='store_true')
    opts=parser.parse_args()
    if opts.dry_run:
        print(json.dumps(dict(profiles=PROFILES,screen_trials=18,validation_trials_max=48,model_loaded=False),indent=2))
        return
    import numpy as np
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from std_msgs.msg import String
    from z1_isaac_bridge.smolvla_sim_rollout import SmolVLASimRollout, parse_args

    class Replay(SmolVLASimRollout):
        def _load_policy(self,checkpoint):
            self.camera_keys=['wrist','global','global_right'];self.state_dim=7
            return SimpleNamespace(reset=lambda:None,close=lambda:None)
        def _policy_batch(self,*args):return None
        def _predict(self,batch):
            if self.index==0:self.trace=[]
            i=min(self.index,len(self.actions)-1);a=self.actions[i].copy()
            a[-1]=float(self.label['close_frame']<=i<self.label['open_frame'])
            self.index+=1
            if self.index>len(self.actions)+50:raise RuntimeError('reference_complete_without_success')
            return a[None]
        def _randomize_scene(self,target,fixed_seed=None):
            original=self._randomization_command_pub
            grid=self.recorded_scene['target_grid']
            class Publisher:
                def publish(inner,message):
                    data=json.loads(message.data);data['target_grid']=grid
                    message.data=json.dumps(data);original.publish(message)
            self._randomization_command_pub=Publisher()
            try:return super()._randomize_scene(target,fixed_seed)
            finally:self._randomization_command_pub=original
        def _snapshot(self):
            sample=super()._snapshot()
            if getattr(self,'trace',None) is not None:
                st=sample[2]
                self.trace.append(dict(time=self._joint_sim_stamp,joints=sample[1].tolist(),
                    status={k:st.get(k) for k in ['object_position_m','physical_grasp_ok','bilateral_contact',
                        'left_finger_touching_table','right_finger_touching_table','target_finger_max_penetration_m',
                        'release_sequence_complete','task_success','object_speed_m_s']}))
            return sample
        def profile(self,mode,values=None):
            request=str(uuid.uuid4());self.parameter_reply={}
            message=String();message.data=json.dumps(dict(command='comparison_parameters',mode=mode,profile=values,request_id=request))
            deadline=time.monotonic()+8
            while time.monotonic()<deadline:
                self._scene_grasp_pub.publish(message)
                for _ in range(10):
                    reply=self.parameter_reply
                    if reply.get('request_id')==request:
                        if not reply.get('ok'):raise RuntimeError(reply.get('error','parameter update failed'))
                        return reply
                    time.sleep(.05)
            raise RuntimeError('parameter interface unavailable; start the experimental scene.py launcher')

    output=ROOT/('results_'+time.strftime('%Y%m%d_%H%M%S'));output.mkdir(exist_ok=False)
    labels={Path(x['raw_root']).name:x for x in json.loads(Path('/home/lp531/pi05_lora_workspace/binary_gripper_20261007/episode_labels.json').read_text())}
    args=parse_args(['--checkpoint',str(ROOT),'--objects','beverage_bottle','--wrist-rgb-topic','/left_wrist_camera/color/image_raw',
        '--reject-blank-cameras','--execute-steps','1','--rate','10','--execution-rate','60','--prelift-contact-stable-s','0',
        '--no-freeze-target-until-close','--binary-gripper','--gripper-action-encoding','binary','--timeout','100',
        '--startup-timeout','30','--output',str(output)])
    args.training_manifest=None
    rclpy.init();node=None;executor=None;thread=None;original=None;rows=[]
    try:
        node=Replay(args);executor=MultiThreadedExecutor(num_threads=4);executor.add_node(node)
        def reply_callback(message):
            try:node.parameter_reply=json.loads(message.data)
            except ValueError:pass
        node.parameter_reply={}
        node.create_subscription(String,'/smolvla/grasp_comparison_status',reply_callback,10)
        thread=threading.Thread(target=executor.spin,daemon=True);thread.start()
        node._disable_other_control()
        original=node.profile('probe')['original']
        if abs(original['stiffness']-6000)>1e-3 or abs(original['damping']-240)>1e-3:
            raise RuntimeError('expected fixed stiffness=6000,damping=240; got '+str(original))
        (output/'manifest.json').write_text(json.dumps(dict(profiles=PROFILES,original=original,model_loaded=False,episodes=EPISODES),indent=2))

        def trial(stage,name,episode,repeat):
            label=labels[episode];source=Path(label['raw_root'])
            node.label=label;node.recorded_scene=json.loads((source/'episode_start.json').read_text())['scene_randomization']
            node.actions=np.array([json.loads(x)['action'] for x in (source/'frames.jsonl').read_text().splitlines()])
            node.index=0;node.trace=None
            actual=node.profile('set',PROFILES[name])['actual']
            directory=output/stage/f'{name}_{episode}_repeat{repeat}';directory.mkdir(parents=True)
            node.trace=[]
            result=node.run_episode('beverage_bottle',len(rows)+1,directory,fixed_seed=node.recorded_scene['seed'])
            trace=node.trace;node.trace=None
            expected=np.array(node.recorded_scene['object_positions_m']['beverage_bottle'])
            position_error=float(np.linalg.norm(np.array(result['target_position_m'])-expected))
            if position_error>.005:raise RuntimeError('initial layout mismatch '+str(position_error))
            # Stream faults and operator interruption are infrastructure failures, not scored trials.
            if result['error_detail'] not in ['', 'reference_complete_without_success'] or result['stop_reason'] in ['operator_stop','stale_stream','input_sync_timeout','sim_clock_timeout']:
                raise RuntimeError('invalid trial '+json.dumps(result))
            grasped=any(x['status'].get('physical_grasp_ok') for x in trace)
            collision=any(x['status'].get('left_finger_touching_table') or x['status'].get('right_finger_touching_table') for x in trace)
            penetration=max([float(x['status'].get('target_finger_max_penetration_m') or 0) for x in trace]+[0])
            row=dict(stage=stage,profile=name,episode=episode,repeat=repeat,actual_parameters=actual,
                success=bool(result['success']),strict_success=strict_placement(result) and not collision,
                grasped=grasped,penetration_m=penetration,table_contact=collision,
                stop_reason=result['stop_reason'],initial_position_error_m=position_error,path=str(directory))
            (directory/'state_trace.json').write_text(json.dumps(trace))
            (directory/'comparison_metrics.json').write_text(json.dumps(row,indent=2))
            rows.append(row);(output/'trials.json').write_text(json.dumps(rows,indent=2))
            print('TRIAL',len(rows),name,episode,'strict=',row['strict_success'],'grasp=',grasped,flush=True)

        randomizer=random.Random(20261008)
        tasks=[(name,'episode_000160',repeat) for name in PROFILES for repeat in range(2)]
        randomizer.shuffle(tasks)
        for values in tasks:trial('screen',*values)
        effort={k:v['effort'] for k,v in PROFILES.items()}
        screen=rank_profiles(rows,2,effort)
        candidates=list(dict.fromkeys([x['profile'] for x in screen[:3]]+[BASELINE]))
        if not screen:
            (output/'recommendation.json').write_text(json.dumps(dict(recommended=None,reason='no candidate passed strict placement screening'),indent=2))
            return
        tasks=[(name,ep,repeat) for name in candidates for ep in EPISODES for repeat in range(3)]
        randomizer.shuffle(tasks)
        for values in tasks:trial('validation',*values)
        ranking=rank_profiles([x for x in rows if x['stage']=='validation'],12,effort)
        recommendation=ranking[0]['profile'] if ranking and ranking[0]['strict_successes']>=9 else None
        record=dict(recommended=recommendation,parameters=PROFILES[recommendation] if recommendation else None,
            ranking=ranking,minimum_strict_successes=9,scope='sampled 4 scenes, 3 repeats each; strict reference-point proxy, video review required',
            production_parameters_changed=False)
        (output/'recommendation.json').write_text(json.dumps(record,indent=2))
        report=['# 摩擦与夹爪力度对照结果','',
            '未加载模型。四个固定示范场景，每组各重复三次。参数推荐只适用于本候选范围。','',
            '| 参数组合 | 严格放置代理成功 | 物理抓取 | 原生成功 | 最大接触穿透 |',
            '|---|---:|---:|---:|---:|']
        for row in ranking:
            report.append(f"| {row['profile']} | {row['strict_successes']}/12 | {row['grasps']}/12 | {row['successes']}/12 | {row['penetration_max_m']*1000:.2f} mm |")
        report+=['', '候选推荐：'+str(recommendation),
            '最终采纳前需查看对应三路录像，确认完整瓶子位于盒内。生产参数未自动修改。']
        (output/'结果报告.md').write_text('\n'.join(report)+'\n')
        print('RECOMMENDATION',json.dumps(record),flush=True)
    finally:
        if node and original:
            try:
                node._set_active(False);node._disable_other_control()
                restored=node.profile('restore')
                (output/'restored.json').write_text(json.dumps(restored,indent=2))
            except Exception as exc:
                (output/'RESTORE_FAILED.txt').write_text(str(exc));print('RESTORE_FAILED',exc,flush=True)
        if executor:executor.shutdown()
        if thread:thread.join(timeout=2)
        if node:node.destroy_node()
        if rclpy.ok():rclpy.shutdown()

if __name__=='__main__':main()
