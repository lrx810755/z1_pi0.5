"""Run all paired trials unattended and calculate results only after each batch."""
import collections,csv,datetime,hashlib,json,math,os,signal,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R))
from workflow_core import classify_error,in_box_released
CHECKPOINT=Path('/home/lp531/smolvla_training/act_cup_minus90_full_cached_chunk16_20261023/checkpoints/050000/pretrained_model')
SCENE=Path('/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/generated_dual_camera/smolvla_cup_only_frustum_20261025.usd')
RAW='/home/lp531/smolvla_data/cup_minus90_rerecord_20261022/basic160'

def save(name,data):
 p=R/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def stamp():return datetime.datetime.now().astimezone().isoformat()
def main():
 seeds=json.loads((R/'paired_seeds.json').read_text());baseline=json.loads((R/'baseline_results.json').read_text())
 expected_scene=json.loads((R/'scene_verified.json').read_text())['sha256']
 scene_hash=hashlib.sha256(SCENE.read_bytes()).hexdigest();assert scene_hash==expected_scene
 policy_hash=hashlib.sha256((CHECKPOINT/'model.safetensors').read_bytes()).hexdigest()
 assert policy_hash=='924f051170835d85b351de784296b08f2cf0080448490b975a86acfa2c194dcf'
 save('experiment.json',dict(checkpoint=str(CHECKPOINT),policy_sha256=policy_hash,scene=str(SCENE),scene_sha256=scene_hash,collision='solid convex frustum, bottom radius 0.018m, top radius 0.0252m, height 0.061m',task_objects=['cup'],removed_objects=['apple','peach','mouse','beverage_bottle'],cup_pose_mode='replay recorded initial cup pose of each cluttered baseline seed',episodes=10,execute_steps=15,action_rate_hz=10,execution_rate_hz=60,minimum_close_hold_s=1,bounds=[.36,.69,-.27,.074],seed_mode='first 10 ordered seeds from previous cup-only 30 trials',baseline='/home/lp531/smolvla_eval/act_cup_only_frustum_steps15_paired30_20261025',success_criterion='final cup root in x/y +/-0.055m about (0.32,0.14), z [0,0.12], released and not held',renderer_world_eps=0.0001,renderer_helper='/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/scripts/render_occlusion.py',progress_monitoring=False,started=stamp()))
 valid={};invalid=[];pending=list(seeds)
 try:
  for batch in range(1,4):
   if not pending:break
   folder=R/f'batch_{batch:02d}'
   if folder.exists():raise RuntimeError(f'Output already exists: {folder}')
   cmd=['/home/lp531/miniconda3/envs/unifolm-vla/bin/python',str(R/'eval_entry.py'),'--policy-type','act','--checkpoint',str(CHECKPOINT),'--objects','cup',
        '--target-xy-bounds','0.36','0.69','-0.27','0.074','--execute-steps','15','--rate','10','--execution-rate','60','--gripper-min-close-hold-s','1',
        '--binary-gripper','--gripper-action-encoding','binary','--no-freeze-target-until-close','--training-dataset-root',RAW,
        '--max-training-target-distance-m','0.03','--reject-blank-cameras','--wrist-rgb-topic','/left_wrist_camera/color/image_raw',
        '--prelift-contact-stable-s','0','--prelift-contact-timeout-s','2.0','--episodes-per-object',str(len(pending)),'--timeout','60','--output',str(folder),
        '--eval-seeds',*[str(s) for s in pending]]
   save(f'command_batch_{batch:02d}.json',cmd)
   save('status.json',dict(status='running',total=10,batch=batch,started=stamp(),progress_monitoring=False))
   with (R/f'batch_{batch:02d}.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
   files=list(folder.glob('act_sim_eval_*/episodes/*/result.json'))
   if len(files)!=len(pending):raise RuntimeError(f'Returned {len(files)}/{len(pending)} trial records')
   retry=[];seen=set()
   for p in files:
    d=json.loads(p.read_text());kind=classify_error(d);seed=d.get('seed')
    if seed not in pending or seed in seen:raise RuntimeError(f'Unmatched seed in {p}: {seed}; {d.get("error_detail")}')
    seen.add(seed)
    if kind=='fatal':raise RuntimeError(f'Evaluator error at {p}: {d.get("error_detail")}; {d.get("cleanup_error")}')
    if kind=='replace':invalid.append(str(p));retry.append(seed)
    else:valid[seed]=p
   pending=[s for s in seeds if s in retry]
  if len(valid)!=10:raise RuntimeError(f'Only {len(valid)}/10 valid paired trials')
  rows=[];pairs=[];health=[]
  before={r['seed']:r for r in baseline['episodes']}
  for number,seed in enumerate(seeds,1):
   p=valid[seed];d=json.loads(p.read_text());s=d['final_status'];x,y,z=s['object_position_m']
   assert d['target_object']=='cup' and d['execute_steps']==15 and classify_error(d)=='valid'
   videos={v:(p.parent/'videos'/v).stat().st_size for v in ['wrist.mp4','global_third_person.mp4','global_right.mp4']};assert min(videos.values())>0
   assert (p.parent/'prediction_chunks.jsonl').stat().st_size and (p.parent/'input_sync.jsonl').stat().st_size
   n=0;deltas=[]
   with (p.parent/'trace.jsonl').open() as f:
    for line in f:
     t=json.loads(line);n+=1;c=t.get('contact_status',{})
     if c.get('gripper_tf_stamp_s') is not None and c.get('object_pose_stamp_s') is not None:deltas.append(abs(c['gripper_tf_stamp_s']-c['object_pose_stamp_s']))
   assert n and deltas and max(deltas)<=.100001,(p,max(deltas,default=-1))
   row=dict(episode=number,source_episode=d['episode_number'],seed=seed,success=in_box_released(s),inside_box=abs(x-.32)<=.055 and abs(y-.14)<=.055 and 0<=z<=.12,final_position_m=[x,y,z],released=s.get('released'),holding=s.get('holding_observed'),legacy_success=d['success'],legacy_failure_category=d['failure_category'],source_result=str(p));rows.append(row)
   old=before[seed];old_d=json.loads(Path(old['source_result']).read_text())
   old_positions=old_d['initial_scene_randomization']['object_positions_m'];new_positions=d['initial_scene_randomization']['object_positions_m']
   assert set(new_positions)=={'cup'},new_positions
   errors={'cup':math.dist(old_positions['cup'],new_positions['cup'])}
   assert errors['cup']<=.002,errors
   old_q=old_d['initial_scene_randomization']['object_orientations_xyzw']['cup'];new_q=d['initial_scene_randomization']['object_orientations_xyzw']['cup']
   dot=abs(sum(a*b for a,b in zip(old_q,new_q)))/(math.sqrt(sum(a*a for a in old_q))*math.sqrt(sum(a*a for a in new_q)))
   angle=math.degrees(2*math.acos(min(1.,dot)));assert angle<=2.,angle
   pairs.append(dict(episode=number,seed=seed,before_fix_success=old['success'],after_fix_success=row['success'],before_fix_category=old['legacy_failure_category'],after_fix_category=row['legacy_failure_category'],initial_position_delta_m=errors,max_initial_position_delta_m=max(errors.values()),cup_initial_orientation_delta_deg=angle))
   health.append(dict(episode=number,trace_rows=n,max_tf_object_delta_s=max(deltas),videos=videos))
  assert hashlib.sha256(SCENE.read_bytes()).hexdigest()==scene_hash,'Scene changed during evaluation'
  successes=sum(r['success'] for r in rows)
  summary=dict(completed=True,total=10,execute_steps=15,successes=successes,success_rate=successes/10,episodes=rows,excluded_invalid_trials=invalid,finished=stamp())
  comparison=dict(total=10,before_fix_successes=baseline['successes'],after_fix_successes=successes,improved=sum(not p['before_fix_success'] and p['after_fix_success'] for p in pairs),regressed=sum(p['before_fix_success'] and not p['after_fix_success'] for p in pairs),max_initial_position_delta_m=max(p['max_initial_position_delta_m'] for p in pairs),after_fix_categories=dict(collections.Counter(r['legacy_failure_category'] for r in rows)),pairs=pairs)
  save('valid_results.json',[str(valid[s]) for s in seeds]);save('invalid_results.json',invalid);save('trace_video_tf_verified.json',health)
  save('final_box_results.json',summary);save('paired_comparison.json',comparison)
  with (R/'paired_comparison.csv').open('w') as f:
   fields=['episode','seed','before_fix_success','after_fix_success','before_fix_category','after_fix_category','max_initial_position_delta_m'];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(pairs)
  comparison['max_cup_initial_orientation_delta_deg']=max(p['cup_initial_orientation_delta_deg'] for p in pairs)
  save('paired_comparison.json',comparison)
  report=f"# 杯子渲染修复后：10次配对评估\n\n同一ACT最终模型，沿用上一轮前10个种子及杯子初始位姿，每轮执行15步。杯子独立场景和倒圆台碰撞体保持一致，启用worldEps=0.0001。\n\n- 修复后：{successes}/10（{successes/10:.1%}）。\n- 修复前相同10个种子：{baseline['successes']}/10。\n- 转成功：{comparison['improved']}；转失败：{comparison['regressed']}。\n- 初始化最大位置差：{comparison['max_initial_position_delta_m']*1000:.3f}毫米；朝向差：{comparison['max_cup_initial_orientation_delta_deg']:.3f}度。\n- 原始过程判定分类：{json.dumps(comparison['after_fix_categories'],ensure_ascii=False)}。\n\n成功以杯子最终位于蓝盒范围、已松爪、未被夹持为准。全部10次结束后统一读取结果，没有逐次监控。\n"
  (R/'评估结果.md').write_text(report)
  save('completed.json',summary);save('status.json',dict(status='complete',completed=10,successes=successes,success_rate=successes/10,finished=stamp()))
 except BaseException as e:
  save('status.json',dict(status='failed',error=str(e),finished=stamp()));raise
if __name__=='__main__':main()
