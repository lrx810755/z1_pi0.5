import json,statistics,collections
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
def read(path):return json.loads(path.read_text())
def main():
 arms={};pairs=[]
 for arm in ['baseline','uniform','weighted']:
  p=ROOT/'corrected_comparison'/arm/'summary.json'
  if not p.exists():continue
  results=read(p);items=[]
  for r in results:
   path=ROOT/'corrected_comparison'/arm/'episodes'/f"beverage_bottle_{r['episode_number']:03d}_seed_{r['seed']}"/'trace.jsonl'
   trace=[json.loads(l) for l in path.read_text().splitlines()]
   close=any(row['command'][-1]>=-.75 for row in trace)
   hz=(len(trace)-1)/(trace[-1]['simulation_elapsed_s']-trace[0]['simulation_elapsed_s']) if len(trace)>1 else None
   contacts=sum(row['contact_status'].get('left_finger_contact_object')=='beverage_bottle' or row['contact_status'].get('right_finger_contact_object')=='beverage_bottle' for row in trace)
   items.append({'episode':r['episode_number'],'seed':r['seed'],'repetition':r['repetition'],'success':r['success'],'closed':close,'physical_grasp':bool(r['physical_predicates'].get('physical_grasp_ok')),'stop_reason':r['stop_reason'] or 'timeout','command_hz':hz,'target_contact_frames':contacts,'collision_mode':r['frozen_target_collision_mode']})
  arms[arm]={'trials':len(results),'successes':sum(i['success'] for i in items),'closed_trials':sum(i['closed'] for i in items),'physical_grasp_trials':sum(i['physical_grasp'] for i in items),'stops':dict(collections.Counter(i['stop_reason'] for i in items)),'median_command_hz':statistics.median(i['command_hz'] for i in items if i['command_hz'] is not None),'trials_detail':items}
 if all((ROOT/'corrected_comparison'/a/'completed.json').exists() for a in ['baseline','uniform','weighted']):
  base=read(ROOT/'corrected_comparison'/'baseline'/'summary.json')
  for arm in ['uniform','weighted']:
   for b,r in zip(base,read(ROOT/'corrected_comparison'/arm/'summary.json')):
    assert b['seed']==r['seed'] and b['episode_number']==r['episode_number']
    bs=b['initial_scene_randomization'];rs=r['initial_scene_randomization']
    layout=max(float(np.linalg.norm(np.asarray(p)-rs['object_positions_m'][n])) for n,p in bs['object_positions_m'].items())
    joints=max(abs(a-z) for a,z in zip(b['initial_joint_state'],r['initial_joint_state']))
    pairs.append({'arm':arm,'episode':r['episode_number'],'seed':r['seed'],'all_object_max_position_delta_m':layout,'max_joint_delta_rad':joints})
 supplemental=read(ROOT/'supplement_collection_result.json') if (ROOT/'supplement_collection_result.json').exists() else None
 payload={'arms':arms,'matched_scene_checks':pairs,'supplement':supplemental,'all_evaluation_arms_completed':len(pairs)==18,'tests':{'focused_passed':68,'full_passed':343,'full_failed':14,'known_existing_failures':True}}
 (ROOT/'four_steps_results.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False))
 lines=['# 四步验证结果','', '## 1. 修正评估物理','', '评估固定瓶子时保持碰撞开启，闭爪前等待动力学恢复确认。状态记录包含碰撞开关；示教采集新增同样的保留碰撞选项。另修正示教命令确认被定期遥测覆盖的问题。','', '## 2. 成功示范回放','', '选取89、235、53、0、120五条成功示范，以原始种子和方格重置。所有物体布局误差不超过几微米，回放采用绝对仿真时间的10Hz。','', '现有安全执行：5/5因跟踪保护停止，每条停止前25个动作被0.1rad步长限制改写。取消步长限制的诊断：5/5执行完保存的动作仍无任务成功。关节限位保留，仅初始姿态有至多0.0003rad限位修正。','', '因此，不能把问题全部归因于采样权重或10Hz与约8Hz的差异。当前示教代码默认固定时关闭碰撞，评估已改为开启；旧数据没有逐帧保存这个物理开关，尚不能证明纯动作可在新物理下复现。','', '## 3. 修正物理后的三模型对照','', '| 模型 | 已完成 | 成功 | 闭爪 | 物理抓取 | 停止原因 |','|---|---:|---:|---:|---:|---|']
 for name,a in arms.items():lines.append(f"| {name} | {a['trials']}/9 | {a['successes']} | {a['closed_trials']} | {a['physical_grasp_trials']} | {a['stops']} |")
 lines += ['', '每个模型使用相同3个布局，各重复3次，采样随机种子固定为20261003。重复运行考察物理与复位差异；不代表不同策略采样随机性的全面统计。']
 if pairs:lines += ['',f"三模型场景配对检查18组：物体位置最大差异{max(p['all_object_max_position_delta_m'] for p in pairs):.8f}m，初始关节最大差异{max(p['max_joint_delta_rad'] for p in pairs):.6f}rad。初始关节存在残余复位抖动，不能声称完全一致。"]
 lines += ['', '## 4. 有针对性的连续补录','']
 if supplemental:
  attempt_files=['supplement_collection_progress.json','supplement_verified_layout_attempts.json','supplement_verified_layout_attempts_round2.json','supplement_linear_approach_attempts.json']
  attempt_counts={}
  for filename in attempt_files:
   p=ROOT/filename
   if p.exists():
    content=read(p)
    attempt_counts[filename]=(sum(item['attempts'] for item in content) if filename=='supplement_collection_progress.json' else len(content)) if isinstance(content,list) else None
  payload['supplement_attempt_files']=attempt_counts
  payload['supplement_total_attempts']=sum(count for count in attempt_counts.values() if count is not None)
  lines += ['补录遇到两个已定位的问题：策略回放命令原先使用墙钟时间，导致使用仿真时间的录制缓冲区无法匹配动作；现已统一为观测关节的仿真时间，独立检查成功写入32帧后丢弃诊断样本。真实碰撞下自动示教仍出现关节跟踪保护及最终位姿误差；改用既有Pilz直线最终接近进行验证，仍出现跟踪保护中止；不能据此宣称示教执行链路已修好。', '', f'重试记录：{attempt_counts}。第4步只有在完整物理成功并通过无辅助动作回放后才算完成。', '']
  if (ROOT/'camera_cadence_probe.json').exists():
   camera=read(ROOT/'camera_cadence_probe.json')
   payload['camera_cadence_probe']=camera
   lines += ['最后6次直线接近尝试中，2次完成物理抓取与放置，但全局相机缺帧导致连续录制保存失败，已丢弃。15秒独立图像订阅检查：全局图像最大间隔约167ms、24处间隔超过90ms；腕部最大间隔约117ms。45ms同步容差无法覆盖这些空洞。第4步当前被相机流连续性和剩余运动跟踪问题阻断，尚未完成。', '']
  lines += [f"新录制成功示范：{supplemental['total_successes']}条；数据目录：`{supplemental['dataset_root']}`。",'', '目标失败方格为[1,1]、[2,0]、[3,0]。只保存完整成功轨迹，失败录制丢弃；未删除任何中间帧。逐帧物理状态另存supplement_physics_history.jsonl。','', f"训练准入：{supplemental['training_eligible']}；{supplemental['eligibility_reason']}"]
 else:lines.append('针对[1,1]、[2,0]、[3,0]方格的采集正在等待对照结束；计划每格2条成功轨迹，固定状态保留真实碰撞。')
 (ROOT/'four_steps_results.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False))
 lines += ['', '## 验证与限制','', '相关回归测试68项通过。全套343项通过、14项失败，失败项与本轮前已有问题一致，详见full_tests.log。新数据必须通过不借助示教辅助的动作回放，再合并和转换训练集。旧数据及全部中间帧保持原样。']
 (ROOT/'四步验证报告.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps({k:{x:v for x,v in a.items() if x!='trials_detail'} for k,a in arms.items()},ensure_ascii=False))
if __name__=='__main__':main()
