"""Compare the independently saved ten-demo model with both original controls."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parent
B=Path('/home/lp531/smolvla_eval/act_cup_train_closedloop10_20261021')
FIVE=Path('/home/lp531/smolvla_eval/act_cup_train_closedloop10_steps5_20261021')
T=Path('/home/lp531/pi05_lora_workspace/act_cup_overfit10_20261021')

def compare():
 training=json.loads((T/'completed.json').read_text());fit=json.loads((T/'fit_history.json').read_text());best=next(r for r in fit if r['step']==training['best_step'])
 original_plan=json.loads((B/'selected_scenes.json').read_text());plan=json.loads((R/'selected_scenes.json').read_text());assert plan['scenes']==original_plan['scenes']
 assert hashlib.sha256((Path(original_plan['checkpoint'])/'model.safetensors').read_bytes()).hexdigest()==original_plan['policy_sha256']
 assert hashlib.sha256((Path(plan['checkpoint'])/'model.safetensors').read_bytes()).hexdigest()==training['best_model_sha256']
 groups={};pairs={};proofs={}
 for key,folder in [('original15',B),('original5',FIVE),('overfit15',R)]:
  d=json.loads((folder/'diagnostic_results.json').read_text());assert d['completed'] and d['total']==10
  groups[key]={k:d[k] for k in ['total','successes','success_rate','ever_grasped','never_closed','never_entered_grasp_envelope','failure_categories']}
  check=[json.loads(l) for l in (folder/'scene_reproduction.jsonl').open()];assert len(check)==10 and all(c['verified'] for c in check)
  proofs[key]=dict(count=10,max_position_error_m=max(max(c['position_errors_m'].values()) for c in check),max_orientation_error_deg=max(max(c['orientation_errors_deg'].values()) for c in check),max_initial_joint_error_rad=max(c['max_initial_joint_error_rad'] for c in check))
  for ep in d['episodes']:
   seed=ep['seed'];p=pairs.setdefault(seed,dict(seed=seed,order=ep['order'],training_episode=ep['training_episode'],grid=ep['grid']));assert p['order']==ep['order'] and p['training_episode']==ep['training_episode']
   raw=json.loads(Path(ep['source_result']).read_text());assert raw['execute_steps']==(5 if key=='original5' else 15)
   p[key]=dict(success=ep['success'],physical_grasp=ep['ever_physical_grasp'],closed_once=ep['closed_once'],entered_grasp_envelope=ep['ever_grasp_envelope'],close_count=sum(e['kind']=='close' for e in ep['events']),open_count=sum(e['kind']=='open' for e in ep['events']),failure_category=ep['failure_category'],final_position_m=ep['final_position_m'],source_result=ep['source_result'])
 assert len(pairs)==10 and all(set(p)>=set(['original15','original5','overfit15']) for p in pairs.values())
 enough=best['normalized_l1']<=.008 and max(best['joint_mae_rad'])<=.008
 result=dict(completed=True,diagnostic_only=True,fit_checks_are_training_data=True,training=training,baseline_fit=fit[0],selected_fit=best,preset_fit_target_met=enough,metrics=groups,scene_verification=proofs,original_model_unchanged=True,pairs=sorted(pairs.values(),key=lambda p:p['order']))
 (R/'overfit_comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 lines=['# 杯子 ACT：10 段示范过拟合诊断与闭环结果','','这是在原050000权重上，仅用之前选定的10段完整训练示范继续训练的诊断。主干及策略学习率保持1e-5，batch16、原网络、损失、二值标签、原统计量和图像处理保持一致，采用新AdamW状态。检查点只依照固定训练窗口的推理误差选择，未依据闭环测试结果选择。','','这些误差来自训练数据，不是独立验证集，不能代表泛化。闭环也复用了训练场景，不能当作正式随机评估成功率。','','## 拟合结果','',f'- 原始固定窗口归一化L1：{fit[0]["normalized_l1"]:.6f}。',f'- 选定检查点固定窗口归一化L1：{best["normalized_l1"]:.6f}，相对下降{training["relative_fit_improvement"]*100:.1f}%。',f'- 追加更新{training["optimizer_updates"]}次，所选检查点为追加第{training["best_step"]}次更新。',f'- 预先设定的充分拟合目标（归一化L1≤0.008，且六关节各自MAE≤0.008弧度）：{"达到" if enough else "未达到"}。','','| 关节 | 原模型MAE/弧度 | 诊断模型MAE/弧度 |','|---|---:|---:|']
 for i,(a,b) in enumerate(zip(fit[0]['joint_mae_rad'],best['joint_mae_rad']),1):lines.append(f'| {i} | {a:.6f} | {b:.6f} |')
 lines+=['','## 相同10个训练场景的闭环结果','','| 指标 | 原模型15步 | 原模型5步 | 10段诊断模型15步 |','|---|---:|---:|---:|']
 for title,k in [('最终入盒成功','successes'),('稳定抓取并抬升','ever_grasped'),('从未闭爪','never_closed'),('从未进入抓取包络','never_entered_grasp_envelope')]:lines.append(f'| {title} | {groups["original15"][k]}/10 | {groups["original5"][k]}/10 | {groups["overfit15"][k]}/10 |')
 lines+=['','| 场景 | 训练示范 | 原模型15步结果 | 诊断模型15步结果 | 原模型闭爪次数 | 诊断模型闭爪次数 |','|---|---|---|---|---|---|']
 for p in result['pairs']:lines.append(f'| {p["order"]} | {p["training_episode"]} | {p["original15"]["failure_category"]} | {p["overfit15"]["failure_category"]} | {p["original15"]["close_count"]} | {p["overfit15"]["close_count"]} |')
 lines+=['','## 证据范围','','所有场景的种子、网格、物体位姿和初始关节均通过核验。原始模型哈希未变化。详细阶段统计、原始轨迹和视频路径见 `overfit_comparison.json` 与 `diagnostic_results.json`。','',f'诊断模型：`{plan["checkpoint"]}`。','', '若预设拟合目标未达到，不能用本轮闭环结果排除训练拟合不足；即使拟合误差下降，也仍需区分小样本记忆、闭环恢复与泛化。']
 (R/'report.md').write_text('\n'.join(lines)+'\n');return result
if __name__=='__main__':
 d=compare();print(json.dumps(dict(training=d['training'],fit_target_met=d['preset_fit_target_met'],metrics=d['metrics']),ensure_ascii=False,indent=2))
