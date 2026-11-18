"""Verify paired scenes and summarize the 5-vs-15 execution experiment."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parent
B=Path('/home/lp531/smolvla_eval/act_cup_train_closedloop10_20261021')

def compare():
 plans=[json.loads((p/'selected_scenes.json').read_text()) for p in [B,R]]
 assert plans[0]['scenes']==plans[1]['scenes'] and plans[0]['policy_sha256']==plans[1]['policy_sha256']
 assert hashlib.sha256((Path(plans[0]['checkpoint'])/'model.safetensors').read_bytes()).hexdigest()==plans[0]['policy_sha256']
 a=json.loads((B/'diagnostic_results.json').read_text());b=json.loads((R/'diagnostic_results.json').read_text());assert a['completed'] and b['completed'] and a['total']==b['total']==10
 proofs=[]
 for folder in [B,R]:
  checks=[json.loads(l) for l in (folder/'scene_reproduction.jsonl').open()];assert len(checks)==10 and all(c['verified'] for c in checks)
  proofs.append(dict(folder=str(folder),scenes=10,max_position_error_m=max(max(c['position_errors_m'].values()) for c in checks),max_orientation_error_deg=max(max(c['orientation_errors_deg'].values()) for c in checks),max_initial_joint_error_rad=max(c['max_initial_joint_error_rad'] for c in checks)))
 pairs=[]
 for x,y in zip(a['episodes'],b['episodes']):
  assert (x['order'],x['seed'],x['training_episode'],x['grid'])==(y['order'],y['seed'],y['training_episode'],y['grid'])
  pair=dict(order=x['order'],seed=x['seed'],training_episode=x['training_episode'],grid=x['grid'])
  for key,e in [('steps15',x),('steps5',y)]:
   raw=json.loads(Path(e['source_result']).read_text());assert raw['execute_steps']==(15 if key=='steps15' else 5)
   for flag,value in [('action_rate_hz',10.),('execution_rate_hz',60.),('gripper_min_close_hold_s',1.),('freeze_target_until_close',False),('gripper_action_encoding','binary')]:assert raw[flag]==value,(flag,raw[flag],value)
   closes=[ev for ev in e['events'] if ev['kind']=='close'];opens=[ev for ev in e['events'] if ev['kind']=='open']
   predictions=[json.loads(l) for l in (Path(e['source_result']).parent/'prediction_chunks.jsonl').open()]
   assert all(p['execute_steps']==raw['execute_steps'] for p in predictions)
   pair[key]=dict(success=e['success'],physical_grasp=e['ever_physical_grasp'],entered_envelope=e['ever_grasp_envelope'],closed_once=e['closed_once'],close_count=len(closes),open_count=len(opens),first_close_s=closes[0]['simulation_s'] if closes else None,failure_category=e['failure_category'],source_result=e['source_result'],predictions=len(predictions),first_close_outside_execution_window=sum(p['first_close_step_06'] is not None and p['first_close_step_06']>=raw['execute_steps'] for p in predictions),first_close_inside_execution_window=sum(p['first_close_step_06'] is not None and p['first_close_step_06']<raw['execute_steps'] for p in predictions))
  pairs.append(pair)
 metrics={}
 for key,d in [('steps15',a),('steps5',b)]:
  metrics[key]={k:d[k] for k in ['total','successes','success_rate','ever_grasped','never_closed','never_entered_grasp_envelope','failure_categories']}
  metrics[key]['repeated_closing_episodes']=sum(p[key]['close_count']>1 for p in pairs)
 result=dict(completed=True,diagnostic_only=True,paired=True,only_control_change='execute_steps 15 vs 5',same_model_sha256=plans[0]['policy_sha256'],scene_verification=proofs,metrics=metrics,pairs=pairs)
 (R/'paired_comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 labels={'grasp_failed':'抓取失败','finger_table_contact':'手指碰桌','transport_or_release_failed':'搬运/释放未完成','placement_failed':'放置未完成','success':'成功','joint_tracking_fault':'关节跟踪故障'}
 lines=['# 杯子 ACT：固定10个训练场景，执行5步与15步对照','','同一050000模型、相同10个训练场景及顺序、10Hz动作、60Hz插值、闭爪保持1秒、二值阈值0.4/0.6。两组均单独复现并核验物体位姿与初始关节状态，唯一控制参数差异是每次执行的动作数。','','| 指标 | 15步 | 5步 |','|---|---:|---:|']
 for label,k in [('最终放置成功','successes'),('稳定物理抓取','ever_grasped'),('全程未闭爪','never_closed'),('未进入抓取包络','never_entered_grasp_envelope'),('反复闭合/张开','repeated_closing_episodes')]:lines.append(f'| {label} | {metrics["steps15"][k]}/10 | {metrics["steps5"][k]}/10 |')
 lines+=['','| 场景 | 训练示范 | 网格 | 15步结果 | 5步结果 | 15步闭爪次数 | 5步闭爪次数 |','|---|---|---|---|---|---|---|']
 for p in pairs:lines.append(f'| {p["order"]} | {p["training_episode"]} | {tuple(p["grid"])} | {labels.get(p["steps15"]["failure_category"],p["steps15"]["failure_category"])} | {labels.get(p["steps5"]["failure_category"],p["steps5"]["failure_category"])} | {p["steps15"]["close_count"]} | {p["steps5"]["close_count"]} |')
 lines+=['','本结果属于10个训练场景的配对诊断，不代表随机场景整体成功率。预测块中闭爪在执行窗口外的计数见JSON；该计数是轨迹现象，不能单独证明因果。','','数据：`paired_comparison.json`、两组各自的`diagnostic_results.json`和完整`rollouts/`。']
 (R/'paired_report.md').write_text('\n'.join(lines)+'\n');return result
if __name__=='__main__':
 d=compare();print(json.dumps(d['metrics'],ensure_ascii=False,indent=2))
