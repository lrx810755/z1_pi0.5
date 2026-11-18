import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;data=json.loads((R/'comparison.json').read_text());seeds=[437216237,1126247665,1328542479];pairs=[]
for seed in seeds:
 a=next(x for x in data if x['seed']==seed and x['steps']==8);b=next(x for x in data if x['seed']==seed and x['steps']==16)
 ra=json.loads(Path(a['result']).read_text());rb=json.loads(Path(b['result']).read_text())
 apos=ra['initial_scene_randomization']['object_positions_m'];bpos=rb['initial_scene_randomization']['object_positions_m'];delta=max(abs(apos[k][i]-bpos[k][i]) for k in apos for i in range(3))
 pairs.append(dict(seed=seed,steps8_close_s=a['first_actual_close_s'],steps16_close_s=b['first_actual_close_s'],steps8_success=a['success'],steps16_success=b['success'],steps8_failure=a['failure'],steps16_failure=b['failure'],requested_positions_equal=ra['initial_scene_randomization']['requested_object_positions_m']==rb['initial_scene_randomization']['requested_object_positions_m'],max_settled_object_position_difference_m=delta,max_initial_joint_difference_rad=float(np.max(np.abs(np.array(ra['initial_joint_state'])-np.array(rb['initial_joint_state']))))))
(R/'paired_summary.json').write_text(json.dumps(pairs,indent=2))
fig,axes=plt.subplots(3,2,figsize=(12,9),sharey=True)
allqueries=[]
for i,seed in enumerate(seeds):
 for j,steps in enumerate([8,16]):
  x=next(x for x in data if x['seed']==seed and x['steps']==steps);queries=json.loads(Path(x['query_file']).read_text());times=[q['sim_time_relative_first_query_s'] for q in queries];indices=[q['first_close_action_1based'] if q['first_close_action_1based'] is not None else 18 for q in queries];ax=axes[i,j];ax.plot(times,indices,'o-',ms=3,lw=.8);ax.axhline(steps+.5,ls='--',color='orange');ax.set_title(f'Seed {seed}, execute {steps}');ax.set_ylim(0,19);ax.set_yticks([1,4,8,12,16,18],['1','4','8','12','16','No close']);ax.set_xlabel('Simulation time relative to first query (s)');ax.set_ylabel('First closing action in 16-step prediction');ax.grid(alpha=.2)
  for q in queries:allqueries.append(dict(steps=steps,seed=seed,**q))
fig.tight_layout();fig.savefig(R/'first_close_positions.png',dpi=160)
with (R/'all_prediction_queries.csv').open('w') as f:
 keys=['steps','seed','query','sim_time_relative_first_query_s','first_close_action_1based','within_execution_window','first_gripper_prediction','gripper_prediction'];w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(allqueries)
lines=['# ACT 8步与16步执行配对实验','', 'ACT 050000；每个种子分别执行8步和16步。三相机、10Hz动作、60Hz控制、0.4/0.6双阈值、60秒仿真超时保持一致。三个种子对应先前失败/成功场景；8步组三轮结束后，16步组三轮依次执行。', '', '|种子|8步首次闭爪秒|16步首次闭爪秒|8步结果|16步结果|','|---|---:|---:|---|---|']
for p in pairs:lines.append(f"|{p['seed']}|{p['steps8_close_s']:.2f}|{p['steps16_close_s']:.2f}|{p['steps8_failure']}|{p['steps16_failure']}|")
lines += ['', '## 结论','', '三个种子均有首次闭爪时间对照。16步在三个种子中均提前闭爪，支持执行窗口截断及重规划会延迟部分闭爪。8步种子1126247665中15次把闭爪安排到执行窗口之外，实际首次闭爪57.17秒；16步12.82秒闭爪。不能将这种延迟泛化成所有失败的主因：8步组0/3成功，16步组1/3成功；第三组16步完成任务，但另外两组仍抓取失败。扩大窗口改善了这组样本的闭爪时机，但没有解决全部抓取问题。', '', '完整记录每次16步预测的夹爪原值和首次>=0.6的动作位置，见all_prediction_queries.csv和每个episode的prediction_chunks.jsonl。没有闭爪预测时用null，而不是假定第17步闭爪。首次实际闭爪时间取执行目标首次变为0rad的仿真时间，不等同于夹爪实际闭合或物理抓取成功。', '', '## 场景配对与限制','', '三组requested_object_positions完全一致，实际稳定位置及初始关节差异见paired_summary.json。物理仿真和相机时序存在细小变化，同种子不能保证每个观测完全相同。改变执行步数也会改变机械臂运动与后续观测，因此是闭环系统对照，不是固定状态下单独切掉未来动作的因果试验。每种子每设置仅一次，不能作为稳健成功率统计。', '', '预测日志的sim_stamp为推理结束附近的仿真时间；名义未来闭爪时间按0.1秒每步计算，只用于查看重新预测的时间推移。完整日志见steps8.log及steps16.log。']
(R/'实验结果.md').write_text('\n'.join(lines)+'\n');print(json.dumps(pairs,indent=2))
