from pathlib import Path
import json
import numpy as np
OUT=Path(__file__).resolve().parent
d=json.loads((OUT/'summary.json').read_text());assert d['completed'] and len(d['episodes'])==8
rs=d['episodes'];lines=[]
for steps in (5,10):
 group=[r for r in rs if r['steps']==steps]
 lines.append(f"| {steps}帧 | {sum(r['success'] for r in group)}/4 | {sum(r['failure']=='finger_table_contact' for r in group)}/4 | {sum(r['failure']=='grasp_failed' for r in group)}/4 |")
rows=[]
for r in sorted(rs,key=lambda r:(r['seed'],r['steps'],r['group'])):
 rows.append(f"| {r['seed']} | {r['group']} | {'碰桌' if r['failure']=='finger_table_contact' else r['failure']} | {r['sim_s']:.2f}s | {'无' if r['first_close_sim_s'] is None else format(r['first_close_sim_s'],'.2f')+'s'} | {r['raw_risk_predictions']} / {r['clipped_risk_predictions']} |")
metrics=f"总计{sum(r['predictions'] for r in rs)}轮模型推理；每轮完整15帧预测、实际三相机输入NPZ、同步日志、60Hz发布轨迹、视频和终止状态均已保存。"
text='''# 执行5帧与10帧的闭环对照（2026-10-07）

## 实际完成8次验证

当前080000二值夹爪/Attention+FFN LoRA模型、chunk15，10Hz动作/60Hz插值；两个曾早期碰桌的场景seed分别489361075、774196979；每个seed、每种执行窗口重复两次。推理随机种子123在每episode开始重置，顺序为5/10/10/5，每组测两个seed。瓶子自由、接触等待关闭、其他控制参数与当前验证SH一致。

'''+metrics+'''

| 执行窗口 | 成功 | 碰桌终止 | 60秒未完成抓取 |
|---|---:|---:|---:|
'''+ '\n'.join(lines)+'''

| seed | 组别 | 结果 | 仿真时长 | 首次闭爪命令 | 危险预测轮数（原始/裁剪后） |
|---|---|---|---:|---:|---:|
'''+ '\n'.join(rows)+'''

危险预测指选取的前5或10帧有URDF夹爪网格最低点不足4mm桌面间隙，不等同于实际碰桌计数。

## 能据此决定什么

这8次没有一次形成任务成功。缩短到5帧没有解决接近/抓取问题，结果不支持把默认执行窗口改为5帧作为修复。样本量小，也不能据1与2次碰桌差异认定10帧统计上更安全。没有测试15帧，不据此推荐加长。

seed489361075两次5帧均碰桌；10帧两次运行到60秒，虽然预测了闭爪但未抓住瓶子。其首次闭爪时的瓶子局部位置分别约(227,−26,−13)mm、(203,−42,−15)mm；抓取允许区x=110..154、y=±40、z=6..30mm。这里明确存在闭爪命令，只是位置已经不合适。

第一轮seed489361075的5帧试验中，第26轮预测首次出现限幅后不足4mm的几何间隙：选中前5帧原始目标最低点约7.20mm，限幅后约3.19mm；第5关节已超限。随后出现原始预测自身也危险的轮次。这与“先出现不可执行输出、裁剪使间隙减少、反馈状态继续偏离”的机制相符，但单次轨迹没有做不裁剪的物理反事实，不能宣称已证明唯一因果。

seed774196979第一轮10帧与两轮5帧的详细结果见表；重复结果有差异，说明即使固定场景/推理种子也需要考虑物理、初始状态和传感器采样时刻的差别。

## 配对和记录校验

'''+ '\n'.join(f"- seed {c['seed']}：四次目标位置最大差{c['target_position_spread_mm']:.4f}mm，初始六轴关节最大差{c['initial_arm_joint_spread_rad']:.6f}rad。" for c in d['pairing_checks'])+f'''
- 8次同步观测最大sensor_skew={max(r['sync_max_skew'] for r in rs):.6f}s，最大观测年龄{max(r['sync_max_age'] for r in rs)*1000:.2f}ms；这些采集配对日志不支持时间错配是主因。
- 各轮推理时间中位数范围{min(r['infer_median_ms'] for r in rs):.1f}..{max(r['infer_median_ms'] for r in rs):.1f}ms；保存无损输入额外耗时中位数范围{min(r['record_median_ms'] for r in rs):.1f}..{max(r['record_median_ms'] for r in rs):.1f}ms。这是带记录的串行执行对照，额外记录开销会影响反馈间隔，尤其5帧更频繁触发记录，不能当成无记录SH的严格性能测量。
- 每轮predictions.jsonl与input_sync.jsonl及输入NPZ数量一致，记录完整chunk形状15×7。
- 所有结果为实际闭环仿真，几何高度重建仍是URDF近似，不包含Isaac碰撞代理和接触动力学。

## 建议下一步

优先改善模型的接近精度，并处理预测超限带来的执行差异。已有离线对照显示10000次更新的模型仍在改善拟合；可做从080000继续10000次优化更新的对照，用相同离线观测和本轮两个固定seed检查是否改善。暂不需要把160条全部重录，也不应只围绕5/10/15帧继续调参。

执行端需保留机械关节上限，并在发布前检查限幅后及插值路径的桌面间隙；目前碰桌检查是在接触后停止。这项保护可阻止继续撞桌，但被保护停止的episode也应算失败，不能当作模型学会了抓取。

本轮未追加训练，未改变原SH默认参数。验证进程结束，模型worker关闭，仿真保留。
'''
(OUT/'实验报告.md').write_text(text)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(2,2,figsize=(12,8))
for si,seed in enumerate((489361075,774196979)):
 for r in [x for x in rs if x['seed']==seed]:
  rows=[json.loads(x) for x in (Path(r['result_path']).parent/'trace.jsonl').read_text().splitlines()][::6]
  times=[x['simulation_elapsed_s'] for x in rows];q5=[np.rad2deg(x['joint_state'][4]) for x in rows]
  error=[]
  for x in rows:
   p=x['contact_status']['relative_position_gripper_m'];error.append(np.linalg.norm([max(.11-p[0],p[0]-.154,0),max(abs(p[1])-.04,0),max(.006-p[2],p[2]-.03,0)])*1000)
  color='tab:blue' if r['steps']==5 else 'tab:orange';style='--' if 'repeat2' in r['group'] else '-'
  for ax,values in zip(axs[si],(q5,error)):ax.plot(times,values,color=color,linestyle=style,label=r['group'])
 axs[si,0].axhline(77,color='red',alpha=.5);axs[si,0].set_ylabel('Actual joint5 (deg)');axs[si,1].set_ylabel('Distance outside grasp envelope (mm)')
 for ax in axs[si]:ax.set_title(f'Seed {seed}');ax.set_xlabel('Simulation time (s)');ax.grid(alpha=.3);ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(OUT/'paired_trajectories.png',dpi=160)
print('Saved report and paired trajectory plot')
