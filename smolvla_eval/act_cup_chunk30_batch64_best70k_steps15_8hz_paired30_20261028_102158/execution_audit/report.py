from check import *
from scipy.spatial.transform import Rotation
import hashlib
A=R/'execution_audit';d=json.loads((A/'results.json').read_text());ref=np.array(d['reference_point_local_m'])
# Independent FK cross-check using scipy homogeneous transforms on deterministic joint samples.
rng=np.random.default_rng(20261028);q=rng.uniform(-.5,.5,(100,6));p,rot=fk(q,ref)
for k,row in enumerate(q):
 T=np.eye(4)
 for i,name in enumerate(['joint1','joint2','joint3','joint4','joint5','joint6','gripperStatorLink']):
  j=J[name];o=j.find('origin');H=np.eye(4);H[:3,3]=np.fromstring(o.attrib['xyz'],sep=' ')
  H[:3,:3]=Rotation.from_euler('xyz',np.fromstring(o.attrib['rpy'],sep=' ')).as_matrix();T=T@H
  if i<6:
   H=np.eye(4);H[:3,:3]=Rotation.from_rotvec(np.fromstring(j.find('axis').attrib['xyz'],sep=' ')*row[i]).as_matrix();T=T@H
 np.testing.assert_allclose((T@np.r_[ref,1])[:3],p[k],atol=1e-12)
 np.testing.assert_allclose(T[:3,:3],rot[k],atol=1e-12)
assert len(d['episodes'])==30 and d['largest_sent_joint_step_rad']<=.100001
m=d['metrics'];lines=['# ACT杯子：目标、指令和实际运动查验','',
'检查对象：训练chunk30、batch64的70k模型；30次评估，执行15步，动作8Hz、指令64Hz。', '',
'结论：已有记录未显示足以解释普遍厘米级偏移的执行跟踪故障。接近阶段目标和实际运动相差数毫米，而多个失败的模型目标与实际位置都偏离杯子。下一步应检查图像到动作的预测及其训练标签；本次尚不能区分视觉定位、动作学习和状态覆盖的具体原因。','',
'## 数值证据','',
f"- 指令到关节状态的离线最佳拟合延迟：中位数{d['best_command_lag_ms']['median']:.1f}ms，范围约42–50ms。它包含状态采样/记录的延迟，不是单独测得的电机延迟。",
f"- 按拟合延迟对齐后，下发指令与实际抓取参考点的位置差：中位数{m['lag_command_mm']['median']:.2f}mm，P95={m['lag_command_mm']['p95']:.2f}mm。",
f"- 每个动作最后一个执行子步，模型目标与当前实际抓取参考点的位置差：中位数{m['endpoint_target_mm']['median']:.2f}mm，P95={m['endpoint_target_mm']['p95']:.2f}mm。当前状态在该子步指令下发前读取，故不能把此差值当作最终稳态误差。",
f"- 图像/状态同步偏差：P95={m['skew_ms']['p95']:.2f}ms，最大{m['skew_ms']['max']:.2f}ms。预测输出时，相对输入图像的年龄：后续块中位数{d['prediction_output_observation_age_ms']['subsequent']['median']:.1f}ms，P95={d['prediction_output_observation_age_ms']['subsequent']['p95']:.1f}ms。",
f"- 0.1rad安全步长限制：逐次动作起点触发比例的中位数{d['step_limit_fraction_per_episode']['median']:.1%}，最大{d['step_limit_fraction_per_episode']['max']:.1%}。未观察到整轮长期被限幅。",
f"- 关节绝对范围裁剪：逐次动作起点触发比例中位数{d['joint_limit_fraction_per_episode']['median']:.1%}，最大{d['joint_limit_fraction_per_episode']['max']:.1%}；该现象仍应在后续标签审查中关注。",
f"- 预抬升等待或动作暂停：{d['episodes_policy_pause']}/30轮。",'',
'## 失败样例','',
'以下只分析杯子首次移动超过3cm之前。选取每轮实际参考点距离杯子中心最近的时刻，比较同一时刻的模型目标与实际位置。','',
'|试验|实际参考点至杯心|模型目标至实际参考点|含义|','|---:|---:|---:|---|']
for e in d['episodes']:
 if e['episode'] in [2,5,6,25]:lines.append(f"|{e['episode']}|{e['nearest_actual_reference_mm']/10:.2f}cm|{e['at_nearest_goal_actual_mm']:.2f}mm|实际运动接近模型给定位置，但该位置偏离杯子|")
lines+=['','## 测量方法与限制','',
'- 使用当前相机版URDF的六轴正向运动学，分别计算模型目标、下发指令和实际关节对应的位置；未对目标位置做拟合校准。另用Scipy齐次变换对100组关节状态独立交叉核对。',
'- 抓取参考点采用当前杯子配置的局部偏移(0.12083045,-0.0028452,0.0194797)m；该配置注明几何推导、待仿真验证。因此它不是新标定的两指精确中点。实际/目标跟踪差的比较可信度高于绝对“杯心距离”；后者仅作粗定位证据，不能用作精密闭爪判定。',
'- 接近段优先取杯子首次位移超过3cm前、参考点距杯心12cm内的动作末子步。若该段不足5个样本，回退到位移前全部动作末子步；每轮near_reference_rows保留覆盖情况。',
'- 记录里的TF和当前关节状态并非完全同时刻。未对齐FK与TF差值的中位数约0.37mm、P95约2.77mm，最大约28.81mm，不能将最大值解释为运动学模型误差。',
'- 本次检查没有证明视觉网络本身是根因，也未排除训练动作/状态对齐或自主运行后的分布偏移。','',
'[完整数值](results.json) · [逐次表格](episodes.csv)']
(A/'查验报告.md').write_text('\n'.join(lines)+'\n')
paths=[URDF,*A.glob('*.py'),R/'experiment.json',R/'selected_checkpoint.json']
(A/'verification.json').write_text(json.dumps(dict(passed=True,independent_fk_samples=100,fk_absolute_tolerance_m=1e-12,episodes=30,command_step_limit_verified=True,hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}),indent=2))
print(A/'查验报告.md')
