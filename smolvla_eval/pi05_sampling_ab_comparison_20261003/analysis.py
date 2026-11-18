"""Summarize matched live trials; closure times are in simulation seconds."""
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent
ARMS = ['baseline', 'uniform', 'weighted']
NAMES = {'baseline': '原模型', 'uniform': 'A：均匀采样', 'weighted': 'B：阶段加权'}


def episode_metrics(result, trace):
    closed = False
    first_close = None
    preclose_contact = False
    table = False
    clipped = 0
    for row in trace:
        status = row.get('contact_status', {})
        table |= any(bool(status.get(k)) for k in ['left_finger_touching_table', 'right_finger_touching_table'])
        # The status was captured BEFORE this command was sent.
        if not closed:
            preclose_contact |= 'beverage_bottle' in [status.get('left_finger_contact_object'), status.get('right_finger_contact_object')]
        if row['command'][-1] >= -.75 and not closed:
            first_close = row.get('simulation_elapsed_s')
            closed = True
        clipped += max(abs(a-b) for a, b in zip(row['command'], row['policy_action'])) > 1e-6
    final = result.get('final_status', {})
    table |= any(bool(final.get(k)) for k in ['left_finger_touching_table', 'right_finger_touching_table'])
    if not closed:
        preclose_contact |= 'beverage_bottle' in [final.get('left_finger_contact_object'), final.get('right_finger_contact_object')]
    evidence = result.get('physical_predicates', {})
    return {
        'seed': result['seed'], 'success': result['success'],
        'closed_once': closed, 'first_close_sim_s': first_close,
        'preclose_target_contact': preclose_contact, 'table_contact': table,
        'preclose_target_contact_measurable': not result.get('freeze_target_until_close', False),
        'physical_grasp': bool(evidence.get('physical_grasp_ok')),
        'lifted': bool(evidence.get('lifted')),
        'failure_category': result['failure_category'],
        'simulation_duration_s': result['simulation_duration_s'],
        'actions': len(trace), 'clipped_actions': clipped,
        'min_policy_gripper_rad': min((r['policy_action'][-1] for r in trace), default=None),
        'max_policy_gripper_rad': max((r['policy_action'][-1] for r in trace), default=None),
        'target_position_m': result.get('target_position_m'),
        'contact_report_available': final.get('contact_report_available'),
    }


def distance(a, b):
    return math.sqrt(sum((x-y)**2 for x, y in zip(a, b)))


def summarize():
    metrics, results, outputs = {}, {}, {}
    seeds = json.loads((ROOT / 'paired_seeds.json').read_text())
    for arm in ARMS:
        marker = json.loads((ROOT / f'{arm}_completed.json').read_text())
        output = Path(marker['output'])
        outputs[arm] = str(output)
        results[arm] = {}
        metrics[arm] = []
        for path in sorted((output / 'episodes').glob('*/result.json')):
            result = json.loads(path.read_text())
            trace = [json.loads(line) for line in (path.parent / 'trace.jsonl').read_text().splitlines()]
            metrics[arm].append(episode_metrics(result, trace))
            results[arm][result['seed']] = result
        assert len(metrics[arm]) == 10 and set(results[arm]) == set(seeds)
    pairing = []
    for seed in seeds:
        base = results['baseline'][seed]
        for arm in ['uniform', 'weighted']:
            other = results[arm][seed]
            a, b = base['initial_scene_randomization'], other['initial_scene_randomization']
            position_error = max(distance(pos, b['object_positions_m'][name]) for name, pos in a['object_positions_m'].items())
            orientation_error = 0
            for name, q in a['object_orientations_xyzw'].items():
                r = b['object_orientations_xyzw'][name]
                dot = abs(sum(x*y for x, y in zip(q, r))) / math.sqrt(sum(x*x for x in q)*sum(y*y for y in r))
                orientation_error = max(orientation_error, math.degrees(2*math.acos(min(1, dot))))
            joint_error = max(abs(x-y) for x, y in zip(base['initial_joint_state'], other['initial_joint_state']))
            pairing.append({'seed': seed, 'arm': arm, 'max_all_objects_position_difference_m': position_error,
                            'max_all_objects_orientation_difference_deg': orientation_error,
                            'max_initial_joint_difference_rad': joint_error,
                            'within_tolerance': position_error <= .002 and orientation_error <= 1 and joint_error <= .002})
    aggregate = {}
    for arm, rows in metrics.items():
        closed_times = [r['first_close_sim_s'] for r in rows if r['first_close_sim_s'] is not None]
        rates = [r['actions']/r['simulation_duration_s'] for r in rows if r['simulation_duration_s'] > 0]
        aggregate[arm] = {'episodes': len(rows), **{key: sum(r[key] for r in rows) for key in ['success', 'closed_once', 'preclose_target_contact', 'table_contact', 'physical_grasp', 'lifted']},
                          'median_first_close_sim_s': median(closed_times) if closed_times else None,
                          'policy_gripper_range_rad': [min((r['min_policy_gripper_rad'] for r in rows if r['min_policy_gripper_rad'] is not None), default=None), max((r['max_policy_gripper_rad'] for r in rows if r['max_policy_gripper_rad'] is not None), default=None)],
                          'median_effective_actions_per_sim_s': median(rates) if rates else None,
                          'failure_categories': dict(Counter(r['failure_category'] for r in rows))}
    summary = {'aggregate': aggregate, 'episode_metrics': metrics, 'pairing': pairing, 'all_pairs_within_tolerance': all(r['within_tolerance'] for r in pairing), 'outputs': outputs,
               'inference_seed_per_episode': 20261003, 'freeze_until_close': True,
               'interpretation_limits': 'Ten paired layouts, one inference seed. Current freeze disables target collisions until unfreeze, so preclose contact and knockdown are unmeasurable. Initial joints have small jitter; some pairs fail the strict .002rad tolerance. Closed_once and table contact are observable; do not infer an unfrozen knockdown rate.'}
    (ROOT / 'comparison.json').write_text(json.dumps(summary, indent=2))
    all_rows = [{**row, 'arm': arm} for arm in ARMS for row in metrics[arm]]
    with (ROOT / 'episode_metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    report = [
        '# π0.5 瓶子阶段采样对照评估', '',
        '原模型（108000）与 A/B 最终检查点（012000，即新增 3000 次参数更新）各完成 10 个相同种子的布局。每条轨迹重置推理随机种子为 20261003；每次执行 5 帧，配置频率 10Hz，最大关节步幅 0.1 rad，单条最多 60 秒仿真时间。推理期间仿真仍继续，实际动作频率可能低于 10Hz，数值保存在 comparison.json。沿用用户 SH 的训练覆盖格子、距离训练瓶子位置不超过 3cm、首闭爪前固定瓶子和触桌停止设置。没有另外添加 XY 矩形限制。', '',
        '| 模型 | 任务成功 | 曾发出闭爪 | 闭爪前触瓶 | 触桌 | 物理抓取成立 | 高度曾越过阈值 | 首闭爪中位仿真秒 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for arm in ARMS:
        s = aggregate[arm]
        t = s['median_first_close_sim_s']
        values = [str(s[key])+'/10' for key in ['success', 'closed_once', 'preclose_target_contact', 'table_contact', 'physical_grasp', 'lifted']]
        if not all(r['preclose_target_contact_measurable'] for r in metrics[arm]):
            values[2] = '冻结期间不可测'
        report.append('| '+NAMES[arm]+' | '+' | '.join(values)+' | '+(f'{t:.2f}' if t is not None else '未闭爪')+' |')
    if all(s[k] == 0 for s in aggregate.values() for k in ['closed_once', 'success', 'physical_grasp', 'lifted']):
        report += ['', '**这次最终检查点对照没有观察到闭爪或抓取改善：三个模型的 30 条轨迹均未发出闭爪指令，也未物理抓住或抬起瓶子。阶段加权没有解决张爪停滞。触桌与超时之间的变化不能作为抓取能力改善的证据。**', '',
                   'A/B 各增加了 3000 次参数更新，初始化、batch、学习率、种子及模型层设置相同；两组仅改变锚点采样概率。此次只能评价这两个最终检查点，不能据此断言所有阶段采样或所有 LoRA 配置都无效。']
    elif all(s['success'] == 0 and s['physical_grasp'] == 0 for s in aggregate.values()):
        report += ['', '**三个模型均未完成有效抓取或任务。B 组出现一次自主闭爪，原模型与 A 组均未闭爪；这是闭爪行为的有限变化，但尚未转化为稳定抓取能力。B 比 A 少两次触桌，同时有四次超时未抓取，不能据此认定问题已经解决。**', '',
                   'A/B 最终检查点均增加了 3000 次参数更新；两组仅改变锚点采样概率。本次不依据离线损失或触桌次数直接替换默认模型。']
    report += ['', '夹爪原始预测范围（rad，闭爪阈值为 -0.75）：']
    for arm in ARMS:
        report.append(f"- {NAMES[arm]}：{aggregate[arm]['policy_gripper_range_rad']}")
    report += ['', '## 同场景检查', '',
               f"20 个模型与原模型的布局配对中，{sum(r['within_tolerance'] for r in pairing)} 个通过 2mm 位置 / 1° 姿态 / 0.002rad 初始关节容差检查。完整数值见 comparison.json。",
               f"全部物体最大位置差 {max(r['max_all_objects_position_difference_m'] for r in pairing)*1000:.4f}mm，最大姿态差 {max(r['max_all_objects_orientation_difference_deg'] for r in pairing):.4f}°；最大初始关节差 {max(r['max_initial_joint_difference_rad'] for r in pairing):.4f}rad。超出关节容差的配对仍列出原始结果，不能视为逐状态完全相同的仿真。",
               '', '## 逐布局结果', '', '| 种子 | 模型 | 结果 | 首闭爪仿真秒 | 闭爪前触瓶 | 初始关节差 rad | 严格配对容差 |', '|---:|---|---|---:|---|---:|---|']
    for seed in seeds:
        for arm in ARMS:
            m = next(r for r in metrics[arm] if r['seed'] == seed)
            t = m['first_close_sim_s']
            close_text = f'{t:.2f}' if t is not None else '未闭爪'
            pair = next((r for r in pairing if r['seed'] == seed and r['arm'] == arm), None)
            joint_text = f"{pair['max_initial_joint_difference_rad']:.4f}" if pair else '基准'
            tolerance_text = str(pair['within_tolerance']) if pair else '基准'
            contact_text = str(m['preclose_target_contact']) if m['preclose_target_contact_measurable'] else '不可测'
            report.append(f"| {seed} | {NAMES[arm]} | {m['failure_category']} | {close_text} | {contact_text} | {joint_text} | {tolerance_text} |")
    report += ['', '## 结果适用范围', '',
               '瓶子在首次闭爪前被固定，而且 Isaac 的 _freeze_object 会关闭目标碰撞。因而这里不能测出冻结期间的触瓶率或自由瓶子的撞倒率；原始触瓶计数为 0 不代表没有几何重叠，也不代表安全。触桌碰撞仍保持启用，可以比较。解冻时恢复碰撞，若夹爪已经和瓶子重叠，接触响应可能使瓶子移位；此次评估没有改变该机制。', '',
               '指令阈值 -0.75 表示开始闭爪，不等于已夹住瓶子。曾抬起只表示物体高度曾越过阈值，不能单独证明抓住。接触按动作发送前的状态计数，采样时刻之间的短暂接触可能漏记。', '',
               '只测试 10 个配对布局、一个推理随机种子，不能证明更大范围的成功率。训练集只有成功示范，这次评估不测试恢复能力。', '',
               '初始化重放的第一轮由于固定种子路径未完整复位而失败，已保存为 invalid_reset_attempt1_* 并排除；随后修复固定种子重放，在每条轨迹前完整复位机械臂、物体、判定器、助手和运动故障状态。', '',
               '评估相关 35 项测试通过。完整项目测试为 337 通过、14 失败，失败涉及未修改的抓取助手、相机校准、路径及采集配置；逐项见 verification.json。此次没有修改这些模块，也没有用规则强制闭爪或修正模型动作。', '',
               '## 视频与原始数据', '']
    for arm in ARMS:
        report.append(f'- {NAMES[arm]}：[{arm}]({outputs[arm]})')
    (ROOT / '对照评估报告.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(aggregate, indent=2))
    return summary


if __name__ == '__main__':
    summarize()
