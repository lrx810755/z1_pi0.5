"""Export all recorded failures from this conversation without changing sources."""
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
import csv
import hashlib
import json
import shutil
import zipfile

OUT = Path(__file__).resolve().parent
ROOT15 = Path('/home/lp531/smolvla_eval/pi05_final_four_objects_50each_20261117_145347')
ROOTNEW = Path('/home/lp531/smolvla_eval/pi05_final_execute25_20_25each_20261117_173702')
GROUPS = [(15, ROOT15), (25, ROOTNEW / 'execute_25'), (20, ROOTNEW / 'execute_20')]
OBJECTS = {'apple': '苹果', 'peach': '桃子', 'cup': '杯子', 'beverage_bottle': '瓶子'}
CATEGORIES = {
    'grasp_failed': ('抓取判据未通过', '默认60秒仿真时限结束，物理抓取与离桌抬升条件未同时通过；不等同于完全没有闭爪或物体没有移动。'),
    'finger_table_contact': ('手指触桌停止', '评估器检测左指或右指接触桌面，执行器触发停止。'),
    'bottle_tipped': ('瓶子倾倒停止', '瓶子倾角大于70度并持续至少1秒，倾倒保护触发；末次result中的姿态可能早于触发时刻。'),
    'joint_tracking_fault': ('关节跟踪误差停止', '前六轴实际关节与上一命令的最大偏差超过0.08rad，持续至少0.5秒，跟踪保护触发。'),
    'transport_or_release_failed': ('搬运或释放未完成', '仿真时限结束前已有物理抓取和抬升证据，但没有完成释放序列。'),
    'placement_failed': ('最终放置未成功', '仿真时限结束前已记录抓取、抬升及释放序列，但未达到稳定入盒的任务成功条件。'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def yes(value):
    return '是' if value is True else '否' if value is False else '未知'


def observation_reason(data):
    category = data.get('failure_category', 'unknown')
    p = data.get('physical_predicates', {})
    s = data.get('final_status', {})
    reason = CATEGORIES.get(category, (category, '见原始错误字段'))[1]
    if category == 'grasp_failed':
        if p.get('gripper_commanded_closed') is False:
            reason += ' 本条未记录到闭爪命令。'
        elif p.get('bilateral_contact') is False:
            reason += ' 本条曾发出闭爪命令，但未记录到目标双侧接触。'
        elif p.get('bilateral_contact') is True:
            reason += ' 本条曾记录目标双侧接触，但物理抓取证明未通过。'
        reason += ' 曾有离桌抬升证据：' + yes(p.get('lifted')) + '。'
    if category == 'finger_table_contact':
        fingers = [label for key, label in [('left_finger_touching_table', '左指'), ('right_finger_touching_table', '右指')] if s.get(key) is True]
        reason += ' 末次状态触桌：' + ('、'.join(fingers) or '末次状态未保留，按停止原因记录') + '。'
    if s.get('failure_reason'):
        reason += ' 评估器末次原因：' + str(s['failure_reason']) + '。'
    if data.get('error_detail'):
        reason += ' 错误详情：' + data['error_detail']
    return reason


def md_cell(value):
    return str(value).replace('|', '\\|').replace('\n', ' ')


def main():
    rows, failures, interrupted = [], [], []
    pins, video_count, missing_video_cases = {}, 0, []
    raw = OUT / '失败原始结果JSON'
    raw.mkdir(exist_ok=True)
    for steps, source in GROUPS:
        runs = sorted(source.glob('pi05_sim_eval_*'))
        if len(runs) != 1:
            raise ValueError(f'Expected one run for execute {steps}, got {len(runs)}')
        run = runs[0]
        for episode in sorted((run / 'episodes').iterdir()):
            if not episode.is_dir():
                continue
            path = episode / 'result.json'
            if not path.is_file():
                interrupted.append(dict(execute_steps=steps, episode=episode.name, reason='用户手动停止时尚未写出完整result.json，不计成功或模型失败', directory=str(episode), videos=[str(p) for p in sorted((episode / 'videos').glob('*.mp4'))]))
                continue
            data = json.loads(path.read_text())
            if data.get('execute_steps') != steps:
                raise ValueError('Execution-window mismatch: ' + str(path))
            pins[str(path)] = sha(path)
            p, s = data.get('physical_predicates', {}), data.get('final_status', {})
            videos = {f.stem: str(f) for f in sorted((episode / 'videos').glob('*.mp4'))}
            row = dict(组别=f'执行{steps}步', 执行步数=steps, 物体=OBJECTS.get(data['target_object'], data['target_object']),
                       物体标识=data['target_object'], 轮次=data['episode_number'], 随机种子=data.get('seed'),
                       结果='成功' if data.get('success') is True else '失败', 失败代码=data.get('failure_category', ''),
                       失败分类=CATEGORIES.get(data.get('failure_category'), ('成功' if data.get('success') else '未知', ''))[0],
                       记录原因='' if data.get('success') else observation_reason(data),
                       停止原因=data.get('stop_reason', ''), 错误详情=data.get('error_detail', ''), 清理错误=data.get('cleanup_error', ''),
                       评估器末次原因=s.get('failure_reason', ''), 搬运末次原因=s.get('transport_failure_reason', ''),
                       仿真时长秒=data.get('simulation_duration_s'), 实际耗时秒=data.get('duration_s'),
                       曾发出闭爪命令=yes(p.get('gripper_commanded_closed')), 曾有目标双侧接触=yes(p.get('bilateral_contact')),
                       曾通过物理抓取=yes(p.get('physical_grasp_ok')), 曾离桌抬升=yes(p.get('lifted')),
                       曾完成释放=yes(p.get('release_sequence_complete')), 末次放置稳定=yes(s.get('placement_stable')),
                       左指末次触桌=yes(s.get('left_finger_touching_table')), 右指末次触桌=yes(s.get('right_finger_touching_table')),
                       左指末次接触物=s.get('left_finger_contact_object'), 右指末次接触物=s.get('right_finger_contact_object'),
                       末次夹爪角rad=s.get('gripper_position_rad'), 末次夹爪距物体m=s.get('gripper_object_distance_m'),
                       末次倾角度=s.get('cylinder_tilt_deg'), 原始结果=str(path), 原始结果SHA256=pins[str(path)],
                       动作轨迹=str(episode / 'trace.jsonl'), 输入同步=str(episode / 'input_sync.jsonl'),
                       预测记录=str(episode / 'prediction_chunks.jsonl'), 腕部视频=videos.get('wrist', ''),
                       外部视频=videos.get('global_third_person', ''), 右侧视频=videos.get('global_right', ''), 试验目录=str(episode))
            rows.append(row)
            if not data.get('success'):
                failures.append(row)
                video_count += len(videos)
                if len(videos) != 3:
                    missing_video_cases.append(str(path))
                target = raw / f'执行{steps}步' / f'{row["物体"]}_{episode.name}.json'
                target.parent.mkdir(exist_ok=True)
                shutil.copyfile(path, target)
    if len({r['原始结果'] for r in rows}) != len(rows):
        raise ValueError('Duplicate source results')
    for name, items in [('全部完整结果.csv', rows), ('全部失败明细.csv', failures)]:
        with (OUT / name).open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(items)
    with (OUT / '全部失败原始记录.jsonl').open('w') as stream:
        for row in failures:
            stream.write(json.dumps(dict(source=row['原始结果'], sha256=row['原始结果SHA256'], result=json.loads(Path(row['原始结果']).read_text())), ensure_ascii=False) + '\n')
    group_summary, category_summary, object_summary = {}, {}, {}
    for steps, _ in GROUPS:
        selected = [r for r in rows if r['执行步数'] == steps]
        fail = [r for r in selected if r['结果'] == '失败']
        group_summary[str(steps)] = dict(completed=len(selected), successes=len(selected)-len(fail), failures=len(fail), success_rate=(len(selected)-len(fail))/len(selected) if selected else None)
        category_summary[str(steps)] = dict(Counter(r['失败代码'] for r in fail))
        object_summary[str(steps)] = {name: dict(completed=sum(r['物体标识']==obj for r in selected), successes=sum(r['物体标识']==obj and r['结果']=='成功' for r in selected), failures=sum(r['物体标识']==obj for r in fail)) for obj, name in OBJECTS.items()}
    counts = Counter(r['失败代码'] for r in failures)
    grasp = [r for r in failures if r['失败代码'] == 'grasp_failed']
    detail_counts = dict(grasp_failed_total=len(grasp), grasp_failed_close_command_seen=sum(r['曾发出闭爪命令']=='是' for r in grasp),
                        grasp_failed_no_bilateral_record=sum(r['曾有目标双侧接触']=='否' for r in grasp),
                        grasp_failed_bilateral_but_no_proven_grasp=sum(r['曾有目标双侧接触']=='是' and r['曾通过物理抓取']=='否' for r in grasp))
    summary = dict(generated_at=datetime.now().astimezone().isoformat(), scope='本会话最终四物体模型的执行15、25、20步三轮，历史杯子单模型不在范围内',
                   total_completed=len(rows), total_successes=len(rows)-len(failures), total_failures=len(failures),
                   groups=group_summary, categories=dict(counts), categories_by_group=category_summary, objects_by_group=object_summary,
                   unfinished=interrupted, grasp_observations=detail_counts, failure_video_files=video_count,
                   missing_three_view_cases=missing_video_cases, source_result_sha256=pins)
    (OUT / '汇总数据.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    md = ['# 全部失败结果及记录原因', '', f'生成时间：{summary["generated_at"]}。当前评估已按用户要求停止。', '',
          '范围：本会话使用最终100000次更新模型进行的执行15、25、20步三轮评估。原始结果、动作记录和视频均保留；不包含历史单杯模型试验。', '',
          f'完整结果 **{len(rows)}** 条：成功 **{len(rows)-len(failures)}** 条、失败 **{len(failures)}** 条；另有手动停止时未完成的 **{len(interrupted)}** 条，单独列出，不算模型失败。', '',
          '| 执行窗口 | 完整结果 | 成功 | 失败 | 成功率 |', '|---|---:|---:|---:|---:|']
    for steps, _ in GROUPS:
        g=group_summary[str(steps)];md.append(f'| {steps}步 | {g["completed"]} | {g["successes"]} | {g["failures"]} | {g["success_rate"]:.2%} |')
    md += ['', '## 失败原因分类', '', '| 记录原因 | 15步 | 25步 | 20步 | 合计 | 判定依据 |', '|---|---:|---:|---:|---:|---|']
    for cat, count in counts.most_common():
        label, explanation=CATEGORIES.get(cat,(cat,'见原记录'))
        md.append('| '+ ' | '.join([label,*[str(category_summary[str(s)].get(cat,0)) for s,_ in GROUPS],str(count),explanation])+' |')
    md += ['', '## 各物体结果', '', '| 执行窗口 | 物体 | 完整结果 | 成功 | 失败 |', '|---|---|---:|---:|---:|']
    for steps,_ in GROUPS:
        for name,g in object_summary[str(steps)].items():md.append(f'| {steps}步 | {name} | {g["completed"]} | {g["successes"]} | {g["failures"]} |')
    md += ['', '## 抓取失败的已有证据', '',
           f'在 {len(grasp)} 条抓取失败中，{detail_counts["grasp_failed_close_command_seen"]} 条曾发出闭爪命令；{detail_counts["grasp_failed_no_bilateral_record"]} 条未记录到目标双侧接触；{detail_counts["grasp_failed_bilateral_but_no_proven_grasp"]} 条曾有双侧接触，但仍未通过物理抓取证明。', '',
           '这些是程序判据和已保存的物理观测，不是对353条视频逐一人工诊断后的机械根因。“抓取失败”不能直接等同为“没闭爪”或“完全没抬升”；具体偏位、推偏、穿入或打滑需结合对应视频和trace复核。缺失的error_detail保持为空，不虚构更细原因。', '',
           '## 手动中断、没有完整结果的试验', '']
    for item in interrupted:md.append(f'- 执行{item["execute_steps"]}步：[{item["episode"]}]({item["directory"]})。{item["reason"]}。')
    md += ['', '## 全部逐条明细与原始结果', '',
           '- [全部失败明细.csv](全部失败明细.csv)：一行一条，含原因、物理证据、原始结果、trace及三视角视频绝对路径。',
           '- [全部失败逐条说明.md](全部失败逐条说明.md)：按窗口及物体分组，每条都有原始结果与视频链接。',
           '- [全部完整结果.csv](全部完整结果.csv)：包括成功样本，便于核对分母。',
           '- [全部失败原始记录.jsonl](全部失败原始记录.jsonl)：原始result完整内容与来源SHA256。',
           '- `失败原始结果JSON/`：每条失败result.json的独立副本。', '',
           f'完整失败记录共关联 {video_count} 个原始视频文件；三视角文件不足的失败样本数为 {len(missing_video_cases)}。视频不重复复制，使用逐条清单内的本机原路径访问。', '']
    (OUT/'失败结果与原因汇总.md').write_text('\n'.join(md))
    detail=['# 全部失败逐条说明', '', f'共{len(failures)}条完整失败结果。每条保留程序分类、原始停止原因与观测说明，视频链接指向本机原文件。', '']
    for steps,_ in GROUPS:
        for obj,name in OBJECTS.items():
            selected=[r for r in failures if r['执行步数']==steps and r['物体标识']==obj]
            detail += [f'## 执行{steps}步 · {name} · {len(selected)}条失败', '']
            for row in selected:
                detail += [f'### 第{row["轮次"]:02d}次，seed={row["随机种子"]}：{row["失败分类"]}', '', row['记录原因'], '',
                           f'- 原始字段：`failure_category={row["失败代码"]}`；`stop_reason={row["停止原因"]}`；`error_detail={row["错误详情"]}`。',
                           f'- 累计证据：闭爪={row["曾发出闭爪命令"]}；双侧接触={row["曾有目标双侧接触"]}；物理抓取={row["曾通过物理抓取"]}；抬升={row["曾离桌抬升"]}；释放完成={row["曾完成释放"]}。',
                           f'- [原始结果]({row["原始结果"]}) · [动作记录]({row["动作轨迹"]}) · [输入同步]({row["输入同步"]})。',
                           '- 视频：'+' · '.join(f'[{label}]({row[key]})' for key,label in [('腕部视频','腕部'),('外部视频','外部'),('右侧视频','右侧')] if row[key]), '']
    (OUT/'全部失败逐条说明.md').write_text('\n'.join(detail))
    # Snapshot only reports and individual result JSON; avoid duplicating large videos/traces.
    with zipfile.ZipFile(OUT/'全部失败结果与原因.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for file in sorted(OUT.rglob('*')):
            if file.is_file() and file.suffix!='.zip':z.write(file,str(file.relative_to(OUT)))
    assert len(failures)==sum(counts.values())==sum(g['failures'] for g in group_summary.values())
    print(json.dumps({k:v for k,v in summary.items() if k!='source_result_sha256'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
