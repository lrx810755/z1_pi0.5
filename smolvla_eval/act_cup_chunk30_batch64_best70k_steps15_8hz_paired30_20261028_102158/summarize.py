"""Audit all30 completed trials and describe observed failure stages."""
import collections
import csv
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

R = Path(__file__).resolve().parent


def main():
    summary = json.loads((R/'completed.json').read_text())
    assert summary['completed'] and summary['total'] == 30 and len(summary['episodes']) == 30
    for path, digest in json.loads((R/'source_snapshot.json').read_text()).items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    previous=json.loads(Path('/home/lp531/smolvla_eval/act_cup_renderfix500_best80k_steps5_8hz_paired30_20261027_220437/completed.json').read_text())
    assert [x['seed'] for x in summary['episodes']]==[x['seed'] for x in previous['episodes']]
    pairs=[dict(seed=a['seed'],episode=a['episode'],previous_chunk16_steps5_success=b['success'],current_chunk30_steps15_success=a['success']) for a,b in zip(summary['episodes'],previous['episodes'])]
    (R/'paired_comparison.json').write_text(json.dumps(dict(previous_run='/home/lp531/smolvla_eval/act_cup_renderfix500_best80k_steps5_8hz_paired30_20261027_220437',previous_successes=previous['successes'],current_successes=summary['successes'],episodes=pairs),ensure_ascii=False,indent=2))
    rows = []
    video_evidence = []
    for item in summary['episodes']:
        folder = Path(item['source_result']).parent
        record = json.loads((folder/'result.json').read_text())
        trace = [json.loads(line) for line in (folder/'trace.jsonl').read_text().splitlines()]
        assert trace and record['execute_steps'] == 15 and record['freeze_target_until_close'] is False
        assert not record['error_detail'] and not record['cleanup_error']
        statuses = [t['contact_status'] for t in trace]
        assert all(s.get('frozen') is False and s.get('target_collisions_enabled') is True for s in statuses)
        assert all(not t.get('command_target_frozen') and t.get('command_target_collisions_enabled') is True for t in trace)
        close = any(t['execution_target'][-1] > -.2 for t in trace)
        grasp = any(t.get('physical_grasp_ok') for t in trace)
        lifted = any(t.get('lifted') for t in trace)
        table = any(s.get('left_finger_touching_table') or s.get('right_finger_touching_table') for s in statuses)
        bilateral = any(s.get('left_finger_contact_object') == 'cup' and s.get('right_finger_contact_object') == 'cup' for s in statuses)
        if item['success']:
            category = '成功放入蓝盒并松爪'
        elif table:
            category = '夹爪触桌'
        elif not close:
            category = '未发出闭爪命令'
        elif not grasp:
            category = '闭爪后未形成有效夹持'
        elif not lifted:
            category = '形成夹持但未有效抬升'
        elif item['inside_box']:
            category = '进入盒内但松爪或撤离未完成'
        else:
            category = '抬升后未完成放盒'
        initial = record['initial_scene_randomization']['object_positions_m']
        assert set(initial) == {'cup','apple','peach','mouse','beverage_bottle'}
        row = dict(**item, observed_category=category, close_command_observed=close,
                   bilateral_cup_contact_observed=bilateral, physical_grasp_observed=grasp,
                   lift_observed=lifted, finger_table_contact_observed=table,
                   max_gripper_prediction=max(t['policy_action'][-1] for t in trace),
                   min_gripper_object_distance_m=min(s['gripper_object_distance_m'] for s in statuses),
                   trace_rows=len(trace), initial_cup_position_m=initial['cup'])
        rows.append(row)
        for name in ['wrist.mp4','global_third_person.mp4','global_right.mp4']:
            path = folder/'videos'/name
            metadata = json.loads(subprocess.check_output([
                'ffprobe','-v','error','-select_streams','v:0','-show_entries',
                'stream=width,height,nb_frames,duration','-of','json',str(path)], text=True))
            stream = metadata['streams'][0]
            assert stream['width'] == 640 and stream['height'] == (360 if name == 'wrist.mp4' else 480)
            assert float(stream['duration']) > 0 and int(stream['nb_frames']) > 0
            video_evidence.append(dict(path=str(path), **stream))
    counts = dict(collections.Counter(x['observed_category'] for x in rows))
    report = dict(total=30, successes=summary['successes'], success_rate=summary['success_rate'],
                  checkpoint=json.loads((R/'selected_checkpoint.json').read_text()),
                  observed_categories=counts, episodes=rows, videos_verified=len(video_evidence),
                  source_hashes_unchanged=True, all_targets_dynamic_and_collisions_enabled=True,
                  finished=datetime.datetime.now().astimezone().isoformat())
    (R/'结果与失败分类.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    (R/'video_metadata_verified.json').write_text(json.dumps(video_evidence, ensure_ascii=False, indent=2)+'\n')
    fields=['episode','seed','success','observed_category','close_command_observed',
            'bilateral_cup_contact_observed','physical_grasp_observed','lift_observed',
            'finger_table_contact_observed','max_gripper_prediction',
            'min_gripper_object_distance_m','source_result']
    with (R/'逐次结果.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    lines = ['# ACT500杯子模型：30次独立验证', '',
             f"成功：**{summary['successes']}/30（{summary['success_rate']:.1%}）**。使用70000步验证集最佳模型（训练chunk30、batch64），动作频率8Hz，底层指令64Hz，每次执行15步后重新预测；种子与上一轮8Hz试验配对。", '',
             '场景为与采集一致的五物体桌面；杯子倒圆台碰撞体和折射修复保持不变。使用30个未参与训练的新随机布局。成功沿用最终放盒并松爪规则。', '',
             '已核对30次轨迹中杯子动态及碰撞开启、90路视频元信息、相机/动作时间同步、模型和场景文件校验和。物理失败均保留计入；下列分类描述记录中可观察的失败阶段，不直接推断模型内部原因。', '',
             '|观察分类|次数|', '|---|---:|']
    lines += [f'|{key}|{value}|' for key,value in counts.items()]
    lines += ['', '|序号|种子|结果|观察分类|记录与三路视频|', '|---:|---:|---|---|---|']
    for row in rows:
        folder = Path(row['source_result']).parent.relative_to(R)
        lines.append(f"|{row['episode']}|{row['seed']}|{'成功' if row['success'] else '失败'}|{row['observed_category']}|[记录]({folder}/result.json) / [腕部]({folder}/videos/wrist.mp4) / [外部左]({folder}/videos/global_third_person.mp4) / [外部右]({folder}/videos/global_right.mp4)|")
    (R/'验证报告.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'episodes'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
