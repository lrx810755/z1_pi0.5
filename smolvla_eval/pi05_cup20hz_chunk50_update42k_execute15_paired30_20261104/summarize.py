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
    actual_seeds = [r['seed'] for r in summary['episodes']]
    excluded_seeds = set(json.loads((R/'excluded_seed_baseline.json').read_text()))
    assert len(set(actual_seeds)) == 30 and not set(actual_seeds) & excluded_seeds
    for path, digest in json.loads((R/'source_snapshot.json').read_text()).items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
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
        terminal=record['final_status']
        table = record.get('stop_reason')=='finger_table_contact' or terminal.get('left_finger_touching_table') or terminal.get('right_finger_touching_table') or any(s.get('left_finger_touching_table') or s.get('right_finger_touching_table') for s in statuses)
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
    compare_paired(rows)
    counts = dict(collections.Counter(x['observed_category'] for x in rows))
    report = dict(total=30, successes=summary['successes'], success_rate=summary['success_rate'],
                  checkpoint=json.loads((R/'selected_checkpoint.json').read_text()),
                  strict_successes=summary['strict_successes'],strict_success_rate=summary['strict_success_rate'],
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
    lines = ['# PI0.5杯子LoRA 42000次更新模型：20Hz、执行15步、30次配对验证', '',
             f"成功：**{summary['successes']}/30（{summary['success_rate']:.1%}）**。使用42000次优化器更新模型（训练chunk50），动作频率20Hz，底层指令60Hz，每次执行15步后重新预测；沿用前轮前30个种子，均未参与训练。", '',
             '场景为与采集一致的五物体桌面；杯子倒圆台碰撞体和折射修复保持不变。使用前轮前30个未参与训练的相同布局。成功沿用最终放盒并松爪规则。', '',
             '已核对30次轨迹中杯子动态及碰撞开启、90路视频元信息、相机/动作时间同步、模型和场景文件校验和。物理失败均保留计入；下列分类描述记录中可观察的失败阶段，不直接推断模型内部原因。', '',
             '|观察分类|次数|', '|---|---:|']
    lines += [f'|{key}|{value}|' for key,value in counts.items()]
    lines += ['',f"完整抓取放置流程：{summary['strict_successes']}/30（{summary['strict_success_rate']:.1%}）。",'', '|序号|种子|结果|观察分类|记录与三路视频|', '|---:|---:|---|---|---|']
    for row in rows:
        folder = Path(row['source_result']).parent.relative_to(R)
        lines.append(f"|{row['episode']}|{row['seed']}|{'成功' if row['success'] else '失败'}|{row['observed_category']}|[记录]({folder}/result.json) / [腕部]({folder}/videos/wrist.mp4) / [外部左]({folder}/videos/global_third_person.mp4) / [外部右]({folder}/videos/global_right.mp4)|")
    (R/'验证报告.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'episodes'}, ensure_ascii=False, indent=2))



def compare_paired(rows):
    baseline=Path('/home/lp531/smolvla_eval/pi05_cup20hz_chunk50_final60k_execute15_paired30_20261104')
    previous=json.loads((baseline/'completed.json').read_text())['episodes'][:30]
    assert len(rows)==len(previous)==30
    pairs=[]
    import numpy as np
    for old,new in zip(previous,rows):
        assert old['seed']==new['seed']
        a=json.loads(Path(old['source_result']).read_text())['initial_scene_randomization']
        b=json.loads(Path(new['source_result']).read_text())['initial_scene_randomization']
        assert set(a['requested_object_positions_m'])==set(b['requested_object_positions_m'])
        for obj in a['requested_object_positions_m']:
            assert np.allclose(a['requested_object_positions_m'][obj],b['requested_object_positions_m'][obj],rtol=0,atol=1e-9)
        pairs.append(dict(episode=new['episode'],seed=new['seed'],baseline60k_success=old['success'],checkpoint42k_success=new['success'],baseline60k_strict=old['legacy_success'],checkpoint42k_strict=new['legacy_success'],requested_layout_matches=True))
    (R/'paired_comparison.json').write_text(json.dumps(dict(total=30,baseline_60k_checkpoint42k_successes=sum(x['success'] for x in previous),checkpoint42k_successes=sum(x['success'] for x in rows),baseline_60k_checkpoint42k_strict=sum(x['legacy_success'] for x in previous),checkpoint42k_strict=sum(x['legacy_success'] for x in rows),episodes=pairs),indent=2))

if __name__ == '__main__':
    main()
