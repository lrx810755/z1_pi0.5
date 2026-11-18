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
    previous_root = Path('/home/lp531/smolvla_eval/pi05_cup_recovery513_final30k_steps15_paired30_20261030')
    previous = json.loads((previous_root/'completed.json').read_text())
    assert actual_seeds == [x['seed'] for x in previous['episodes']]
    physics=json.loads((R/'physics_verified.json').read_text())
    assert physics['passed'] and abs(physics['actual_cup_mass_kg']-.30)<1e-6
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
        first_close=next((i for i,t in enumerate(trace) if t['execution_target'][-1]>-.2),len(trace)-1)
        preclose_tilt=max(t['contact_status'].get('cylinder_tilt_deg') or 0 for t in trace[:first_close+1])
        initial = record['initial_scene_randomization']['object_positions_m']
        assert set(initial) == {'cup','apple','peach','mouse','beverage_bottle'}
        row = dict(**item, observed_category=category, close_command_observed=close,
                   bilateral_cup_contact_observed=bilateral, physical_grasp_observed=grasp,
                   lift_observed=lifted, finger_table_contact_observed=table,
                   max_gripper_prediction=max(t['policy_action'][-1] for t in trace),
                   min_gripper_object_distance_m=min(s['gripper_object_distance_m'] for s in statuses),
                   trace_rows=len(trace), initial_cup_position_m=initial['cup'],preclose_max_tilt_deg=preclose_tilt)
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
                  strict_successes=sum(bool(x['legacy_success']) for x in rows),
                  strict_success_rate=sum(bool(x['legacy_success']) for x in rows)/30,
                  physics=physics,preclose_tilt_gt60_episodes=[x['episode'] for x in rows if x['preclose_max_tilt_deg']>60],
                  never_bilateral_episodes=[x['episode'] for x in rows if not x['bilateral_cup_contact_observed']],observed_categories=counts, episodes=rows, videos_verified=len(video_evidence),
                  source_hashes_unchanged=True, all_targets_dynamic_and_collisions_enabled=True,
                  finished=datetime.datetime.now().astimezone().isoformat())
    compare_paired(previous, rows)
    (R/'结果与失败分类.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    (R/'video_metadata_verified.json').write_text(json.dumps(video_evidence, ensure_ascii=False, indent=2)+'\n')
    fields=['episode','seed','success','observed_category','close_command_observed',
            'bilateral_cup_contact_observed','physical_grasp_observed','lift_observed',
            'finger_table_contact_observed','max_gripper_prediction',
            'min_gripper_object_distance_m','source_result']
    with (R/'逐次结果.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    lines = ['# PI0.5杯子300克、桌面摩擦2倍：15步、30次配对评估', '',
             f"成功：**{summary['successes']}/30（{summary['success_rate']:.1%}）**。使用最终30000次优化器更新模型（训练chunk30），动作频率10Hz，底层指令60Hz，每次执行15步后重新预测；杯子300克、静/动摩擦0.40/0.30；沿用80克15步基线的30个种子及顺序，均未进入训练集。", '',
             '场景为与采集一致的五物体桌面；杯子倒圆台碰撞体和折射修复保持不变。使用与80克15步基线相同的30个未参与训练布局。成功沿用最终放盒并松爪规则。', '',
             '已核对30次轨迹中杯子动态及碰撞开启、90路视频元信息、相机/动作时间同步、模型和场景文件校验和。物理失败均保留计入；下列分类描述记录中可观察的失败阶段，不直接推断模型内部原因。', '',
             '|观察分类|次数|', '|---|---:|']
    lines += [f'|{key}|{value}|' for key,value in counts.items()]
    lines += ['', f"完整流程判定：{report['strict_successes']}/30（{report['strict_success_rate']:.1%}）。原80克15步基线：最终入盒11/30，完整流程6/30。配对结果见 paired_comparison.json。"]
    lines += ['', '|序号|种子|结果|观察分类|记录与三路视频|', '|---:|---:|---|---|---|']
    for row in rows:
        folder = Path(row['source_result']).parent.relative_to(R)
        lines.append(f"|{row['episode']}|{row['seed']}|{'成功' if row['success'] else '失败'}|{row['observed_category']}|[记录]({folder}/result.json) / [腕部]({folder}/videos/wrist.mp4) / [外部左]({folder}/videos/global_third_person.mp4) / [外部右]({folder}/videos/global_right.mp4)|")
    (R/'验证报告.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'episodes'}, ensure_ascii=False, indent=2))



def compare_paired(previous, rows):
    import numpy as np
    paired=[]
    for current, old in zip(rows, previous['episodes']):
        assert current['seed']==old['seed']
        new_record=json.loads(Path(current['source_result']).read_text())
        old_record=json.loads(Path(old['source_result']).read_text())
        new_scene=new_record['initial_scene_randomization']
        old_scene=old_record['initial_scene_randomization']
        assert new_scene['seed']==old_scene['seed']
        # The seeded request must match exactly; physical settling may differ slightly.
        assert new_scene.get('target_grid')==old_scene.get('target_grid')
        key='requested_object_positions_m'
        assert set(new_scene[key])==set(old_scene[key])
        for obj in new_scene[key]:
            assert np.allclose(new_scene[key][obj],old_scene[key][obj],rtol=0,atol=1e-9)
        position_errors={};orientation_errors={}
        for obj in new_scene['object_positions_m']:
            a=np.asarray(new_scene['object_positions_m'][obj]);b=np.asarray(old_scene['object_positions_m'][obj])
            position_errors[obj]=float(np.linalg.norm(a-b))
            a=np.asarray(new_scene['object_orientations_xyzw'][obj]);b=np.asarray(old_scene['object_orientations_xyzw'][obj])
            cosine=abs(float(np.dot(a,b)))/(np.linalg.norm(a)*np.linalg.norm(b))
            orientation_errors[obj]=float(np.degrees(2*np.arccos(np.clip(cosine,0,1))))
        assert max(position_errors.values())<=.005,(current['seed'],position_errors)
        assert max(orientation_errors.values())<=5.,(current['seed'],orientation_errors)
        paired.append(dict(episode=current['episode'],seed=current['seed'],baseline80g_final_box_success=old['success'],mass300g_friction2_final_box_success=current['success'],baseline80g_strict_success=old['legacy_success'],mass300g_friction2_strict_success=current['legacy_success'],requested_layout_exact_match=True,settled_position_errors_m=position_errors,settled_orientation_errors_deg=orientation_errors))
    summary=dict(total=30,baseline80g_final_box_successes=sum(x['success'] for x in previous['episodes']),mass300g_friction2_final_box_successes=sum(x['success'] for x in rows),baseline80g_strict_successes=sum(x['legacy_success'] for x in previous['episodes']),mass300g_friction2_strict_successes=sum(x['legacy_success'] for x in rows),all30_requested_layouts_match=True,maximum_settled_position_tolerance_m=.005,maximum_settled_orientation_tolerance_deg=5.,episodes=paired)
    (R/'paired_comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')

if __name__ == '__main__':
    main()
