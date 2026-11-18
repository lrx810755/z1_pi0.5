#!/usr/bin/env python3
"""Screen the 400 cup episodes registered for current PI0.5 training, read-only."""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.ndimage import median_filter

import review
import screen_fruit_motion as motion
from screen_fruit_repetition import repeated_motion,RepeatRules

TRAINING=Path('/home/lp531/pi05_lora_workspace/pi05_current_round_20261111/data_sources.json')
OLD_RUN=Path('/home/lp531/pi05_lora_workspace/pi05_cup_recovery513_lora_20261029')
HOLD_TYPES={'长停顿候选','末尾停留偏长','缓慢漂移或低进展'}


def cup_command_labels(rows):
    """A cup can be securely closed at -.65 rad; fruit's -.60 is unsuitable."""
    closed=False;labels=[]
    for row in rows:
        value=float(row['action'][6])
        if value>=-.82:closed=True
        elif value<=-.875:closed=False
        labels.append(int(closed))
    return np.asarray(labels)


def evaluate(path,cohort,chain,fixed,old_records,old_actions,rules):
    root=path.parent;physical=root.resolve()
    pins={name:motion.digest(root/name) for name in ['frames.jsonl','episode_start.json','episode_summary.json']}
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    summary=review.read_json(root/'episode_summary.json')
    info=review.read_json(root.parent/'dataset_info.json')
    assert info['joint_names'][:7]==[f'joint{i}' for i in range(1,7)]+['jointGripper']
    for key in ['task_success','physical_grasp_ok','bilateral_contact_ever',
                'object_lifted_clear_of_table_ever','release_sequence_complete','placement_stable']:
        if summary.get(key) is not True:raise ValueError('Missing physical success: '+key)
    assert summary['dropped_frames']==0 and summary['frame_count']==len(rows) and summary['fps']==10
    times=np.asarray([r['simulation_time'] for r in rows],float)
    q=np.asarray([r['observation.state'] for r in rows],float)
    action=np.asarray([r['action'] for r in rows],float)
    assert q.shape==action.shape==(len(rows),7) and np.isfinite(q).all() and np.isfinite(action).all()
    assert np.isfinite(times).all() and np.allclose(np.diff(times),.1,atol=1e-5,rtol=0)
    assert all(r['target_object']=='cup' and r['action_stamp']<=r['simulation_time']+1e-5 for r in rows)
    inferred=cup_command_labels(rows)
    labels=inferred;label_source='cup_command_hysteresis_close_-0.82_open_-0.875'
    inferred_difference=None
    if cohort=='previous_selected200':
        record=old_records[str(physical)]
        for name,expected in record['source_hashes'].items():assert motion.digest(root/name)==expected
        a,b=record['dataset_from_index'],record['dataset_to_index']
        binary=old_actions[a:b]
        assert len(binary)==len(rows) and np.isin(binary[:,-1],[0,1]).all()
        assert np.allclose(binary[:,:6],action[:,:6],atol=1e-6,rtol=0)
        labels=binary[:,-1].astype(int);label_source='original_513_training_binary_labels'
        inferred_difference=int(np.sum(labels!=inferred))
    phases=np.asarray([r.get('phase','UNKNOWN') for r in rows],dtype=object)
    expert=np.ones(len(rows),bool)
    position,rotation=review.forward(q[:,:6],chain)
    gripper=position+np.einsum('nij,j->ni',rotation,fixed)
    modes=motion.height_modes(phases,labels)
    if np.sum(modes=='带载抬升搬运')<10:raise ValueError('Unable to identify >=1s loaded cup motion')
    base=review.analyze(path,chain,review.Settings())
    result=dict(path=str(root),physical_path=str(physical),dataset=str(root.parent),episode=root.name,
        cohort=cohort,object='cup',fragment=False,kind='basic',
        grid=summary['scene_randomization']['target_grid'],frames=len(rows),fps=10.,duration_s=len(rows)/10.,
        source_hashes=pins,gripper_label_source=label_source,
        inferred_vs_historical_label_different_frames=inferred_difference,
        loaded_frames=int(np.sum(modes=='带载抬升搬运')),findings=[],
        stationary_s=base['stationary_s'],read_error='')
    for f in base['findings']:
        if f['type'] not in HOLD_TYPES or f['recorded_s']<5:continue
        finding=dict(f,tier='重点' if f['recorded_s']>=8 else '一般',phase='近静止筛查',needs_correction_context=False)
        finding['detail']+='；本次重点线8秒，一般线5秒'
        result['findings'].append(finding)
    result['motion_segments']=motion.reverse_segments(gripper,phases,labels,expert,times,10.,False,rules)
    for segment in result['motion_segments']:
        if not segment['tier']:continue
        a=segment['reverse_start_frame'];b=segment['reverse_end_frame_exclusive']
        result['findings'].append(dict(type='反向移动候选',tier=segment['tier'],
            start_frame=a,end_frame_exclusive=b,start_s=a/10.,end_s=b/10.,
            phase=segment['phase'],needs_correction_context=False,
            detail=f"{segment['metric']}投影累计反向{segment['backward_m']*1000:.1f}mm，"
                f"最大一次回退{segment['max_reverse_excursion_m']*1000:.1f}mm，占本段路径{segment['backward_fraction']:.1%}"))
    new,repeat_metrics=repeated_motion(gripper,q[:,:6],phases,modes,expert,10.,RepeatRules())
    for finding in new:finding['needs_correction_context']=False
    result['findings'].extend(new);result['repetition_findings']=new;result['repetition_metrics']=repeat_metrics
    z=median_filter(gripper[:,2],size=3,mode='nearest')
    heights=[]
    for mode in sorted(set(modes)):
        inds=np.flatnonzero(modes==mode);peak=int(inds[np.argmax(z[inds])])
        heights.append(dict(mode=mode,peak_z_m=float(z[peak]),raw_peak_z_m=float(gripper[inds,2].max()),
            peak_frame=peak,initial_z_m=float(z[inds[0]]),rise_m=float(z[peak]-z[inds[0]])))
    result['height_metrics']=heights
    view=dict(times=(times-times[0]).round(3).tolist(),positions=gripper.round(5).tolist(),
        phases=phases.tolist(),modes=modes.tolist(),
        images=[[(root/r['observation.images.'+camera]).resolve().as_uri()
                 for camera in ['wrist','global','global_right']] for r in rows])
    data=dict(z=z,modes=modes,expert=expert)
    return result,data,view


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    out=args.out
    if out.exists() and any(out.iterdir()):raise FileExistsError('Refuse overwrite')
    out.mkdir(parents=True,exist_ok=True)
    train_hash=motion.digest(TRAINING);train=review.read_json(TRAINING)
    roots=[(Path(x['root']),'current_round200') for x in train['base_sources'] if Path(x['root']).name=='cup_cylinder_horizontal200_20261108']
    roots += [(Path(x['root']),'previous_selected200') for x in train['additions'] if x.get('object')=='cup']
    assert len(roots)==2
    old_records={str(Path(r['source_path']).resolve()):r for r in review.read_json(OLD_RUN/'split_manifest.json')['source_records'] if r['source']=='basic'}
    with np.load(OLD_RUN/'lowdim.npz') as data:old_actions=data['actions']
    chain=review.chain_from_urdf(review.DEFAULT_URDF)
    robot=ET.parse(review.DEFAULT_URDF).getroot()
    fixed=np.fromstring(robot.find("joint[@name='gripperStatorLink']/origin").get('xyz'),sep=' ')
    files=[]
    for root,cohort in roots:
        found,aliases=review.discover([root]);assert len(found)==200 and not aliases
        files.extend((path,cohort) for path in found)
    assert len({str(path.resolve()) for path,_ in files})==400
    items=[];arrays={};display={};errors=[];rules=motion.Rules()
    for i,(path,cohort) in enumerate(files,1):
        try:
            item,data,view=evaluate(path,cohort,chain,fixed,old_records,old_actions,rules)
            items.append(item);arrays[item['path']]=data;display[item['path']]=view
        except Exception as exc:errors.append(dict(path=str(path),error=str(exc)))
        if i%50==0:print('Analyzed',i,'/400',flush=True)
    # Each camera/geometry/approach version has its own matched height baseline.
    for cohort in ['current_round200','previous_selected200']:
        motion.height_screen([e for e in items if e['cohort']==cohort],arrays,rules)
    for item in items:
        for name,expected in item['source_hashes'].items():assert motion.digest(Path(item['path'])/name)==expected
    assert motion.digest(TRAINING)==train_hash
    assert len({e['source_hashes']['frames.jsonl'] for e in items})==len(items)
    candidates=[e for e in items if e['findings']]
    focus=[e for e in candidates if any(f['tier']=='重点' for f in e['findings'])]
    repeated=[dict(e,findings=e['repetition_findings']) for e in items if e['repetition_findings']]
    csv=motion.csv_rows(items)
    review.write_csv(out/'杯子全部候选清单.csv',csv)
    review.write_csv(out/'杯子重点复核清单.csv',[r for r in csv if r['复核级别']=='重点'])
    review.write_csv(out/'杯子反复移动清单.csv',motion.csv_rows(repeated))
    review.write_csv(out/'读取问题.csv',errors)
    review.write_csv(out/'全部400条指标.csv',[dict(来源=e['cohort'],Episode=e['episode'],网格=e['grid'],
        时长秒=e['duration_s'],近静止合计秒=e['stationary_s'],带载帧数=e['loaded_frames'],
        候选项数=len(e['findings']),候选类型='；'.join(sorted({f['type'] for f in e['findings']})),
        原始路径=e['path'],物理源路径=e['physical_path']) for e in items])
    kinds=sorted({f['type'] for e in candidates for f in e['findings']})
    report=dict(passed=not errors,analyzed=len(items),read_errors=errors,scope='当前训练来源中的杯子：本轮200＋旧集优选200',
        training_sources_manifest=str(TRAINING),training_manifest_sha256=train_hash,
        unique_candidate_episodes=len(candidates),focus_episodes=len(focus),review_only_episodes=len(candidates)-len(focus),
        by_source={cohort:dict(root=str(root),analyzed=sum(e['cohort']==cohort for e in items),
            candidates=sum(e['cohort']==cohort for e in candidates),focus=sum(e['cohort']==cohort for e in focus)) for root,cohort in roots},
        by_type={kind:dict(episodes=sum(any(f['type']==kind for f in e['findings']) for e in candidates),
            focus_episodes=sum(any(f['type']==kind and f['tier']=='重点' for f in e['findings']) for e in candidates)) for kind in kinds},
        loaded_repetition_candidate_episodes=sum(any(f.get('motion_mode')=='带载抬升搬运' for f in e['repetition_findings']) for e in candidates),
        height_baselines_separate_by_source=True,wait_focus_s=8,wait_review_s=5,
        cup_command_hysteresis=dict(close_rad=-.82,open_rad=-.875),
        historical200_use_original_binary_labels=True,
        motion_rules=asdict(rules),repeat_rules=asdict(RepeatRules()),
        source_code_sha256=motion.digest(__file__),urdf_sha256=motion.digest(review.DEFAULT_URDF),
        helper_source_hashes={str(Path(x.__file__)):motion.digest(x.__file__) for x in [review,motion]},
        source_files_changed=False,training_manifest_changed=False,results=items)
    review.atomic_json(out/'report.json',report)
    motion.write_html(out/'合并交互复核.html',items,display)
    motion.write_html(out/'反复移动交互复核.html',repeated,display)
    for name in ['合并交互复核.html','反复移动交互复核.html']:
        path=out/name;text=path.read_text().replace('苹果与桃子轨迹复核','杯子轨迹复核').replace('仅本轮四个源集','当前训练清单：新录200条＋旧集优选200条')
        path.write_text(text)
    (out/'说明.txt').write_text('只检查当前训练清单的400条杯子：本轮200条＋从原513条中优选的200条。\n'
        '不混入历史未入选的另外200条、历史纠偏集或20Hz副本；按物理路径和frames哈希去重。\n'
        '旧集使用原513条训练时的二值夹爪标签；本轮用杯子动作角滞回阈值：>=-0.82闭爪，<=-0.875开爪。\n'
        '等待重点线8秒、一般线5秒；反向、高度、往返、腕部反复、启停与绕行规则沿用苹果/桃子筛查，详见report。\n'
        '高度基准按新旧两批分别计算，物体几何和接近方式不同，不混用；高度不是杯底或物体中心离桌高度。\n'
        '所有候选均需人工判断避障、对中、搬运必要性，没有自动剔除或修改训练来源。\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['results','helper_source_hashes','motion_rules','repeat_rules']},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
