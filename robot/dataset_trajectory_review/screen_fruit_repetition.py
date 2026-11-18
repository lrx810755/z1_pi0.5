#!/usr/bin/env python3
"""Supplement frozen fruit screening with repeated-motion review candidates."""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict,dataclass
import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

import review
import screen_fruit_motion as fruit


@dataclass
class RepeatRules:
    sample_s:float=.2
    window_s:float=6.
    stride_s:float=1.
    minimum_segment_s:float=2.
    positional_prominence_m:float=.006
    minimum_extremum_spacing_s:float=.3
    review_turns:int=4
    focus_turns:int=6
    minimum_repeated_travel_m:float=.04
    wrist_prominence_rad:float=.035
    wrist_minimum_repeated_travel_rad:float=.20
    detour_ratio:float=2.5
    detour_extra_m:float=.15
    detour_min_path_m:float=.25
    active_speed_m_s:float=.004
    minimum_pause_s:float=.5
    minimum_bout_travel_m:float=.008
    review_move_bouts:int=5
    focus_move_bouts:int=7
    box_xy:tuple=(.32,.14)
    approximate_box_radius_m:float=.18


def extrema_metrics(values,fps,prominence,rules):
    distance=max(1,round(rules.minimum_extremum_spacing_s*fps))
    highs=find_peaks(values,prominence=prominence,distance=distance)[0]
    lows=find_peaks(-values,prominence=prominence,distance=distance)[0]
    # Alternating prominent extrema, rather than raw derivative sign changes,
    # keep controller noise and densely sampled corners from inflating counts.
    candidates=sorted([(int(i),1) for i in highs]+[(int(i),-1) for i in lows])
    kept=[]
    for index,sign in candidates:
        if kept and kept[-1][1]==sign:
            old=kept[-1][0]
            if (sign==1 and values[index]>values[old]) or (sign==-1 and values[index]<values[old]):
                kept[-1]=(index,sign)
        elif not kept or abs(values[index]-values[kept[-1][0]])>=prominence:
            kept.append((index,sign))
    indices=[i for i,_ in kept]
    travel=float(np.abs(np.diff(values[indices])).sum()) if len(indices)>1 else 0.
    return dict(turns=len(indices),repeated_travel=travel,
        extent=float(np.ptp(values)),extrema=indices)


def movement_bouts(points,fps,rules):
    distance=np.linalg.norm(np.diff(points,axis=0),axis=1)
    active=distance*fps>rules.active_speed_m_s
    # A brief slowing at a continuous corner is not a new movement bout.
    for a,b in review.runs(~active):
        if a>0 and b<len(active) and (b-a)/fps<rules.minimum_pause_s:active[a:b]=True
    bouts=[]
    for a,b in review.runs(active):
        travel=float(distance[a:b].sum())
        if travel>=rules.minimum_bout_travel_m and (b-a)/fps>=.2:
            bouts.append(dict(start=int(a),end=int(b+1),travel_m=travel))
    return bouts


def repeated_motion(points,joints,phases,modes,expert,fps,rules):
    points=median_filter(points,size=(3,1),mode='nearest')
    joints=median_filter(joints,size=(3,1),mode='nearest')
    boundaries=np.flatnonzero(np.r_[True,(phases[1:]!=phases[:-1])|
        (modes[1:]!=modes[:-1])|(expert[1:]!=expert[:-1]),True])
    findings=[];metrics=[]
    def add(kind,tier,a,b,phase,detail,extra):
        findings.append(dict(type=kind,tier=tier,start_frame=int(a),end_frame_exclusive=int(b),
            start_s=float(a/fps),end_s=float(b/fps),phase=phase,detail=detail,**extra))
    for a,b in zip(boundaries[:-1],boundaries[1:]):
        if not expert[a] or (b-a)/fps<rules.minimum_segment_s:continue
        phase=str(phases[a]);mode=str(modes[a]);loaded=mode=='带载抬升搬运'
        near=float(np.median(np.linalg.norm(points[a:b,:2]-rules.box_xy,axis=1)))<=rules.approximate_box_radius_m
        context=dict(motion_mode=mode,near_box_approx=bool(near))
        # Diagnose whole segments as well as local six-second windows. Long
        # intervals can contain well-spaced repeated corrections with pauses.
        windows=[(int(a),int(b))]
        width=round(rules.window_s*fps);stride=max(1,round(rules.stride_s*fps))
        if b-a>width:
            windows += [(left,min(int(b),left+width)) for left in range(int(a),int(b)-width+1,stride)]
        interval_findings=[]
        for left,right in windows:
            inds=np.unique(np.r_[np.arange(left,right,max(1,round(rules.sample_s*fps))),right-1])
            p=points[inds]
            sample_fps=fps/max(1,round(rules.sample_s*fps))
            signals=[('X',p[:,0]),('Y',p[:,1]),('Z',p[:,2])]
            # The dominant direction also catches diagonally oriented shuttles.
            centered=p-p.mean(0)
            if np.linalg.norm(centered)>1e-9:
                axis=np.linalg.svd(centered,full_matrices=False)[2][0]
                signals.append(('主方向',centered@axis))
            candidates=[]
            for axis,values in signals:
                m=extrema_metrics(values,sample_fps,rules.positional_prominence_m,rules)
                if m['turns']>=rules.review_turns and m['repeated_travel']>=rules.minimum_repeated_travel_m:
                    candidates.append((axis,m))
            if candidates:
                axis,m=max(candidates,key=lambda c:(c[1]['turns'],c[1]['repeated_travel']))
                start=int(inds[max(0,m['extrema'][0]-1)]);end=int(inds[min(len(inds)-1,m['extrema'][-1]+1)])+1
                tier='重点' if m['turns']>=rules.focus_turns else '一般'
                interval_findings.append(dict(type='多次往返或局部抖动候选',tier=tier,
                    start=start,end=end,axis=axis,**m))
            # Repeated wrist articulation can be visible even when TCP stays
            # almost stationary. Judge actual joint angles, not held commands.
            for j in range(3,6):
                m=extrema_metrics(joints[inds,j],sample_fps,rules.wrist_prominence_rad,rules)
                if m['turns']>=rules.review_turns and m['repeated_travel']>=rules.wrist_minimum_repeated_travel_rad:
                    start=int(inds[max(0,m['extrema'][0]-1)]);end=int(inds[min(len(inds)-1,m['extrema'][-1]+1)])+1
                    interval_findings.append(dict(type='腕部关节反复调整候选',
                        tier='重点' if m['turns']>=rules.focus_turns else '一般',
                        start=start,end=end,axis=f'joint{j+1}',**m))
        # Merge overlapping windows, retaining maximum supported counts. Counts
        # from different axes/windows are never summed into artificial cycles.
        merged=[]
        for f in sorted(interval_findings,key=lambda x:(x['type'],x['axis'],x['start'],x['end'])):
            if merged and f['type']==merged[-1]['type'] and f['axis']==merged[-1]['axis'] and f['start']<=merged[-1]['end']:
                old=merged[-1];old['end']=max(old['end'],f['end'])
                if f['turns']>old['turns']:old.update({k:v for k,v in f.items() if k not in ['start','end']})
            else:merged.append(copy.deepcopy(f))
        for f in merged:
            angular=f['type']=='腕部关节反复调整候选'
            unit='rad' if angular else 'mm';factor=1 if angular else 1000
            add(f['type'],f['tier'],f['start'],f['end'],phase,
                f"{mode}，{f['axis']}至少{f['turns']}次有幅度的方向反转；"
                f"相邻极值间累计运动{f['repeated_travel']*factor:.1f}{unit}；"
                f"{'盒附近（按夹爪位置近似）' if near else '非盒附近'}；需结合避障/对中目的复核",
                dict(**context,turns=f['turns'],repeated_travel=f['repeated_travel'],axis=f['axis']))
        path_points=points[a:b,:2] if loaded else points[a:b]
        lengths=np.linalg.norm(np.diff(path_points,axis=0),axis=1)
        path=float(lengths[lengths>=.0003].sum());net=float(np.linalg.norm(path_points[-1]-path_points[0]))
        ratio=path/max(net,.03)
        bouts=movement_bouts(points[a:b],fps,rules)
        metrics.append(dict(start_frame=int(a),end_frame_exclusive=int(b),phase=phase,**context,
            path_m=path,net_m=net,path_ratio=ratio,movement_bouts=len(bouts),
            max_position_turns=max([f['turns'] for f in interval_findings if f['type']=='多次往返或局部抖动候选'],default=0),
            max_wrist_turns=max([f['turns'] for f in interval_findings if f['type']=='腕部关节反复调整候选'],default=0)))
        if path>=rules.detour_min_path_m and path-net>=rules.detour_extra_m and ratio>=rules.detour_ratio:
            add('绕行或低效路径候选','重点' if ratio>=4 and path-net>=.3 else '一般',a,b,phase,
                f"{mode}，{'水平XY' if loaded else 'XYZ'}路径{path*100:.1f}cm、起终点距离{net*100:.1f}cm、"
                f"路径比{ratio:.2f}；此项可能是单次大绕行，不能直接视作多次往返",context)
        if loaded and len(bouts)>=rules.review_move_bouts:
            add('搬运频繁启停候选','重点' if len(bouts)>=rules.focus_move_bouts else '一般',
                a+bouts[0]['start'],a+bouts[-1]['end'],phase,
                f"带载同一阶段出现{len(bouts)}段有效移动；每段移动≥8mm，间隔停顿≥0.5s；"
                "正常抬升/平移/下降通常已有多段，本项仅列过多启停供复核",dict(**context,movement_bouts=len(bouts)))
    return findings,metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-report',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):raise FileExistsError('Refuse overwrite')
    args.out.mkdir(parents=True,exist_ok=True)
    prior=review.read_json(args.previous_report);assert prior['passed'] and prior['analyzed']==779
    chain=review.chain_from_urdf(review.DEFAULT_URDF)
    robot=ET.parse(review.DEFAULT_URDF).getroot()
    fixed=np.fromstring(robot.find("joint[@name='gripperStatorLink']/origin").get('xyz'),sep=' ')
    rules=RepeatRules();items=[];display={};errors=[]
    for i,old in enumerate(prior['results'],1):
        try:
            root=Path(old['path'])
            for filename,expected in old['source_hashes'].items():
                if fruit.digest(root/filename)!=expected:raise ValueError('Changed frozen source: '+filename)
            _,data,view=fruit.read_episode(root/'frames.jsonl',chain,fixed,fruit.Rules())
            rows=[json.loads(line) for line in (root/'frames.jsonl').read_text().splitlines()]
            joints=np.asarray([r['observation.state'][:6] for r in rows])
            new,metrics=repeated_motion(np.asarray(view['positions']),joints,np.asarray(view['phases']),
                np.asarray(view['modes']),data['expert'],old['fps'],rules)
            item=copy.deepcopy(old)
            for finding in new:finding['needs_correction_context']=item['fragment']
            item['repetition_findings']=new;item['repetition_metrics']=metrics
            item['findings'].extend(new)
            items.append(item);display[item['path']]=view
        except Exception as exc:errors.append(dict(path=old['path'],error=str(exc)))
        if i%100==0:print('Analyzed',i,'/779',flush=True)
    for item in items:
        for filename,expected in item['source_hashes'].items():
            assert fruit.digest(Path(item['path'])/filename)==expected
    repetition=[e for e in items if e['repetition_findings']]
    focus=[e for e in repetition if any(f['tier']=='重点' for f in e['repetition_findings'])]
    previous_paths={e['path'] for e in prior['results'] if e['findings']}
    new_paths={e['path'] for e in repetition}-previous_paths
    repetition_only=[dict(e,findings=e['repetition_findings']) for e in repetition]
    review.write_csv(args.out/'反复移动候选清单.csv',fruit.csv_rows(repetition_only))
    review.write_csv(args.out/'合并全部候选清单.csv',fruit.csv_rows(items))
    review.write_csv(args.out/'新增候选清单.csv',fruit.csv_rows([e for e in items if e['path'] in new_paths]))
    review.write_csv(args.out/'读取问题.csv',errors)
    kinds=sorted({f['type'] for e in repetition for f in e['repetition_findings']})
    report=dict(passed=not errors,analyzed=len(items),read_errors=errors,
        prior_report=str(args.previous_report),prior_report_sha256=fruit.digest(args.previous_report),
        repetition_candidate_episodes=len(repetition),repetition_focus_episodes=len(focus),
        newly_flagged_episode_count=len(new_paths),combined_candidate_episodes=sum(bool(e['findings']) for e in items),
        by_type={k:dict(episodes=sum(any(f['type']==k for f in e['repetition_findings']) for e in repetition),
            focus_episodes=sum(any(f['type']==k and f['tier']=='重点' for f in e['repetition_findings']) for e in repetition)) for k in kinds},
        by_object={obj:dict(analyzed=sum(e['object']==obj for e in items),
            repetition_candidates=sum(e['object']==obj for e in repetition),
            repetition_focus=sum(e['object']==obj for e in focus),
            new_candidates=sum(e['object']==obj and e['path'] in new_paths for e in items)) for obj in ['apple','peach']},
        loaded_repetition_episodes=sum(any(f['motion_mode']=='带载抬升搬运' for f in e['repetition_findings']) for e in repetition),
        fragment_repetition_episodes=sum(e['fragment'] for e in repetition),
        thresholds=asdict(rules),source_code_sha256=fruit.digest(__file__),
        helper_code_sha256=fruit.digest(fruit.__file__),
        source_files_changed=False,training_manifest_changed=False,results=items)
    review.atomic_json(args.out/'report.json',report)
    fruit.write_html(args.out/'反复移动交互复核.html',repetition_only,display)
    fruit.write_html(args.out/'合并交互复核.html',items,display)
    (args.out/'说明.txt').write_text('本轮779条苹果/桃子；在上一轮反向/高度筛查基础上补充反复移动。\n'
        '反复移动候选清单按区间列出；同一条轨迹可因多个部位/规则出现多行，报告按唯一轨迹计数。\n'
        '多次往返：同一阶段同夹爪状态内，位置极值显著性≥6mm，至少4次反转，极值间累计运动≥4cm；6次反转列重点。\n'
        '腕部关节反复调整：joint4/5/6极值显著性≥0.035rad，至少4次反转，极值间累计转动≥0.20rad。\n'
        '检查完整阶段和6秒局部窗口，重叠区间合并，跨窗口/轴的反转次数不相加。\n'
        '搬运启停：同阶段至少5次有效移动，每段≥8mm，停顿≥0.5秒；7次列重点。\n'
        '绕行项可能只是一次大绕行，与多次往返分开列出；带载绕行只算水平路径，避免正常抬升下降误报。\n'
        '纠偏按已记录的步骤拆分，但仍需人工判断必要的退让和对中。盒附近按夹爪XY距(0.32,0.14)≤0.18m近似。\n'
        '没有自动删数据、合成视频或修改训练清单。原先发现的夹爪标签长度不符问题仍保留在各轨迹记录里。\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['results']},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
