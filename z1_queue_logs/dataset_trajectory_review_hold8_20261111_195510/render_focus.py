#!/usr/bin/env python3
"""Render only previously screened long stationary intervals, preserving full episodes."""
import argparse
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

from review import Settings, atomic_json, checklist, read_json, render, write_csv


def select_holds(metrics, minimum):
    selected=[]
    for entry in metrics:
        matches=[f for f in entry.get('findings',[])
                 if f['type']=='长停顿候选' and f['recorded_s']>=minimum]
        if not matches:continue
        item=copy.deepcopy(entry)
        item['findings']=copy.deepcopy(matches)
        item.pop('video_start_s',None)
        item.pop('video_end_s',None)
        selected.append(item)
    selected.sort(key=lambda e:(-max(f['recorded_s'] for f in e['findings']),e['dataset'],e['episode']))
    return selected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metrics',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--min-hold',type=float,default=8.)
    args=p.parse_args();out=args.output.absolute();out.mkdir(parents=True,exist_ok=True)
    if args.min_hold<=0:raise ValueError('Positive hold threshold required')
    if (out/'completed.json').exists():raise RuntimeError('Output already completed')
    metrics=json.loads(args.metrics.read_text())
    selected=select_holds(metrics,args.min_hold)
    if not selected:raise RuntimeError('No intervals meet the requested threshold')
    # Confirm that the prior screening still points at the same raw records.
    for item in selected:
        path=Path(item['path'])/'frames.jsonl'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['frames_sha256']:
            raise RuntimeError('Source frames changed since screening: '+str(path))
    cfg=Settings()
    request=dict(source_metrics=str(args.metrics.absolute()),minimum_continuous_hold_s=args.min_hold,
        selection='长停顿候选 only; no duration-outlier, sampling-gap, or detour-only candidates',
        scanned_source_records=len(metrics),selected_episodes=len(selected),speed=2.,
        video_mode='full episodes',input_mutation=False,
        started_at=dt.datetime.now().astimezone().isoformat())
    atomic_json(out/'request.json',request)
    atomic_json(out/'candidate_manifest.json',selected)
    atomic_json(out/'status.json',dict(stage='rendering',completed_episodes=0,total_candidates=len(selected)))
    write_csv(out/'异常候选清单.csv',checklist(selected))
    print(json.dumps(request,ensure_ascii=False),flush=True)
    result=render(selected,out,cfg)
    atomic_json(out/'candidate_manifest.json',selected)
    write_csv(out/'异常候选清单.csv',checklist(selected))
    (out/'汇总.md').write_text(f'''# 连续近静止≥{args.min_hold:g}秒重点复核

从既有{len(metrics)}条筛查结果中筛选{len(selected)}条记录，仅保留连续近静止至少{args.min_hold:g}秒的区间。数据源frames文件SHA256已与初筛逐条核对，原始数据未修改。

视频保留每条完整回合，轨迹2倍速，片头0.8秒；清单顺序与视频章节一致。画面标出长停顿区间，便于结合前后动作判断。只有采样间断、阶段偏长或绕行的记录不进入本次视频。

长停顿是人工复核候选，可能包含规划或夹持验收等待，不直接等于需要重录。人工结论、是否重录和备注栏保留空白供填写。
''',encoding='utf-8')
    completed=dict(verified=True,selected_episodes=len(selected),minimum_hold_s=args.min_hold,
        video=result,completed_at=dt.datetime.now().astimezone().isoformat())
    atomic_json(out/'completed.json',completed)
    atomic_json(out/'status.json',dict(stage='complete',**completed))


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if '--output' in sys.argv:
            out=Path(sys.argv[sys.argv.index('--output')+1]).absolute()
            atomic_json(out/'status.json',dict(stage='error',error=str(exc)))
        raise
