#!/usr/bin/env python3
"""Read-only screening of current-round apple/peach reverse motion and height.

Outputs review candidates, never rejection decisions or a modified train list.
Heights refer to gripperStator in the URDF robot base, not object centre height.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import html
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.ndimage import median_filter

import review


@dataclass
class Rules:
    sample_s: float = .3
    spatial_noise_m: float = .0015
    minimum_segment_s: float = 1.5
    minimum_goal_axis_m: float = .03
    focus_reverse_excursion_m: float = .05
    absolute_focus_reverse_excursion_m: float = .08
    review_reverse_excursion_m: float = .03
    reverse_path_fraction: float = .08
    correction_reverse_path_fraction: float = .25
    height_focus_margin_m: float = .08
    height_review_margin_m: float = .05
    height_mad_factor: float = 4.
    height_focus_min_s: float = .3
    height_review_min_s: float = .5
    minimum_cell_peers: int = 8


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command_labels(rows, root):
    warnings=[]
    labels = review.read_json(root/'gripper_labels.json',{}).get('labels')
    if labels is not None:
        labels=np.asarray(labels)
        if len(labels)==len(rows) and np.isin(labels,[0,1]).all():
            return labels.astype(int), 'saved_gripper_labels', warnings
        warnings.append(dict(type='夹爪标签与帧不一致',frames=len(rows),labels=len(labels),
            detail='仅本次轨迹筛查改用原始动作的滞回判定；原始标签未修改，不应据此认为训练标签已修复。'))
    # Hysteresis excludes small commanded angle fluctuations during contact.
    values=[];closed=False
    for row in rows:
        value=row['action'][6]
        if value>=-.60:closed=True
        elif value<=-.75:closed=False
        values.append(int(closed))
    return np.asarray(values), 'raw_command_hysteresis_close_-0.60_open_-0.75', warnings


def reverse_segments(points, phases, labels, expert, times, fps, fragment, rules):
    boundaries=np.flatnonzero(np.r_[True,
        (phases[1:]!=phases[:-1]) | (labels[1:]!=labels[:-1]) |
        (expert[1:]!=expert[:-1]) | (np.diff(times)>1.6/fps),True])
    segments=[]
    smooth=median_filter(points,size=(3,1),mode='nearest')
    for a,b in zip(boundaries[:-1],boundaries[1:]):
        if not expert[a] or (b-a)/fps<rules.minimum_segment_s:continue
        # A loaded transport legitimately rises and descends. Horizontal goal
        # progress is used there so normal vertical placement isn't backtracking.
        planar=bool(labels[a] and str(phases[a]).split('#')[0] in
                    ['TRANSPORT','PHYSICAL_TEST_LIFT','PLACING'])
        dimensions=2 if planar else 3
        inds=np.unique(np.r_[np.arange(a,b,max(1,round(rules.sample_s*fps))),b-1])
        kept=[int(inds[0])]
        for i in inds[1:]:
            if np.linalg.norm(smooth[i,:dimensions]-smooth[kept[-1],:dimensions])>=rules.spatial_noise_m:
                kept.append(int(i))
        if len(kept)<3:continue
        p=smooth[kept,:dimensions];v=np.diff(p,axis=0)
        length=float(np.linalg.norm(v,axis=1).sum())
        net=float(np.linalg.norm(p[-1]-p[0]))
        if net<rules.minimum_goal_axis_m:continue
        axis=(p[-1]-p[0])/net
        progress=(p-p[0])@axis
        delta=np.diff(progress)
        back=float(-delta[delta<0].sum())
        drawdown=np.maximum.accumulate(progress)-progress
        bottom=int(np.argmax(drawdown));top=int(np.argmax(progress[:bottom+1]))
        fraction=back/max(length,1e-12)
        threshold=rules.correction_reverse_path_fraction if fragment else rules.reverse_path_fraction
        tier=''
        if ((not fragment and drawdown[bottom]>=rules.absolute_focus_reverse_excursion_m)
                or (fraction>=threshold and drawdown[bottom]>=rules.focus_reverse_excursion_m)):
            tier='重点'
        elif ((not fragment or fraction>=threshold)
                and drawdown[bottom]>=rules.review_reverse_excursion_m):
            tier='一般'
        segments.append(dict(start_frame=int(a),end_frame_exclusive=int(b),
            phase=str(phases[a]),jaw_closed=bool(labels[a]),metric='XY' if planar else 'XYZ',
            path_m=length,net_m=net,backward_m=back,backward_fraction=fraction,
            max_reverse_excursion_m=float(drawdown[bottom]),
            reverse_start_frame=kept[top],reverse_end_frame_exclusive=kept[bottom]+1,
            tier=tier))
    return segments


def height_modes(phases, labels):
    n=len(labels)
    first_closed=int(np.flatnonzero(labels)[0]) if np.any(labels) else n
    base_phases=np.asarray([str(p).split('#')[0] for p in phases])
    loaded=(labels==1)&np.isin(base_phases,['PHYSICAL_TEST_LIFT','TRANSPORT','PLACING'])
    first_loaded=int(np.flatnonzero(loaded)[0]) if np.any(loaded) else n
    modes=np.full(n,'空爪接近',dtype=object)
    modes[(labels==1)&~loaded]='闭爪准备'
    modes[loaded]='带载抬升搬运'
    modes[(np.arange(n)>first_loaded)&(labels==0)]='释放后撤离'
    if first_closed==n:modes[:]='空爪接近'
    return modes


def read_episode(path, chain, fixed, rules):
    root=path.parent
    pins={p.name:digest(p) for p in [path,root/'episode_summary.json',root/'episode_start.json',
        root/'correction.json',root/'gripper_labels.json',root/'training_mask.json'] if p.exists()}
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows)<3:raise ValueError('Too few frames')
    summary=review.read_json(root/'episode_summary.json')
    meta=review.read_json(root/'correction.json',{})
    q=np.asarray([row['observation.state'] for row in rows],float)
    action=np.asarray([row['action'] for row in rows],float)
    if q.shape!=(len(rows),7) or action.shape!=q.shape or not np.isfinite(q).all() or not np.isfinite(action).all():
        raise ValueError('State/action is not finite 7D')
    times=np.asarray([r['simulation_time'] for r in rows],float)
    fps=float(summary['fps'])
    if not np.isfinite(times).all() or np.any(np.diff(times)<=0):raise ValueError('Invalid timestamps')
    if abs(fps-10)>1e-6 or not np.allclose(np.diff(times),1/fps,atol=1e-5,rtol=0):
        raise ValueError('Sampling is not regular 10 Hz')
    if summary['frame_count']!=len(rows) or summary['dropped_frames']!=0:
        raise ValueError('Frame count/drop mismatch')
    info=review.read_json(root.parent/'dataset_info.json')
    if info['joint_names'][:7]!=[f'joint{i}' for i in range(1,7)]+['jointGripper']:
        raise ValueError('Joint order mismatch')
    fragment=bool(summary.get('fragment_success') or meta.get('fragment_goal'))
    if not summary.get('physical_grasp_ok') or not (summary.get('task_success') or summary.get('fragment_success')):
        raise ValueError('Missing success for declared episode goal')
    position,rotation=review.forward(q[:,:6],chain)
    gripper=position+np.einsum('nij,j->ni',rotation,fixed)
    labels,label_source,label_warnings=command_labels(rows,root)
    phases=np.asarray([r.get('phase','UNKNOWN') for r in rows],dtype=object)
    expert=review.event_mask(meta,times)
    saved=review.read_json(root/'training_mask.json',{}).get('expert_frame_mask')
    if saved is not None:
        if len(saved)!=len(rows):raise ValueError('Expert mask length mismatch')
        expert &= np.asarray(saved,bool)
    # Documented correction legs are intentional changes of direction. Don't
    # score their preparation lift and following approach as one reverse leg.
    stages=meta.get('stages',[])
    for stage in stages:
        if 'start_sim_time' not in stage or 'end_sim_time' not in stage:continue
        chosen=(times>=stage['start_sim_time'])&(times<stage['end_sim_time'])
        phases[chosen]=[str(p)+'#'+stage['stage'] for p in phases[chosen]]
    fk_checks=[]
    for stage in stages:
        if 'actual_position' not in stage or 'end_sim_time' not in stage:continue
        i=int(np.argmin(abs(times-stage['end_sim_time'])))
        delta=float(abs(times[i]-stage['end_sim_time']))
        if delta<=.11:
            fk_checks.append(dict(frame=i,stage=stage['stage'],stamp_delta_s=delta,
                error_m=float(np.linalg.norm(position[i]-stage['actual_position']))))
    motion=reverse_segments(gripper,phases,labels,expert,times,fps,fragment,rules)
    modes=height_modes(phases,labels)
    smooth_z=median_filter(gripper[:,2],size=3,mode='nearest')
    height=[]
    for mode in sorted(set(modes)):
        mask=(modes==mode)&expert
        inds=np.flatnonzero(mask)
        if not len(inds):continue
        peak=int(inds[np.argmax(smooth_z[inds])])
        height.append(dict(mode=mode,peak_z_m=float(smooth_z[peak]),
            raw_peak_z_m=float(gripper[inds,2].max()),peak_frame=peak,
            initial_z_m=float(smooth_z[inds[0]]),
            rise_m=float(smooth_z[peak]-smooth_z[inds[0]])))
    grid=summary.get('scene_randomization',{}).get('target_grid')
    result=dict(path=str(root),dataset=str(root.parent),episode=root.name,
        object=summary['target_object'],fragment=fragment,
        kind=meta.get('task',{}).get('kind','basic'),grid=grid,
        frames=len(rows),fps=fps,duration_s=len(rows)/fps,
        source_hashes=pins,gripper_label_source=label_source,data_quality_issues=label_warnings,
        motion_segments=motion,height_metrics=height,fk_checks=fk_checks,findings=[])
    for segment in motion:
        if not segment['tier']:continue
        a=segment['reverse_start_frame'];b=segment['reverse_end_frame_exclusive']
        result['findings'].append(dict(type='反向移动候选',tier=segment['tier'],
            start_frame=a,end_frame_exclusive=b,start_s=a/fps,end_s=b/fps,
            phase=segment['phase'],detail=(f"{segment['metric']}投影累计反向{segment['backward_m']*1000:.1f}mm，"
                f"最大一次回退{segment['max_reverse_excursion_m']*1000:.1f}mm，"
                f"占本段路径{segment['backward_fraction']:.1%}"),
            needs_correction_context=fragment))
    display=dict(times=(times-times[0]).round(3).tolist(),
        positions=gripper.round(5).tolist(),phases=phases.tolist(),modes=modes.tolist(),
        images=[[str((root/r['observation.images.'+c]).resolve().as_uri())
                 for c in ['wrist','global','global_right']] for r in rows])
    return result,dict(z=smooth_z,modes=modes,expert=expert),display


def height_screen(items, arrays, rules):
    for item in items:
        for metric in item['height_metrics']:
            mode=metric['mode']
            # Baselines always separate fruit, full/fragment and mode. Full
            # examples use the same spatial cell where enough peers exist.
            pool=[e for e in items if e['object']==item['object'] and e['fragment']==item['fragment']]
            same_kind=[e for e in pool if e['kind']==item['kind']]
            if item['fragment'] and len(same_kind)>=rules.minimum_cell_peers:pool=same_kind
            cell=[e for e in pool if e['grid']==item['grid']]
            scope='同物体同任务类型'
            if len(cell)>=rules.minimum_cell_peers:
                pool=cell;scope+='同网格'
            values=[h['peak_z_m'] for e in pool for h in e['height_metrics'] if h['mode']==mode]
            if len(values)<rules.minimum_cell_peers:
                metric.update(baseline_available=False);continue
            center=float(np.median(values));mad=float(np.median(np.abs(np.asarray(values)-center)))
            focus=center+max(rules.height_focus_margin_m,rules.height_mad_factor*1.4826*mad)
            lower=center+max(rules.height_review_margin_m,3*1.4826*mad)
            metric.update(baseline_available=True,baseline_scope=scope,baseline_n=len(values),
                baseline_median_z_m=center,baseline_mad_m=mad,
                focus_threshold_z_m=focus,review_threshold_z_m=lower)
            data=arrays[item['path']];chosen=(data['modes']==mode)&data['expert']
            focus_runs=[(a,b) for a,b in review.runs(chosen&(data['z']>focus))
                        if (b-a)/item['fps']+1e-6>=rules.height_focus_min_s]
            lower_runs=[(a,b) for a,b in review.runs(chosen&(data['z']>lower))
                        if (b-a)/item['fps']+1e-6>=rules.height_review_min_s]
            intervals=focus_runs or lower_runs
            tier='重点' if focus_runs else '一般'
            for a,b in intervals:
                item['findings'].append(dict(type='抬升高度偏高候选',tier=tier,
                    start_frame=a,end_frame_exclusive=b,start_s=a/item['fps'],end_s=b/item['fps'],
                    phase=mode,detail=(f"{mode}夹爪原点峰值{metric['peak_z_m']*100:.1f}cm，"
                        f"{scope}{len(values)}条中位数{center*100:.1f}cm，"
                        f"筛查线{(focus if focus_runs else lower)*100:.1f}cm；"
                        "高度相对机器人基坐标系，不是物体中心离桌高度"),
                    needs_correction_context=item['fragment']))


def csv_rows(items):
    rows=[]
    for item in items:
        for finding in item['findings']:
            rows.append(dict(复核级别=finding['tier'],物体=item['object'],
                示范类型='纠偏片段' if item['fragment'] else '基础完整示范',
                数据集=Path(item['dataset']).name,Episode=item['episode'],网格=item['grid'],
                候选类型=finding['type'],阶段=finding['phase'],
                开始秒=round(finding['start_s'],2),结束秒=round(finding['end_s'],2),
                依据=finding['detail'],纠偏需结合恢复目的=item['fragment'],
                原始路径=item['path'],人工结论='',是否重录='',备注=''))
    return sorted(rows,key=lambda r:(r['复核级别']!='重点',r['物体'],r['数据集'],r['Episode'],r['开始秒']))


def write_html(path, items, display):
    candidates=sorted([e for e in items if e['findings']],
        key=lambda e:(not any(f['tier']=='重点' for f in e['findings']),e['object'],e['path']))
    data=[dict(**{k:e[k] for k in ['path','episode','object','fragment','grid','findings']},
               **display[e['path']]) for e in candidates]
    payload=json.dumps(data,ensure_ascii=False,
        default=lambda value:value.item() if isinstance(value,np.generic) else str(value)).replace('<','\\u003c')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>苹果与桃子轨迹复核</title>
<style>body{font:16px sans-serif;margin:24px;background:#f5f6fa;color:#172539}select{width:100%;padding:9px}#plots{display:flex;gap:12px}svg{width:49%;height:260px;background:white;border:1px solid #ddd}#images{display:flex;gap:8px}img{width:32%;object-fit:contain;background:#222}input{width:100%}li{margin:8px 0}a{color:#1474b5}small{color:#536478}</style>
<h2>苹果与桃子轨迹复核</h2><p>仅本轮四个源集。候选供人工复核，原始数据与训练清单未改动。带载阶段按水平运动评价反向，避免正常抬升与下降被计入；纠偏片段按记录的恢复步骤分段。</p>
<p>高度是 gripperStator 原点相对机器人基坐标系的 Z，不是物体离桌高度。蓝线为全程轨迹，红段为筛查区间。拖动滑块可看原始三路图像；本页完全离线。</p>
<select id="pick"></select><ul id="notes"></ul><div id="plots"><svg id="xy" viewBox="0 0 600 260"></svg><svg id="z" viewBox="0 0 600 260"></svg></div>
<input id="slider" type="range" min="0" value="0"><p id="stamp"></p><div id="images"><img id="cam0"><img id="cam1"><img id="cam2"></div><p id="location"></p>
<script>const DATA=__DATA__;const $=id=>document.getElementById(id);let current=null;const esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
DATA.forEach((e,i)=>{let o=document.createElement('option');o.value=i;o.textContent=`${e.findings.some(f=>f.tier==='重点')?'重点':'一般'} | ${e.object} | ${e.fragment?'纠偏':'基础'} | ${e.path.split('/').slice(-2).join('/')} | 网格${e.grid}`;$('pick').append(o)});
function graph(id, pts, xlabel,ylabel){let x0=Math.min(...pts.map(p=>p[0])),x1=Math.max(...pts.map(p=>p[0])),y0=Math.min(...pts.map(p=>p[1])),y1=Math.max(...pts.map(p=>p[1]));if(x1-x0<.01){x0-=.01;x1+=.01}if(y1-y0<.01){y0-=.01;y1+=.01}let xx=x=>55+(x-x0)/(x1-x0)*515,yy=y=>218-(y-y0)/(y1-y0)*180;let line=part=>part.map((p,i)=>(i?'L':'M')+xx(p[0])+','+yy(p[1])).join(' ');let s=`<text x="25" y="23">${esc(ylabel)}</text><text x="460" y="252">${esc(xlabel)}</text>`;[0,.5,1].forEach(f=>{let x=x0+(x1-x0)*f,y=y0+(y1-y0)*f;s+=`<path d="M${xx(x)},38V218M55,${yy(y)}H570" stroke="#e7edf1" fill="none"/><text x="${xx(x)-18}" y="238" font-size="12">${x.toFixed(2)}</text><text x="4" y="${yy(y)+4}" font-size="12">${y.toFixed(2)}</text>`});s+=`<path d="${line(pts)}" fill="none" stroke="#2683bd" stroke-width="2"/>`;current.findings.forEach(f=>{s+=`<path d="${line(pts.slice(f.start_frame,f.end_frame_exclusive))}" fill="none" stroke="#dc4159" stroke-width="3"/>`});let p=pts[+$('slider').value];s+=`<circle cx="${xx(p[0])}" cy="${yy(p[1])}" r="5" fill="#111"/>`;$(id).innerHTML=s}
function frame(){if(!current)return;let i=+$('slider').value;current.images[i].forEach((src,k)=>$('cam'+k).src=src);$('stamp').textContent=`第 ${i} 帧 | 录制时间 ${current.times[i].toFixed(2)} 秒 | ${current.phases[i]} | ${current.modes[i]} | 夹爪原点 Z ${(current.positions[i][2]*100).toFixed(1)} cm`;graph('xy',current.positions.map(p=>[p[0],p[1]]),'X (m)','Y (m)');graph('z',current.positions.map((p,j)=>[current.times[j],p[2]]),'录制时间 (s)','夹爪原点 Z (m)')}
function choose(){current=DATA[+$('pick').value];if(!current){$('notes').textContent='无候选';return}$('slider').max=current.times.length-1;$('slider').value=current.findings[0].start_frame;$('notes').innerHTML=current.findings.map(f=>`<li><b>${esc(f.tier)} · ${esc(f.type)} · ${f.start_s.toFixed(1)}–${f.end_s.toFixed(1)}秒</b>：${esc(f.detail)}${f.needs_correction_context?'；纠偏示范需结合恢复目的判断。':''}</li>`).join('');$('location').innerHTML=`<a href="${esc('file://'+current.path)}">打开原始 episode 目录</a>`;frame()}
$('pick').onchange=choose;$('slider').oninput=frame;choose();</script></html>'''
    path.write_text(page.replace('__DATA__',payload))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();out=args.out
    if out.exists() and any(out.iterdir()):raise FileExistsError('Refuse to overwrite output')
    out.mkdir(parents=True,exist_ok=True)
    rules=Rules();chain=review.chain_from_urdf(review.DEFAULT_URDF)
    robot=ET.parse(review.DEFAULT_URDF).getroot()
    fixed=np.fromstring(robot.find("joint[@name='gripperStatorLink']/origin").get('xyz'),sep=' ')
    assert np.allclose(fixed,[.051,0,0])
    scope=review.read_json(Path(__file__).with_name('current_round_sources.json'))
    roots=[r for r in scope['roots'] if Path(r).name.startswith(('apple_scale110_','peach_scale120_'))]
    assert len(roots)==4
    files,aliases=review.discover(roots)
    items=[];arrays={};display={};errors=[]
    for i,path in enumerate(files,1):
        try:
            item,data,view=read_episode(path,chain,fixed,rules)
            items.append(item);arrays[item['path']]=data;display[item['path']]=view
        except Exception as exc:errors.append(dict(path=str(path),error=str(exc)))
        if i%100==0:print('Analyzed',i,'/',len(files),flush=True)
    height_screen(items,arrays,rules)
    for item in items:
        for filename,expected in item['source_hashes'].items():
            if digest(Path(item['path'])/filename)!=expected:
                raise ValueError('Input changed during analysis: '+item['path'])
    candidates=[e for e in items if e['findings']]
    focus=[e for e in candidates if any(f['tier']=='重点' for f in e['findings'])]
    csv=csv_rows(items)
    review.write_csv(out/'候选轨迹清单.csv',csv)
    review.write_csv(out/'重点复核清单.csv',[r for r in csv if r['复核级别']=='重点'])
    review.write_csv(out/'读取问题.csv',errors)
    quality_issues=[dict(物体=e['object'],数据集=Path(e['dataset']).name,Episode=e['episode'],
        问题=w['type'],Frames帧数=w['frames'],标签数=w['labels'],说明=w['detail'],
        原始路径=e['path']) for e in items for w in e['data_quality_issues']]
    review.write_csv(out/'数据一致性问题.csv',quality_issues)
    review.write_csv(out/'全部779条指标.csv',[dict(物体=e['object'],类型='纠偏' if e['fragment'] else '基础',
        数据集=Path(e['dataset']).name,Episode=e['episode'],网格=e['grid'],
        时长秒=e['duration_s'],候选项数=len(e['findings']),
        峰值夹爪高度厘米=round(max(h['peak_z_m'] for h in e['height_metrics'])*100,2),
        最大一次阶段回退毫米=round(max([m['max_reverse_excursion_m'] for m in e['motion_segments']],default=0)*1000,2),
        原始路径=e['path']) for e in items])
    checks=[c['error_m'] for e in items for c in e['fk_checks']]
    report=dict(passed=not errors,scope='本轮苹果/桃子；不含历史训练集',roots=roots,
        discovered=len(files),analyzed=len(items),read_errors=errors,data_quality_issues=quality_issues,
        unique_candidate_episodes=len(candidates),focus_episodes=len(focus),
        review_only_episodes=len(candidates)-len(focus),
        by_source={Path(root).name:dict(analyzed=sum(e['dataset']==root for e in items),
            candidates=sum(e['dataset']==root for e in candidates),
            focus=sum(e['dataset']==root for e in focus)) for root in roots},
        by_type={kind:dict(episodes=sum(any(f['type']==kind for f in e['findings']) for e in candidates),
            focus_episodes=sum(any(f['type']==kind and f['tier']=='重点' for f in e['findings']) for e in candidates))
            for kind in ['反向移动候选','抬升高度偏高候选']},
        height_modes=Counter(f['phase'] for e in candidates for f in e['findings'] if f['type']=='抬升高度偏高候选'),
        fk_vs_recorded_correction_pose=dict(samples=len(checks),median_error_m=float(np.median(checks)) if checks else None,
            p95_error_m=float(np.quantile(checks,.95)) if checks else None,max_error_m=max(checks,default=None)),
        thresholds=asdict(rules),coordinate='URDF link00 base / gripperStator origin; NOT object height',
        kinematic_urdf=review.DEFAULT_URDF,urdf_sha256=digest(review.DEFAULT_URDF),
        source_code_sha256=digest(__file__),source_files_changed=False,training_manifest_changed=False,
        results=items)
    review.atomic_json(out/'report.json',report)
    write_html(out/'交互复核.html',items,display)
    (out/'说明.txt').write_text('本轮苹果基础323＋纠偏59，桃子基础340＋纠偏57，共779条。\n'
        '重点复核清单优先看；候选轨迹清单还包含一般候选。同一条轨迹可有多行区间。\n'
        '双击交互复核.html，选择 Episode 并拖动滑块查看轨迹位置、高度与三路原始图像。\n'
        '反向是同阶段同夹爪命令状态内的几何代理；带载搬运只按XY平面判断回退。\n'
        '纠偏已有恢复步骤单独分段，仍需人工判断必要退让、对中和避障。\n'
        '高度按同物体、基础/纠偏、阶段模式比较；样本足够时优先同网格，纠偏优先同类型。\n'
        '重点高度线：组中位数＋max(8cm,4×1.4826×MAD)，至少持续0.3秒；一般线：组中位数＋max(5cm,3×1.4826×MAD)，至少0.5秒。\n'
        '基础重点反向线：最大一次回退≥8cm，或≥5cm且累计反向路径比例≥8%；一般回退≥3cm。纠偏始终要求路径比例≥25%。\n'
        '单独保留绝对回退规则，避免很长的绕行/抬高路径稀释反向比例。\n'
        '数据一致性问题另列清单；有标签长度问题时仅筛查改用原始动作推断夹爪状态，不修改或修复训练标签。\n'
        '高度为夹爪原点相对机器人基坐标系的Z，不能当成物体离桌高度。没有自动删数据或调整训练集。\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['results']},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
