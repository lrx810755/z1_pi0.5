#!/usr/bin/env python3
"""Screen every saved frames.jsonl and concatenate full candidate episodes at2x.

Read-only with respect to inputs. Heuristics produce review candidates, not a
decision to reject or rerecord. Run with --help for paths and thresholds.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import datetime as dt
import csv
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.ndimage import median_filter
from scipy.spatial.transform import Rotation

DEFAULT_URDF = '/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/generated_dual_camera/z1_with_dual_d435_resolved.urdf'
CAMERAS = ('wrist', 'global', 'global_right', 'vla_wrist')
SKIP = {'images', 'videos', 'collection_logs', 'runs', 'rejected_sources',
        'interrupted', '__pycache__', 'node_modules'}


@dataclass
class Settings:
    hold_window_s: float = 1.0
    hold_min_s: float = 3.0
    hold_high_s: float = 5.0
    ee_window_m: float = .002
    joint_window_rad: float = .006
    orientation_window_deg: float = 1.0
    gripper_window_rad: float = .012
    tail_allowance_s: float = 2.5
    tail_screen_s: float = 4.0
    reverse_distance_m: float = .030
    reverse_fraction: float = .25
    jitter_turns: int = 4
    jitter_path_m: float = .025
    jitter_extent_m: float = .050
    duration_ratio: float = 1.5
    duration_mad_factor: float = 3.0
    phase_min_samples: int = 8
    video_fps: int = 20
    speed: float = 2.0
    width: int = 1440
    height: int = 600
    title_s: float = .8
    ffmpeg_threads: int = 2


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
        default=lambda x: x.item() if isinstance(x, np.generic) else str(x)))
    os.replace(temp, path)


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def runs(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def discover(roots, errors=None):
    files, aliases, seen = [], [], {}
    errors=[] if errors is None else errors
    def walk(folder, ancestors):
        try:
            stat=folder.stat()
            key=(stat.st_dev,stat.st_ino)
            if key in ancestors:return
            entries=sorted(folder.iterdir())
        except OSError as exc:
            errors.append(dict(path=str(folder),error=str(exc)))
            return
        frame=folder/'frames.jsonl'
        if frame.is_file():
            yield frame
            return
        if folder.name.startswith('episode_'):return
        for child in entries:
            if child.name.startswith('.') or child.name in SKIP:continue
            if child.is_symlink() and not child.exists():
                errors.append(dict(path=str(child),error='Broken directory/file link'))
            if child.is_dir():yield from walk(child,ancestors|{key})
    for root in roots:
        root = Path(root).expanduser().absolute()
        if root.is_file():
            possible = [root]
        else:
            possible = list(walk(root,set()))
        for path in sorted(possible,key=lambda p:(p!=p.resolve(),str(p))):
            if path.name != 'frames.jsonl':
                continue
            stat = path.stat()
            key = (str(path.parent.resolve()),stat.st_dev,stat.st_ino)
            if key in seen:
                aliases.append(dict(path=str(path), same_file_as=str(seen[key])))
            else:
                seen[key] = path
                files.append(path)
    return files, aliases


def chain_from_urdf(path):
    if path is None or not Path(path).exists():
        return None
    robot = ET.parse(path).getroot()
    chain = []
    for i in range(1, 7):
        joint = robot.find(f"joint[@name='joint{i}']")
        if joint is None:
            raise ValueError('URDF lacks joint' + str(i))
        origin = joint.find('origin')
        transform = np.eye(4)
        transform[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
        transform[:3, :3] = Rotation.from_euler('xyz',
            np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
        chain.append((transform, np.fromstring(joint.find('axis').get('xyz'), sep=' ')))
    return chain


def forward(q, chain):
    transforms = np.repeat(np.eye(4)[None], len(q), axis=0)
    for i, (fixed, axis) in enumerate(chain):
        moving = np.repeat(np.eye(4)[None], len(q), axis=0)
        moving[:, :3, :3] = Rotation.from_rotvec(q[:, i, None] * axis).as_matrix()
        transforms = transforms @ fixed @ moving
    return transforms[:, :3, 3], transforms[:, :3, :3]


def hold_mask(q, action, times, ee, rotations, fps, cfg):
    n = len(q)
    width = max(3, round(cfg.hold_window_s * fps) + 1)
    found = np.zeros(n, bool)
    # First cheaply reject windows with clearly moving arm/jaw/command.
    for a in range(n - width + 1):
        b = a + width
        if np.max(np.diff(times[a:b])) > 1.6/fps:
            continue
        if np.max(np.ptp(q[a:b, :6], axis=0)) > cfg.joint_window_rad:
            continue
        if np.ptp(q[a:b, 6]) > cfg.gripper_window_rad:
            continue
        if action is not None and (
                np.max(np.ptp(action[a:b, :6], axis=0)) > cfg.joint_window_rad
                or np.ptp(action[a:b, 6]) > .015):
            continue
        if ee is not None:
            if np.linalg.norm(np.ptp(ee[a:b], axis=0)) > cfg.ee_window_m:
                continue
            angles = Rotation.from_matrix(rotations[a].T @ rotations[a:b]).magnitude()
            if angles.max() > math.radians(cfg.orientation_window_deg):
                continue
        found[a:b] = True
    return found


def event_mask(meta, times):
    expert = np.ones(len(times), bool)
    start = None
    for event in sorted(meta.get('events', []), key=lambda e: e.get('time', 0)):
        if event.get('event') == 'intervention_start':
            start = event['time']
        elif event.get('event') == 'intervention_end' and start is not None:
            expert[(times >= start-1e-7) & (times <= event['time']+1e-7)] = False
            start = None
    if start is not None:
        expert[times >= start-1e-7] = False
    return expert


def issue(name, a, b, times, fps, detail, priority='review'):
    a, b = int(a), int(b)
    return dict(type=name, start_frame=a, end_frame_exclusive=b,
        start_s=round(float(times[a]-times[0]), 3),
        end_s=round(float(times[b-1]-times[0]+1/fps), 3),
        recorded_s=round((b-a)/fps, 3), priority=priority, detail=detail)


def jitter_windows(positions,a,b,fps,cfg):
    """Local oscillations can be hidden inside an otherwise long carry phase."""
    intervals=[]
    width=max(6,round(4*fps));stride=max(1,round(fps));sample=max(1,round(.2*fps))
    for left in range(int(a),int(b),stride):
        right=min(int(b),left+width)
        if (right-left)/fps<2.5:continue
        points=positions[np.r_[np.arange(left,right,sample),right-1]]
        kept=[points[0]]
        for point in points[1:]:
            if np.linalg.norm(point-kept[-1])>=.0015:kept.append(point)
        if len(kept)<6:continue
        points=np.asarray(kept);v=np.diff(points,axis=0);length=np.linalg.norm(v,axis=1)
        turns=int(np.sum(np.sum(v[:-1]*v[1:],axis=1)/(length[:-1]*length[1:])<-.3))
        distance=float(length.sum());extent=float(np.linalg.norm(np.ptp(points,axis=0)))
        if turns>=cfg.jitter_turns and distance>=cfg.jitter_path_m and extent<=cfg.jitter_extent_m:
            near=float(np.median(np.linalg.norm(points[:,:2]-[.32,.14],axis=1)))<.22
            if intervals and left<=intervals[-1]['end']:
                intervals[-1]['end']=right
                intervals[-1]['turns']=max(intervals[-1]['turns'],turns)
            else:intervals.append(dict(start=left,end=right,turns=turns,near=near))
    return intervals


def analyze(path, chain, cfg):
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if len(rows) < 2:
        raise ValueError('Fewer than two readable frames')
    folder = path.parent
    summary = read_json(folder/'episode_summary.json', {})
    info = read_json(folder.parent/'dataset_info.json', {})
    meta = read_json(folder/'correction.json', {})
    q = np.asarray([r['observation.state'][:7] for r in rows], float)
    if q.shape != (len(rows), 7) or not np.isfinite(q).all():
        raise ValueError('State must begin with six finite arm joints plus a gripper')
    names = info.get('joint_names', [])
    if names and names[:7] != [f'joint{i}' for i in range(1,7)]+['jointGripper']:
        raise ValueError('Joint order differs; refusing to apply Z1 FK with wrong order')
    action = None
    if all('action' in r and len(r['action']) >= 7 for r in rows):
        action = np.asarray([r['action'][:7] for r in rows], float)
        if not np.isfinite(action).all():
            raise ValueError('Nonfinite action')
    time_key = 'simulation_time' if all('simulation_time' in r for r in rows) else 'timestamp'
    times = np.asarray([r[time_key] for r in rows], float)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Nonfinite or nonincreasing timestamps; no automatic sorting applied')
    measured_fps = 1/float(np.median(np.diff(times)))
    fps = float(summary.get('fps', info.get('fps', measured_fps)))
    if not .1 <= fps <= 240:
        raise ValueError('Invalid declared sampling frequency')
    t = times-times[0]
    phases = np.asarray([str(r.get('phase', 'UNKNOWN')) for r in rows])
    expert = event_mask(meta, times)
    saved_mask = read_json(folder/'training_mask.json', {})
    if 'expert_frame_mask' in saved_mask:
        saved = np.asarray(saved_mask['expert_frame_mask'], bool)
        if len(saved) != len(rows):
            raise ValueError('Training mask length differs from frames')
        expert &= saved
    ee, rotations = forward(q[:, :6], chain) if chain is not None else (None, None)
    still = hold_mask(q, action, times, ee, rotations, fps, cfg)
    findings = []
    eligible = expert.copy()
    eligible[t <= .35] = False
    # Tail holds are a separate rule, so legitimate settling is not hidden in
    # generic long holds; unusually long tails still become review candidates.
    tail_a = len(rows)
    if (still & expert)[-1]:
        tail_a = runs(still & expert)[-1][0]
        tail_s = (len(rows)-tail_a)/fps
        eligible[tail_a:] = False
        if tail_s > cfg.tail_screen_s:
            findings.append(issue('末尾停留偏长', tail_a, len(rows), times, fps,
                f'尾部近静止{tail_s:.2f}s；常规尾部保留量约{cfg.tail_allowance_s}s，需人工核实'))
    for a,b in runs(still & eligible):
        seconds = (b-a)/fps
        if seconds < cfg.hold_min_s:
            continue
        extent = float(np.linalg.norm(np.ptp(ee[a:b], axis=0))) if ee is not None else None
        joint_extent = float(np.max(np.ptp(q[a:b, :6], axis=0)))
        drift = joint_extent > .012 or (extent is not None and extent > .004)
        findings.append(issue('缓慢漂移或低进展' if drift else '长停顿候选', a,b,times,fps,
            f'近静止{seconds:.2f}s；末端范围{extent*1000:.2f}mm；最大关节范围{math.degrees(joint_extent):.2f}deg'
                if extent is not None else f'近静止{seconds:.2f}s；无FK，仅关节筛查',
            'high' if seconds >= cfg.hold_high_s and not drift else 'review'))
    gaps = np.flatnonzero(np.diff(times) > 1.6/fps)
    if len(gaps):
        a=int(gaps[np.argmax(np.diff(times)[gaps])])
        findings.append(issue('采样时间间断',a,a+2,times,fps,
            f'{len(gaps)}处间断；最大{np.diff(times).max():.3f}s；可能来自旧预处理，需结合版本核实'))
    if summary.get('frame_count') not in (None, len(rows)):
        findings.append(issue('摘要帧数不符',0,len(rows),times,fps,
            f"summary={summary.get('frame_count')}, actual={len(rows)}", 'high'))
    # Motion directions are judged inside one evaluator phase and one gripper
    # command-intent state, with deliberate interventions removed. Corrective
    # retreat can still trigger this heuristic and must be reviewed by a human.
    if ee is not None:
        smooth = median_filter(ee, size=(3,1), mode='nearest')
        labels = read_json(folder/'gripper_labels.json', {}).get('labels')
        if labels is None or len(labels) != len(rows):
            labels = (q[:,6] > -.82).astype(int)
        else:
            labels = np.asarray(labels)
        boundary = np.r_[True, (phases[1:] != phases[:-1]) | (labels[1:] != labels[:-1])
                         | (expert[1:] != expert[:-1]) | (np.diff(times) > 1.6/fps), True]
        changes = np.flatnonzero(boundary)
        for a,b in zip(changes[:-1],changes[1:]):
            if not expert[a] or (b-a)/fps < 2.:
                continue
            for candidate in jitter_windows(smooth,a,b,fps,cfg):
                findings.append(issue('盒附近往返抖动候选' if candidate['near'] else '局部往返抖动候选',
                    candidate['start'],candidate['end'],times,fps,
                    f"局部4秒窗口出现至少{candidate['turns']}次明显反转；排除小于1.5mm微动；盒附近按末端位置近似"))
            step = max(1, round(.3*fps))
            points = smooth[np.r_[np.arange(a,b,step),b-1]]
            kept = [points[0]]
            for p in points[1:]:
                if np.linalg.norm(p-kept[-1]) >= .0015:
                    kept.append(p)
            if len(kept)<5:
                continue
            points = np.asarray(kept)
            v = np.diff(points,axis=0);lengths=np.linalg.norm(v,axis=1)
            path_length=float(lengths.sum());net=float(np.linalg.norm(points[-1]-points[0]))
            extent=float(np.linalg.norm(np.ptp(points,axis=0)))
            cos=np.sum(v[:-1]*v[1:],axis=1)/(lengths[:-1]*lengths[1:])
            turns=int(np.sum(cos < -.3))
            if turns>=cfg.jitter_turns and path_length>=cfg.jitter_path_m and extent<=cfg.jitter_extent_m:
                near=float(np.median(np.linalg.norm(points[:,:2]-[.32,.14],axis=1))) < .22
                findings.append(issue('盒附近往返抖动候选' if near else '局部往返抖动候选',a,b,times,fps,
                    f'{turns}次明显方向反转；范围{extent*1000:.1f}mm；路径{path_length*1000:.1f}mm；盒附近按末端位置近似'))
            if net > .03:
                projection=v@((points[-1]-points[0])/net)
                backward=float(-projection[projection<0].sum())
                if backward>=cfg.reverse_distance_m and backward/path_length>=cfg.reverse_fraction:
                    findings.append(issue('反向移动较多',a,b,times,fps,
                        f'相对本阶段起终点方向累计反向{backward*1000:.1f}mm，占路径{backward/path_length:.1%}；必要退让也可能命中'))
            if path_length>=.15 and path_length-net>=.10 and path_length/max(net,.02)>=2.5:
                findings.append(issue('绕行或反复调整候选',a,b,times,fps,
                    f'路径{path_length*1000:.1f}mm、起终点距离{net*1000:.1f}mm；按相同阶段判断'))
    kind=meta.get('task',{}).get('kind',summary.get('episode_goal','basic'))
    target=summary.get('target_object',rows[0].get('target_object','unknown'))
    fragment=bool(summary.get('fragment_success',False) or meta.get('fragment_goal'))
    group=f'{folder.parent}|{target}|{kind}|{fragment}|{round(fps)}'
    return dict(dataset=str(folder.parent),episode=folder.name,path=str(folder),frames=len(rows),fps=fps,
        target=target,kind=kind,fragment=fragment,group=group,read_error='',
        duration_s=round(len(rows)/fps,3),simulation_span_s=round(float(t[-1]+1/fps),3),
        expert_s=round(float(np.sum(expert))/fps,3),
        intervention_s=round(float(np.sum(~expert))/fps,3),
        stationary_s=round(float(np.sum(still & expert))/fps,3),
        phase_durations={p:round(int(n)/fps,3) for p,n in Counter(phases[expert]).items()},
        cartesian_available=chain is not None,findings=findings,
        frames_sha256=hashlib.sha256(raw).hexdigest(),cameras=[c for c in CAMERAS if 'observation.images.'+c in rows[0]])


def duration_outliers(items,cfg):
    groups=defaultdict(list)
    for item in items:
        if not item['read_error']:groups[item['group']].append(item)
    for group in groups.values():
        if len(group)<cfg.phase_min_samples:
            continue
        for phase in ['__total__']+sorted({p for e in group for p in e['phase_durations']}):
            present=[e for e in group if phase=='__total__' or phase in e['phase_durations']]
            if len(present)<cfg.phase_min_samples:continue
            values=np.asarray([e['expert_s'] if phase=='__total__' else e['phase_durations'][phase] for e in present])
            median=float(np.median(values));mad=float(np.median(np.abs(values-median)))
            threshold=max(median*cfg.duration_ratio,median+cfg.duration_mad_factor*1.4826*mad,median+3.)
            for item,value in zip(present,values):
                if value>threshold:
                    item['findings'].append(dict(type='同类总耗时偏长' if phase=='__total__' else '同类阶段耗时偏长',
                        start_frame=0,end_frame_exclusive=item['frames'],start_s=0.,end_s=item['simulation_span_s'],
                        recorded_s=item['duration_s'],priority='review',
                        detail=f'{phase}: {value:.2f}s；同组中位数{median:.2f}s；筛查线{threshold:.2f}s；同组{len(present)}条'))


def clock_text(seconds):
    seconds=max(0.,seconds)
    return f'{int(seconds//3600):02}:{int(seconds%3600//60):02}:{seconds%60:06.3f}'


def write_csv(path,rows):
    if not rows:
        path.write_text('\ufeff无记录\n',encoding='utf-8');return
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        fields=list(rows[0]);w=csv.DictWriter(stream,fields);w.writeheader()
        for row in rows:
            w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def checklist(items):
    return [dict(序号=i+1,数据集=e['dataset'],Episode=e['episode'],物体=e['target'],类型=e['kind'],
        原轨迹秒=e['duration_s'],异常项='；'.join(dict.fromkeys(x['type'] for x in e['findings'])),
        异常区间及依据='；'.join(f"{x['start_s']:.2f}–{x['end_s']:.2f}s {x['type']}: {x['detail']}" for x in e['findings']),
        视频开始=clock_text(e['video_start_s']) if 'video_start_s' in e else '',
        视频结束=clock_text(e['video_end_s']) if 'video_end_s' in e else '',
        Frames路径=str(Path(e['path'])/'frames.jsonl'),人工结论='',是否重录='',备注='')
        for i,e in enumerate(items)]


@lru_cache(maxsize=12)
def font(size):
    for p in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
        if Path(p).exists():return ImageFont.truetype(p,size)
    return ImageFont.load_default()


def wrap(draw,text,f,max_width):
    lines=['']
    for char in str(text):
        if char=='\n' or draw.textlength(lines[-1]+char,font=f)>max_width:
            lines.append('' if char=='\n' else char)
        else:lines[-1]+=char
    return lines


def title_image(item,index,total,cfg):
    image=Image.new('RGB',(cfg.width,cfg.height),'#10202e');d=ImageDraw.Draw(image)
    d.text((30,30),f'人工复核候选 {index}/{total} · 后续轨迹2倍速',font=font(34),fill='white')
    lines=[Path(item['dataset']).name,item['episode']+' | '+item['target']+' | '+str(item['kind']),
        '；'.join(dict.fromkeys(x['type'] for x in item['findings'])),
        '候选不等于错误，请结合抓取、恢复、避障和放置目的判断。']
    y=110
    for text in lines:
        for line in wrap(d,text,font(27),cfg.width-60)[:3]:
            d.text((30,y),line,font=font(27),fill='#ffe197');y+=44
    return image


def render(items,out,cfg):
    partial=out/'异常轨迹汇总_2倍速.partial.mp4'
    movie=out/'异常轨迹汇总_2倍速.mp4'
    error_rows=[];frame_count=0;chapters=[]
    ffmpeg=['ffmpeg','-hide_banner','-loglevel','warning','-y','-f','rawvideo','-pix_fmt','rgb24',
        '-s',f'{cfg.width}x{cfg.height}','-r',str(cfg.video_fps),'-i','pipe:0','-an',
        '-c:v','libx264','-preset','veryfast','-crf','24','-threads',str(cfg.ffmpeg_threads),
        '-pix_fmt','yuv420p','-g',str(cfg.video_fps*2),
        '-movflags','+frag_keyframe+empty_moov+default_base_moof',str(partial)]
    with (out/'ffmpeg.log').open('w') as log:
        process=subprocess.Popen(ffmpeg,stdin=subprocess.PIPE,stderr=log)
        try:
            for index,item in enumerate(items,1):
                begin=frame_count/cfg.video_fps
                card=title_image(item,index,len(items),cfg).tobytes()
                for _ in range(round(cfg.title_s*cfg.video_fps)):
                    process.stdin.write(card);frame_count+=1
                item['video_start_s']=frame_count/cfg.video_fps
                path=Path(item['path'])
                try:
                    rows=[json.loads(l) for l in (path/'frames.jsonl').read_text().splitlines() if l.strip()]
                except Exception as exc:
                    rows=[];error_rows.append(dict(episode=str(path),frame=-1,camera='',error=str(exc)))
                cameras=[c for c in CAMERAS if rows and 'observation.images.'+c in rows[0]][:3]
                fps=item['fps'] or 10.
                output_count=math.ceil(len(rows)*cfg.video_fps/(fps*cfg.speed))
                # Original frame order at recorded fps, always2x. A timestamp
                # gap is displayed and reported, never silently repaired.
                previous=-1;image=None
                for output_index in range(output_count):
                    source_index=min(len(rows)-1,int(output_index*fps*cfg.speed/cfg.video_fps))
                    if source_index!=previous:
                        row=rows[source_index]
                        image=Image.new('RGB',(cfg.width,cfg.height),'#101820');d=ImageDraw.Draw(image)
                        d.text((18,8),f'#{index}/{len(items)} {Path(item["dataset"]).name}',font=font(23),fill='white')
                        d.text((18,43),f'{item["episode"]} | {item["target"]} | 2倍速 | 记录时间 {source_index/fps:.2f}s',font=font(23),fill='#aadfff')
                        panel_w=cfg.width//max(1,len(cameras))
                        for k,camera in enumerate(cameras):
                            x=k*panel_w
                            try:
                                relative=Path(row['observation.images.'+camera])
                                image_path=relative if relative.is_absolute() else path/relative
                                with Image.open(image_path) as source:
                                    resampling=getattr(Image,'Resampling',Image)
                                    panel=ImageOps.contain(source.convert('RGB'),(panel_w-8,360),resampling.BILINEAR)
                                image.paste(panel,(x+(panel_w-panel.width)//2,95+(360-panel.height)//2))
                            except Exception as exc:
                                d.text((x+12,180),'图像缺失/无法解码',font=font(23),fill='#ff7777')
                                error_rows.append(dict(episode=str(path),frame=source_index,camera=camera,error=str(exc)))
                            d.text((x+10,460),camera,font=font(20),fill='white')
                        active=[f for f in item['findings'] if f['start_frame']<=source_index<f['end_frame_exclusive']]
                        text='当前区间：'+'；'.join(dict.fromkeys(f['type'] for f in active)) if active else '当前区间：轨迹上下文（保留完整回合供判断）'
                        for line_no,line in enumerate(wrap(d,text,font(24),cfg.width-36)[:2]):
                            d.text((18,500+line_no*34),line,font=font(24),fill='#ffbf66' if active else '#a8c2cd')
                        previous=source_index
                    process.stdin.write(image.tobytes());frame_count+=1
                item['video_end_s']=frame_count/cfg.video_fps
                chapters.append((begin,item['video_end_s'],f'{index:04d} {Path(item["dataset"]).name} {item["episode"]}'))
                if index%10==0 or index==len(items):
                    write_csv(out/'异常候选清单.csv',checklist(items))
                    write_csv(out/'图像读取问题.csv',error_rows)
                    atomic_json(out/'status.json',dict(stage='rendering',completed_episodes=index,
                        total_candidates=len(items),video_seconds=frame_count/cfg.video_fps,
                        updated_at=dt.datetime.now().astimezone().isoformat()))
                    print(f'VIDEO {index}/{len(items)} output={clock_text(frame_count/cfg.video_fps)}',flush=True)
            process.stdin.close()
            if process.wait()!=0:raise RuntimeError('ffmpeg failed; inspect ffmpeg.log')
        except BaseException:
            if process.stdin and not process.stdin.closed:process.stdin.close()
            process.terminate();process.wait();raise
    metadata=out/'chapters.ffmetadata'
    with metadata.open('w') as stream:
        stream.write(';FFMETADATA1\n')
        for a,b,title in chapters:
            title=title.replace('\\','\\\\').replace('=','\\=').replace(';','\\;').replace('#','\\#').replace('\n',' ')
            stream.write(f'[CHAPTER]\nTIMEBASE=1/1000\nSTART={round(a*1000)}\nEND={round(b*1000)}\ntitle={title}\n')
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(partial),'-i',str(metadata),
        '-map','0:v:0','-map_chapters','1','-c','copy','-movflags','+faststart',str(movie)],check=True)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration:stream=codec_name,width,height',
                                             '-of','json',str(movie)],text=True))
    if abs(float(probe['format']['duration'])-frame_count/cfg.video_fps)>.15:
        raise RuntimeError('Final video duration differs from generated frame count')
    partial.unlink()
    write_csv(out/'图像读取问题.csv',error_rows)
    return dict(path=str(movie),duration_s=frame_count/cfg.video_fps,frames=frame_count,
                image_read_errors=len(error_rows),ffprobe=probe)


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--roots',nargs='+',default=json.loads(Path('/home/lp531/robot/dataset_trajectory_review/current_round_sources.json').read_text())['roots'],help='默认仅本轮新录制；可显式指定目录或frames.jsonl')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--urdf',default=DEFAULT_URDF,help='Z1六轴FK；文件不存在时只检查关节和时间')
    parser.add_argument('--thresholds',type=Path,help='JSON，字段见输出thresholds.json')
    parser.add_argument('--no-video',action='store_true',help='仅筛查，不生成视频')
    args=parser.parse_args();out=args.output.expanduser().absolute();out.mkdir(parents=True,exist_ok=True)
    cfg=Settings(**(read_json(args.thresholds,{}) if args.thresholds else {}))
    if cfg.speed!=2.0 or cfg.width%2 or cfg.height%2 or cfg.video_fps<=0 or cfg.hold_min_s<=0:
        raise ValueError('Require2x speed, positive thresholds/fps and even video dimensions')
    if (out/'completed.json').exists():raise RuntimeError('Completed output already exists; choose a new output directory')
    atomic_json(out/'thresholds.json',asdict(cfg))
    atomic_json(out/'request.json',dict(roots=args.roots,output=str(out),urdf=args.urdf,
        speed=2.,video_mode='full flagged episodes',input_mutation=False,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        started_at=dt.datetime.now().astimezone().isoformat()))
    shutil.copy2(Path(__file__),out/'review_source.py')
    discovery_errors=[]
    files,aliases=discover(args.roots,discovery_errors)
    write_csv(out/'目录读取问题.csv',discovery_errors)
    if not files:raise RuntimeError('No saved frames.jsonl files found')
    atomic_json(out/'input_snapshot.json',dict(frames_files=[str(p) for p in files],same_file_aliases=aliases,
        note='Snapshot at launch; includes historical and prepared training versions with frames.jsonl.'))
    print(f'DISCOVERED {len(files)} episodes across {len(set(p.parent.parent for p in files))} datasets',flush=True)
    chain=chain_from_urdf(args.urdf);items=[]
    for i,path in enumerate(files,1):
        try:item=analyze(path,chain,cfg)
        except Exception as exc:
            try:
                fallback=read_json(path.parent/'episode_summary.json',{})
                fallback_info=read_json(path.parent.parent/'dataset_info.json',{})
                fallback_fps=float(fallback.get('fps',fallback_info.get('fps',10.)))
                if not .1<=fallback_fps<=240:fallback_fps=10.
            except Exception:
                fallback={};fallback_fps=10.
            try:
                with path.open() as stream:line_count=sum(bool(line.strip()) for line in stream)
            except OSError:line_count=int(fallback.get('frame_count',0))
            duration=line_count/fallback_fps
            item=dict(dataset=str(path.parent.parent),episode=path.parent.name,path=str(path.parent),frames=line_count,fps=fallback_fps,
                target=fallback.get('target_object','unknown'),kind='unknown',fragment=False,group='',read_error=str(exc),duration_s=duration,
                simulation_span_s=duration,expert_s=0.,intervention_s=0.,stationary_s=0.,phase_durations={},
                cartesian_available=False,frames_sha256='',cameras=[],findings=[dict(type='数据无法正常分析',
                    start_frame=0,end_frame_exclusive=line_count,start_s=0.,end_s=duration,recorded_s=duration,priority='high',detail=str(exc))])
        items.append(item)
        if i%100==0 or i==len(files):
            atomic_json(out/'status.json',dict(stage='analyzing',completed=i,total=len(files)))
            print(f'ANALYZED {i}/{len(files)}',flush=True)
    duration_outliers(items,cfg)
    candidates=[e for e in items if e['findings']]
    candidates.sort(key=lambda e:(not any(f['priority']=='high' for f in e['findings']),e['dataset'],e['episode']))
    atomic_json(out/'all_metrics.json',items)
    write_csv(out/'所有样本指标.csv',[{k:v for k,v in e.items() if k!='findings'}|{'异常数':len(e['findings'])} for e in items])
    write_csv(out/'异常候选清单.csv',checklist(candidates))
    write_csv(out/'无法分析.csv',[dict(path=e['path'],error=e['read_error']) for e in items if e['read_error']])
    grouped=defaultdict(list)
    for e in items:grouped[e['dataset']].append(e)
    coverage=[dict(数据集=k,扫描条数=len(v),候选条数=sum(bool(e['findings']) for e in v),
                   无法分析=sum(bool(e['read_error']) for e in v),同源别名条数=0) for k,v in grouped.items()]
    indexed={str(Path(e['path'])/'frames.jsonl'):e for e in items}
    alias_groups=defaultdict(list)
    alias_report=[]
    candidate_ids={str(Path(e['path'])/'frames.jsonl'):i+1 for i,e in enumerate(candidates)}
    for alias in aliases:
        alias_groups[str(Path(alias['path']).parent.parent)].append(alias)
        alias_report.append(dict(路径=alias['path'],同一记录原路径=alias['same_file_as'],
            对应视频及清单序号=candidate_ids.get(alias['same_file_as'],''),
            说明='同一episode目录的路径别名，已复用检查，不重复拼接视频'))
    for name,group in alias_groups.items():
        row=next((c for c in coverage if c['数据集']==name),None)
        if row is None:
            row=dict(数据集=name,扫描条数=0,候选条数=0,无法分析=0,同源别名条数=0);coverage.append(row)
        row['同源别名条数']=len(group)
    write_csv(out/'同源路径别名.csv',alias_report)
    write_csv(out/'数据集覆盖.csv',coverage)
    atomic_json(out/'candidate_manifest.json',candidates)
    result=dict(scanned=len(items),datasets=len(coverage),same_episode_aliases=len(aliases),candidates=len(candidates),
        discovery_errors=len(discovery_errors),
        analysis_errors=sum(bool(e['read_error']) for e in items),input_files_modified=False,
        scope='All saved frames.jsonl under requested roots; historical and prepared versions included.',
        rules='Heuristic candidates for manual review. Necessary correction/avoidance can trigger direction rules.')
    if candidates and not args.no_video:
        if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):raise RuntimeError('ffmpeg/ffprobe required for video')
        result['video']=render(candidates,out,cfg)
        write_csv(out/'异常候选清单.csv',checklist(candidates))
        atomic_json(out/'candidate_manifest.json',candidates)
    (out/'汇总.md').write_text(f'''# 训练集轨迹人工复核

扫描{len(items)}条、{len(grouped)}个数据集，筛出{len(candidates)}条候选。完整视频按《异常候选清单.csv》序号排列，轨迹2倍速，片头单独显示0.8秒，含章节索引。清单保留人工结论、是否重录和备注空列。

检查长停顿、低进展、过长尾部、同数据集/同物体/同类任务的总时长和阶段时长、反向移动、往返抖动、绕行及采样时间间断。已知人为扰动排除；夹爪动作和机械臂动作分开判断。阈值见thresholds.json，可修改后在新输出目录重跑。

所有结果只是候选，不自动删除、改写、剪裁、修改训练掩码或决定重录。必要恢复和避障可能命中方向规则；没有被标记也不等于轨迹已通过人工验收。盒附近判断按末端位置近似，不能替代物体真值。

范围包括输入根目录下的历史录制和预处理版本；只合并同一真实文件的路径别名，不把不同版本静默当成重复数据。未完成的.recording目录和诊断目录跳过。范围快照见input_snapshot.json。

视频按原始帧顺序和记录fps播放为2倍速；缺采样的时间段不补造动作，真实simulation_time间断单列清单。缺失图像用警示占位并写入《图像读取问题.csv》。无法解析的轨迹只能显示片头，详细错误见《无法分析.csv》。
''',encoding='utf-8')
    atomic_json(out/'completed.json',dict(result,completed_at=dt.datetime.now().astimezone().isoformat()))
    atomic_json(out/'status.json',dict(stage='complete',**result))
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        print(traceback.format_exc(),flush=True)
        if '--output' in sys.argv:
            output=Path(sys.argv[sys.argv.index('--output')+1]).expanduser().absolute()
            atomic_json(output/'status.json',dict(stage='error',error=str(exc),
                updated_at=dt.datetime.now().astimezone().isoformat()))
        # The log and partial fragmented video are deliberately retained.
        raise
