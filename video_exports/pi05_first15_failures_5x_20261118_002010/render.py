"""First 15-step evaluation: all recorded failures, category reels, full-frame 5x."""
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import csv
import hashlib
import json
import subprocess
import time

ROOT = Path('/home/lp531/smolvla_eval/pi05_final_four_objects_50each_20261117_145347/pi05_sim_eval_20261117_150326')
WORK = Path(__file__).resolve().parent
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
CAMERAS = ['global_third_person', 'global_right', 'wrist']
NAMES = {'apple': '苹果', 'peach': '桃子', 'cup': '杯子', 'beverage_bottle': '瓶子'}
ORDER = {'apple': 0, 'peach': 1, 'cup': 2, 'beverage_bottle': 3}
CATEGORIES = {
    'grasp_failed': ('抓取失败', 79),
    'finger_table_contact': ('手指触桌', 76),
    'bottle_tipped': ('瓶子倾倒', 14),
    'transport_or_release_failed': ('搬运或释放失败', 3),
    'placement_failed': ('放置失败', 2),
}


def probe(path, chapters=False):
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
           'stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration,codec_name:format=duration,size', '-of', 'json']
    if chapters:
        cmd += ['-show_chapters']
    return json.loads(subprocess.check_output(cmd + [str(path)], text=True))


def write(name, data):
    dest = WORK / name
    tmp = dest.with_suffix(dest.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(dest)


def render_clip(item):
    number, total, source, result = item
    category = result['failure_category']
    label = CATEGORIES[category][0]
    directory = WORK / 'clips' / category
    directory.mkdir(exist_ok=True)
    name = f'{number:03d}_{result["target_object"]}_{result["episode_number"]:03d}'
    dest = directory / (name + '.mp4')
    source_dir = source.parent / 'videos'
    metadata = {cam: probe(source_dir / (cam + '.mp4'))['streams'][0] for cam in CAMERAS}
    lengths = {int(m['nb_frames']) for m in metadata.values()}
    assert len(lengths) == 1 and all(m['r_frame_rate'] == '10/1' for m in metadata.values()), source
    assert [(metadata[c]['width'], metadata[c]['height']) for c in CAMERAS] == [(640,480),(640,480),(640,360)]
    frames = lengths.pop()
    assert frames > 0
    text1 = directory / (name + '_title.txt')
    text2 = directory / (name + '_detail.txt')
    text1.write_text(f'首次15步评估  |  {label}  |  片段 {number:02d}/{total:02d}  |  5倍速')
    text2.write_text(f'物体：{NAMES[result["target_object"]]}  |  原第 {result["episode_number"]:02d} 次  |  seed {result["seed"]}')
    cmd = ['ffmpeg', '-nostdin', '-y', '-v', 'error', '-xerror', '-filter_complex_threads', '2']
    for camera in CAMERAS:
        cmd += ['-threads', '2', '-i', str(source_dir / (camera + '.mp4'))]
    filters = [f'[{i}:v]settb=AVTB,setpts=(PTS-STARTPTS)/5,setsar=1[v{i}]' for i in range(3)]
    filters += ['[v0][v1]hstack=inputs=2:shortest=1[top]',
                '[v2]pad=1280:360:320:0:black[bottom]',
                '[top][bottom]vstack=inputs=2:shortest=1,pad=1280:904:0:64:black[panel]']
    filters.append(
        f'[panel]drawtext=fontfile={FONT}:textfile={text1}:expansion=none:x=18:y=5:fontsize=24:fontcolor=white,'
        f'drawtext=fontfile={FONT}:textfile={text2}:expansion=none:x=18:y=35:fontsize=20:fontcolor=white,'
        f"drawtext=fontfile={FONT}:text='外部左':x=12:y=77:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.6,"
        f"drawtext=fontfile={FONT}:text='外部右':x=652:y=77:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.6,"
        f"drawtext=fontfile={FONT}:text='腕部':x=334:y=557:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.6,"
        f"drawtext=fontfile={FONT}:text='5倍速':x=32:y=675:fontsize=34:fontcolor=white[out]")
    cmd += ['-filter_complex', ';'.join(filters), '-map', '[out]', '-an', '-r', '50', '-vsync', 'cfr',
            '-frames:v', str(frames), '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '21',
            '-threads', '4', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(dest)]
    cached = False
    if dest.is_file():
        try:
            old = probe(dest)['streams'][0]
            cached = (old['width'], old['height'], old['avg_frame_rate'], int(old['nb_frames'])) == (1280, 904, '50/1', frames)
        except (subprocess.CalledProcessError, KeyError, ValueError):
            cached = False
    if not cached:
        with (directory / (name + '.log')).open('w') as log:
            subprocess.run(cmd, stdout=log, stderr=log, check=True)
    info = probe(dest); stream = info['streams'][0]
    assert stream['width'] == 1280 and stream['height'] == 904 and stream['avg_frame_rate'] == '50/1'
    assert int(stream['nb_frames']) == frames
    duration = float(info['format']['duration'])
    assert abs(duration - frames / 50) < .025
    return dict(order=number, category=category, object=result['target_object'], episode=result['episode_number'],
                seed=result['seed'], source_result=str(source), source_result_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                source_videos={c: str(source_dir/(c+'.mp4')) for c in CAMERAS},
                file=str(dest), frames=frames, source_duration_s=frames/10, duration_s=duration)


def concatenate(category, clips):
    label = CATEGORIES[category][0]
    dest = WORK / 'exports' / f'首次15步评估_{label}_{len(clips)}次_三视角_5倍速.mp4'
    listing = WORK / ('concat_' + category + '.txt')
    listing.write_text(''.join(f"file '{clip['file']}'\n" for clip in clips))
    metadata = [';FFMETADATA1', f'title=首次15步评估 {label} 三视角 5倍速',
                'comment=Recorded video frames at10fps played at50fps; full views; no source frames removed']
    cursor = 0
    for clip in clips:
        start = cursor; cursor += clip['frames']
        metadata += ['[CHAPTER]', 'TIMEBASE=1/50', f'START={start}', f'END={cursor}',
                     f'title={NAMES[clip["object"]]} 第{clip["episode"]}次 seed{clip["seed"]}']
        clip.update(start_s=start/50, end_s=cursor/50)
    chapters = WORK / ('chapters_' + category + '.ffmeta')
    chapters.write_text('\n'.join(metadata) + '\n')
    building = dest.with_name(dest.stem + '.building.mp4')
    subprocess.run(['ffmpeg','-nostdin','-y','-v','error','-f','concat','-safe','0','-i',str(listing),
                    '-i',str(chapters),'-map','0:v:0','-map_metadata','1','-map_chapters','1','-c','copy',
                    '-movflags','+faststart',str(building)], check=True)
    info = probe(building, chapters=True); stream = info['streams'][0]
    assert stream['width']==1280 and stream['height']==904 and stream['avg_frame_rate']=='50/1'
    assert int(stream['nb_frames']) == cursor and abs(float(info['format']['duration'])-cursor/50)<.08
    assert len(info['chapters']) == len(clips)
    # Full decode of the final reel verifies playable packets across concatenation boundaries.
    subprocess.run(['ffmpeg','-nostdin','-v','error','-xerror','-threads','2','-i',str(building),
                    '-map','0:v:0','-f','null','-'], check=True)
    building.replace(dest)
    return dict(category=category,label=label,output=str(dest),episodes=len(clips),speed=5,
                source_fps=10,output_fps=50,width=1280,height=904,frames=cursor,duration_s=cursor/50,
                chapters=len(clips),source_frames_preserved=True,full_decode_verified=True,
                size_bytes=dest.stat().st_size,clips=clips)


def main():
    selected=defaultdict(list);unfinished=[]
    for ep in (ROOT/'episodes').iterdir():
        if not ep.is_dir():continue
        source=ep/'result.json'
        if not source.exists():unfinished.append(str(ep));continue
        result=json.loads(source.read_text())
        assert result['execute_steps']==15
        if result['success']:continue
        assert result['failure_category'] in CATEGORIES
        selected[result['failure_category']].append((source,result))
    assert sum(map(len,selected.values()))==174
    for category,(label,expected) in CATEGORIES.items():
        selected[category].sort(key=lambda item:(ORDER[item[1]['target_object']],item[1]['episode_number']))
        assert len(selected[category])==expected
    write('source_manifest.json',dict(source=str(ROOT),selected_failures=174,excluded_incomplete=unfinished,
         categories={c:[dict(source=str(p),object=d['target_object'],episode=d['episode_number'],seed=d['seed']) for p,d in items] for c,items in selected.items()}))
    started=time.monotonic();finished=[];completed=0
    # Small categories finish first, while retaining source order within every category.
    for category in sorted(CATEGORIES,key=lambda c:CATEGORIES[c][1]):
        items=selected[category];clips=[]
        write('progress.json',dict(stage='rendering',category=category,label=CATEGORIES[category][0],completed_episodes=completed,total_episodes=174,finished_categories=[r['category'] for r in finished]))
        with ThreadPoolExecutor(max_workers=3) as pool:
            for clip in pool.map(render_clip,[(i,len(items),p,d) for i,(p,d) in enumerate(items,1)]):
                clips.append(clip);completed+=1
                write('progress.json',dict(stage='rendering',category=category,label=CATEGORIES[category][0],completed_episodes=completed,total_episodes=174,elapsed_s=time.monotonic()-started,finished_categories=[r['category'] for r in finished]))
                print(json.dumps(dict(completed=completed,total=174,category=category,object=clip['object'],episode=clip['episode']),ensure_ascii=False),flush=True)
        reel=concatenate(category,clips);finished.append(reel);write(category+'_verified.json',reel)
        print(json.dumps(dict(category_finished=category,episodes=len(clips),duration_s=reel['duration_s'],output=reel['output']),ensure_ascii=False),flush=True)
    assert completed==174 and sum(r['episodes'] for r in finished)==174
    rows=[]
    for reel in finished:
        for clip in reel['clips']:rows.append(dict(category=reel['label'],output=reel['output'],object=NAMES[clip['object']],episode=clip['episode'],seed=clip['seed'],start_s=clip['start_s'],end_s=clip['end_s'],source_result=clip['source_result']))
    with (WORK/'视频时间索引.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    brief=[{k:v for k,v in reel.items() if k!='clips'} for reel in finished]
    write('completed.json',dict(passed=True,source=str(ROOT),episodes=174,categories=5,speed=5,source_fps=10,output_fps=50,
         all_source_frames_preserved=True,excluded_incomplete=unfinished,outputs=brief,elapsed_s=time.monotonic()-started))
    lines=['# 首次15步评估：全部失败分类视频','', '174条完整失败，按程序记录原因分成5个合集。每条三视角同步并列排布，无裁剪；原始10fps的每帧均保留，以50fps播放，实现精确5倍速。原始视频未改动。','',
           '顺序：同一分类内按苹果、桃子、杯子、瓶子，再按原试验轮次排序。画面标注分类、物体、轮次、seed及片段序号，MP4含逐条章节。','',
           '这里的速度基于保存的视频帧率；不包含原运行时没有录入视频的模型推理等待，也不代表完整墙钟时间除以五。','',
           '| 分类 | 失败条数 | 时长 | 视频 |','|---|---:|---:|---|']
    for reel in finished:
        sec=round(reel['duration_s']);lines.append(f'| {reel["label"]} | {reel["episodes"]} | {sec//60}:{sec%60:02d} | [{Path(reel["output"]).name}]({reel["output"]}) |')
    lines+=['','[逐条时间索引CSV](视频时间索引.csv)。手动停止时没有完整result的一条瓶子轨迹未计入失败合集。分类沿用原评估器标签，不把“抓取失败”进一步虚构成具体机械根因。','']
    (WORK/'视频说明.md').write_text('\n'.join(lines))
    write('progress.json',dict(stage='completed',completed_episodes=174,total_episodes=174,categories=5,elapsed_s=time.monotonic()-started))
    print(json.dumps(dict(done=True,episodes=174,categories=5,elapsed_s=time.monotonic()-started),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
