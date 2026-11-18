"""Peach ACT execution window 30, grasp failures, three views at 5x."""
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import csv
import hashlib
import json
import subprocess
import time

ROOT = Path('/home/lp531/smolvla_eval/act_peach_final_execute30_30_20261118/act_sim_eval_20261118_101625')
WORK = Path(__file__).resolve().parent
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
CAMERAS = ['global_third_person', 'global_right', 'wrist']
NAMES = {'apple': '苹果', 'peach': '桃子', 'cup': '杯子', 'beverage_bottle': '瓶子'}
ORDER = {'apple': 0, 'peach': 1, 'cup': 2, 'beverage_bottle': 3}
CATEGORIES = {'grasp_failed': ('抓取失败', 27)}


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
    text1.write_text(f'桃子ACT_执行30步  |  {label}  |  片段 {number:02d}/{total:02d}  |  5倍速')
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
    dest = WORK / 'exports' / f'桃子ACT_执行30步_{label}_{len(clips)}次_三视角_5倍速.mp4'
    listing = WORK / ('concat_' + category + '.txt')
    listing.write_text(''.join(f"file '{clip['file']}'\n" for clip in clips))
    metadata = [';FFMETADATA1', f'title=桃子ACT_执行30步 {label} 三视角 5倍速',
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
    selected = []
    for path in sorted((ROOT / 'episodes').glob('*/result.json')):
        result = json.loads(path.read_text())
        if result['failure_category'] == 'grasp_failed':
            assert result['target_object'] == 'peach' and result['execute_steps'] == 30 and not result['success']
            selected.append((path, result))
    selected.sort(key=lambda item: item[1]['episode_number'])
    assert len(selected) == 27 and len({r['seed'] for _, r in selected}) == 27
    clips = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for clip in pool.map(render_clip, [(i, 27, p, r) for i, (p, r) in enumerate(selected, 1)]):
            clips.append(clip)
            print(f"Encoded {len(clips)}/27 (trial {clip['episode']})", flush=True)
    reel = concatenate('grasp_failed', clips)
    original = Path(reel['output'])
    destination = ROOT.parents[2] / original.name
    assert destination.parent == Path('/home/lp531')
    original.replace(destination)
    reel['output'] = str(destination)
    write('verified.json', reel)
    print(json.dumps({k: v for k, v in reel.items() if k != 'clips'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
