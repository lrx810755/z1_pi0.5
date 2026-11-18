"""Render a pinned random20 raw demonstration selection at10x, three views."""
import collections
import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys

R = Path(sys.argv[1])
selection = json.loads((R/'selection.json').read_text())
parts = R/'parts'; parts.mkdir(exist_ok=True)
font = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def probe(path):
    return json.loads(subprocess.check_output([
        'ffprobe','-v','error','-select_streams','v:0','-show_entries',
        'stream=width,height,nb_frames,r_frame_rate:format=duration,size',
        '-of','json',str(path)], text=True))


def make(item):
    number, record = item
    root = Path(record['source_path']); frames = record['frames']
    rows = [json.loads(line) for line in (root/'frames.jsonl').read_text().splitlines()]
    assert len(rows) == frames
    dest = parts/f'{number:02d}.mp4'
    cmd = ['ffmpeg','-y','-v','error','-filter_complex_threads','2']
    cameras = [('global','GLOBAL'),('global_right','RIGHT'),('wrist','WRIST')]
    for cam, _ in cameras:
        for i,row in enumerate(rows):
            expected = root/'images'/cam/f'frame_{i:06d}.jpg'
            assert root/row['observation.images.'+cam] == expected and expected.is_file()
        cmd += ['-threads','2','-framerate','100','-start_number','0',
                '-i',str(root/'images'/cam/'frame_%06d.jpg')]
    filters=[]
    for i, (_,label) in enumerate(cameras):
        text=f"Demo {number:02d}/20 | {record['source'].upper()} {root.name} | {label} | 10x"
        filters.append(f"[{i}:v]pad=640:480:0:(oh-ih)/2:black,setsar=1,drawtext=fontfile={font}:text='{text}':x=10:y=12:fontsize=17:fontcolor=white:box=1:boxcolor=black@0.65[v{i}]")
    filters.append('[v0][v1][v2]hstack=inputs=3:shortest=1[out]')
    cmd += ['-filter_complex',';'.join(filters),'-map','[out]','-frames:v',str(frames),
            '-an','-r','100','-c:v','libx264','-preset','veryfast','-crf','22',
            '-threads','3','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)]
    subprocess.run(cmd,check=True)
    info=probe(dest)
    assert int(info['streams'][0]['nb_frames']) == frames
    assert abs(float(info['format']['duration'])-frames/100)<.025
    print(f'Completed demo {number:02d}/20: {record["source"]}/{root.name}',flush=True)
    return dest,info


with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    clips=list(pool.map(make,enumerate(selection['episodes'],1)))
listing=parts/'concat.txt'
listing.write_text(''.join(f"file '{path}'\n" for path,_ in clips))
out=Path(selection['output'])
subprocess.run(['ffmpeg','-y','-v','error','-f','concat','-safe','0','-i',str(listing),
                '-c','copy','-movflags','+faststart',str(out)],check=True)
info=probe(out);expected=sum(x['frames'] for x in selection['episodes'])
assert info['streams'][0]['width']==1920 and info['streams'][0]['height']==480
assert int(info['streams'][0]['nb_frames'])==expected
assert abs(float(info['format']['duration'])-expected/100)<.05
verified=dict(verified=True,total=20,speed=10,source_fps=10,output_fps=100,
              random_seed=selection['random_seed'],output=str(out),
              source_counts=dict(collections.Counter(x['source'] for x in selection['episodes'])),
              all_source_frames_preserved=True,total_frames=expected,**info)
(R/'video_verified.json').write_text(json.dumps(verified,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(verified,ensure_ascii=False),flush=True)
