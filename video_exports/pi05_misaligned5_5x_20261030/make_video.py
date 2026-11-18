"""Three synchronized recorded views;5 selected misalignment failures,5x source playback."""
import concurrent.futures
import json
from pathlib import Path
import subprocess

R=Path('/home/lp531/smolvla_eval/pi05_cup_recovery513_final30k_steps15_paired30_20261030')
W=Path(__file__).resolve().parent
OUT=Path('/home/lp531/杯子PI05_15步_5次未对准失败_三视角_5倍速.mp4')
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
CAMERAS=['global_third_person','global_right','wrist']


def probe(path):
    return json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries',
        'format=duration,size:stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration',
        '-of','json',str(path)],text=True))


def make(item):
    number,entry=item
    episode=entry['episode'];source=Path(entry['source_result']).parent/'videos'
    metadata={cam:probe(source/f'{cam}.mp4')['streams'][0] for cam in CAMERAS}
    frames={int(v['nb_frames']) for v in metadata.values()}
    assert len(frames)==1 and all(v['r_frame_rate']=='10/1' for v in metadata.values())
    n=frames.pop()
    title=W/f'trial_{episode:02d}_title.txt'
    title.write_text(f'Failure {number:02d}/5  |  Original trial {episode:02d}  |  5x playback')
    dest=W/f'trial_{episode:02d}_5x.mp4'
    cmd=['ffmpeg','-y','-v','error','-filter_complex_threads','2']
    for camera in CAMERAS:
        cmd+=['-threads','2','-i',str(source/f'{camera}.mp4')]
    filters=[]
    for i in range(3):
        filters.append(f'[{i}:v]settb=AVTB,setpts=(PTS-STARTPTS)/5,setsar=1[v{i}]')
    filters += ['[v0][v1]hstack=inputs=2:shortest=1[top]',
                '[v2]pad=1280:360:320:0:black[bottom]',
                '[top][bottom]vstack=inputs=2:shortest=1,pad=1280:904:0:64:black[panel]']
    filters.append(f"[panel]drawtext=fontfile={FONT}:textfile={title}:x=20:y=18:fontsize=27:fontcolor=white,"
        f"drawtext=fontfile={FONT}:text='External left':x=12:y=78:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.6,"
        f"drawtext=fontfile={FONT}:text='External right':x=652:y=78:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.6,"
        f"drawtext=fontfile={FONT}:text='Wrist':x=334:y=558:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.6,"
        f"drawtext=fontfile={FONT}:text='Trial {episode:02d}':x=30:y=620:fontsize=30:fontcolor=white,"
        f"drawtext=fontfile={FONT}:text='5x':x=30:y=670:fontsize=46:fontcolor=white[out]")
    # Limit CFR output to the source frame count: FFmpeg otherwise pads the
    # shortened last packet duration with four duplicate frames at50fps.
    cmd+=['-filter_complex',';'.join(filters),'-map','[out]','-an','-r','50','-vsync','cfr','-frames:v',str(n),
          '-c:v','libx264','-preset','veryfast','-crf','21','-threads','4','-pix_fmt','yuv420p',
          '-movflags','+faststart',str(dest)]
    subprocess.run(cmd,check=True)
    result=probe(dest);stream=result['streams'][0]
    assert stream['width']==1280 and stream['height']==904 and stream['avg_frame_rate']=='50/1'
    assert int(stream['nb_frames'])==n,(episode,n,stream['nb_frames'])
    duration=float(result['format']['duration'])
    assert abs(duration-n/50)<.025
    item=dict(episode=episode,source_result=entry['source_result'],file=str(dest),frames=n,
              duration_s=duration,source_duration_s=n/10,speed=5,camera_metadata=metadata)
    print(json.dumps(dict(completed=number,episode=episode,duration_s=duration)),flush=True)
    return item


def main():
    ids=json.loads((R/'failure_analysis_20261030/mechanism_summary.json').read_text())['no_contact_no_major_preclose_tilt']
    assert ids==[2,12,18,26,28]
    rows={e['episode']:e for e in json.loads((R/'completed.json').read_text())['episodes']}
    chosen=[rows[i] for i in ids]
    assert len(chosen)==5 and all(not row['success'] for row in chosen)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        clips=list(pool.map(make,enumerate(chosen,1)))
    concat=W/'concat.txt'
    concat.write_text(''.join(f"file '{c['file']}'\n" for c in clips))
    cursor=0
    metadata=[';FFMETADATA1','title=PI05 cup -5 misalignment failures -5x','comment=15-step execution; 3 synchronized recorded views; 5x source-video speed']
    for clip in clips:
        start=round(cursor*1000);cursor+=clip['duration_s'];end=round(cursor*1000)
        metadata+=['[CHAPTER]','TIMEBASE=1/1000',f'START={start}',f'END={end}',f'title=Original trial {clip["episode"]:02d}']
        clip.update(start_s=start/1000,end_s=end/1000)
    chapter=W/'chapters.txt';chapter.write_text('\n'.join(metadata)+'\n')
    building=OUT.with_name(OUT.stem+'.building.mp4')
    subprocess.run(['ffmpeg','-y','-v','error','-f','concat','-safe','0','-i',str(concat),
        '-i',str(chapter),'-map','0:v','-map_metadata','1','-map_chapters','1','-c','copy',
        '-movflags','+faststart',str(building)],check=True)
    info=probe(building)
    assert abs(float(info['format']['duration'])-cursor)<.08
    assert int(info['streams'][0]['nb_frames'])==sum(x['frames'] for x in clips)
    building.replace(OUT)
    verified=dict(passed=True,output=str(OUT),episodes=ids,speed=5,views=3,
        duration_s=float(info['format']['duration']),width=1280,height=904,fps=50,
        source_frames_preserved=True,clips=clips)
    (W/'verified.json').write_text(json.dumps(verified,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in verified.items() if k!='clips'},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
