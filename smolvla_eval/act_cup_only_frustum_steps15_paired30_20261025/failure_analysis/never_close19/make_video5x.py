import concurrent.futures,json,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent
rows=json.loads((R/'metrics.json').read_text())
parts=R/'video5x_parts';parts.mkdir(exist_ok=True)
font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
def make(row):
 n=row['episode'];dest=parts/f'{n:02d}.mp4';src=Path(row['path'])/'videos'
 cmd=['ffmpeg','-y','-v','error','-filter_complex_threads','2']
 for name in ['global_third_person.mp4','global_right.mp4','wrist.mp4']:cmd+=['-i',str(src/name)]
 filters=[]
 for i,cam in enumerate(['GLOBAL','RIGHT','WRIST']):
  filters.append(f"[{i}:v]setpts=(PTS-STARTPTS)/5,fps=50,pad=640:480:0:(oh-ih)/2:black,setsar=1,drawtext=fontfile={font}:text='Trial {n:02d} | {cam} | 5x':x=12:y=12:fontsize=23:fontcolor=white:box=1:boxcolor=black@0.65[v{i}]")
 filters.append('[v0][v1][v2]hstack=inputs=3:shortest=1[out]')
 cmd+=['-filter_complex',';'.join(filters),'-map','[out]','-an','-c:v','libx264','-preset','veryfast','-crf','22','-threads','3','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)]
 subprocess.run(cmd,check=True);print(f'Completed {n:02d}',flush=True)
 return dest
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:result=list(pool.map(make,rows))
listing=parts/'concat.txt';listing.write_text(''.join(f"file '{p}'\n" for p in result))
out=Path('/home/lp531/杯子ACT_19次未闭爪失败_三视角_5倍速.mp4')
subprocess.run(['ffmpeg','-y','-v','error','-f','concat','-safe','0','-i',str(listing),'-c','copy','-movflags','+faststart',str(out)],check=True)
probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration,size:stream=width,height,r_frame_rate','-of','json',str(out)],text=True))
(R/'video5x_verified.json').write_text(json.dumps(dict(output=str(out),episodes=[x['episode'] for x in rows],speed=5,**probe),indent=2))
print(json.dumps(dict(output=str(out),**probe),ensure_ascii=False),flush=True)
