import json,subprocess,concurrent.futures,time
from pathlib import Path
R=Path(__file__).parent
SOURCE=Path('/home/lp531/smolvla_eval/act_gentle_15vs8_repaired_20261015_133937/steps15/act_sim_eval_20261015_133942/episodes')
OUT=Path('/home/lp531/瓶子ACT_原30次评估_三视角_3倍速.mp4')
VIEWS=['global_third_person','global_right','wrist']
font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

def probe(p):
 d=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=width,height,r_frame_rate,nb_frames,duration:format=duration','-of','json',str(p)]));s=d['streams'][0];return dict(width=s['width'],height=s['height'],fps=s['r_frame_rate'],frames=s.get('nb_frames'),duration_s=float(d['format']['duration']))

def encode(row):
 i=row['trial'];target=R/f'trial_{i:02d}.mp4';filters=[]
 for k in range(3):filters.append(f'[{k}:v]setpts=(PTS-STARTPTS)/3,fps=30,scale=640:480:force_original_aspect_ratio=decrease,pad=640:480:(ow-iw)/2:(oh-ih)/2,setsar=1,tpad=stop_mode=clone:stop_duration=1[v{k}]')
 filters.append('[v0][v1][v2]hstack=inputs=3:shortest=1[stack]')
 labels=[f'Trial {i:02d}/30 | GLOBAL | 3x','GLOBAL RIGHT','WRIST']
 chain='[stack]'+','.join(f"drawtext=fontfile={font}:text='{label}':x={k*640+12}:y=10:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.65:boxborderw=5" for k,label in enumerate(labels))+'[out]';filters.append(chain)
 cmd=['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-y','-filter_complex_threads','1']
 for name in VIEWS:cmd+=['-i',row['videos'][name]['path']]
 cmd+=['-filter_complex',';'.join(filters),'-map','[out]','-an','-t',str(row['output_duration_s']),'-c:v','libx264','-preset','veryfast','-crf','20','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(target)]
 result=subprocess.run(cmd,capture_output=True,text=True)
 if result.returncode:raise RuntimeError(f'Trial {i}: '+result.stderr)
 actual=probe(target);assert abs(actual['duration_s']-row['output_duration_s'])<.10,(i,actual,row['output_duration_s']);return i,target,actual

if __name__=='__main__':
 assert not OUT.exists(),OUT
 folders=sorted(SOURCE.glob('beverage_bottle_*'));assert len(folders)==30
 rows=[]
 for i,d in enumerate(folders,1):
  assert int(d.name.split('_')[2])==i,d
  videos={name:dict(path=str(d/'videos'/(name+'.mp4')),**probe(d/'videos'/(name+'.mp4'))) for name in VIEWS}
  durations=[v['duration_s'] for v in videos.values()];assert max(durations)-min(durations)<1,(i,durations)
  rows.append(dict(trial=i,episode=str(d),videos=videos,output_duration_s=max(durations)/3))
 manifest=dict(source=str(SOURCE),cohort='ACT gentle sampling, execute15 actions, 16/30 final-position-and-release successes',speed=3,views=VIEWS,output=str(OUT),trials=rows)
 (R/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print('Verified 30 trials, 90 source videos; expected output seconds',sum(a['output_duration_s'] for a in rows),flush=True)
 results={}
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  for f in concurrent.futures.as_completed([pool.submit(encode,row) for row in rows]):
   i,target,actual=f.result();results[i]=dict(path=str(target),**actual);print('Encoded',i,'completed',len(results),'/30',flush=True)
 concat=R/'concat.txt';concat.write_text(''.join("file '"+str(R/f'trial_{i:02d}.mp4')+"'\n" for i in range(1,31)))
 temporary=R/'combined.mp4';subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-y','-f','concat','-safe','0','-i',str(concat),'-c','copy','-movflags','+faststart',str(temporary)],check=True)
 out_probe=probe(temporary);expected=sum(r['duration_s'] for r in results.values());assert abs(out_probe['duration_s']-expected)<.15,(out_probe,expected);temporary.replace(OUT)
 (R/'encoded_manifest.json').write_text(json.dumps(dict(output=str(OUT),metadata=out_probe,clips=results,source_manifest=str(R/'source_manifest.json')),ensure_ascii=False,indent=2));print('DONE',str(OUT),out_probe,flush=True)
