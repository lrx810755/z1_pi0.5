"""Concatenate exactly the 25 grasp-failed evaluations at 5x, with episode chapters."""
import json,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent
report=json.loads((R/'final_box_results.json').read_text())
rows=[e for e in report['episodes'] if e['legacy_failure_category']=='grasp_failed']
assert report['completed'] and report['total']==30 and len(rows)==25
lines=[];filters=[];metadata=[';FFMETADATA1'];chapters=[];elapsed=0.
for order,e in enumerate(rows,1):
 source=Path(e['source_result']).parent/'videos/global_third_person.mp4'
 probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,r_frame_rate:format=duration','-of','json',str(source)],text=True))
 stream=probe['streams'][0];assert (stream['width'],stream['height'],stream['r_frame_rate'])==(640,480,'10/1')
 duration=float(probe['format']['duration']);assert duration>0
 lines.append("file '"+str(source).replace("'","'\\''")+"'")
 label=f'GRASP FAILED {order:02d}/25 | Eval {e["episode"]:02d}/30 | 5x'
 filters.append(f"drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='{label}':fontsize=18:fontcolor=white:box=1:boxcolor=black@0.8:boxborderw=6:x=10:y=12:enable='gte(t,{elapsed:.6f})*lt(t,{elapsed+duration:.6f})'")
 metadata+=['[CHAPTER]','TIMEBASE=1/1000',f'START={round(elapsed*200)}',f'END={round((elapsed+duration)*200)}',f'title=Grasp failure {order:02d} - Eval {e["episode"]:02d} - seed {e["seed"]}']
 chapters.append(dict(order=order,evaluation_episode=e['episode'],seed=e['seed'],start_s=elapsed/5,duration_s=duration/5,source=str(source)))
 elapsed+=duration
(R/'grasp_failed25_concat.txt').write_text('\n'.join(lines)+'\n')
(R/'grasp_failed25_filters.ffscript').write_text(','.join(filters+['setpts=(PTS-STARTPTS)/5']))
(R/'grasp_failed25_chapters.ffmeta').write_text('\n'.join(metadata)+'\n')
output=R/'cup_grasp_failed25_5x.mp4'
with (R/'grasp_failed25_encode.log').open('w') as log:
 subprocess.run(['ffmpeg','-hide_banner','-y','-f','concat','-safe','0','-i',str(R/'grasp_failed25_concat.txt'),'-i',str(R/'grasp_failed25_chapters.ffmeta'),'-map','0:v:0','-map_metadata','1','-map_chapters','1','-filter_script:v',str(R/'grasp_failed25_filters.ffscript'),'-r','30','-c:v','libx264','-preset','veryfast','-crf','20','-threads','4','-pix_fmt','yuv420p','-an','-movflags','+faststart',str(output)],stdout=log,stderr=log,check=True)
probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_chapters','-show_entries','format=duration','-of','json',str(output)],text=True))
assert len(probe['chapters'])==25 and abs(float(probe['format']['duration'])-elapsed/5)<.25
result=dict(file=str(output),speed=5,trials=25,duration_s=float(probe['format']['duration']),chapters=chapters,verified=True)
(R/'grasp_failed25_review_manifest.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='chapters'},indent=2))
