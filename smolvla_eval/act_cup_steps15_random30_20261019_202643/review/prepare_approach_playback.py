import json,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent
metrics={m['episode']:m for m in json.loads((R/'metrics.json').read_text())}
ids=[1,3,4,7,8,10,11,14,16,17,20,22,24,26,29]
playlist=[];filters=[];manifest=[];chapters=[';FFMETADATA1'];elapsed=0.;frames=0
for index,ep in enumerate(ids,1):
 p=Path(metrics[ep]['path'])/'videos/global_third_person.mp4'
 assert p.is_file(),p
 probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=width,height,r_frame_rate,nb_frames:format=duration','-of','json',str(p)],text=True))
 s=probe['streams'][0];assert (s['width'],s['height'],s['r_frame_rate'])==(640,480,'10/1')
 duration=float(probe['format']['duration']);n=int(s['nb_frames']);frames+=n
 playlist.append("file '"+str(p).replace("'","'\\''")+"'")
 text=f'{index:02d}/15   CUP EVAL {ep:02d}   APPROACH FAILURE'
 filters.append(f"drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='{text}':fontsize=20:fontcolor=white:box=1:boxcolor=black@0.8:boxborderw=8:x=12:y=14:enable='gte(t,{elapsed:.6f})*lt(t,{elapsed+duration:.6f})'")
 chapters.extend(['[CHAPTER]','TIMEBASE=1/1000',f'START={round(elapsed*1000)}',f'END={round((elapsed+duration)*1000)}',f'title=Episode {ep:02d}'])
 manifest.append({'order':index,'episode':ep,'start_s':elapsed,'duration_s':duration,'source':str(p),'frames':n})
 elapsed+=duration
(R/'approach_15_concat.txt').write_text('\n'.join(playlist)+'\n')
(R/'approach_15_labels.ffscript').write_text(','.join(filters))
(R/'approach_15_chapters.ffmeta').write_text('\n'.join(chapters)+'\n')
output=R/'cup_approach_failures_15_in_order.mp4'
with (R/'approach_15_encode.log').open('w') as log:
 subprocess.run(['ffmpeg','-hide_banner','-y','-f','concat','-safe','0','-i',str(R/'approach_15_concat.txt'),'-i',str(R/'approach_15_chapters.ffmeta'),'-map','0:v:0','-map_metadata','1','-map_chapters','1','-filter_script:v',str(R/'approach_15_labels.ffscript'),'-c:v','libx264','-preset','veryfast','-crf','20','-threads','4','-pix_fmt','yuv420p','-an','-movflags','+faststart',str(output)],stdout=log,stderr=log,check=True)
probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_chapters','-show_format','-of','json',str(output)],text=True))
assert int(probe['streams'][0]['nb_frames'])==frames
assert abs(float(probe['format']['duration'])-elapsed)<.11
assert len(probe['chapters'])==15
info={'file':str(output),'episodes':ids,'frames':frames,'duration_s':elapsed,'chapters':manifest,'verified':True}
(R/'approach_15_playback_manifest.json').write_text(json.dumps(info,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in info.items() if k!='chapters'},ensure_ascii=False))
