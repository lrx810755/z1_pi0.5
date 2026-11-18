"""Review only the nine completed fine-tuning diagnostics, excluding interrupted trial 10."""
import json,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent

def main():
 d=json.loads((R/'partial_comparison.json').read_text());episodes=d['episodes'];assert len(episodes)==9 and [e['order'] for e in episodes]==list(range(1,10))
 lines=[];filters=[];metadata=[';FFMETADATA1'];chapters=[];elapsed=0.
 for e in episodes:
  p=Path(e['source_result']).parent/'videos/global_third_person.mp4';r=json.loads(Path(e['source_result']).read_text());assert r.get('stop_reason')!='operator_stop'
  probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=width,height,r_frame_rate:format=duration','-of','json',str(p)],text=True));s=probe['streams'][0];assert (s['width'],s['height'],s['r_frame_rate'])==(640,480,'10/1');dur=float(probe['format']['duration'])
  lines.append("file '"+str(p).replace("'","'\\''")+"'")
  label=f'FINETUNED CUP {e["order"]:02d}/09 | demo {e["training_episode"]:03d} | {"SUCCESS" if e["new_success"] else "FAILED"} | 5x'
  filters.append(f"drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='{label}':fontsize=18:fontcolor=white:box=1:boxcolor=black@0.8:boxborderw=6:x=10:y=12:enable='gte(t,{elapsed:.6f})*lt(t,{elapsed+dur:.6f})'")
  metadata+=['[CHAPTER]','TIMEBASE=1/1000',f'START={round(elapsed*200)}',f'END={round((elapsed+dur)*200)}',f'title=Trial {e["order"]} - {e["category"]}']
  chapters.append(dict(order=e['order'],training_episode=e['training_episode'],start_s=elapsed/5,duration_s=dur/5,source=str(p),success=e['new_success']));elapsed+=dur
 (R/'review9_concat.txt').write_text('\n'.join(lines)+'\n');(R/'review9_filters.ffscript').write_text(','.join(filters+['setpts=(PTS-STARTPTS)/5']));(R/'review9_chapters.ffmeta').write_text('\n'.join(metadata)+'\n');output=R/'finetuned_cup_completed9_5x.mp4'
 with (R/'review9_encode.log').open('w') as log:
  subprocess.run(['ffmpeg','-hide_banner','-y','-f','concat','-safe','0','-i',str(R/'review9_concat.txt'),'-i',str(R/'review9_chapters.ffmeta'),'-map','0:v:0','-map_metadata','1','-map_chapters','1','-filter_script:v',str(R/'review9_filters.ffscript'),'-r','30','-c:v','libx264','-preset','veryfast','-crf','20','-threads','4','-pix_fmt','yuv420p','-an','-movflags','+faststart',str(output)],stdout=log,stderr=log,check=True)
 probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_chapters','-show_entries','format=duration','-of','json',str(output)],text=True));assert len(probe['chapters'])==9 and abs(float(probe['format']['duration'])-elapsed/5)<.2
 result=dict(file=str(output),speed=5,completed_trials=9,duration_s=float(probe['format']['duration']),chapters=chapters,verified=True);(R/'review9_manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='chapters'},indent=2))
if __name__=='__main__':main()
