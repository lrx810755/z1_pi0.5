"""Export two full-length, synchronized three-camera reviews at exactly 3x."""
import json,subprocess,hashlib,cv2,time,os
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;HOME=Path('/home/lp531')
TRAIN=Path('/home/lp531/pi05_lora_workspace/act_cup_minus90_full_cached_20261023')
EVAL=Path('/home/lp531/smolvla_eval/act_cup_minus90_full_steps15_random30_20261023')
CAMERAS=['global','global_right','wrist'];LABELS=['GLOBAL LEFT','GLOBAL RIGHT','WRIST'];WIDTH=1920;HEIGHT=544;FPS=30
cv2.setNumThreads(2)
records=json.loads((TRAIN/'split_manifest.json').read_text())['source_records'];bygrid={}
for rec in records:
 if rec['source']=='original':bygrid.setdefault(tuple(rec['grid']),rec)
assert len(bygrid)==20
selected=[bygrid[(c,r)] for r in range(4) for c in range(5)]
evaluations=json.loads((EVAL/'final_box_results.json').read_text())['episodes'];ids=[1,3,6,8,13,16,19,21,22,24];evals=[e for e in evaluations if e['episode'] in ids];assert len(evals)==10

def header(title):
 out=np.zeros((64,WIDTH,3),np.uint8)
 cv2.putText(out,title,(12,24),cv2.FONT_HERSHEY_SIMPLEX,.72,(255,255,255),1,cv2.LINE_AA)
 for i,name in enumerate(LABELS):cv2.putText(out,name,(i*640+12,52),cv2.FONT_HERSHEY_SIMPLEX,.58,(0,220,255),1,cv2.LINE_AA)
 return out

def compose(images,head,index,total):
 out=np.zeros((HEIGHT,WIDTH,3),np.uint8);out[:64]=head
 for i,im in enumerate(images):
  h,w=im.shape[:2];assert w==640 and h in (360,480),(h,w)
  out[64+(480-h)//2:64+(480-h)//2+h,i*640:i*640+640]=im
 cv2.putText(out,f'frame {index:04d}/{total:04d}',(1600,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(200,200,200),1,cv2.LINE_AA)
 return out

def encode(kind,episodes,output):
 temporary=R/f'{kind}.encoding.mp4';log=(R/f'{kind}_encode.log').open('w')
 p=subprocess.Popen(['ffmpeg','-hide_banner','-y','-f','rawvideo','-pixel_format','bgr24','-video_size',f'{WIDTH}x{HEIGHT}','-framerate',str(FPS),'-i','pipe:0','-an','-c:v','libx264','-preset','veryfast','-crf','20','-threads','6','-pix_fmt','yuv420p',str(temporary)],stdin=subprocess.PIPE,stdout=log,stderr=log)
 chapters=[];meta=[';FFMETADATA1'];count=0;started=time.monotonic()
 try:
  for order,rec in enumerate(episodes,1):
   start=count
   if kind=='demonstrations':
    root=Path(rec['source_path']);raw=(root/'frames.jsonl').read_bytes();assert hashlib.sha256(raw).hexdigest()==rec['source_frames_sha256'];rows=[json.loads(l) for l in raw.splitlines()];n=len(rows)
    title=f'DEMONSTRATION {order:02d}/20 | grid ({rec["grid"][0]},{rec["grid"][1]}) | episode {rec["episode_index"]:03d} | {rec["split"].upper()} | 3x'
    head=header(title)
    for i,row in enumerate(rows):
     images=[cv2.imread(str(root/row['observation.images.'+cam]),cv2.IMREAD_COLOR) for cam in CAMERAS];assert all(im is not None for im in images),(root,i)
     p.stdin.write(compose(images,head,i,n).tobytes());count+=1
    info=dict(episode=rec['episode_index'],grid=rec['grid'],split=rec['split'],seed=rec['seed'],source=str(root),source_frames=n,source_fps=10)
   else:
    root=Path(rec['source_result']).parent;paths=[root/'videos'/name for name in ['global_third_person.mp4','global_right.mp4','wrist.mp4']];caps=[cv2.VideoCapture(str(path)) for path in paths];assert all(cap.isOpened() for cap in caps)
    lengths=[int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps];rates=[cap.get(cv2.CAP_PROP_FPS) for cap in caps];assert all(abs(rate-10)<1e-6 for rate in rates) and len(set(lengths))==1,(lengths,rates)
    n=lengths[0];title=f'EVALUATION {order:02d}/10 | trial {rec["episode"]:02d}/30 | {rec["legacy_failure_category"].upper()} | 3x';head=header(title)
    try:
     for i in range(n):
      frames=[cap.read() for cap in caps];assert all(ok for ok,im in frames),(root,i)
      p.stdin.write(compose([im for ok,im in frames],head,i,n).tobytes());count+=1
    finally:
     for cap in caps:cap.release()
    info=dict(episode=rec['episode'],seed=rec['seed'],category=rec['legacy_failure_category'],success=rec['success'],source=str(root),source_frames=n,source_fps=10)
   chapter=dict(order=order,start_s=start/FPS,duration_s=(count-start)/FPS,**info);chapters.append(chapter)
   meta+=['[CHAPTER]','TIMEBASE=1/30000',f'START={start*1000}',f'END={count*1000}',f'title={title}']
   (R/'progress.json').write_text(json.dumps(dict(stage=kind,completed=order,total=len(episodes),frames=count,elapsed_s=time.monotonic()-started)))
   print(kind,order,'/',len(episodes),flush=True)
  p.stdin.close();code=p.wait()
  if code:raise RuntimeError(f'ffmpeg exited {code}')
 except BaseException:
  p.kill();p.wait();raise
 metapath=R/f'{kind}_chapters.ffmeta';metapath.write_text('\n'.join(meta)+'\n')
 subprocess.run(['ffmpeg','-hide_banner','-y','-i',str(temporary),'-i',str(metapath),'-map','0:v:0','-map_metadata','1','-map_chapters','1','-c:v','copy','-movflags','+faststart',str(output)],stdout=log,stderr=log,check=True)
 probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_chapters','-show_entries','stream=width,height,nb_frames,r_frame_rate:format=duration','-of','json',str(output)],text=True));stream=probe['streams'][0]
 assert stream['width']==WIDTH and stream['height']==HEIGHT and int(stream['nb_frames'])==count and stream['r_frame_rate']=='30/1'
 assert len(probe['chapters'])==len(episodes) and abs(float(probe['format']['duration'])-count/FPS)<.05
 temporary.unlink()
 manifest=dict(file=str(output),verified=True,speed=3,source_fps=10,output_fps=FPS,frames=count,duration_s=float(probe['format']['duration']),views=CAMERAS,layout='three full views side by side; wrist letterboxed, no field-of-view crop',episodes=len(episodes),chapters=chapters)
 (R/f'{kind}_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in manifest.items() if k!='chapters'},ensure_ascii=False),flush=True)
 return manifest

one=encode('demonstrations',selected,HOME/'杯子基础示范_20格_三视角_3倍速.mp4')
two=encode('evaluations',evals,HOME/'杯子ACT评估_10次_三视角_3倍速.mp4')
(R/'completed.json').write_text(json.dumps(dict(complete=True,files=[one['file'],two['file']],demonstration_grid_count=20,evaluation_ids=ids,validation_grids_included=[[1,1],[2,3]],verified=True),ensure_ascii=False,indent=2))
