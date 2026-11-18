import json,subprocess,pathlib,cv2,time
root=pathlib.Path(__file__).resolve().parent
manifest=json.loads((root.parent/'split_manifest.json').read_text())
records=[r for r in manifest['source_records'] if r['episode_index'] in manifest['train_episodes']]
output=root/'act_cup_dynamic_training_224_5x.mp4'
log=(root/'train_5x_encode.log').open('w')
p=subprocess.Popen(['ffmpeg','-hide_banner','-y','-f','rawvideo','-pixel_format','bgr24','-video_size','640x480','-framerate','10','-i','pipe:0','-an','-c:v','libx264','-preset','ultrafast','-crf','21','-threads','4','-pix_fmt','yuv420p','-movflags','+faststart',str(output)],stdin=subprocess.PIPE,stdout=log,stderr=log)
chapters=[];count=0
try:
 for order,r in enumerate(records,1):
  src=pathlib.Path(r['source_path']);rows=[json.loads(line) for line in (src/'frames.jsonl').open()];start=count
  for i in range(0,len(rows),5):
   im=cv2.imread(str(src/rows[i]['observation.images.global']))
   if im is None:raise RuntimeError(f'Missing image {src} frame {i}')
   im=cv2.resize(im,(640,480));cv2.rectangle(im,(0,0),(640,54),(0,0,0),-1)
   cv2.putText(im,f'ACT CUP TRAIN {order:03d}/224 | 5x | {r["source"]}',(10,21),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
   label='CLOSE' if r['close_frame']<=i and (r['open_frame'] is None or i<r['open_frame']) else 'OPEN'
   cv2.putText(im,f'episode {r["episode_index"]:03d} | frame {i:03d}/{len(rows)} | label {label}',(10,44),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,220,255),1,cv2.LINE_AA)
   p.stdin.write(im.tobytes());count+=1
  chapters.append(dict(order=order,episode=r['episode_index'],source=r['source'],source_path=str(src),start_s=start/10,duration_s=(count-start)/10))
  if order%20==0:print(f'{order}/224',flush=True)
 p.stdin.close();code=p.wait()
 if code:raise RuntimeError(f'ffmpeg exited {code}')
except BaseException:
 p.kill();raise
probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=width,height,nb_frames:format=duration','-of','json',str(output)],text=True))
assert int(probe['streams'][0]['nb_frames'])==count
info=dict(file=str(output),episodes=len(records),speed=5,fps=10,frames=count,duration_s=float(probe['format']['duration']),source_frames=sum(r['frames'] for r in records),sampling='Every fifth source frame at 10 fps; per-episode tail rounded up to next output frame',verified=True,chapters=chapters)
(root/'train_5x_manifest.json').write_text(json.dumps(info,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in info.items() if k!='chapters'}),flush=True)
