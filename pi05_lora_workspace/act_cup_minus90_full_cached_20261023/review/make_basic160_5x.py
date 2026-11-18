"""Review all 160 actual training-input basic demonstrations, including validation."""
import json,subprocess,pathlib,cv2,hashlib,time
R=pathlib.Path(__file__).resolve().parent
training=json.loads((R.parent/'split_manifest.json').read_text())
records=[r for r in training['source_records'] if r['source']=='original']
assembly=json.loads(pathlib.Path('/home/lp531/smolvla_data/cup_minus90_rerecord_20261022/basic160/assembly_manifest.json').read_text())
assert len(records)==160 and len(assembly['episodes_mapping'])==160
assert [r['episode_index'] for r in records]==list(range(160))
output=R/'cup_basic160_5x.mp4';temporary=R/'cup_basic160_5x.encoding.mp4'
log=(R/'basic160_5x_encode.log').open('w');cv2.setNumThreads(2)
p=subprocess.Popen(['ffmpeg','-hide_banner','-y','-f','rawvideo','-pixel_format','bgr24','-video_size','640x480','-framerate','10','-i','pipe:0','-an','-c:v','libx264','-preset','veryfast','-crf','20','-threads','4','-pix_fmt','yuv420p',str(temporary)],stdin=subprocess.PIPE,stdout=log,stderr=log)
chapters=[];metadata=[';FFMETADATA1'];count=0;started=time.monotonic()
try:
 for order,(record,mapping) in enumerate(zip(records,assembly['episodes_mapping']),1):
  source=pathlib.Path(record['source_path'])
  assert source.resolve()==pathlib.Path(mapping['source_path']).resolve()
  raw=(source/'frames.jsonl').read_bytes();assert hashlib.sha256(raw).hexdigest()==record['source_frames_sha256']
  rows=[json.loads(line) for line in raw.splitlines()];start=count
  for i in range(0,len(rows),5):
   image=cv2.imread(str(source/rows[i]['observation.images.global']),cv2.IMREAD_COLOR)
   if image is None:raise RuntimeError(f'Unreadable frame: {source}, {i}')
   assert image.shape==(480,640,3)
   cv2.rectangle(image,(0,0),(640,54),(0,0,0),-1)
   grid=record['grid'];label='CLOSED' if record['close_frame']<=i<record['open_frame'] else 'OPEN'
   title=f'BASIC {order:03d}/160 | episode {record["episode_index"]:03d} | grid {grid[0]},{grid[1]} | 5x'
   cv2.putText(image,title,(10,21),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
   cv2.putText(image,f'frame {i:03d}/{len(rows)} | {label} | {record["split"]}',(10,44),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,220,255),1,cv2.LINE_AA)
   p.stdin.write(image.tobytes());count+=1
  chapters.append(dict(order=order,episode=record['episode_index'],grid=grid,source_path=str(source),replaced=mapping['replaced'],start_s=start/10,duration_s=(count-start)/10))
  metadata+=['[CHAPTER]','TIMEBASE=1/1000',f'START={start*100}',f'END={count*100}',f'title=Basic {order:03d} - episode {record["episode_index"]:03d} - grid {grid[0]},{grid[1]}']
  (R/'basic160_5x_progress.json').write_text(json.dumps(dict(completed=order,total=160,elapsed_s=time.monotonic()-started)))
  if order%20==0:print(f'{order}/160',flush=True)
 p.stdin.close();code=p.wait()
 if code:raise RuntimeError(f'ffmpeg exited {code}')
except BaseException:
 p.kill();p.wait();raise
meta=R/'basic160_5x_chapters.ffmeta';meta.write_text('\n'.join(metadata)+'\n')
subprocess.run(['ffmpeg','-hide_banner','-y','-i',str(temporary),'-i',str(meta),'-map','0:v:0','-map_metadata','1','-map_chapters','1','-c:v','copy','-movflags','+faststart',str(output)],stdout=log,stderr=log,check=True)
probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_chapters','-show_entries','stream=width,height,nb_frames:format=duration','-of','json',str(output)],text=True))
assert int(probe['streams'][0]['nb_frames'])==count and len(probe['chapters'])==160
assert abs(float(probe['format']['duration'])-count/10)<.11
temporary.unlink()
info=dict(file=str(output),episodes=160,replacements=36,retained=124,speed=5,fps=10,frames=count,duration_s=float(probe['format']['duration']),source_frames=sum(r['frames'] for r in records),sampling='Every fifth raw source frame at 10 fps, per-episode tail rounded up',includes_training_and_validation=True,verified=True,chapters=chapters)
(R/'basic160_5x_manifest.json').write_text(json.dumps(info,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in info.items() if k!='chapters'},ensure_ascii=False),flush=True)
