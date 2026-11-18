import json,subprocess,io
from pathlib import Path
from PIL import Image,ImageDraw
OUT=Path(__file__).resolve().parent
metrics=json.loads((OUT/'metrics.json').read_text())
for m in metrics:
 p=Path(m['path'])/'videos/global_third_person.mp4'
 info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','json',str(p)]));duration=float(info['format']['duration'])
 canvas=Image.new('RGB',(1280,550),'#181818');draw=ImageDraw.Draw(canvas)
 for j,frac in enumerate([0,.2,.4,.6,.8,.98]):
  seconds=frac*duration
  data=subprocess.check_output(['ffmpeg','-loglevel','error','-ss',str(seconds),'-i',str(p),'-frames:v','1','-f','image2pipe','-vcodec','png','-'])
  im=Image.open(io.BytesIO(data));im.thumbnail((420,240));x=(j%3)*426;y=(j//3)*265+22
  canvas.paste(im,(x,y));draw.text((x+5,y-17),f"{m['run']} video {seconds:.1f}s",fill='white')
 canvas.save(OUT/(m['run']+'_overview.jpg'))
 # Latest and success: detailed 3 camera views, compare aligned frame indices.
 if m['run'] in ['001616','003437']:
  frames=[0,int(duration*10*.5),int(duration*10*.7),int(duration*10*.8),int(duration*10*.9),int(duration*10)-2]
  canvas=Image.new('RGB',(1280,len(frames)*263),'#181818');draw=ImageDraw.Draw(canvas)
  for row,n in enumerate(frames):
   for col,name in enumerate(['global_third_person.mp4','global_right.mp4','wrist.mp4']):
    data=subprocess.check_output(['ffmpeg','-loglevel','error','-ss',str(n/10),'-i',str(p.parent/name),'-frames:v','1','-f','image2pipe','-vcodec','png','-'])
    im=Image.open(io.BytesIO(data));im.thumbnail((420,236));x=col*426;y=row*263+22
    canvas.paste(im,(x,y));draw.text((x+3,y-17),f'{name} video {n/10:.1f}s',fill='white')
  canvas.save(OUT/(m['run']+'_three_views.jpg'))
print('contact sheets saved')
