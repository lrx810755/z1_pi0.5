from pathlib import Path
import json,cv2
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).parent;D=json.loads((R/'comparison.json').read_text());font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
for number in [6,28]:
 e=next(x for x in D['episodes'] if x['episode']==number)
 trace=[json.loads(l) for l in (Path(e['path'])/'trace.jsonl').read_text().splitlines()]
 top=max(trace,key=lambda t:(t['contact_status'].get('object_position_m') or [0,0,-999])[2])
 opens=[x for x in e['events'] if x['kind']=='open' and x['time']>e['first_physical_time']]
 events=[('first close',e['first_close']['time']),('reliable grasp',e['first_physical_time']),('max height',top['simulation_elapsed_s']),('release' if opens else 'late transport',opens[0]['time'] if opens else 57.),('final',trace[-1]['simulation_elapsed_s'])]
 times=[]
 for t in trace:
  if t['execution_substep']==0 and t['policy_action_index']==0:times.append(t['simulation_elapsed_s'])
  if t['execution_substep']==5:times.append(t['simulation_elapsed_s'])
 page=Image.new('RGB',(1920,3*318+35),'#101010');draw=ImageDraw.Draw(page);draw.text((5,5),f'EVAL {number:02d}: transport and release sequence',font=font,fill='#ffff80')
 for x,(label,when) in enumerate(events):
  index=min(range(len(times)),key=lambda i:abs(times[i]-when))
  for y,name in enumerate(['global_third_person.mp4','global_right.mp4','wrist.mp4']):
   cap=cv2.VideoCapture(str(Path(e['path'])/'videos'/name));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.set(cv2.CAP_PROP_POS_FRAMES,min(n-1,index));ok,f=cap.read();cap.release();assert ok
   im=Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB));im.thumbnail((384,288));left=x*384;top_y=35+y*318
   page.paste(im,(left+(384-im.width)//2,top_y+28+(288-im.height)//2));draw.text((left+3,top_y+4),f'{label} {when:.2f}s',font=font,fill='white')
 page.save(R/f'transport_{number:02d}.jpg',quality=94)
 print(number,'peak',top['simulation_elapsed_s'],top['contact_status']['object_position_m'])
