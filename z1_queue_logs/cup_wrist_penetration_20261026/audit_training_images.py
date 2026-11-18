from pathlib import Path
import json,collections,numpy as np,datetime
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).parent/'training_audit';R.mkdir(exist_ok=True)
a=json.loads(Path('/home/lp531/pi05_lora_workspace/act_cup_minus90_full_cached_20261023/split_manifest.json').read_text())['source_records'];base=[r for r in a if r['source']=='original'];cells=collections.defaultdict(list)
for r in base:cells[tuple(r['grid'])].append(r)
chosen=[]
for cell,records in sorted(cells.items(),key=lambda x:(x[0][1],x[0][0])):
 # Alternate preference to cover retained and newly rerecorded basic demonstrations.
 replaced=[r for r in records if 'basic_replacements' in str(Path(r['source_path']).resolve())]
 retained=[r for r in records if 'basic_replacements' not in str(Path(r['source_path']).resolve())]
 chosen.append((replaced if (sum(cell)%2==0 and replaced) else retained or replaced)[0])
for name in ['old_correction','new_correction']:
 group=[r for r in a if r['source']==name]
 chosen.extend(group[i] for i in np.linspace(0,len(group)-1,10,dtype=int))
assert len(chosen)==40 and len({r['episode_index'] for r in chosen})==40
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16);checks=[];rows=[]
for rec in chosen:
 ep=Path(rec['source_path']);frames=[json.loads(l) for l in (ep/'frames.jsonl').read_text().splitlines()];page=Image.new('RGB',(1280,235),'#111');draw=ImageDraw.Draw(page);draw.text((5,3),f"Dataset {rec['episode_index']:03d} {rec['source']} grid={rec['grid']} split={rec['split']}",font=font,fill='white')
 for col,(phase,index) in enumerate([('APPROACH',max(0,rec['close_frame']-10)),('PRECLOSE',max(0,rec['close_frame']-1)),('CLOSE+5',min(len(frames)-1,rec['close_frame']+5)),('RELEASE-5',max(0,rec['open_frame']-5))]):
  path=ep/frames[index]['observation.images.wrist'];im=Image.open(path).convert('RGB');im.thumbnail((320,180));page.paste(im,(col*320,55));draw.text((col*320+5,30),f'{phase} frame {index}',font=font,fill='#ffff80');checks.append(dict(dataset_episode=rec['episode_index'],source=rec['source'],phase=phase,frame=index,path=str(path)))
 rows.append(page)
for start in range(0,40,5):
 pg=Image.new('RGB',(1280,235*5),'black')
 for i,row in enumerate(rows[start:start+5]):pg.paste(row,(0,235*i))
 pg.save(R/f'sample_{start:02d}_{start+4:02d}.jpg',quality=95)
summary=[]
for r in a:
 p=Path(r['source_path']);d=json.loads((p/'episode_start.json').read_text());summary.append(dict(episode=r['episode_index'],source=r['source'],recorded=datetime.datetime.fromtimestamp(d['started_at_unix_s']).astimezone().isoformat(),recorded_unix=d['started_at_unix_s'],resolved=str(p.resolve())))
report=dict(actual_training_episodes=len(a),sources=dict(collections.Counter(r['source'] for r in a)),recorded_time_range=[min(r['recorded'] for r in summary),max(r['recorded'] for r in summary)],sample_episodes=[r['episode_index'] for r in chosen],sample_frames=len(checks),checks=checks,all_episode_recording_times=summary)
(R/'sample_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['checks','all_episode_recording_times']},ensure_ascii=False,indent=2))
# Original three-view close-up for the retained basic demonstration 0.
rec=a[0];ep=Path(rec['source_path']);frames=[json.loads(l) for l in (ep/'frames.jsonl').read_text().splitlines()];idx=rec['close_frame']-1;pg=Image.new('RGB',(1920,515),'black');draw=ImageDraw.Draw(pg)
for i,cam in enumerate(['global','global_right','wrist']):
 im=Image.open(ep/frames[idx]['observation.images.'+cam]).convert('RGB');pg.paste(im,(640*i,35+(480-im.height)//2));draw.text((640*i+5,5),f'Dataset 000, frame {idx}, original {cam}',font=font,fill='white')
pg.save(R/'original_demo000_preclose_three_views.jpg',quality=97)
