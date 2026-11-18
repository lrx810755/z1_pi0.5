from pathlib import Path
import json,collections
import numpy as np
P=Path(__file__).resolve().parent;W=Path('/home/lp531/pi05_lora_workspace');records=json.loads((W/'grip_stage_base_20261011/phase_manifest.json').read_text());rows=[]
for r in records:
 root=Path(r['source_path']);sc=json.loads((root/'episode_start.json').read_text())['scene_randomization'];pos=sc['object_positions_m']['beverage_bottle'];g=sc['target_grid'];c=json.loads((root/'correction.json').read_text()) if r['source']=='correction' else None
 rows.append(dict(episode=r['episode_index'],source=r['source'],split=r['split'],position=pos,grid=g,frames=r['frames'],kind=c['task']['kind'] if c else None,offset_approach_m=c['task']['offset_approach_m'] if c else None,rotation_tool_deg=c['task']['rotation_tool_deg'] if c else None))
old=np.load(W/'gentle_correction_base_20261010/sampling_arrays.npz');new=np.load(W/'grip_stage_base_20261011/sampling_arrays.npz');held=[r['episode'] for r in rows if r['split']=='validation'];train=~np.isin(old['episode'],held)
def probabilities(weights):
 p=np.zeros(len(weights))
 for flag,mass in [(False,.7),(True,.3)]:
  mask=train&(old['correction']==flag);p[mask]=weights[mask]*mass/weights[mask].sum()
 return p
po=probabilities(np.where(old['key'],2.,1.));pn=probabilities(new['phase_weight']);cells=[]
for y in range(5):
 for x in range(5):
  rr=[r for r in rows if r['grid']==[x,y]];ids=[r['episode'] for r in rr];mask=np.isin(old['episode'],ids);cmask=mask&old['correction'];cells.append(dict(grid=[x,y],x_range=[.36+x*.066,.36+(x+1)*.066],y_range=[-.27+y*.086,-.27+(y+1)*.086],original=sum(r['source']=='original' for r in rr),correction_train=sum(r['source']=='correction' and r['split']=='train' for r in rr),correction_validation=sum(r['split']=='validation' for r in rr),old_total_mass=float(po[mask].sum()),new_total_mass=float(pn[mask].sum()),old_correction_mass=float(po[cmask].sum()),new_correction_mass=float(pn[cmask].sum())))
summary={}
for name,rr in [('original',[r for r in rows if r['source']=='original']),('correction_all',[r for r in rows if r['source']=='correction']),('correction_train',[r for r in rows if r['source']=='correction' and r['split']=='train'])]:
 summary[name]=dict(episodes=len(rr),occupied_cells=len(set(tuple(r['grid']) for r in rr)),x_range=[min(r['position'][0] for r in rr),max(r['position'][0] for r in rr)],y_range=[min(r['position'][1] for r in rr),max(r['position'][1] for r in rr)],by_x_grid=dict(collections.Counter(r['grid'][0] for r in rr)),by_y_grid=dict(collections.Counter(r['grid'][1] for r in rr)),kinds=dict(collections.Counter(r['kind'] for r in rr if r['kind'])))
out=dict(summary=summary,cells=cells,episodes=rows,note='Positions are initial bottle positions; correction kind is a gripper offset category, not bottle location. Sampling masses are expected normalized anchor probabilities; not actual sampled counts. Coordinate ranges use current workspace .36-.69 and -.27-.16, 5x5.')
(P/'coverage.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print(json.dumps(summary,indent=2));print('occupied correction cells',[(c['grid'],c['correction_train'],round(c['old_correction_mass']*100,2),round(c['new_correction_mass']*100,2)) for c in cells if c['correction_train']]);print('empty correction in original', [c['grid'] for c in cells if c['original'] and not c['correction_train']])
