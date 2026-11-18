"""Pure safety, split, expert-window and sampling rules for 400+100+20."""
from collections import Counter
import numpy as np
HOLDOUT={(1,1),(2,3)}

def split_for_grid(grid):
 if len(grid)!=2 or any(type(x) not in (int,np.int64,np.int32) for x in grid) or not(0<=grid[0]<5 and 0<=grid[1]<4):raise ValueError('Invalid layout grid')
 return 'validation' if tuple(grid) in HOLDOUT else 'train'

def valid_anchors(expert,horizon=30):
 expert=np.asarray(expert)
 if expert.dtype!=bool or expert.ndim!=1 or horizon<1:raise ValueError('Boolean expert mask and positive horizon required')
 out=np.zeros(len(expert),bool)
 if len(expert)>=horizon:
  bad=np.r_[0,np.cumsum(~expert)];out[:len(expert)-horizon+1]=(bad[horizon:]-bad[:-horizon])==0
 return out

def source_balanced_weights(eligible,transition,pool,recovery):
 eligible=np.asarray(eligible,bool);transition=np.asarray(transition,bool);pool=np.asarray(pool);recovery=np.asarray(recovery,bool)
 if not(eligible.shape==transition.shape==pool.shape==recovery.shape) or eligible.ndim!=1:raise ValueError('Unaligned sampling arrays')
 if not np.isin(pool,[0,1]).all() or np.any(recovery&(pool==0)):raise ValueError('Recovery requires correction source pool')
 multipliers=np.maximum(np.where(transition,1.5,1.),np.where(recovery,3.,1.));w=np.zeros(len(pool),np.float64)
 for source,mass in enumerate([.75,.25]):
  selected=eligible&(pool==source)
  if not selected.any():raise ValueError('No eligible anchors in source pool')
  w[selected]=mass*multipliers[selected]/multipliers[selected].sum()
 return w

def validate_old_records(records):
 if len(records)!=500 or Counter(r['source'] for r in records)!={'basic':400,'correction':100}:raise ValueError('Require exact400 basic +100 old corrections')

def validate_new_collection(records,expected_count):
 if type(expected_count) is not int or not 16<=expected_count<=20 or len(records)!=expected_count:raise ValueError(f'Require verified count16..20 accepted policy recoveries, got {len(records)} expected {expected_count}')
 for r in records:
  if r.get('accepted') is not True or r.get('recovery_complete') is not True or r.get('source_type')!='actual_policy_push_recovery':raise ValueError('Unaccepted or nonactual policy recovery')
  if not r.get('source_identity'):raise ValueError('Missing independent source identity')
 if len({r['source_identity'] for r in records})!=expected_count:raise ValueError('Duplicated source policy episode')

def validate_split(records):
 layouts={}
 for r in records:
  seed=r['seed'];grid=tuple(r['grid'])
  if seed in layouts and layouts[seed]!=grid:raise ValueError('Same source seed must preserve its initial layout grid')
  layouts[seed]=grid
  if r['split']!=split_for_grid(r['grid']):raise ValueError('Spatial split differs from old heldout cells')
  if 'source_grid' in r and tuple(r['grid'])!=tuple(r['source_grid']):raise ValueError('Recovery source layout changed')
 for field in ['seed','source_frames_sha256']:
  if {r[field] for r in records if r['split']=='train'}&{r[field] for r in records if r['split']=='validation'}:raise ValueError(f'Train/validation {field} overlap')

def recovery_mask(rows,meta,expert):
 times=np.asarray([r['simulation_time'] for r in rows],np.float64);expert=np.asarray(expert,bool)
 if times.shape!=expert.shape or not np.isfinite(times).all() or np.any(np.diff(times)<0):raise ValueError('Invalid recovery timeline')
 intervals=[]
 def add(start,end,origin,stage=None):
  if not np.isfinite([start,end]).all() or end<start:raise ValueError('Invalid recovery interval')
  intervals.append(dict(start=float(start),end=float(end),origin=origin,**({'stage':stage} if stage else {})))
 if meta.get('source_type')=='actual_policy_push_recovery':
  if not meta.get('recovery_intervals_s'):raise ValueError('Explicit new recovery intervals required')
  for item in meta['recovery_intervals_s']:add(item['start'],item['end'],'explicit_expert_recovery')
 else:
  events=meta.get('events',[])
  def grasp_end(after):
   close=next((e['time'] for e in events if e.get('event')=='gripper_intent' and e.get('state')=='close' and e['time']>=after),None)
   if close is None:return after
   ids=[i for i,r in enumerate(rows) if times[i]>=close and r.get('physical_grasp_ok') is True and expert[i]]
   return float(times[ids[0]]+.3) if ids else after
  for stage in meta.get('recovery_stages',[]):add(stage['start'],max(stage['end'],grasp_end(stage['end'])),'recovery_stages',stage['stage'])
  for event in events:
   if event.get('event')=='retry_opened':add(event['time'],grasp_end(event['time']),'retry_opened')
   if event.get('event')=='loaded_recovery_start':
    end=next((e['time'] for e in events if e.get('event')=='loaded_recovery_complete' and e['time']>=event['time']),None)
    if end is None:raise ValueError('Incomplete loaded recovery')
    add(event['time'],end,'loaded_recovery')
 out=np.zeros(len(rows),bool)
 for interval in intervals:out|=(times>=interval['start']-1e-8)&(times<=interval['end']+1e-8)
 return out&expert,intervals

def feature_stats(values,mask):
 x=np.asarray(values)[mask].astype(np.float64)
 if not len(x):raise ValueError('No approved training values')
 return dict(mean=x.mean(0).tolist(),std=x.std(0).tolist(),min=x.min(0).tolist(),max=x.max(0).tolist(),count=[len(x)])
