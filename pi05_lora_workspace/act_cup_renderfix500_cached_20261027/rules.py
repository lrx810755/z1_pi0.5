"""Pure audit rules, with no simulator or training side effects."""
from collections import Counter
import numpy as np
PLAN_HASH='6640b372e672501d989500e69aa5fc676c46a47a9ffd6f9f4af3f71aa440f748'
EXPECTED_CATEGORIES={'approach':20,'shifted_target':20,'close_transition':15,'placement':15,'low_clearance':10,'regrasp':10,'transport':10}
HOLDOUT={(1,1),(2,3)}

def split_for_grid(grid):
 if len(grid)!=2 or not (0<=grid[0]<5 and 0<=grid[1]<4):raise ValueError('Invalid grid')
 return 'validation' if tuple(grid) in HOLDOUT else 'train'

def valid_anchors(expert,horizon=16):
 expert=np.asarray(expert,dtype=bool);out=np.zeros(len(expert),bool)
 if len(expert)>=horizon:
  bad=np.r_[0,np.cumsum(~expert)];out[:len(expert)-horizon+1]=(bad[horizon:]-bad[:-horizon])==0
 return out

def correction_labels(labels,mask,n):
 y=np.asarray(labels['labels'],dtype=np.float32)
 if y.shape!=(n,) or not np.isin(y,[0,1]).all():raise ValueError('Correction labels must be aligned binary intent')
 expert=np.asarray(mask['expert_frame_mask']);declared=np.asarray(mask['valid_action_horizon16_anchors'])
 if expert.shape!=(n,) or declared.shape!=(n,) or expert.dtype!=bool or declared.dtype!=bool:raise ValueError('Boolean aligned masks required')
 valid=valid_anchors(expert)
 if not np.array_equal(valid,declared):raise ValueError('Declared horizon16 mask differs from complete expert windows')
 return y,expert,valid

def validate_collection(basic_count,records):
 if basic_count!=400 or len(records)!=100:raise ValueError(f'Require EXACT 400 basic + 100 corrections; got {basic_count}+{len(records)}')
 if any(r['plan_hash']!=PLAN_HASH for r in records):raise ValueError('Correction plan hash mismatch')
 if Counter(r['kind'] for r in records)!=EXPECTED_CATEGORIES:raise ValueError('Correction categories mismatch')
 if len({r['task_id'] for r in records})!=100:raise ValueError('Duplicate correction tasks')

def validate_reference(grid,reference_grid,seed,reference_seed):
 if tuple(grid)!=tuple(reference_grid) or seed!=reference_seed:raise ValueError('Correction reference must use same grid and seed')

def feature_stats(values,mask):
 x=np.asarray(values)[mask].astype(np.float64)
 if not len(x):raise ValueError('No approved training values')
 return dict(mean=x.mean(0).tolist(),std=x.std(0).tolist(),min=x.min(0).tolist(),max=x.max(0).tolist(),count=[len(x)])

def full_labels(rows):
 """Single close/release basic demonstrations only; corrections use stored intent."""
 a=np.asarray([r['action'][-1] for r in rows]);strong=np.flatnonzero(a>-.75)
 if not len(strong):raise ValueError('No confirmed closing ramp')
 held=int(strong[0]);before=np.flatnonzero(a[:held]<=-.895)
 if not len(before):raise ValueError('No open command before close')
 close=int(before[-1]+1);fully=np.flatnonzero((np.arange(len(a))>held)&(a<=-.899))
 if not len(fully):raise ValueError('No final release')
 end=int(fully[0]);start=end
 while start>held and a[start]<a[start-1]-1e-4:start-=1
 release=start+1
 if not(close<held<release<=end) or np.max(a[end:])>=-.88:raise ValueError('Ambiguous opening')
 # A second closing segment would violate the single-close assumption.
 if np.any(a[held:release]<=-.899):raise ValueError('Multiple basic close/release cycles')
 labels=np.zeros(len(rows),np.float32);labels[close:release]=1
 return labels,close,release
