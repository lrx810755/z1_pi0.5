"""Single-close full demonstrations: preserve the proven previous-cup intent rule."""
import numpy as np

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
