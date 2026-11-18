import numpy as np

def source_balanced_weights(eligible,key,pool):
 eligible=np.asarray(eligible,dtype=bool);key=np.asarray(key,dtype=bool);pool=np.asarray(pool)
 if not (eligible.shape==key.shape==pool.shape):raise ValueError('Unaligned sampling arrays')
 if not np.isin(pool,[0,1]).all():raise ValueError('Unknown source pool')
 w=np.zeros(len(pool),dtype=np.float64)
 for p,mass in enumerate([.75,.25]):
  mask=eligible&(pool==p)
  if not mask.any():raise ValueError(f'No approved training windows in source pool {p}')
  values=np.where(key[mask],1.5,1.);w[mask]=mass*values/values.sum()
 return w
