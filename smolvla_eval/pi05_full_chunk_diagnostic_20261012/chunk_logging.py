"""Record the complete postprocessed chunk before selecting its execution prefix."""
import json
import numpy as np

def record_prediction(chunk,steps,path,metadata):
 a=np.asarray(chunk)
 if a.ndim==3:
  if a.shape[0]!=1:raise ValueError('batch must be one')
  a=a[0]
 if a.ndim!=2 or a.shape[1]!=7 or not np.isfinite(a).all() or steps<=0:
  raise ValueError('invalid prediction chunk')
 n=min(steps,len(a));closed=a[:,-1]>=.5
 row=dict(metadata,actions=a.tolist(),chunk_length=len(a),execute_steps=n,gripper_raw=a[:,-1].tolist(),
  first_close_index=next((i for i,x in enumerate(closed) if x),None),
  executed_prefix_close_indices=[i for i in range(n) if closed[i]],
  discarded_tail_close_indices=[i for i in range(n,len(a)) if closed[i]],
  discarded_tail_open_indices=[i for i in range(n,len(a)) if not closed[i]])
 with path.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
 return a[:steps].copy()
