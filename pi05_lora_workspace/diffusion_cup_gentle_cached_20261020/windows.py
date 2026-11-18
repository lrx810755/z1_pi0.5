"""Diffusion horizon starts one frame before the current observation."""
import numpy as np
def indices(start,end,current):
 if not start<=current<end:raise ValueError('Anchor outside its episode')
 obs=np.maximum([current-1,current],start)
 action=np.arange(current-1,current+15)
 padding=(action<start)|(action>=end)
 return obs,np.clip(action,start,end-1),padding
