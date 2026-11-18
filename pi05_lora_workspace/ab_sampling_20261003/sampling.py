"""Sample intact observation/action windows by their anchor's phase."""
import numpy as np
import torch
from lerobot.datasets.sampler import EpisodeAwareSampler


def classify_frames(state, action):
    state,action=np.asarray(state),np.asarray(action)
    if state.shape!=action.shape or state.ndim!=2 or state.shape[1]!=7:
        raise ValueError('Expected matching Nx7 state/action arrays')
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError('Non-finite state/action')
    closed=np.flatnonzero(action[:,-1]>=-.75)
    if not len(closed):raise ValueError('No first closure in successful demonstration')
    close=int(closed[0]);weights=np.ones(len(state));labels=np.full(len(state),'normal',dtype='<U16')
    for i in range(close):
        lo,hi=max(0,i-2),min(close,i+3)
        if hi-lo<3:continue
        motion=np.ptp(state[lo:hi,:6],axis=0)
        if motion.max()<=.01 and np.all(action[lo:hi,-1]<-.85):
            weights[i]=.25;labels[i]='open_wait'
        elif motion[3:6].max()>=.02:
            weights[i]=2;labels[i]='wrist_approach'
    lo,hi=max(0,close-5),min(len(state),close+6)
    weights[lo:hi]=3;labels[lo:hi]='close_transition'
    return weights,labels


def sampler_class(global_weights):
    """Both arms use seeded replacement sampling; only probabilities differ."""
    global_weights=np.asarray(global_weights,dtype=np.float64)
    if global_weights.ndim!=1 or not np.isfinite(global_weights).all() or np.any(global_weights<=0):
        raise ValueError('Every source frame must have finite, positive sampling weight')
    class PhaseSampler(EpisodeAwareSampler):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            lengths=np.diff(np.r_[0,self._cum_lengths])
            absolute=np.concatenate([np.arange(s,s+n) for s,n in zip(self._starts,lengths)])
            self.weights=torch.as_tensor(global_weights[absolute].copy(),dtype=torch.double)
        def _iter_epoch(self,epoch,start):
            if not self.shuffle:
                yield from super()._iter_epoch(epoch,start)
                return
            order=torch.multinomial(self.weights,self._num_frames,replacement=True,generator=self._epoch_generator(epoch))
            for position in order[start:].tolist():yield self._frame_index(position)
    return PhaseSampler
