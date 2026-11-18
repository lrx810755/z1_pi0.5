"""Each 16 anchors: 4/object, 3 basic + 1 correction; every pair differs."""
import numpy as np
from lerobot.datasets.sampler import EpisodeAwareSampler
from config import RUN

PAIRS=np.array([[0,1],[2,3],[0,2],[1,3],[0,3],[1,2],[0,1],[2,3]])

def draw_block(rng, pools, probabilities):
    objects=rng.permutation(4)[PAIRS[rng.permutation(8)]].reshape(-1)
    output=np.empty(16,dtype=np.int64)
    for obj in range(4):
        slots=np.flatnonzero(objects==obj);rng.shuffle(slots)
        for pool,positions in [(2*obj,slots[:3]),(2*obj+1,slots[3:])]:
            output[positions]=rng.choice(pools[pool],size=len(positions),replace=False,p=probabilities[pool])
    return output

class BalancedSampler(EpisodeAwareSampler):
    def __init__(self,*args,**kwargs):
        mapping=kwargs.pop('absolute_to_relative_idx',None)
        super().__init__(*args,absolute_to_relative_idx=None,**kwargs)
        with np.load(RUN/'sampling_arrays.npz') as a:
            ids=np.asarray(super().indices);ids=ids[a['eligible_train'][ids]]
            if mapping is not None:ids=np.array([i for i in ids if int(i) in mapping])
            self.pools=[];self.probabilities=[]
            for k in range(8):
                selected=ids[a['source_pool'][ids]==k]
                if len(selected)<3:raise ValueError('Too few eligible anchors in pool '+str(k))
                weights=a['weights'][selected];self.probabilities.append(weights/weights.sum())
                self.pools.append(np.array([mapping[int(i)] for i in selected]) if mapping is not None else selected)
        self._num_frames=max(16,((len(ids)+15)//16)*16)
    @property
    def indices(self):return np.concatenate(self.pools).tolist()
    def _iter_epoch(self,epoch,start):
        # Per-block seeds avoid regenerating all preceding blocks on resume.
        for block in range(start//16,self._num_frames//16):
            rng=np.random.default_rng(np.random.SeedSequence([self.seed,epoch,block]))
            order=draw_block(rng,self.pools,self.probabilities)
            for k in range(max(0,start-block*16),16):yield int(order[k])
