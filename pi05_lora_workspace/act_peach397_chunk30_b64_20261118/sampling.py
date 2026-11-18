"""Deterministic 48 basic +16 correction windows in every actual batch64."""
import numpy as np
from lerobot.datasets.sampler import EpisodeAwareSampler
from common import RUN


def draw_block(rng, pools, probabilities):
    result = np.r_[rng.choice(pools[0],48,replace=False,p=probabilities[0]),
                   rng.choice(pools[1],16,replace=False,p=probabilities[1])]
    rng.shuffle(result)
    return result


class BalancedSampler(EpisodeAwareSampler):
    def __init__(self, *args, **kwargs):
        mapping = kwargs.pop('absolute_to_relative_idx', None)
        super().__init__(*args, absolute_to_relative_idx=None, **kwargs)
        with np.load(RUN/'sampling_arrays.npz') as a:
            indices = np.asarray(super().indices)
            indices = indices[a['eligible_train'][indices]]
            self.pools, self.probabilities = [], []
            for pool in [0,1]:
                ids = indices[a['source_pool'][indices] == pool]
                if mapping is not None:
                    ids = np.array([i for i in ids if int(i) in mapping])
                if len(ids) < 64:
                    raise ValueError('Insufficient eligible anchors')
                weights = a['weights'][ids]
                self.probabilities.append(weights / weights.sum())
                self.pools.append(np.array([mapping[int(i)] for i in ids]) if mapping is not None else ids)
        self._num_frames = ((len(indices)+63)//64)*64

    @property
    def indices(self):
        return np.concatenate(self.pools).tolist()

    def _iter_epoch(self, epoch, start):
        for block in range(start//64, self._num_frames//64):
            rng = np.random.default_rng(np.random.SeedSequence([self.seed,epoch,block]))
            order = draw_block(rng,self.pools,self.probabilities)
            for offset in range(max(0,start-block*64),64):
                yield int(order[offset])
