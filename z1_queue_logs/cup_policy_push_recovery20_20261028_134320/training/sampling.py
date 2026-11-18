"""Deterministic source-balanced sampling of complete, approved action windows."""
from pathlib import Path
import numpy as np
import torch
from lerobot.datasets.sampler import EpisodeAwareSampler
RUN=Path(__file__).resolve().parent

class BalancedSampler(EpisodeAwareSampler):
    def __init__(self,*args,**kwargs):
        mapping=kwargs.pop('absolute_to_relative_idx',None)
        super().__init__(*args,absolute_to_relative_idx=None,**kwargs)
        with np.load(RUN/'sampling_arrays.npz') as a:
            candidates=np.asarray(self._indices,dtype=np.int64) if hasattr(self,'_indices') else np.asarray(super().indices,dtype=np.int64)
            candidates=candidates[a['eligible_train'][candidates]]
            if mapping is not None:candidates=np.asarray([i for i in candidates if int(i) in mapping],dtype=np.int64)
            if not len(candidates):raise ValueError('No valid training anchors')
            self.absolute_indices=candidates
            self.relative_indices=np.asarray([mapping[int(i)] for i in candidates] if mapping is not None else candidates)
            self.probabilities=torch.tensor(a['weights'][candidates],dtype=torch.double)
            self.probabilities/=self.probabilities.sum()
        self._num_frames=len(candidates)

    @property
    def indices(self):return self.relative_indices.tolist()

    def _iter_epoch(self,epoch,start):
        order=torch.multinomial(self.probabilities,self._num_frames,replacement=True,generator=self._epoch_generator(epoch)).tolist()
        for k in range(start,self._num_frames):yield int(self.relative_indices[order[k]])
