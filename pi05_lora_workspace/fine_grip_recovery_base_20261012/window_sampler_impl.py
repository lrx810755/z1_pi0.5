"""Window-restricted sampler; no training is launched by importing it."""
from pathlib import Path
import numpy as np
import torch
from lerobot.datasets.sampler import EpisodeAwareSampler
W=Path(__file__).resolve().parent


class WindowSampler(EpisodeAwareSampler):
    """Compatible constructor with explicit eligibility. Apply windows before pool normalization.

    Default arrays refer only to the existing 192-episode dataset. New raw fragment masks
    must be mapped into the actual converted dataset before use; this class rejects a
    dataset whose episode boundaries differ from its arrays.
    """
    def __init__(self,*args,eligibility_path=W/'existing_192_sampling_arrays.npz',**kwargs):
        # The base sampler may receive a sparse absolute->relative dataset mapping.
        # Gather absolute candidates first, then translate only retained candidates.
        mapping=kwargs.pop('absolute_to_relative_idx',None)
        if len(args)>7:
            raise ValueError('Pass absolute_to_relative_idx by keyword')
        super().__init__(*args,absolute_to_relative_idx=None,**kwargs)
        with np.load(eligibility_path) as archive:
            a={k:archive[k] for k in archive.files}
        if len(a['episode'])!=len(a['eligible_train_anchor']):
            raise ValueError('Eligibility shape mismatch')
        ends=np.asarray(args[1] if len(args)>1 else kwargs['dataset_to_indices'])
        if not len(ends) or int(ends[-1])!=len(a['episode']) or len(ends)!=len(np.unique(a['episode'])):
            raise ValueError('Dataset size differs from eligibility arrays')
        # Episode-aware indices must refer to the dataset used to generate the mask.
        for j,(start,stop) in enumerate(zip(args[0] if args else kwargs['dataset_from_indices'],
                                          args[1] if len(args)>1 else kwargs['dataset_to_indices'])):
            start=int(start);stop=int(stop)
            if not (0<=start<stop<=len(a['episode'])):
                raise ValueError('Dataset bounds differ from eligibility arrays')
            if a['episode'][start]!=j or a['episode'][stop-1]!=j or a['frame'][start]!=0 or a['frame'][stop-1]!=stop-start-1:
                raise ValueError('Dataset episode identity differs from eligibility arrays')
        candidates=np.asarray(super().indices,dtype=np.int64)
        allowed=a['eligible_train_anchor'][candidates]
        if mapping is not None:
            allowed &= np.asarray([int(i) in mapping for i in candidates])
        self._absolute_indices=candidates[allowed]
        if not len(self._absolute_indices):
            raise ValueError('No training anchors remain inside sampling windows')
        self._relative=np.asarray([mapping[int(i)] for i in self._absolute_indices] if mapping is not None
                                  else self._absolute_indices,dtype=np.int64)
        self._num_frames=len(self._relative)
        self._absolute_to_relative=mapping
        weights=np.array(a['phase_weight'][self._absolute_indices],dtype=np.float64,copy=True)
        correction=a['correction'][self._absolute_indices]
        if not np.isfinite(weights).all() or (weights<=0).any():
            raise ValueError('Invalid phase weights')
        for pool,mass in [(False,.7),(True,.3)]:
            mask=correction==pool
            if not mask.any():
                if self.shuffle:raise ValueError('Both pools must remain for weighted training')
                continue
            weights[mask]*=mass/weights[mask].sum()
        weights/=weights.sum()
        self._probabilities=torch.tensor(weights,dtype=torch.double)

    @property
    def indices(self):return self._relative.tolist()

    def _iter_epoch(self,epoch,start):
        order=(torch.multinomial(self._probabilities,self._num_frames,replacement=True,
                                generator=self._epoch_generator(epoch)).tolist()
               if self.shuffle else range(self._num_frames))
        for k in range(start,self._num_frames):yield int(self._relative[order[k]])
