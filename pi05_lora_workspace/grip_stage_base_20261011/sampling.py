"""Pool-normalized phase sampling with deterministic sample-exact resume."""
from pathlib import Path
import json
import numpy as np
import torch
from lerobot.datasets.sampler import EpisodeAwareSampler
from phases import probabilities

RUN = Path(__file__).resolve().parent


class GentleSampler(EpisodeAwareSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.shuffle:
            return
        a = np.load(RUN / 'sampling_arrays.npz')
        relative = self.indices
        inverse = None if self._absolute_to_relative is None else {v: k for k, v in self._absolute_to_relative.items()}
        absolute = np.array(relative if inverse is None else [inverse[i] for i in relative], dtype=int)
        correction = a['correction'][absolute]
        phase = a['phase_weight'][absolute]
        self._probabilities = torch.tensor(probabilities(correction, phase), dtype=torch.double)
        self._relative = np.asarray(relative)
        records = json.loads((RUN / 'phase_manifest.json').read_text())
        held = {r['episode_index'] for r in records if r['split'] == 'validation'}
        assert not (set(a['episode'][absolute]) & held)
        p = self._probabilities.numpy()
        report = dict(training_windows=len(p), original_mass=float(p[~correction].sum()),
                      correction_mass=float(p[correction].sum()), heldout_excluded=True, loss_weight=1,
                      within_pool_weights={'other': 1, 'preclose_1s': 1.5, 'close_to_stable_lift': 2},
                      key_mass=float(p[phase != 1].sum()), natural_key_fraction=float((phase != 1).mean()),
                      phase_mass={str(w): float(p[phase == w].sum()) for w in [1., 1.5, 2.]},
                      pool_phase_mass={name: {str(w): float(p[(correction == flag) & (phase == w)].sum())
                                             for w in [1., 1.5, 2.]}
                                       for name, flag in [('original', False), ('correction', True)]})
        (RUN / 'sampling_audit.json').write_text(json.dumps(report, indent=2))

    def _iter_epoch(self, epoch, start):
        if not self.shuffle:
            yield from super()._iter_epoch(epoch, start)
            return
        order = torch.multinomial(self._probabilities, self._num_frames, replacement=True,
                                  generator=self._epoch_generator(epoch))
        for k in range(start, self._num_frames):
            yield int(self._relative[int(order[k])])
