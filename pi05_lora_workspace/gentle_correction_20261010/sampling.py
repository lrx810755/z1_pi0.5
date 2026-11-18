"""Pool-normalized moderate weighting with deterministic sample-exact resume."""
from pathlib import Path
import json
import numpy as np
import torch
from lerobot.datasets.sampler import EpisodeAwareSampler
RUN=Path(__file__).resolve().parent

def probabilities(correction,key):
 correction=np.asarray(correction,dtype=bool);key=np.asarray(key,dtype=bool)
 weights=np.where(key,2.,1.)
 for group,mass in [(False,.7),(True,.3)]:
  mask=correction==group
  if not mask.any():raise ValueError('Both source pools must contain training windows')
  weights[mask]*=mass/weights[mask].sum()
 return weights

class GentleSampler(EpisodeAwareSampler):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  if not self.shuffle:return
  a=np.load(RUN/'sampling_arrays.npz')
  relative=self.indices
  inverse=None if self._absolute_to_relative is None else {v:k for k,v in self._absolute_to_relative.items()}
  absolute=np.array(relative if inverse is None else [inverse[i] for i in relative],dtype=int)
  self._probabilities=torch.tensor(probabilities(a['correction'][absolute],a['key'][absolute]),dtype=torch.double)
  self._relative=np.asarray(relative)
  records=json.loads((RUN/'phase_manifest.json').read_text())
  held={e['episode_index'] for e in records if e['split']=='validation'}
  assert not (set(a['episode'][absolute]) & held)
  p=self._probabilities.numpy();c=a['correction'][absolute];k=a['key'][absolute]
  report=dict(training_windows=len(p),original_mass=float(p[~c].sum()),correction_mass=float(p[c].sum()),key_mass=float(p[k].sum()),
              natural_key_fraction=float(k.mean()),within_pool_key_weight=2,loss_weight=1,heldout_excluded=True,
              pool_key_mass={name:float(p[(c==flag)&k].sum()) for name,flag in [('original',False),('correction',True)]})
  (RUN/'sampling_audit.json').write_text(json.dumps(report,indent=2))
 def _iter_epoch(self,epoch,start):
  if not self.shuffle:yield from super()._iter_epoch(epoch,start);return
  order=torch.multinomial(self._probabilities,self._num_frames,replacement=True,generator=self._epoch_generator(epoch))
  for k in range(start,self._num_frames):yield int(self._relative[int(order[k])])
