"""Training uses a frozen snapshot of the verified fine-grip + recovery weights."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
RUN=Path(__file__).resolve().parent
# Import under a distinct module name to avoid this file's sampling module name.
import importlib.util
spec=importlib.util.spec_from_file_location('window_sampler_impl',RUN/'window_sampler_impl.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class GentleSampler(module.WindowSampler):
 def __init__(self,*args,**kwargs):
  arrays=RUN/'sampling_arrays.npz'
  super().__init__(*args,eligibility_path=arrays,**kwargs)
  with np.load(arrays) as a:
   absolute=self._absolute_indices;p=self._probabilities.numpy();ep=a['episode'][absolute]
   assert a['eligible_train_anchor'][absolute].all()
   records=json.loads((RUN/'phase_manifest.json').read_text());held={r['episode_index'] for r in records if r['split']=='validation'}
   assert not set(ep)&held
   report=dict(training_windows=len(absolute),original_mass=float(p[ep<160].sum()),correction_mass=float(p[ep>=160].sum()),
    recovery_mass=float(p[a['recovery'][absolute]].sum()),heldout_excluded=True,intervals_enforced=True,
    sampling_arrays_sha256=hashlib.sha256(arrays.read_bytes()).hexdigest(),
    phase_weight_max=float(a['phase_weight'].max()),loss_weight=1,recovery_weight=2,overlap_rule='maximum')
  (RUN/'sampling_audit.json').write_text(json.dumps(report,indent=2)+'\n')
